"""
utils/fk_numpy.py — PUMA 560 Forward Kinematics in pure NumPy.

This module is used during:
  - Dataset generation (fast, no PyTorch overhead)
  - Validation of IK solutions
  - Unit tests (compared against roboticstoolbox)

The exact same mathematics are replicated in fk_torch.py for gradient-based
training. Both must produce identical results — enforced by test_fk.py.

Convention: Standard DH (Denavit-Hartenberg)
  T_i = Rz(theta_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DH_PARAMS, JOINT_LIMITS, POSE_DIM, JOINT_DIM


# ── Single-link DH transform ──────────────────────────────────────────────────

def dh_transform_numpy(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Compute a single DH link homogeneous transform (4x4).

    Standard DH convention:
        T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)

    Args:
        a     : link length (metres)
        d     : link offset (metres)
        alpha : link twist (radians)
        theta : joint angle (radians) — the variable for revolute joints

    Returns:
        T : (4, 4) numpy array, dtype float64
    """
    ct = np.cos(theta);  st = np.sin(theta)
    ca = np.cos(alpha);  sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa,  a * ct],
        [st,  ct * ca, -ct * sa,  a * st],
        [0.0,       sa,       ca,       d],
        [0.0,      0.0,      0.0,     1.0],
    ], dtype=np.float64)


# ── Full-chain FK ─────────────────────────────────────────────────────────────

def fk_numpy(q: np.ndarray) -> np.ndarray:
    """Compute PUMA 560 end-effector transform from joint angles.

    Args:
        q : (6,) array of joint angles in radians

    Returns:
        T : (4, 4) homogeneous transform — end-effector in world frame
    """
    q = np.asarray(q, dtype=np.float64)
    assert q.shape == (JOINT_DIM,), f"Expected q shape ({JOINT_DIM},), got {q.shape}"

    T = np.eye(4, dtype=np.float64)
    for i in range(JOINT_DIM):
        a, d, alpha, theta_offset = DH_PARAMS[i]
        Ti = dh_transform_numpy(a, d, alpha, q[i] + theta_offset)
        T = T @ Ti
    return T


def fk_numpy_batch(Q: np.ndarray) -> np.ndarray:
    """Batched FK for multiple joint configurations.

    Args:
        Q : (N, 6) array of joint angle sets

    Returns:
        Ts : (N, 4, 4) array of homogeneous transforms
    """
    Q = np.asarray(Q, dtype=np.float64)
    assert Q.ndim == 2 and Q.shape[1] == JOINT_DIM
    N = Q.shape[0]
    Ts = np.empty((N, 4, 4), dtype=np.float64)
    for i in range(N):
        Ts[i] = fk_numpy(Q[i])
    return Ts


# ── Pose vector encoding / decoding ──────────────────────────────────────────

def T_to_pose_vector(T: np.ndarray) -> np.ndarray:
    """Extract a 9-dimensional pose vector from a 4x4 SE(3) transform.

    Representation: [x, y, z, r11, r21, r31, r12, r22, r32]
      - Position (3 values): the translation column
      - Rotation columns 0 and 1 (6 values): first two columns of R

    Why not Euler angles or quaternions?
      - Euler angles have gimbal lock and discontinuities.
      - Quaternions have a sign ambiguity (q and -q represent the same rotation).
      - The first two rotation columns uniquely determine R (the third is their
        cross product) and are smooth everywhere on SO(3).
      - This makes the cINN conditioning signal globally differentiable.

    Args:
        T : (4, 4) homogeneous transform

    Returns:
        v : (9,) float64 pose vector
    """
    pos    = T[:3, 3]      # [x, y, z]
    r_col0 = T[:3, 0]      # [r11, r21, r31]
    r_col1 = T[:3, 1]      # [r12, r22, r32]
    return np.concatenate([pos, r_col0, r_col1]).astype(np.float64)


def pose_vector_to_T(v: np.ndarray) -> np.ndarray:
    """Reconstruct a (4,4) SE(3) transform from a 9-dim pose vector.

    The third rotation column is recovered as the cross product of the first two,
    which enforces SO(3) membership (det=1, orthonormal).

    Args:
        v : (9,) pose vector [x, y, z, r11, r21, r31, r12, r22, r32]

    Returns:
        T : (4, 4) homogeneous transform
    """
    v = np.asarray(v, dtype=np.float64)
    assert v.shape == (POSE_DIM,), f"Expected (9,), got {v.shape}"

    pos    = v[0:3]
    r_col0 = v[3:6]
    r_col1 = v[6:9]

    # Gram-Schmidt: orthonormalise the two columns, then compute third via cross
    r0 = r_col0 / (np.linalg.norm(r_col0) + 1e-12)
    r1 = r_col1 - np.dot(r_col1, r0) * r0   # subtract projection onto r0
    r1 = r1 / (np.linalg.norm(r1) + 1e-12)
    r2 = np.cross(r0, r1)

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = r0
    T[:3, 1] = r1
    T[:3, 2] = r2
    T[:3, 3] = pos
    return T


# ── Rotation error metric ─────────────────────────────────────────────────────

def rotation_geodesic_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angular distance between two rotation matrices (radians).

    Uses: angle = arccos( (trace(R1^T R2) - 1) / 2 )
    Numerically clamped to [0, π].

    Args:
        R1, R2 : (3, 3) rotation matrices

    Returns:
        angle : scalar in [0, π] radians
    """
    R_rel = R1.T @ R2
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    # Avoid tiny non-zero angles from floating point noise when rotations match.
    if cos_angle >= 1.0 - 1e-12:
        return 0.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


# ── IK solution validation ────────────────────────────────────────────────────

def validate_ik_solution(
    q_sol: np.ndarray,
    T_target: np.ndarray,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    margin: float = 1e-3,
) -> tuple[bool, float, float]:
    """Validate a proposed IK solution for geometric and limit correctness.

    Checks (in order, short-circuits on first failure):
      1. No NaN or Inf in q_sol
      2. All joints within limits (with margin)
      3. FK round-trip position error < pos_tol
      4. FK round-trip rotation error < rot_tol

    Args:
        q_sol    : (6,) proposed joint angles
        T_target : (4,4) target end-effector transform
        pos_tol  : metres
        rot_tol  : radians
        margin   : joint-limit safety margin (radians)

    Returns:
        (is_valid, pos_error, rot_error)
    """
    if not np.all(np.isfinite(q_sol)):
        return False, np.inf, np.inf

    lo = JOINT_LIMITS[0] + margin
    hi = JOINT_LIMITS[1] - margin
    if np.any(q_sol < lo) or np.any(q_sol > hi):
        return False, np.inf, np.inf

    T_check = fk_numpy(q_sol)
    pos_err = float(np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]))
    rot_err = rotation_geodesic_error(T_check[:3, :3], T_target[:3, :3])

    if pos_err > pos_tol or rot_err > rot_tol:
        return False, pos_err, rot_err

    return True, pos_err, rot_err


# ── Manipulability ────────────────────────────────────────────────────────────

def jacobian_numpy(q: np.ndarray) -> np.ndarray:
    """Compute the 6×6 geometric Jacobian of PUMA 560 numerically.

    Uses central finite differences with a small perturbation.
    Accurate to O(h²).  Used only for manipulability scoring at inference —
    not in the training loop (where we don't need gradients through J).

    Args:
        q : (6,) joint angles

    Returns:
        J : (6, 6) geometric Jacobian  [v; w] columns
    """
    h = 1e-6
    J = np.zeros((6, 6), dtype=np.float64)
    T0 = fk_numpy(q)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]

    for i in range(JOINT_DIM):
        q_plus  = q.copy(); q_plus[i]  += h
        q_minus = q.copy(); q_minus[i] -= h
        T_plus  = fk_numpy(q_plus)
        T_minus = fk_numpy(q_minus)

        # Linear velocity Jacobian column
        J[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * h)

        # Angular velocity Jacobian column via skew-symmetric part of dR/dq
        dR = (T_plus[:3, :3] - T_minus[:3, :3]) / (2 * h)
        S  = dR @ R0.T   # skew-symmetric matrix
        J[3, i] = S[2, 1]
        J[4, i] = S[0, 2]
        J[5, i] = S[1, 0]

    return J


def manipulability(q: np.ndarray) -> float:
    """Yoshikawa manipulability measure: sqrt(det(J J^T)).

    Higher value = further from singularity = better.
    Returns 0.0 at singularities (det = 0).

    Args:
        q : (6,) joint angles

    Returns:
        w : scalar manipulability index >= 0
    """
    J = jacobian_numpy(q)
    val = np.linalg.det(J @ J.T)
    return float(np.sqrt(max(val, 0.0)))


def joint_limit_margin(q: np.ndarray) -> float:
    """Minimum normalised distance from any joint limit.

    Returns a value in [0, 0.5]:  0 = at a limit,  0.5 = perfectly centred.

    Args:
        q : (6,) joint angles

    Returns:
        margin : scalar in [0, 0.5]
    """
    lo   = JOINT_LIMITS[0]
    hi   = JOINT_LIMITS[1]
    span = hi - lo
    # Fraction of range from each boundary
    frac_lo = (q - lo) / span
    frac_hi = (hi - q) / span
    return float(np.min(np.minimum(frac_lo, frac_hi)))
