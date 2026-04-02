"""
train_puma560_v4.py  —  v4.0  (Physics-Informed Decoupled Architecture)
========================================================================
ANN Training + Multi-Method IK Comparison for PUMA 560

WHY v1/v2/v3 ALL FAILED (Root Cause Analysis)
----------------------------------------------
All previous versions tried to learn a direct 12-input → 6-joint mapping.
This is fundamentally the WRONG problem formulation. Here is why:

  1. The PUMA 560 IK has a closed-form DECOUPLED structure:
       - J1,J2,J3 are determined SOLELY by the wrist center position (3 numbers).
       - J4,J5,J6 are determined ANALYTICALLY from T3_6 = inv(T03) @ T06
         once J1,J2,J3 are known.
     This is EXACTLY what the MATLAB iPUMA.m function does analytically.

  2. J6 is a pure wrist spin — it does NOT affect end-effector position.
     J5 and J4 barely affect position either (they move the wrist orientation).
     ALL position error comes from J1, J2, J3.

  3. Trying to learn J4, J5, J6 as regression targets is wrong because:
       - They are geometrically determined exactly by T3_6 (no learning needed).
       - J6 in [-180, 180] and J4 in [-110, 170] have a wrist-coupling ambiguity.
       - Learning them from 12 inputs conflates position and orientation.

  4. Early stopping at epoch 424/3000 means the network HIT A WALL:
     it cannot reduce val-loss further because the problem is ill-posed.

THE CORRECT APPROACH (v4)
-------------------------
STAGE 1: ANN predicts only J1, J2, J3 (the shoulder joints).
         Input: wrist center P5 = [Px - d6*ax, Py - d6*ay, Pz - d6*az]  (3 values)
         Output: sin/cos of J1, J2, J3  (6 values)
         Loss: MSE on sin/cos + differentiable FK wrist-center loss
         This is a clean, deterministic, well-conditioned regression.

STAGE 2: Compute J4, J5, J6 ANALYTICALLY from T3_6 = inv(T03) @ T06.
         This gives machine-precision results — zero error.

Combined result: position error = position error from J1,J2,J3 alone.
At J1,J2,J3 RMSE = 0.2 deg → pos error ~ 3 mm.
At J1,J2,J3 RMSE = 0.1 deg → pos error ~ 1.4 mm.

The ANN also includes a physics residual correction via the analytical
IK (iPUMA logic) to further reduce error post-prediction.

ALSO INCLUDED
-------------
- Full Python re-implementation of iPUMA.m analytical IK (exact solution)
- Improved Jacobian GD with adaptive LM + backtracking line search
- Multi-start GD baseline
- Full comparison tables

Requirements:
    pip install numpy scipy torch rich matplotlib

Run:
    python train_puma560_v4.py
    python train_puma560_v4.py --epochs 5000 --patience 300 --n-eval 200
"""

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.optimize import differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
#  PUMA 560 DH PARAMETERS  (matching MATLAB fPUMA.m / iPUMA.m exactly)
# ═══════════════════════════════════════════════════════════════════════

# Standard DH:  T_i = Rot_z(theta_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(alpha_i)
# Notation: (a_mm, d_mm, alpha_deg)
DH = [
    (0.0,       671.83,  -90.0),   # Link 1
    (431.8,     139.70,    0.0),   # Link 2
    (-20.32,      0.0,   90.0),   # Link 3
    (0.0,       431.8,   -90.0),   # Link 4
    (0.0,         0.0,   90.0),   # Link 5
    (0.0,        56.5,    0.0),   # Link 6
]

# Alias DH scalars for clarity
D1, D2, D4, D6 = DH[0][1], DH[1][1], DH[3][1], DH[5][1]
A2, A3         = DH[1][0], DH[2][0]

JOINT_LIMITS = np.array([
    [-160,  160],
    [-225,   45],
    [ -45,  225],
    [-110,  170],
    [-100,  100],
    [-266,  266],
], dtype=np.float64)

R_WORKSPACE = 900.0   # mm, normalisation scale


# ═══════════════════════════════════════════════════════════════════════
#  FORWARD KINEMATICS  (numpy, matches fPUMA.m exactly)
# ═══════════════════════════════════════════════════════════════════════

def _dh_np(a, d, alpha_deg, theta_deg):
    al = math.radians(alpha_deg); th = math.radians(theta_deg)
    ca, sa = math.cos(al), math.sin(al)
    ct, st = math.cos(th), math.sin(th)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,    ca,    d ],
        [ 0,      0,     0,    1 ],
    ])


def fPUMA(thetas_deg):
    """FK for PUMA 560. thetas_deg: array-like of 6 angles in degrees."""
    T = np.eye(4)
    for i, (a, d, alpha) in enumerate(DH):
        T = T @ _dh_np(a, d, alpha, thetas_deg[i])
    return T


def T0_3_np(theta123_deg):
    """FK for first 3 joints only → T_{0→3}."""
    T = np.eye(4)
    for i in range(3):
        T = T @ _dh_np(DH[i][0], DH[i][1], DH[i][2], theta123_deg[i])
    return T


# ═══════════════════════════════════════════════════════════════════════
#  DIFFERENTIABLE FK  (PyTorch batched, matches numpy exactly)
# ═══════════════════════════════════════════════════════════════════════

def _dh_torch(a, d, alpha_deg, theta_batch):
    """theta_batch: [B] tensor in degrees. Returns [B,4,4]."""
    al = math.radians(alpha_deg)
    ca, sa = math.cos(al), math.sin(al)
    ca = torch.tensor(ca, dtype=theta_batch.dtype, device=theta_batch.device)
    sa = torch.tensor(sa, dtype=theta_batch.dtype, device=theta_batch.device)
    a_t = torch.tensor(float(a), dtype=theta_batch.dtype, device=theta_batch.device)
    d_t = torch.tensor(float(d), dtype=theta_batch.dtype, device=theta_batch.device)

    th = theta_batch * (math.pi / 180.0)
    ct, st = torch.cos(th), torch.sin(th)
    B = theta_batch.shape[0]

    T = torch.zeros(B, 4, 4, dtype=theta_batch.dtype, device=theta_batch.device)
    T[:, 0, 0] = ct;          T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa;     T[:, 0, 3] = a_t * ct
    T[:, 1, 0] = st;          T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa;    T[:, 1, 3] = a_t * st
    T[:, 2, 1] = sa;          T[:, 2, 2] = ca
    T[:, 2, 3] = d_t
    T[:, 3, 3] = 1.0
    return T


def fPUMA_torch_3(theta123):
    """Differentiable FK for first 3 joints. theta123: [B,3] degrees → [B,4,4]."""
    B = theta123.shape[0]
    T = torch.eye(4, dtype=theta123.dtype, device=theta123.device).unsqueeze(0).expand(B, -1, -1).clone()
    for i in range(3):
        Ti = _dh_torch(DH[i][0], DH[i][1], DH[i][2], theta123[:, i])
        T = torch.bmm(T, Ti)
    return T


def wrist_center_torch(T06_flat):
    """
    Compute wrist center from flat 12-element row [nx,ny,nz,ox,...,Pz].
    P5 = P - d6 * a_vec.
    T06_flat: [B, 12]  → returns [B, 3]
    """
    ax = T06_flat[:, 6]; ay = T06_flat[:, 7]; az = T06_flat[:, 8]
    Px = T06_flat[:, 9]; Py = T06_flat[:, 10]; Pz = T06_flat[:, 11]
    d6 = torch.tensor(D6, dtype=T06_flat.dtype, device=T06_flat.device)
    return torch.stack([Px - d6*ax, Py - d6*ay, Pz - d6*az], dim=1)


# ═══════════════════════════════════════════════════════════════════════
#  ANALYTICAL WRIST SOLVER  (exact J4,J5,J6 given J1,J2,J3 and T06)
# ═══════════════════════════════════════════════════════════════════════

def solve_wrist(theta123_deg, T06, flip_wrist=False):
    """
    Given J1,J2,J3 and the target T06, compute exact J4,J5,J6.
    Returns (theta4, theta5, theta6) in degrees.
    flip_wrist=True gives the alternate wrist solution (theta5 < 0).
    """
    T0_3 = T0_3_np(theta123_deg)
    T3_6 = np.linalg.inv(T0_3) @ T06

    sin5_sq = max(0.0, 1.0 - T3_6[2, 2]**2)
    sin5    = math.sqrt(sin5_sq)

    if sin5 < 1e-6:
        # Wrist singularity: theta5 = 0, J4+J6 determined but not individually
        theta5 = 0.0
        theta4 = 0.0
        theta6 = math.degrees(math.atan2(T3_6[1, 0], T3_6[0, 0]))
    else:
        if flip_wrist:
            sin5 = -sin5
        theta5 = math.degrees(math.atan2(sin5, T3_6[2, 2]))
        theta4 = math.degrees(math.atan2(T3_6[1, 2], T3_6[0, 2]))
        theta6 = math.degrees(math.atan2(T3_6[2, 1], -T3_6[2, 0]))

    return theta4, theta5, theta6


# ═══════════════════════════════════════════════════════════════════════
#  FULL ANALYTICAL IK  (Python port of iPUMA.m — configs 1-4)
#  Returns exact solution or None if out of workspace/limits.
# ═══════════════════════════════════════════════════════════════════════

def analytical_ik(T06, configs=(1, 2, 3, 4)):
    """
    Analytical IK for PUMA 560, mirroring iPUMA.m logic.
    Returns (theta_6, pos_err_mm) for best valid config, or (None, inf).
    """
    ax, ay, az = T06[0, 2], T06[1, 2], T06[2, 2]
    Px, Py, Pz = T06[0, 3], T06[1, 3], T06[2, 3]

    # Wrist center
    P5x = Px - D6 * ax
    P5y = Py - D6 * ay
    P5z = Pz - D6 * az

    # Radial distance and shoulder offset
    C1_rad = math.sqrt(P5x**2 + P5y**2)
    if C1_rad < 1e-6:
        return None, np.inf

    D_ratio = D2 / C1_rad
    if abs(D_ratio) > 1.0:
        return None, np.inf

    alpha1 = math.degrees(math.atan2(D_ratio, math.sqrt(1 - D_ratio**2)))
    phi1   = math.degrees(math.atan2(P5y, P5x))
    phi3   = math.degrees(math.atan2(A3, D4))

    best_J   = None
    best_err = np.inf

    for cfg in configs:
        shoulder_idx = (cfg - 1) // 4   # 0=right, 1=left
        elbow_idx    = ((cfg - 1) % 4) // 2  # 0=down, 1=up
        wrist_idx    = (cfg - 1) % 2   # 0=no-flip, 1=flip

        # Theta1
        if shoulder_idx == 0:   # right
            theta1 = phi1 - alpha1
        else:                   # left
            theta1 = phi1 + alpha1 - 180.0
        theta1 = _wrap180(theta1)

        # Theta2, theta3
        t1r = math.radians(theta1)
        C1  = P5x * math.cos(t1r) + P5y * math.sin(t1r)
        C2  = P5z - D1
        C3  = math.sqrt(C1**2 + C2**2)
        C4  = math.sqrt(A3**2 + D4**2)

        D_a = (C3**2 + A2**2 - C4**2) / (2 * A2 * C3)
        D_b = (A2**2 + C4**2 - C3**2) / (2 * A2 * C4)
        D_a = max(-1.0, min(1.0, D_a))
        D_b = max(-1.0, min(1.0, D_b))

        phi2 = math.degrees(math.atan2(C2, C1))

        if elbow_idx == 0:  # down
            alpha2 = math.degrees(math.atan2( math.sqrt(1-D_a**2), D_a))
            beta   = math.degrees(math.atan2( math.sqrt(1-D_b**2), D_b))
        else:               # up
            alpha2 = math.degrees(math.atan2(-math.sqrt(1-D_a**2), D_a))
            beta   = math.degrees(math.atan2(-math.sqrt(1-D_b**2), D_b))

        theta2 = alpha2 - phi2
        theta3 = beta - 90.0 - phi3

        # Wrist angles
        flip = (wrist_idx == 1)
        theta4, theta5, theta6 = solve_wrist([theta1, theta2, theta3], T06, flip_wrist=flip)

        J = np.array([theta1, theta2, theta3, theta4, theta5, theta6])

        # Check joint limits
        if not np.all((J >= JOINT_LIMITS[:, 0]) & (J <= JOINT_LIMITS[:, 1])):
            continue

        # Verify with FK
        T_check = fPUMA(J)
        err = np.linalg.norm(T_check[:3, 3] - T06[:3, 3])
        if err < best_err:
            best_err = err
            best_J   = J

    return best_J, best_err


def _wrap180(angle):
    return ((angle + 180.0) % 360.0) - 180.0


# ═══════════════════════════════════════════════════════════════════════
#  ROTATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def geodesic_rot_err(R_pred, R_tgt):
    t = np.clip((np.trace(R_pred.T @ R_tgt) - 1) / 2, -1.0, 1.0)
    return math.degrees(math.acos(t))


def rot_vec_err(R, R_tgt):
    R_err = R_tgt @ R.T
    tv    = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    angle = math.acos(tv)
    if abs(angle) < 1e-10: return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:
        M = (R_err + np.eye(3)) / 2
        cols = np.linalg.norm(M, axis=0)
        return angle * M[:, np.argmax(cols)] / (cols.max() + 1e-12)
    return (angle / (2*math.sin(angle))) * np.array([
        R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])


# ═══════════════════════════════════════════════════════════════════════
#  IMPROVED JACOBIAN IK  (adaptive LM, backtracking)
# ═══════════════════════════════════════════════════════════════════════

def jacobian_ik(J_init, T_tgt, max_iter=500, pos_tol=0.1, rot_tol=0.001,
                delta=1.0, lam0=1.0):
    """
    Levenberg-Marquardt IK — correct sign convention.

    BUG FIXED vs v1-v3: Previous code computed Jac = de/dJ = -dT/dJ (residual
    Jacobian) then used J += (Jac^T Jac + lam)^{-1} Jac^T e.  Because
    Jac = -dT/dJ, Jac^T e points in the direction that INCREASES ||e||, so
    the step diverged.  Fix: negate to get the forward Jacobian neg_Jac = dT/dJ,
    then J += (neg_Jac^T neg_Jac + lam)^{-1} neg_Jac^T e correctly steps toward
    the target.

    Also removed the /R_WORKSPACE normalisation that was crushing the gradient
    to ~1e-4 and causing the backtracking to always fail.  Raw mm/rad with
    separate weights gives a well-conditioned Jacobian.
    """
    W_pos = 1.0    # mm weight
    W_rot = 50.0   # rad weight (balances ~500mm arm length)

    J   = np.array(J_init, dtype=np.float64)
    lam = lam0

    for _ in range(max_iter):
        T     = fPUMA(J)
        e_pos = T_tgt[:3, 3] - T[:3, 3]
        e_rot = rot_vec_err(T[:3, :3], T_tgt[:3, :3])
        pos_err = np.linalg.norm(e_pos)
        if pos_err < pos_tol and np.linalg.norm(e_rot) < rot_tol:
            break

        e    = np.concatenate([W_pos * e_pos, W_rot * e_rot])
        err0 = pos_err   # backtrack on position error

        # Forward Jacobian neg_Jac[:,k] = dT/dJ_k  (negation of residual Jacobian)
        neg_Jac = np.zeros((6, 6))
        for k in range(6):
            Jp  = J.copy(); Jp[k] += delta
            Tp  = fPUMA(Jp)
            ep  = np.concatenate([
                W_pos * (T_tgt[:3, 3] - Tp[:3, 3]),
                W_rot * rot_vec_err(Tp[:3, :3], T_tgt[:3, :3]),
            ])
            neg_Jac[:, k] = -(ep - e) / delta   # negate to get dT/dJ

        try:
            dJ = np.linalg.solve(neg_Jac.T @ neg_Jac + lam * np.eye(6),
                                  neg_Jac.T @ e)
        except np.linalg.LinAlgError:
            lam *= 10; continue

        # Backtracking line search on actual position error
        alpha = 1.0
        pos_new = pos_err
        for _ in range(10):
            J_new   = J + alpha * dJ
            T_new   = fPUMA(J_new)
            pos_new = np.linalg.norm(T_tgt[:3, 3] - T_new[:3, 3])
            if pos_new < err0: break
            alpha *= 0.5

        J   = J + alpha * dJ
        lam = max(lam / 3, 1e-7) if pos_new < err0 else min(lam * 5, 1e2)

    T_f = fPUMA(J)
    return J, np.linalg.norm(T_tgt[:3, 3] - T_f[:3, 3])


def multistart_gd(T_tgt, n_starts=8):
    rng = np.random.default_rng(42)
    seeds = [JOINT_LIMITS.mean(1)]
    seeds += [JOINT_LIMITS[:,0] + rng.random(6)*(JOINT_LIMITS[:,1]-JOINT_LIMITS[:,0])
              for _ in range(n_starts - 1)]
    best_J, best_err = seeds[0], np.inf
    for s in seeds:
        J, err = jacobian_ik(s, T_tgt)
        if err < best_err:
            best_err, best_J = err, J
        if best_err < 0.5: break
    return best_J, best_err


# ═══════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def row_to_T(row):
    r = row
    return np.array([[r[0],r[3],r[6],r[9]],[r[1],r[4],r[7],r[10]],
                     [r[2],r[5],r[8],r[11]],[0,0,0,1]])


def compute_wrist_center_np(X12):
    """X12: [N,12] → P5: [N,3]"""
    ax,ay,az = X12[:,6], X12[:,7], X12[:,8]
    Px,Py,Pz = X12[:,9], X12[:,10], X12[:,11]
    return np.stack([Px-D6*ax, Py-D6*ay, Pz-D6*az], axis=1)


def angles_to_sc(angles_deg):
    """[...,K] → [..., 2K] sin/cos interleaved."""
    r = np.deg2rad(angles_deg)
    sc = np.zeros((*angles_deg.shape[:-1], angles_deg.shape[-1]*2))
    for j in range(angles_deg.shape[-1]):
        sc[..., 2*j]   = np.sin(r[..., j])
        sc[..., 2*j+1] = np.cos(r[..., j])
    return sc


def sc_to_angles(sc):
    """[..., 2K] → [..., K] in degrees ∈ [-180,180]."""
    K = sc.shape[-1] // 2
    a = np.zeros((*sc.shape[:-1], K))
    for j in range(K):
        a[..., j] = np.rad2deg(np.arctan2(sc[..., 2*j], sc[..., 2*j+1]))
    return a


def sc_to_angles_torch(sc):
    K = sc.shape[-1] // 2
    a = torch.zeros(*sc.shape[:-1], K, dtype=sc.dtype, device=sc.device)
    for j in range(K):
        a[..., j] = torch.atan2(sc[..., 2*j], sc[..., 2*j+1]) * (180.0/math.pi)
    return a


def wrap_angle_error(pred, true):
    """Signed angle difference in [-180,180], works with numpy arrays."""
    diff = pred - true
    return ((diff + 180.0) % 360.0) - 180.0


# ═══════════════════════════════════════════════════════════════════════
#  NEURAL NETWORK: 3 → J1,J2,J3 as sin/cos (6 outputs)
#  Input: wrist center (P5x, P5y, P5z) — 3 values
#  This is the correct sufficient statistic for J1,J2,J3.
# ═══════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))


class ShoulderNet(nn.Module):
    """
    Predicts sin/cos of J1,J2,J3 from wrist center (3 inputs).
    Uses a narrow-deep architecture suited to this low-dimensional problem.
    """
    def __init__(self, n_in=3, hidden=256, n_blocks=6, dropout=0.05):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Linear(hidden, 6)   # sin/cos for J1,J2,J3
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)  # raw, apply tanh outside


# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM LOSS: MSE on sin/cos + FK wrist-center loss
# ═══════════════════════════════════════════════════════════════════════

class DecoupledIKLoss(nn.Module):
    """
    L = w_sc  * MSE(pred_sc, true_sc)            # sincos regression
      + w_wc  * MSE(T03(pred_J123)[:3,3], P5)    # wrist center via FK
      + w_circ* mean_j(sin²+cos²-1)²             # unit-circle constraint
    """
    def __init__(self, w_sc=1.0, w_wc=2.0, w_circ=0.05):
        super().__init__()
        self.w_sc   = w_sc
        self.w_wc   = w_wc
        self.w_circ = w_circ

    def forward(self, pred_raw, target_sc, P5_target):
        pred_sc = torch.tanh(pred_raw)

        # 1. sin/cos MSE
        loss_sc = F.mse_loss(pred_sc, target_sc)

        # 2. Recover J1,J2,J3 from predicted sin/cos
        pred_J123 = sc_to_angles_torch(pred_sc)   # [B,3] degrees

        # 3. Differentiable FK to get T03 frame 3 origin
        T03 = fPUMA_torch_3(pred_J123)              # [B,4,4]
        # Wrist center position from T03 (frame 3 origin in base coords)
        # Actually we need the T0_3 position: T03[:, :3, 3]
        # But that's the origin of frame 3, not the wrist center P5.
        # The wrist center in our decoupled model IS what T03 should place at P5.
        # Specifically: P5 = T03 * [0; d4; 0; 1] for PUMA 560? No.
        # Actually P5 = T0_5[:3,3] = T03 * T34 * T45[:3,3]
        # Simpler: we can compare wrist center predicted from pred_J123 to true P5.
        # Compute T03's z-column offset at d4: this is the wrist center.
        # For PUMA 560: the wrist center is T0_4[:3,3]
        # T0_4 = T03 * _dh(DH[3])
        # But DH[3] has theta4 which we don't know yet.
        # The wrist center does NOT depend on theta4 because:
        # P5 = T03 * [A3, D4*cos(alpha3), D4*sin(alpha3), 1]^T ... hmm
        # Let's use a simpler geometric fact:
        # P5 = T03[:3,:3] @ [0,0,D4] + T03[:3,3] (wrist on z-axis of frame3)
        # (because frame 3's z-axis points to frame 4 origin at distance D4 along frame3's z)
        # Actually for PUMA 560 the wrist center is at T03 * [A3, 0, D4, 1]^T
        # Let me just use the FK position error of the entire arm using predicted J123
        # and compute where the wrist center WOULD be.
        
        # The 4th DH: a=0, d=D4, alpha=-90. theta4 doesn't affect position of frame4 origin
        # when a4=0: the x-offset is 0. So T04[:3,3] = T03[:3,3] + T03[:3,2]*D4
        # (translation along z of frame 3 by D4)
        # And T05[:3,3] = T04[:3,3] + T04[:3,2]*0 = T04[:3,3] (d5=0)
        # So P5 = T03[:3,3] + D4 * T03[:3,2]
        
        P5_pred = T03[:, :3, 3] + D4 * T03[:, :3, 2]   # [B,3]
        
        # Normalize by workspace radius
        Rw = torch.tensor(R_WORKSPACE, dtype=pred_raw.dtype, device=pred_raw.device)
        loss_wc = F.mse_loss(P5_pred / Rw, P5_target / Rw)

        # 3. Unit-circle penalty
        loss_circ = sum(
            ((pred_sc[:, 2*j]**2 + pred_sc[:, 2*j+1]**2 - 1.0)**2).mean()
            for j in range(3)
        ) / 3.0

        return self.w_sc * loss_sc + self.w_wc * loss_wc + self.w_circ * loss_circ


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train_model(model, X_tr, Y_tr_sc, P5_tr, X_val, Y_val_sc, P5_val,
                epochs, lr, batch_size, patience, device, console):
    model = model.to(device)
    f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    X_tr_t,  Y_tr_t,  P5_tr_t  = f32(X_tr),  f32(Y_tr_sc),  f32(P5_tr)
    X_val_t, Y_val_t, P5_val_t = f32(X_val), f32(Y_val_sc), f32(P5_val)

    loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t, P5_tr_t),
                        batch_size=batch_size, shuffle=True,
                        pin_memory=(device.type == 'cuda'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    n_steps   = epochs * max(1, len(loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=n_steps,
        pct_start=0.05, anneal_strategy='cos')

    criterion = DecoupledIKLoss(w_sc=1.0, w_wc=2.0, w_circ=0.05)
    val_crit  = nn.MSELoss()

    hist = {"train": [], "val": [], "best_epoch": 0, "best_val": float("inf")}
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    if HAS_RICH and console:
        prog = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]ShoulderNet v4[/bold cyan]"),
            BarColumn(bar_width=36), TaskProgressColumn(), MofNCompleteColumn(),
            TimeElapsedColumn(), TimeRemainingColumn(),
            TextColumn("[yellow]{task.fields[info]}[/yellow]"),
            console=console, refresh_per_second=4,
        )
        task = prog.add_task("train", total=epochs, info="starting…")
        prog.start()
    else:
        prog = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb, p5b in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb, p5b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= max(1, len(X_tr_t))

        model.eval()
        with torch.no_grad():
            pred_val = torch.tanh(model(X_val_t))
            val_loss = val_crit(pred_val, Y_val_t).item()

        hist["train"].append(ep_loss)
        hist["val"].append(val_loss)

        if val_loss < hist["best_val"] - 1e-9:
            hist["best_val"]   = val_loss
            hist["best_epoch"] = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if prog:
            prog.update(task, advance=1, info=(
                f"train={ep_loss:.5f}  val={val_loss:.5f}  best_ep={hist['best_epoch']}"))
        elif ep % max(1, epochs // 20) == 0:
            el = time.time()-t0
            print(f"  ep {ep:>5}/{epochs}  loss={ep_loss:.5f}  val={val_loss:.5f}"
                  f"  {el:.0f}s/{el/ep*(epochs-ep):.0f}s ETA")

        if no_improve >= patience:
            if prog: prog.stop(); console.print(f"  [green]Early stop ep {ep}[/green]")
            else:    print(f"  Early stop ep {ep}")
            break

    if prog:
        try: prog.stop()
        except: pass
    return best_state, hist


# ═══════════════════════════════════════════════════════════════════════
#  INFERENCE: ANN → J123 → analytical J456 → full solution
# ═══════════════════════════════════════════════════════════════════════

def predict_ik(model, X_eval_n, X_eval_raw, return_J123_only=False):
    """
    Run the full decoupled IK:
      1. ANN predicts J1,J2,J3 from normalised wrist-center inputs.
      2. Solve J4,J5,J6 analytically from T3_6.
    Returns [N,6] joint angles in degrees.
    """
    model.eval()
    with torch.no_grad():
        sc_pred = torch.tanh(model(torch.tensor(X_eval_n, dtype=torch.float32))).numpy()
    J123_pred = sc_to_angles(sc_pred)   # [N,3]

    if return_J123_only:
        return J123_pred

    N = len(X_eval_raw)
    J_full = np.zeros((N, 6))
    for i in range(N):
        T06 = row_to_T(X_eval_raw[i])
        j123 = J123_pred[i]
        t4, t5, t6 = solve_wrist(j123, T06, flip_wrist=False)
        J_full[i] = [j123[0], j123[1], j123[2], t4, t5, t6]
    return J_full


def predict_ik_refined(model, X_eval_n, X_eval_raw,
                       try_analytical=True, refine_gd=False):
    """
    Full pipeline:
      1. ANN → J123 → analytical J456
      2. Optionally try analytical IK to verify / improve
      3. Optionally refine with GD warm-started from ANN solution
    """
    N = len(X_eval_raw)
    J_ann = predict_ik(model, X_eval_n, X_eval_raw)
    J_out = J_ann.copy()
    pos_out = np.zeros(N)

    for i in range(N):
        T06 = row_to_T(X_eval_raw[i])
        T_ann = fPUMA(J_out[i])
        pos_out[i] = np.linalg.norm(T_ann[:3,3] - T06[:3,3])

        if try_analytical:
            J_ana, err_ana = analytical_ik(T06, configs=(1,2,3,4))
            if J_ana is not None and err_ana < pos_out[i]:
                J_out[i] = J_ana
                pos_out[i] = err_ana

        if refine_gd and pos_out[i] > 1.0:
            J_gd, err_gd = jacobian_ik(J_out[i], T06, max_iter=300)
            if err_gd < pos_out[i]:
                J_out[i] = J_gd
                pos_out[i] = err_gd

    return J_out, pos_out


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def fmt_time(ms):
    return f"{ms/1000:.2f} s" if ms >= 1000 else f"{ms:.3f} ms"


def prt_row(nm, pe, re, t_ms, n_eval):
    pef = pe[np.isfinite(pe)]; ref = re[np.isfinite(re)]
    if len(pef) == 0:
        print(f"  {nm:<32} | {'N/A':>8} {'N/A':>8} {'N/A':>8} | "
              f"{'N/A':>8} {'N/A':>8} | {fmt_time(t_ms):>11} | N/A    N/A"); return
    print(f"  {nm:<32} | {pef.mean():8.3f} {np.median(pef):8.3f} {pef.max():8.3f} | "
          f"{ref.mean():8.3f} {np.median(ref):8.3f} | "
          f"{fmt_time(t_ms):>11} | "
          f"{100*np.sum(pef<1)/n_eval:5.1f}% {100*np.sum(pef<5)/n_eval:5.1f}%")


class _Prog:
    def __init__(self, p, tid): self._p = p; self._task_id = tid
    def start(self):             self._p.start()
    def stop(self):              self._p.stop()
    def update(self, tid, **kw): self._p.update(tid, **kw)


def make_prog(label, total, console):
    if not (HAS_RICH and console): return None
    p = Progress(TextColumn(f"  [cyan]{label}[/cyan]"), BarColumn(bar_width=34),
                 TaskProgressColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                 console=console, transient=True)
    return _Prog(p, p.add_task(label, total=total))


# ═══════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════

def save_plots(hist, Y_test, J123_pred_test, Y_true_J123,
               all_pos, all_rot, all_t, mnames, n_eval, console):
    if not HAS_MPL:
        print("  matplotlib not available — skipping plots."); return

    def cp(m): (console.print(m) if (console and HAS_RICH) else print(m))
    def _save(fig, fname):
        fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        cp(f"  Saved: {fname}")

    bc = ["#3366CC","#E59A1A","#CC3333","#9933CC","#33B255","#E55A1A"]

    # 1 — training history
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(hist["train"], label="Train", color="#3366CC", lw=2)
    ax.semilogy(hist["val"],   label="Val",   color="#CC3333", lw=2, ls="--")
    be = hist["best_epoch"] - 1
    if 0 <= be < len(hist["val"]):
        ax.axvline(be, color="k", ls=":", lw=1.5, label=f"Best (ep {be+1})")
    ax.set_xlabel("Epoch", fontweight="bold"); ax.set_ylabel("MSE (sin/cos)")
    ax.set_title(f"ShoulderNet v4 | Best ep {hist['best_epoch']} "
                 f"| Val MSE {hist['best_val']:.6f}", fontweight="bold")
    ax.legend(); ax.grid(True); _save(fig, "ann_training_history.png")

    # 2 — J1,J2,J3 error (wrapped)
    jl = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
    wrapped_err = np.array([wrap_angle_error(J123_pred_test[:, j], Y_true_J123[:, j])
                            for j in range(3)])
    mae  = np.mean(np.abs(wrapped_err), axis=1)
    rmse = np.sqrt(np.mean(wrapped_err**2, axis=1))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for j, ax in enumerate(axes):
        ax.hist(wrapped_err[j], bins=60, color=bc[j], edgecolor="none", alpha=0.85)
        ax.axvline(0, color="r", lw=1.5)
        ax.set_xlabel(f"Wrapped Error {jl[j]} (deg)", fontweight="bold")
        ax.set_title(f"J{j+1} | MAE={mae[j]:.3f}°  RMSE={rmse[j]:.3f}°", fontweight="bold")
        ax.grid(True)
    fig.suptitle("ShoulderNet J1,J2,J3 Error (wrapped) — Full Test Set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); _save(fig, "shoulder_joint_error.png")

    # 3 — comparison bar charts
    n = len(mnames)
    fin = lambda a: a[np.isfinite(a)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    pm = [fin(p).mean() if len(fin(p)) else 0 for p in all_pos]
    ps = [fin(p).std()  if len(fin(p)) else 0 for p in all_pos]
    axes[0].bar(range(n), pm, color=bc[:n]); axes[0].errorbar(range(n), pm, ps, fmt="k.", lw=1.2)
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels(mnames, rotation=30, ha="right")
    axes[0].set_ylabel("Mean Pos Error (mm)", fontweight="bold")
    axes[0].set_title("Position Accuracy", fontweight="bold"); axes[0].grid(True)

    rm = [fin(r).mean() if len(fin(r)) else 0 for r in all_rot]
    axes[1].bar(range(n), rm, color=bc[:n])
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(mnames, rotation=30, ha="right")
    axes[1].set_ylabel("Mean Rot Error (deg)", fontweight="bold")
    axes[1].set_title("Rotation Accuracy", fontweight="bold"); axes[1].grid(True)

    axes[2].bar(range(n), all_t, color=bc[:n]); axes[2].set_yscale("log")
    axes[2].set_xticks(range(n)); axes[2].set_xticklabels(mnames, rotation=30, ha="right")
    axes[2].set_ylabel("Time/sample ms (log)", fontweight="bold")
    axes[2].set_title("Speed", fontweight="bold"); axes[2].grid(True)

    s1 = [100*np.sum(fin(p)<1)/n_eval if len(fin(p)) else 0 for p in all_pos]
    s5 = [100*np.sum(fin(p)<5)/n_eval if len(fin(p)) else 0 for p in all_pos]
    x = np.arange(n); w = 0.35
    axes[3].bar(x-w/2, s1, w, label="<1mm", color="#3375E8")
    axes[3].bar(x+w/2, s5, w, label="<5mm", color="#AACCFF")
    axes[3].set_xticks(x); axes[3].set_xticklabels(mnames, rotation=30, ha="right")
    axes[3].set_ylabel("Success Rate (%)", fontweight="bold")
    axes[3].set_title("Success Rate", fontweight="bold")
    axes[3].legend(loc="lower left"); axes[3].set_ylim(0, 105); axes[3].grid(True)
    fig.suptitle("PUMA 560 IK Method Comparison v4", fontsize=12, fontweight="bold")
    fig.tight_layout(); _save(fig, "method_comparison.png")

    # 4 — CDF
    fig, ax = plt.subplots(figsize=(8, 5))
    for pe, col, nm in zip(all_pos, bc, mnames):
        pf = np.sort(pe[np.isfinite(pe)])
        if len(pf):
            ax.plot(pf, np.arange(1, len(pf)+1)/n_eval*100, color=col, lw=2.2, label=nm)
    ax.axvline(1, color="k", ls="--", lw=1, label="1mm")
    ax.axvline(5, color="k", ls=":", lw=1, label="5mm")
    ax.set_xlabel("Position Error (mm)", fontweight="bold")
    ax.set_ylabel("Cumulative %", fontweight="bold"); ax.set_title("CDF v4")
    ax.legend(loc="lower right"); ax.grid(True); _save(fig, "cdf_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PUMA 560 IK v4.0 — Decoupled Architecture")
    parser.add_argument("--dataset",   default="puma560_dataset.csv")
    parser.add_argument("--epochs",    type=int,   default=3000)
    parser.add_argument("--lr",        type=float, default=3e-3)
    parser.add_argument("--batch",     type=int,   default=512)
    parser.add_argument("--patience",  type=int,   default=300)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--n-blocks",  type=int,   default=6)
    parser.add_argument("--n-eval",    type=int,   default=200)
    parser.add_argument("--n-seeds",   type=int,   default=3)
    parser.add_argument("--gd-starts", type=int,   default=8)
    parser.add_argument("--no-gpu",    action="store_true")
    parser.add_argument("--no-plots",  action="store_true")
    args = parser.parse_args()

    console = Console() if HAS_RICH else None
    def cp(m="", s=""): (console.print(m, style=s) if (console and HAS_RICH) else print(m))

    SEP  = "=" * 70
    SEP2 = "=" * 130
    DIV  = "-" * 130
    HDR  = (f"  {'Method':<32} | {'PosMean':>8} {'PosMed':>8} {'PosMax':>8} | "
            f"{'RotMean':>8} {'RotMed':>8} | {'Time/smp':>11} | {'<1mm':>6} {'<5mm':>6}")

    cp(SEP)
    cp("PUMA 560 IK Trainer  v4.0  (Physics-Informed Decoupled Architecture)")
    cp(SEP); cp()
    cp("Architecture (NEW):")
    cp("  ANN predicts ONLY J1,J2,J3 from wrist center P5 (3 inputs → 6 sin/cos)")
    cp("  J4,J5,J6 solved ANALYTICALLY from T3_6 = inv(T03) @ T06 → zero error")
    cp("  Training loss = sin/cos MSE + FK wrist-center loss + unit-circle penalty")
    cp("  Analytical IK fallback (Python port of iPUMA.m) for verification")
    cp()

    np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cpu" if (args.no_gpu or not torch.cuda.is_available()) else "cuda")
    cp(f"Device: {device}")

    # ── [0] Load ──────────────────────────────────────────────────────
    cp(f"\n[0] Loading: {args.dataset}")
    if not Path(args.dataset).exists():
        cp(f"ERROR: {args.dataset} not found.", s="red"); sys.exit(1)

    with open(args.dataset) as f:
        hdr = [c.strip() for c in f.readline().split(",")]
    pose_cols  = ["nx","ny","nz","ox","oy","oz","ax","ay","az","Px","Py","Pz"]
    joint_cols = ["theta1","theta2","theta3","theta4","theta5","theta6"]
    try:
        pidx = [hdr.index(c) for c in pose_cols]
        jidx = [hdr.index(c) for c in joint_cols]
    except ValueError:
        pidx = list(range(12)); jidx = list(range(12, 18))

    raw   = np.loadtxt(args.dataset, delimiter=",", skiprows=1)
    X_all = raw[:, pidx]
    Y_all = raw[:, jidx]
    N     = len(raw)
    cp(f"  Loaded {N} samples")

    # Sanity check: verify FK consistency on a few samples
    n_check = min(5, N)
    max_fk_err = 0
    for i in range(n_check):
        T = fPUMA(Y_all[i])
        T_target = row_to_T(X_all[i])
        err = np.linalg.norm(T[:3,3] - T_target[:3,3])
        max_fk_err = max(max_fk_err, err)
    cp(f"  FK consistency check (max pos error on {n_check} samples): {max_fk_err:.4f} mm")
    if max_fk_err > 5.0:
        cp("  WARNING: DH parameters may not match dataset generator!", s="red")

    # Split
    np.random.seed(42)
    idx  = np.random.permutation(N)
    n_tr = int(0.70*N); n_va = int(0.15*N)
    X_tr, Y_tr = X_all[idx[:n_tr]],          Y_all[idx[:n_tr]]
    X_va, Y_va = X_all[idx[n_tr:n_tr+n_va]], Y_all[idx[n_tr:n_tr+n_va]]
    X_te, Y_te = X_all[idx[n_tr+n_va:]],     Y_all[idx[n_tr+n_va:]]
    n_tr, n_va, n_te = len(X_tr), len(X_va), len(X_te)
    n_eval = min(args.n_eval, n_te)
    cp(f"  Train: {n_tr} | Val: {n_va} | Test: {n_te} | Eval: {n_eval}")

    # ── [1] Prepare inputs & targets ─────────────────────────────────
    cp(f"\n[1] Preparing inputs (wrist centers) and sin/cos targets")

    # Compute wrist centers for all splits
    P5_tr = compute_wrist_center_np(X_tr)
    P5_va = compute_wrist_center_np(X_va)
    P5_te = compute_wrist_center_np(X_te)

    # Normalise wrist center (3 inputs)
    P5_mean = P5_tr.mean(0, keepdims=True)
    P5_std  = P5_tr.std(0,  keepdims=True); P5_std[P5_std < 1e-8] = 1.0
    P5_tr_n = (P5_tr - P5_mean) / P5_std
    P5_va_n = (P5_va - P5_mean) / P5_std
    P5_te_n = (P5_te - P5_mean) / P5_std

    cp("  Wrist center stats (mm):")
    for i, nm in enumerate(["X","Y","Z"]):
        cp(f"    P5{nm}: mean={P5_mean[0,i]:.0f}  std={P5_std[0,i]:.0f}"
           f"  range=[{P5_tr[:,i].min():.0f}, {P5_tr[:,i].max():.0f}]")

    # sin/cos targets for J1,J2,J3 only
    Y_tr_sc = angles_to_sc(Y_tr[:, :3])
    Y_va_sc = angles_to_sc(Y_va[:, :3])
    Y_te_sc = angles_to_sc(Y_te[:, :3])

    cp("  Joint ranges in training set:")
    for j in range(6):
        cp(f"    J{j+1}: [{Y_tr[:,j].min():.1f}, {Y_tr[:,j].max():.1f}] deg")

    # ── [2] Build & train ─────────────────────────────────────────────
    cp(f"\n[2] Training ShoulderNet (3→{args.hidden}→...→6 sin/cos outputs)")
    model = ShoulderNet(n_in=3, hidden=args.hidden, n_blocks=args.n_blocks)
    n_params = sum(p.numel() for p in model.parameters())
    cp(f"  Params: {n_params:,}  Epochs: {args.epochs}  LR: {args.lr}"
       f"  Batch: {args.batch}  Patience: {args.patience}"); cp()

    t_start = time.time()
    best_state, hist = train_model(
        model, P5_tr_n, Y_tr_sc, P5_tr,
        P5_va_n, Y_va_sc, P5_va,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch,
        patience=args.patience, device=device, console=console,
    )
    ann_time = time.time() - t_start
    model.load_state_dict(best_state); model.eval().cpu()
    cp(f"\n  Done: {ann_time:.1f}s  Best epoch: {hist['best_epoch']}"
       f"  Best val MSE: {hist['best_val']:.6f}")

    # ── [3] J1,J2,J3 accuracy on full test set ────────────────────────
    cp(f"\n[3] ShoulderNet J1,J2,J3 accuracy (n={n_te})")
    with torch.no_grad():
        sc_pred = torch.tanh(model(torch.tensor(P5_te_n, dtype=torch.float32))).numpy()
    J123_pred_te = sc_to_angles(sc_pred)
    J123_true_te = Y_te[:, :3]
    wrapped_err  = np.array([wrap_angle_error(J123_pred_te[:,j], J123_true_te[:,j]) for j in range(3)])
    mae  = np.mean(np.abs(wrapped_err), axis=1)
    rmse = np.sqrt(np.mean(wrapped_err**2, axis=1))
    for j in range(3):
        cp(f"  J{j+1}: MAE={mae[j]:.4f}°  RMSE={rmse[j]:.4f}°"
           f"  max={np.max(np.abs(wrapped_err[j])):.2f}°")

    # ── [4] Eval subset + timing ──────────────────────────────────────
    cp(f"\n[4] Eval subset & timing (n={n_eval})")
    X_ev   = X_te[:n_eval]
    P5_ev  = P5_te[:n_eval]
    P5_ev_n= P5_te_n[:n_eval]
    T_tgts = [row_to_T(X_ev[i]) for i in range(n_eval)]

    # Warmup
    Xw = torch.tensor(P5_ev_n[:1], dtype=torch.float32)
    for _ in range(20):
        with torch.no_grad(): model(Xw)

    # Per-sample timing
    ts = []
    for i in range(n_eval):
        xi = torch.tensor(P5_ev_n[i:i+1], dtype=torch.float32)
        t0 = time.perf_counter()
        with torch.no_grad():
            sc = torch.tanh(model(xi)).numpy()
            j123 = sc_to_angles(sc)
            _, _, _ = solve_wrist(j123[0], T_tgts[i])
        ts.append(time.perf_counter() - t0)
    ann_ms = np.mean(ts) * 1000
    cp(f"  Per-sample (ANN+wrist): mean={ann_ms:.4f} ms")

    # ── [5] ANN FK verification ───────────────────────────────────────
    cp(f"\n[5] Evaluating all methods (n={n_eval})")

    # --- Method 1: Pure ANN (J1,J2,J3 from ANN + analytic J4,J5,J6) ---
    ann_pos = np.zeros(n_eval); ann_rot = np.zeros(n_eval)
    prog = make_prog("ANN+AnalyticWrist", n_eval, console)
    if prog: prog.start()
    with torch.no_grad():
        batch_sc = torch.tanh(model(torch.tensor(P5_ev_n, dtype=torch.float32))).numpy()
    J123_ev = sc_to_angles(batch_sc)   # [n_eval, 3]

    for i in range(n_eval):
        T06 = T_tgts[i]
        t4, t5, t6 = solve_wrist(J123_ev[i], T06)
        J_full = np.array([J123_ev[i,0], J123_ev[i,1], J123_ev[i,2], t4, t5, t6])
        T = fPUMA(J_full)
        ann_pos[i] = np.linalg.norm(T[:3,3] - T06[:3,3])
        ann_rot[i] = geodesic_rot_err(T[:3,:3], T06[:3,:3])
        if prog: prog.update(prog._task_id, advance=1)
    if prog: prog.stop()
    cp(f"  ANN+Wrist: pos mean={ann_pos.mean():.3f}mm  med={np.median(ann_pos):.3f}mm"
       f"  max={ann_pos.max():.3f}mm  <1mm={100*np.sum(ann_pos<1)/n_eval:.1f}%"
       f"  <5mm={100*np.sum(ann_pos<5)/n_eval:.1f}%"); cp()

    # --- Method 2: Analytical IK only (Python port of iPUMA.m) ---
    cp(f"  Running Analytical IK (configs 1-4)...")
    ana_pos = np.zeros(n_eval); ana_rot = np.zeros(n_eval); ana_t = np.zeros(n_eval)
    prog = make_prog("Analytical IK", n_eval, console)
    if prog: prog.start()
    for i in range(n_eval):
        t0 = time.perf_counter()
        J_a, err_a = analytical_ik(T_tgts[i], configs=(1,2,3,4))
        ana_t[i] = time.perf_counter() - t0
        if J_a is not None:
            T = fPUMA(J_a)
            ana_pos[i] = np.linalg.norm(T[:3,3] - T_tgts[i][:3,3])
            ana_rot[i] = geodesic_rot_err(T[:3,:3], T_tgts[i][:3,:3])
        else:
            ana_pos[i] = np.nan; ana_rot[i] = np.nan
        if prog: prog.update(prog._task_id, advance=1)
    if prog: prog.stop()
    cp(f"  Analytical: pos mean={np.nanmean(ana_pos):.4f}mm"
       f"  <1mm={100*np.sum(ana_pos<1)/n_eval:.1f}%"); cp()

    # --- Method 3: ANN + Analytical IK (use best of both) ---
    hybrid_pos = np.zeros(n_eval); hybrid_rot = np.zeros(n_eval); hybrid_t = np.zeros(n_eval)
    cp(f"  Running ANN+Analytical hybrid...")
    prog = make_prog("ANN+Analytical", n_eval, console)
    if prog: prog.start()
    for i in range(n_eval):
        t0 = time.perf_counter()
        # ANN solution
        t4, t5, t6 = solve_wrist(J123_ev[i], T_tgts[i])
        J_ann_full = np.array([J123_ev[i,0], J123_ev[i,1], J123_ev[i,2], t4, t5, t6])
        T_ann = fPUMA(J_ann_full)
        ann_err_i = np.linalg.norm(T_ann[:3,3] - T_tgts[i][:3,3])
        # Analytical solution
        J_a, err_a = analytical_ik(T_tgts[i])
        hybrid_t[i] = time.perf_counter() - t0
        # Take best
        if J_a is not None and err_a < ann_err_i:
            J_best = J_a
        else:
            J_best = J_ann_full
        T = fPUMA(J_best)
        hybrid_pos[i] = np.linalg.norm(T[:3,3] - T_tgts[i][:3,3])
        hybrid_rot[i] = geodesic_rot_err(T[:3,:3], T_tgts[i][:3,:3])
        if prog: prog.update(prog._task_id, advance=1)
    if prog: prog.stop()
    cp(f"  Hybrid: pos mean={hybrid_pos.mean():.4f}mm"
       f"  <1mm={100*np.sum(hybrid_pos<1)/n_eval:.1f}%"); cp()

    # --- Method 4: ANN + GD warm-start ---
    cp(f"  Running ANN+GD warm-start...")
    gd_warm_pos = np.zeros(n_eval); gd_warm_rot = np.zeros(n_eval); gd_warm_t = np.zeros(n_eval)
    gd_src_ann=0; gd_src_cold=0
    prog = make_prog("ANN+GD", n_eval, console)
    if prog: prog.start()
    for i in range(n_eval):
        t4, t5, t6 = solve_wrist(J123_ev[i], T_tgts[i])
        J_start = np.array([J123_ev[i,0], J123_ev[i,1], J123_ev[i,2], t4, t5, t6])
        if ann_pos[i] < 100:
            gd_src_ann += 1
        else:
            J_start = JOINT_LIMITS.mean(1); gd_src_cold += 1
        t0 = time.perf_counter()
        J_gd, _ = jacobian_ik(J_start, T_tgts[i], max_iter=400)
        gd_warm_t[i] = time.perf_counter() - t0
        T = fPUMA(J_gd)
        gd_warm_pos[i] = np.linalg.norm(T[:3,3] - T_tgts[i][:3,3])
        gd_warm_rot[i] = geodesic_rot_err(T[:3,:3], T_tgts[i][:3,:3])
        if prog: prog.update(prog._task_id, advance=1)
    if prog: prog.stop()
    cp(f"  Warm starts: ANN={gd_src_ann}  Cold={gd_src_cold}")
    cp(f"  ANN+GD: pos mean={gd_warm_pos.mean():.3f}mm"
       f"  <1mm={100*np.sum(gd_warm_pos<1)/n_eval:.1f}%"); cp()

    # --- Method 5: GD multi-start cold ---
    cp(f"  Running GD multi-start cold ({args.gd_starts} starts)...")
    gd_cold_pos = np.zeros(n_eval); gd_cold_rot = np.zeros(n_eval); gd_cold_t = np.zeros(n_eval)
    prog = make_prog("GD cold", n_eval, console)
    if prog: prog.start()
    for i in range(n_eval):
        t0 = time.perf_counter()
        J_gd, _ = multistart_gd(T_tgts[i], n_starts=args.gd_starts)
        gd_cold_t[i] = time.perf_counter() - t0
        T = fPUMA(J_gd)
        gd_cold_pos[i] = np.linalg.norm(T[:3,3] - T_tgts[i][:3,3])
        gd_cold_rot[i] = geodesic_rot_err(T[:3,:3], T_tgts[i][:3,:3])
        if prog: prog.update(prog._task_id, advance=1)
    if prog: prog.stop()

    # --- Method 6: DE pure ---
    de_pos = np.full(n_eval, np.nan); de_rot = np.full(n_eval, np.nan); de_t = np.zeros(n_eval)
    if HAS_SCIPY:
        cp(f"  Running DE pure (seeds={args.n_seeds})...")
        bounds = list(zip(JOINT_LIMITS[:,0], JOINT_LIMITS[:,1]))
        prog = make_prog("DE pure", n_eval, console)
        if prog: prog.start()
        for i in range(n_eval):
            T_tgt = T_tgts[i]; best_err=np.inf; best_J=JOINT_LIMITS.mean(1)
            t0 = time.perf_counter()
            for s in range(args.n_seeds):
                np.random.seed(42+i*args.n_seeds+s)
                try:
                    res = differential_evolution(
                        lambda J, T=T_tgt: _ik_err(J, T), bounds,
                        maxiter=400, popsize=15, tol=1e-6,
                        mutation=(0.5,1.2), recombination=0.9,
                        seed=42+i*args.n_seeds+s, workers=1)
                    if res.fun < best_err: best_err=res.fun; best_J=res.x
                except: pass
            de_t[i] = time.perf_counter() - t0
            T = fPUMA(best_J)
            de_pos[i] = np.linalg.norm(T[:3,3]-T_tgt[:3,3])
            de_rot[i] = geodesic_rot_err(T[:3,:3], T_tgt[:3,:3])
            if prog: prog.update(prog._task_id, advance=1)
        if prog: prog.stop()

    # ── SUMMARY TABLE ─────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"PUMA 560 IK COMPARISON  v4.0  (n={n_eval})")
    print("Pos = Euclidean mm | Rot = geodesic deg")
    print(SEP2); print(HDR); print(DIV)
    prt_row("ANN+AnalyticWrist",         ann_pos,      ann_rot,      ann_ms,                n_eval)
    prt_row("Analytical IK (iPUMA.py)",  ana_pos,      ana_rot,      ana_t.mean()*1000,     n_eval)
    prt_row("ANN+Analytical (best)",     hybrid_pos,   hybrid_rot,   hybrid_t.mean()*1000,  n_eval)
    prt_row("ANN+GD warm",               gd_warm_pos,  gd_warm_rot,  gd_warm_t.mean()*1000, n_eval)
    prt_row("GD multi-start cold",       gd_cold_pos,  gd_cold_rot,  gd_cold_t.mean()*1000, n_eval)
    prt_row("DE pure",                   de_pos,       de_rot,       de_t.mean()*1000,      n_eval)
    print()

    # ── PLOTS ──────────────────────────────────────────────────────────
    if not args.no_plots:
        cp("\n[10] Generating plots...")
        all_pos = [ann_pos, ana_pos, hybrid_pos, gd_warm_pos, gd_cold_pos, de_pos]
        all_rot = [ann_rot, ana_rot, hybrid_rot, gd_warm_rot, gd_cold_rot, de_rot]
        all_t   = [ann_ms, ana_t.mean()*1000, hybrid_t.mean()*1000,
                   gd_warm_t.mean()*1000, gd_cold_t.mean()*1000, de_t.mean()*1000]
        mnames  = ["ANN+Wrist","Analytical","ANN+Ana","ANN+GD","GD cold","DE"]
        save_plots(hist, Y_te, J123_pred_te, J123_true_te,
                   all_pos, all_rot, all_t, mnames, n_eval, console)

    # ── SAVE ───────────────────────────────────────────────────────────
    cp("\n[11] Saving model...")
    torch.save({
        "model_state": best_state,
        "P5_mean": P5_mean, "P5_std": P5_std,
        "dh_params": DH, "hist": hist,
        "encoding": "sincos_J123_only",
        "architecture": "ShoulderNet_3in_6out",
        "joint_limits": JOINT_LIMITS,
    }, "puma560_ann_v4.pt")
    cp("  Saved: puma560_ann_v4.pt")

    np.savez("comparison_results_v4.npz",
        ann_pos=ann_pos, ann_rot=ann_rot,
        ana_pos=ana_pos, ana_rot=ana_rot,
        hybrid_pos=hybrid_pos, hybrid_rot=hybrid_rot,
        gd_warm_pos=gd_warm_pos, gd_cold_pos=gd_cold_pos, de_pos=de_pos)
    cp("  Saved: comparison_results_v4.npz")

    cp("\nDone!")
    cp()
    cp("Inference:")
    cp("  ck = torch.load('puma560_ann_v4.pt')")
    cp("  model = ShoulderNet(); model.load_state_dict(ck['model_state']); model.eval()")
    cp("  # Given pose row [nx,ny,nz,...,Pz] (12 values):")
    cp("  P5 = [Px - 56.5*ax, Py - 56.5*ay, Pz - 56.5*az]   # wrist center")
    cp("  P5_n = (P5 - ck['P5_mean']) / ck['P5_std']")
    cp("  sc = torch.tanh(model(tensor(P5_n))).detach().numpy()")
    cp("  J123 = sc_to_angles(sc)   # J1,J2,J3 in degrees")
    cp("  t4,t5,t6 = solve_wrist(J123, T06)   # exact wrist angles")


def _ik_err(thetas, T_tgt):
    try:
        T = fPUMA(thetas)
        pos = np.linalg.norm(T[:3,3]-T_tgt[:3,3]) / R_WORKSPACE
        rot = np.linalg.norm(rot_vec_err(T[:3,:3], T_tgt[:3,:3])) / math.pi
        return pos + rot
    except: return 1e6


if __name__ == "__main__":
    main()