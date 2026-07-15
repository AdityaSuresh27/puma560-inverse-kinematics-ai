"""
utils/fk_torch.py — PUMA 560 Forward Kinematics in PyTorch.

This module provides a fully differentiable FK that can be used inside the
training loop as a geometric consistency loss:
    L_fk = || FK(θ̂) - x_target ||²

Design requirements:
  - Works on batches: input (B, 6), output (B, 4, 4) or (B, 9) pose vectors
  - All operations are PyTorch-native — gradients flow back to θ̂
  - Numerically identical to fk_numpy.py (tested in test_fk.py)
  - Device-agnostic: works on CPU and CUDA

The DH parameters are registered as non-trainable buffers in the Module so
they move to the correct device automatically with .to(device).
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DH_PARAMS, JOINT_DIM, POSE_DIM


class PumaFK(nn.Module):
    """Differentiable forward kinematics for PUMA 560.

    Registered DH parameters as buffers so they:
      1. Move to GPU with .to(device)
      2. Are excluded from optimizer parameter groups
      3. Are saved/loaded with state_dict

    Usage:
        fk = PumaFK()
        T  = fk(q)          # (B, 4, 4)
        pv = fk.to_pose_vector(q)   # (B, 9)
    """

    def __init__(self):
        super().__init__()

        # Register DH params as buffers (float64 for numerical accuracy)
        dh = torch.tensor(DH_PARAMS, dtype=torch.float64)  # (6, 4)
        self.register_buffer('dh_a',     dh[:, 0])   # link lengths
        self.register_buffer('dh_d',     dh[:, 1])   # link offsets
        self.register_buffer('dh_alpha', dh[:, 2])   # link twists
        self.register_buffer('dh_theta_offset', dh[:, 3])  # theta offsets

    # ── Single-link transform ─────────────────────────────────────────────────

    def _dh_transform_batch(
        self,
        a:     torch.Tensor,  # scalar
        d:     torch.Tensor,  # scalar
        alpha: torch.Tensor,  # scalar
        theta: torch.Tensor,  # (B,) joint angles
    ) -> torch.Tensor:
        """Build batched DH transform matrices.

        Args:
            a, d, alpha : scalar DH parameters (buffers, same for all samples)
            theta       : (B,) joint angles

        Returns:
            T : (B, 4, 4) homogeneous transforms
        """
        B = theta.shape[0]
        dtype = theta.dtype
        device = theta.device

        ct = torch.cos(theta)   # (B,)
        st = torch.sin(theta)   # (B,)
        ca = torch.cos(alpha).to(dtype)
        sa = torch.sin(alpha).to(dtype)
        a  = a.to(dtype)
        d  = d.to(dtype)

        zeros = torch.zeros(B, dtype=dtype, device=device)
        ones  = torch.ones(B,  dtype=dtype, device=device)

        # Row-major construction of (B, 4, 4) matrices
        # Row 0: [ct,  -st*ca,  st*sa,  a*ct]
        # Row 1: [st,   ct*ca, -ct*sa,  a*st]
        # Row 2: [ 0,      sa,     ca,     d]
        # Row 3: [ 0,       0,      0,     1]

        row0 = torch.stack([ ct,  -st*ca,  st*sa,  a*ct], dim=1)  # (B, 4)
        row1 = torch.stack([ st,   ct*ca, -ct*sa,  a*st], dim=1)
        row2 = torch.stack([zeros, sa*ones, ca*ones, d*ones], dim=1)
        row3 = torch.stack([zeros, zeros,  zeros,   ones],  dim=1)

        T = torch.stack([row0, row1, row2, row3], dim=1)  # (B, 4, 4)
        return T

    # ── Full-chain FK ─────────────────────────────────────────────────────────

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute end-effector transforms for a batch of joint configurations.

        Args:
            q : (B, 6) joint angles in radians, any floating-point dtype

        Returns:
            T : (B, 4, 4) homogeneous transforms
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)   # handle single input gracefully

        B = q.shape[0]
        assert q.shape[1] == JOINT_DIM, f"Expected 6 joints, got {q.shape[1]}"

        # Work in float64 for numerical accuracy matching NumPy implementation
        q64 = q.double()

        # Initialise with identity matrices
        T = torch.eye(4, dtype=torch.float64, device=q.device).unsqueeze(0).expand(B, -1, -1).clone()

        for i in range(JOINT_DIM):
            theta_i = q64[:, i] + self.dh_theta_offset[i]   # (B,)
            Ti = self._dh_transform_batch(
                self.dh_a[i], self.dh_d[i], self.dh_alpha[i], theta_i
            )                                                  # (B, 4, 4)
            T = torch.bmm(T, Ti)

        # Return in the same dtype as input
        return T.to(q.dtype)

    # ── Pose vector encoding ──────────────────────────────────────────────────

    def to_pose_vector(self, q: torch.Tensor) -> torch.Tensor:
        """Compute 9-dim pose vectors for a batch.

        Representation: [x, y, z, r11, r21, r31, r12, r22, r32]
        Exactly matches T_to_pose_vector() in fk_numpy.py.

        Args:
            q  : (B, 6) joint angles

        Returns:
            pv : (B, 9) pose vectors
        """
        T = self.forward(q)           # (B, 4, 4)
        pos    = T[:, :3, 3]          # (B, 3)   translation
        r_col0 = T[:, :3, 0]          # (B, 3)   first rotation column
        r_col1 = T[:, :3, 1]          # (B, 3)   second rotation column
        return torch.cat([pos, r_col0, r_col1], dim=1)   # (B, 9)

    # ── Pose consistency loss ─────────────────────────────────────────────────

    def consistency_loss(
        self,
        q_pred:    torch.Tensor,   # (B, 6)  predicted joint angles
        pose_target: torch.Tensor, # (B, 9)  target pose vectors
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
    ) -> torch.Tensor:
        """FK consistency loss: MSE between FK(q_pred) and pose_target.

        Separately weights position and rotation terms so we can tune
        the relative importance during training.

        Loss = pos_weight * || FK_pos(q) - p_target ||² 
             + rot_weight * || FK_rot(q) - r_target ||²

        Both terms are mean-reduced over the batch.

        Args:
            q_pred       : (B, 6) predicted joint angles
            pose_target  : (B, 9) target [x,y,z, R_col0, R_col1]
            pos_weight   : weight for position MSE term
            rot_weight   : weight for rotation MSE term

        Returns:
            loss : scalar tensor
        """
        pose_pred = self.to_pose_vector(q_pred)   # (B, 9)

        pos_loss = torch.mean((pose_pred[:, :3] - pose_target[:, :3]) ** 2)
        rot_loss = torch.mean((pose_pred[:, 3:] - pose_target[:, 3:]) ** 2)

        return pos_weight * pos_loss + rot_weight * rot_loss


# ── Module-level convenience singleton ───────────────────────────────────────
# Instantiated once; callers can import and use directly.
# Thread-safe for inference (no shared mutable state).
_fk_singleton: PumaFK | None = None

def get_fk_module(device: str = 'cpu') -> PumaFK:
    """Get (or create) the shared FK module on the specified device."""
    global _fk_singleton
    if _fk_singleton is None:
        _fk_singleton = PumaFK()
    return _fk_singleton.to(device)
