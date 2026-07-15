"""
models/mode_selector.py — Mode Selector MLP for PUMA 560 cINN.

The mode selector is a small multi-label classifier that, given a target
end-effector pose, predicts which of the 8 configuration modes are likely
to yield a valid IK solution.

Why multi-label (not multi-class)?
  A given pose can have multiple valid modes (e.g. both elbow-up and
  elbow-down may be reachable). The selector outputs a probability per mode
  and we query all modes with p > threshold, dramatically reducing the number
  of cINN evaluations while keeping a fallback to all-8 if nothing passes.

Training signal: binary cross-entropy per mode, supervised by the dataset
(where mode i is labelled 1 if the analytical IK returned a valid solution
for that mode at this pose, else 0).

Inference usage:
  1. probs = selector(pose_norm)           # (8,) probabilities
  2. active_modes = (probs > thresh)       # bool mask
  3. If sum(active_modes) == 0: fallback → evaluate all 8 modes
  4. Query cINN only for active modes → FK filter → best solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import POSE_DIM, N_MODES


class ModeSelectorMLP(nn.Module):
    """Small MLP classifier: pose (9-dim) → mode probabilities (8-dim).

    Architecture: 3 hidden layers, 128 units each, Dropout for regularisation.
    Sigmoid output (multi-label: each mode independently).

    Args:
        pose_dim   : input dimension (9)
        n_modes    : number of output classes (8)
        hidden_dim : width of hidden layers
        dropout    : dropout probability (applied after each hidden layer)
    """

    def __init__(
        self,
        pose_dim:   int   = POSE_DIM,
        n_modes:    int   = N_MODES,
        hidden_dim: int   = 128,
        dropout:    float = 0.2,
    ):
        super().__init__()

        self.pose_dim   = pose_dim
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim // 2, n_modes),
            # No sigmoid here — use BCEWithLogitsLoss during training
            # Apply sigmoid at inference for probabilities
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, pose_norm: torch.Tensor) -> torch.Tensor:
        """Return raw logits.

        Args:
            pose_norm : (B, 9) normalised pose vectors

        Returns:
            logits : (B, 8) — pass through sigmoid for probabilities
        """
        return self.net(pose_norm)

    @torch.no_grad()
    def predict_probs(self, pose_norm: torch.Tensor) -> torch.Tensor:
        """Return per-mode probabilities (inference convenience).

        Args:
            pose_norm : (B, 9) or (9,) normalised pose vectors

        Returns:
            probs : (B, 8) or (8,) sigmoid probabilities
        """
        squeeze = pose_norm.dim() == 1
        if squeeze:
            pose_norm = pose_norm.unsqueeze(0)

        self.eval()
        logits = self.forward(pose_norm)
        probs  = torch.sigmoid(logits)

        return probs.squeeze(0) if squeeze else probs

    @torch.no_grad()
    def predict_active_modes(
        self,
        pose_norm: torch.Tensor,   # (9,)
        threshold: float = 0.3,    # lower = recall-biased (safer), higher = precision-biased
        min_modes: int = 1,        # always return at least this many modes
    ) -> list[int]:
        """Return list of mode indices predicted to be active.

        Falls back to all 8 modes if nothing exceeds threshold.

        Args:
            pose_norm : (9,) normalised pose vector (single pose)
            threshold : probability threshold for mode activation
            min_modes : minimum number of modes to return

        Returns:
            active_modes : sorted list of mode indices
        """
        probs = self.predict_probs(pose_norm)   # (8,)
        active = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()

        if len(active) < min_modes:
            # Fall back: take top-k by probability
            topk = torch.topk(probs, k=min_modes).indices.tolist()
            active = list(set(active + topk))

        return sorted(active)

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss(
        self,
        logits: torch.Tensor,   # (B, 8)
        labels: torch.Tensor,   # (B, 8) float — 1.0 if mode valid, 0.0 otherwise
        pos_weight: Optional[torch.Tensor] = None,  # (8,) class imbalance weight
    ) -> torch.Tensor:
        """Binary cross-entropy loss (multi-label).

        Args:
            logits     : (B, 8) raw logits from forward()
            labels     : (B, 8) binary labels per mode
            pos_weight : upweight positive class if modes are rare

        Returns:
            loss : scalar tensor
        """
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"ModeSelectorMLP("
            f"pose_dim={self.pose_dim}, n_modes={self.n_modes}, "
            f"hidden={self.hidden_dim}, params={n:,})"
        )

