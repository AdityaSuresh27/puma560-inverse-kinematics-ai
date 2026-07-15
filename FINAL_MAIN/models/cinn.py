"""
models/cinn.py — Mode-Conditioned Conditional Invertible Neural Network (cINN)
                 for PUMA 560 inverse kinematics.

Architecture overview:
  - Backbone: normalising flow built with FrEIA affine coupling blocks
  - Conditioning: pose vector (9-dim) + mode one-hot (8-dim) = 17-dim condition
  - Forward pass (training): joint_angles → latent z  (used for NLL loss)
  - Inverse pass (inference): z + condition → joint_angles

Each coupling block contains a small MLP subnet conditioned on the full
condition vector. Conditioning is injected at EVERY coupling block, not just
the first — this is critical for mode-specific solution quality.

Key design choices:
  1. Soft-clamping (via tanh) on the log-scale output of affine coupling
     prevents exploding/vanishing scales during early training.
  2. Random permutations between blocks ensure all joint dimensions are
     transformed across all blocks (prevents dead dimensions).
  3. The flow is volume-preserving in the sense that we track log|det J|
     explicitly — no approximations.

Training uses:
  loss = NLL(z, log_det) + lambda(t) * FK_consistency_loss(q_pred, pose_target)
  where lambda(t) is linearly warmed up over FK_LOSS_WARMUP epochs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FrEIA imports
try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
except ImportError:
    raise ImportError(
        "FrEIA not installed. Run: pip install FrEIA\n"
        "GitHub: https://github.com/vislearn/FrEIA"
    )

from config import (
    JOINT_DIM, POSE_DIM, N_MODES, COND_DIM,
    N_COUPLING_BLOCKS, HIDDEN_DIM, N_HIDDEN_LAYERS, CLAMP_VALUE,
)


# ── Coupling subnet factory ───────────────────────────────────────────────────

def make_subnet(in_dim: int, out_dim: int) -> nn.Module:
    """Build the MLP subnet used inside each affine coupling block.

    The subnet maps:  [x_half || condition] → [s, t]   (scale and translate)

    Architecture: Linear → LayerNorm → LeakyReLU (repeated) → Linear
    LayerNorm (not BatchNorm) is used because:
      - Batch-independent: works with any batch size including 1
      - More stable than BatchNorm for flow networks
      - Doesn't require running statistics at inference

    Args:
        in_dim  : input dimension (half of JOINT_DIM + COND_DIM)
        out_dim : output dimension (half of JOINT_DIM, for s and t each)

    Returns:
        subnet : nn.Sequential MLP
    """
    layers: list[nn.Module] = []

    # Input → first hidden
    layers.append(nn.Linear(in_dim, HIDDEN_DIM))
    layers.append(nn.LayerNorm(HIDDEN_DIM))
    layers.append(nn.LeakyReLU(0.2, inplace=True))

    # Hidden → hidden (N_HIDDEN_LAYERS - 1 additional layers)
    for _ in range(N_HIDDEN_LAYERS - 1):
        layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
        layers.append(nn.LayerNorm(HIDDEN_DIM))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    # Final → output
    layers.append(nn.Linear(HIDDEN_DIM, out_dim))

    subnet = nn.Sequential(*layers)

    # Initialise output layer close to zero — this makes the flow start
    # as approximately the identity, which improves early training stability
    nn.init.zeros_(subnet[-1].weight)
    nn.init.zeros_(subnet[-1].bias)

    return subnet


# ── cINN model ────────────────────────────────────────────────────────────────

class PumacINN(nn.Module):
    """Mode-conditioned cINN for PUMA 560 inverse kinematics.

    The model learns a reversible bijection:
        forward:  θ → z          (normalisation direction, used for training NLL)
        inverse:  z → θ          (generation direction, used for inference)

    Both directions are conditioned on c = [pose_vec || mode_onehot].

    Args:
        joint_dim  : dimensionality of joint space (6 for PUMA 560)
        cond_dim   : dimensionality of conditioning vector (17 = 9 + 8)
        n_blocks   : number of affine coupling blocks
        clamp      : soft-clamp value for log-scale outputs
    """

    def __init__(
        self,
        joint_dim: int = JOINT_DIM,
        cond_dim:  int = COND_DIM,
        n_blocks:  int = N_COUPLING_BLOCKS,
        clamp:     float = CLAMP_VALUE,
    ):
        super().__init__()

        self.joint_dim = joint_dim
        self.cond_dim  = cond_dim
        self.n_blocks  = n_blocks
        self.clamp     = clamp

        # Build the FrEIA invertible network
        self.inn = self._build_inn()

    def _build_inn(self) -> Ff.SequenceINN:
        """Construct the invertible network using FrEIA."""

        inn = Ff.SequenceINN(self.joint_dim)

        for i in range(self.n_blocks):
            # AllInOneBlock combines:
            #   - Affine coupling (ActNorm + 1x1 convolution + coupling)
            #   - Soft clamping on scale outputs
            inn.append(
                Fm.AllInOneBlock,
                # SequenceINN expects a condition index (into c=[...]), not a ConditionNode.
                cond=0,
                cond_shape=(self.cond_dim,),
                subnet_constructor=make_subnet,
                affine_clamping=self.clamp,
                global_affine_type='SOFTPLUS',
                permute_soft=True,   # learnable soft permutation per block
            )

        return inn

    # ── Forward (θ → z, for training) ────────────────────────────────────────

    def forward(
        self,
        joints: torch.Tensor,    # (B, 6)  normalised joint angles
        condition: torch.Tensor, # (B, 17) [pose_norm || mode_onehot]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map joint angles to latent space.

        Args:
            joints    : (B, 6) normalised joint angles
            condition : (B, 17) conditioning vector

        Returns:
            z       : (B, 6) latent codes ~ N(0, I) under correct model
            log_det : (B,) log determinant of the Jacobian
        """
        z, log_det = self.inn(joints, c=[condition])
        return z, log_det

    # ── Inverse (z → θ, for inference) ───────────────────────────────────────

    def inverse(
        self,
        z: torch.Tensor,         # (B, 6)  latent codes
        condition: torch.Tensor, # (B, 17) conditioning vector
    ) -> torch.Tensor:
        """Map latent codes to joint angles.

        Args:
            z         : (B, 6) latent codes (typically sampled ~ N(0, I))
            condition : (B, 17) conditioning vector

        Returns:
            joints : (B, 6) normalised joint angle predictions
        """
        joints, _ = self.inn(z, c=[condition], rev=True)
        return joints

    # ── NLL loss ──────────────────────────────────────────────────────────────

    def nll_loss(
        self,
        joints: torch.Tensor,    # (B, 6)
        condition: torch.Tensor, # (B, 17)
    ) -> torch.Tensor:
        """Negative log-likelihood under a standard Gaussian prior.

        Loss = -log p(θ) = -log p_z(z) - log|det J|
             = 0.5 * ||z||² + 0.5 * D * log(2π) - log|det J|

        The constant 0.5 * D * log(2π) is dropped as it doesn't affect training.
        Mean-reduced over the batch.

        Args:
            joints    : (B, 6) normalised joint angles
            condition : (B, 17)

        Returns:
            nll : scalar tensor
        """
        z, log_det = self.forward(joints, condition)

        # Gaussian log-prob: -0.5 * ||z||² (summed over dimensions)
        log_pz = -0.5 * torch.sum(z ** 2, dim=1)   # (B,)

        # ELBO: log p(x) = log p(z) + log|det J|
        log_px = log_pz + log_det                   # (B,)

        return -torch.mean(log_px)

    # ── Sample multiple solutions per mode at inference ───────────────────────

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,   # (B, 17) or (17,)
        n_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample joint angle predictions from the model.

        Args:
            condition   : (B, 17) or (17,) conditioning vector
            n_samples   : number of samples per condition
            temperature : scale the latent Gaussian (1.0 = standard)
                          Lower temperature → less diversity, more "central" solutions

        Returns:
            joints : (B, n_samples, 6) predicted normalised joint angles
                     If input was (17,), returns (n_samples, 6)
        """
        squeeze = condition.dim() == 1
        if squeeze:
            condition = condition.unsqueeze(0)

        B = condition.shape[0]
        device = condition.device

        # Expand condition to (B * n_samples, 17)
        cond_exp = condition.unsqueeze(1).expand(B, n_samples, -1)
        cond_exp = cond_exp.reshape(B * n_samples, self.cond_dim)

        # Sample latent codes
        z = torch.randn(B * n_samples, self.joint_dim, device=device) * temperature

        joints_flat = self.inverse(z, cond_exp)            # (B*n_samples, 6)
        joints = joints_flat.reshape(B, n_samples, self.joint_dim)

        return joints.squeeze(0) if squeeze else joints

    # ── Parameter count ───────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def __repr__(self) -> str:
        p = self.count_parameters()
        return (
            f"PumacINN(\n"
            f"  joint_dim={self.joint_dim}, cond_dim={self.cond_dim},\n"
            f"  n_blocks={self.n_blocks}, clamp={self.clamp},\n"
            f"  hidden_dim={HIDDEN_DIM}, n_hidden={N_HIDDEN_LAYERS},\n"
            f"  params={p['trainable']:,} trainable / {p['total']:,} total\n"
            f")"
        )


# ── Conditioning vector builder ───────────────────────────────────────────────

def build_condition(
    pose_norm: torch.Tensor,    # (B, 9) or (9,)
    mode_onehot: torch.Tensor,  # (B, 8) or (8,)
) -> torch.Tensor:
    """Concatenate normalised pose and mode one-hot into condition vector.

    Args:
        pose_norm   : (B, 9) normalised pose vectors
        mode_onehot : (B, 8) mode one-hot vectors

    Returns:
        condition : (B, 17) concatenated condition
    """
    if pose_norm.dim() == 1:
        pose_norm   = pose_norm.unsqueeze(0)
        mode_onehot = mode_onehot.unsqueeze(0)
    return torch.cat([pose_norm, mode_onehot], dim=1)


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(
    model: PumacINN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
    extra: Optional[dict] = None,
) -> None:
    """Save model + optimizer state with metadata."""
    ckpt = {
        'epoch':      epoch,
        'val_loss':   val_loss,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'model_config': {
            'joint_dim':   model.joint_dim,
            'cond_dim':    model.cond_dim,
            'n_blocks':    model.n_blocks,
            'clamp':       model.clamp,
            'hidden_dim':  HIDDEN_DIM,
            'n_hidden':    N_HIDDEN_LAYERS,
        },
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    device: str = 'cpu',
) -> tuple[PumacINN, dict]:
    """Load a checkpoint and return (model, checkpoint_dict).

    The model is reconstructed from the saved config so the architecture
    always matches — no need to manually specify parameters.

    v3+: checkpoint stores hidden_dim and n_hidden so loading works
    correctly even if config.py has different values.
    """
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt['model_config']

    # Temporarily override global HIDDEN_DIM / N_HIDDEN_LAYERS if checkpoint
    # recorded them (v3+), so make_subnet builds the correct architecture.
    # For older checkpoints without these fields, infer from saved weights.
    import config as _cfg_module
    _orig_hd = _cfg_module.HIDDEN_DIM
    _orig_nh = _cfg_module.N_HIDDEN_LAYERS

    if 'hidden_dim' in cfg:
        _cfg_module.HIDDEN_DIM = cfg['hidden_dim']
    else:
        # Infer hidden_dim from the first subnet's first Linear layer
        state = ckpt['model_state']
        for key in sorted(state.keys()):
            if 'subnet.0.weight' in key:
                _cfg_module.HIDDEN_DIM = state[key].shape[0]
                break

    if 'n_hidden' in cfg:
        _cfg_module.N_HIDDEN_LAYERS = cfg['n_hidden']
    else:
        # Infer n_hidden by counting 2D weight tensors (= Linear layers) in first subnet
        # LayerNorm weights are 1D, so checking ndim == 2 filters them out
        state = ckpt['model_state']
        linear_count = sum(
            1 for key in state.keys()
            if 'module_list.0.subnet' in key
            and key.endswith('.weight')
            and state[key].ndim == 2
        )
        # n_hidden = total_linear_layers - 1 (subtract the final output layer)
        _cfg_module.N_HIDDEN_LAYERS = max(1, linear_count - 1)

    # Rebuild the global ref used by make_subnet (module-level import)
    global HIDDEN_DIM, N_HIDDEN_LAYERS
    HIDDEN_DIM     = _cfg_module.HIDDEN_DIM
    N_HIDDEN_LAYERS = _cfg_module.N_HIDDEN_LAYERS

    model = PumacINN(
        joint_dim=cfg['joint_dim'],
        cond_dim=cfg['cond_dim'],
        n_blocks=cfg['n_blocks'],
        clamp=cfg['clamp'],
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    # Restore original config values so further model creation uses config.py
    _cfg_module.HIDDEN_DIM     = _orig_hd
    _cfg_module.N_HIDDEN_LAYERS = _orig_nh
    HIDDEN_DIM     = _orig_hd
    N_HIDDEN_LAYERS = _orig_nh

    return model, ckpt
