"""
config.py — v3 FINAL: single source of truth for all hyperparameters.

Every script imports from here. Change a value once, it propagates everywhere.

KEY CHANGES from v2:
  - Architecture: 10 blocks × 256 hidden (balanced speed/quality)
  - FK_LOSS_WEIGHT: 10.0 (doubled from v2's 5.0)
  - FK_LATENT_NOISE_STD: 0.01 (tighter from v2's 0.03)
  - INF_TEMPERATURE: 0.5 (sharper from v2's 0.8)
  - N_LATENT_SAMPLES: 25 (more from v2's 15)
  - INF_FK_POS_TOL: 1e-3 (relaxed from v2's 5e-4 for raw flow)
  - N_EPOCHS: 300 (more training for convergence)
  - Checkpoint now saves hidden_dim for portable loading
"""

from pathlib import Path
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
OUTPUT_DIR  = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PATH = OUTPUT_DIR / "puma560_dataset.h5"
NORM_PATH    = OUTPUT_DIR / "normalisation_stats.npz"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── PUMA 560 DH parameters ────────────────────────────────────────────────────
# Standard DH convention. Each row: [a (m), d (m), alpha (rad), theta_offset (rad)]
# Source: Corke & Armstrong, "A search for consensus among model parameters
#         reported for the PUMA 560 robot", ICRA 1994
DH_PARAMS = np.array([
    [ 0.0,      0.67183,   np.pi/2,  0.0],   # link 1
    [ 0.4318,   0.0,       0.0,      0.0],   # link 2
    [ 0.0203,   0.15005,  -np.pi/2,  0.0],   # link 3
    [ 0.0,      0.4318,    np.pi/2,  0.0],   # link 4
    [ 0.0,      0.0,      -np.pi/2,  0.0],   # link 5
    [ 0.0,      0.0,       0.0,      0.0],   # link 6
], dtype=np.float64)

# Joint limits (radians) — from PUMA 560 hardware specification
# Row 0 = lower, Row 1 = upper
JOINT_LIMITS = np.array([
    [-2.7925, -3.9270, -0.7854, -4.7124, -1.7453, -4.7124],  # lower
    [ 2.7925,  0.7854,  3.9270,  4.7124,  1.7453,  4.7124],  # upper
], dtype=np.float64)

# ── Configuration modes (8 analytical IK branches) ───────────────────────────
# 3 binary flags: shoulder (L/R), elbow (U/D), wrist (N/F)
# Encoded as: mode = shoulder_bit | (elbow_bit << 1) | (wrist_bit << 2)
CONFIG_STRINGS = [
    'lun',  # mode 0: shoulder=L, elbow=Up,   wrist=NoFlip
    'run',  # mode 1: shoulder=R, elbow=Up,   wrist=NoFlip
    'ldn',  # mode 2: shoulder=L, elbow=Down, wrist=NoFlip
    'rdn',  # mode 3: shoulder=R, elbow=Down, wrist=NoFlip
    'luf',  # mode 4: shoulder=L, elbow=Up,   wrist=Flip
    'ruf',  # mode 5: shoulder=R, elbow=Up,   wrist=Flip
    'ldf',  # mode 6: shoulder=L, elbow=Down, wrist=Flip
    'rdf',  # mode 7: shoulder=R, elbow=Down, wrist=Flip
]
N_MODES = 8

MODE_NAMES = [
    'L-Up-NoFlip', 'R-Up-NoFlip', 'L-Down-NoFlip', 'R-Down-NoFlip',
    'L-Up-Flip',   'R-Up-Flip',   'L-Down-Flip',   'R-Down-Flip',
]

# ── Dataset generation ────────────────────────────────────────────────────────
N_POSES          = 200_000   # poses to sample — 200k gives ~600k samples (3 modes avg)
FK_POS_TOLERANCE = 1e-4      # metres — max acceptable FK round-trip position error
FK_ROT_TOLERANCE = 1e-3      # radians — max acceptable FK round-trip rotation error
JOINT_LIMIT_MARGIN = 1e-3    # radians — safety buffer inside joint limits
SAVE_EVERY       = 20_000    # checkpoint interval (number of poses processed)

# ── Pose representation ───────────────────────────────────────────────────────
# We represent SE(3) as 9 values: [x, y, z, r11, r21, r31, r12, r22, r32]
# = position (3) + first two columns of rotation matrix (6)
# This is unique, avoids gimbal lock, and is differentiable everywhere.
POSE_DIM  = 9
JOINT_DIM = 6
COND_DIM  = POSE_DIM + N_MODES   # 9 + 8 = 17 (pose + mode one-hot)

# ── cINN architecture ─────────────────────────────────────────────────────────
# v3: 10 blocks × 256 hidden — balanced for precision AND speed
#   v1 was 8×256  → fast but geometrically imprecise (107mm FK error)
#   v2 was 12×384 → precise but too slow on CPU (1200ms p95)
#   v3 is 10×256  → keeps the critical FK loss fixes, fits CPU latency budget
N_COUPLING_BLOCKS = 10       # 8→10: more than v1 but not as heavy as v2
HIDDEN_DIM        = 256      # 256: proven width, much faster than 384
N_HIDDEN_LAYERS   = 3        # depth of each coupling subnet
CLAMP_VALUE       = 2.0      # soft clamping for log-scale in affine coupling

# ── Training (used by colab_train.py) ─────────────────────────────────────────
BATCH_SIZE        = 512      # larger batch for faster throughput on P100
LEARNING_RATE     = 1e-4     # standard for AdamW with this model size
WEIGHT_DECAY      = 1e-5
N_EPOCHS          = 300      # 200→300: more epochs for deeper convergence
WARMUP_EPOCHS     = 15       # warm up for 15 epochs before full LR
LR_SCHEDULER      = 'cosine'  # 'cosine' or 'plateau'

# FK consistency loss — THE CRITICAL SETTINGS for raw flow accuracy
#   v1: weight=1.0, noise=0.20 → 107mm error (catastrophic)
#   v2: weight=5.0, noise=0.03 → 0.47mm mean (good but borderline at 0.5mm tol)
#   v3: weight=10.0, noise=0.01 → target: <0.2mm mean (comfortably inside tolerance)
FK_LOSS_WEIGHT    = 10.0     # 5.0→10.0: double the geometric pressure
FK_LOSS_WARMUP    = 10       # 15→10: start FK loss earlier for more geometric training
FK_LATENT_NOISE_STD = 0.01   # 0.03→0.01: even tighter perturbation for sub-0.5mm

# ── Mode selector training ───────────────────────────────────────────────────
TRAIN_MODE_SELECTOR   = True
MODE_SELECTOR_EPOCHS  = 30    # 25→30 for better convergence
MODE_SELECTOR_LR      = 2e-4
MODE_SELECTOR_BS      = 512
MODE_SELECTOR_DROPOUT = 0.2

GRAD_CLIP_NORM    = 1.0       # gradient clipping max norm

# ── Inference ─────────────────────────────────────────────────────────────────
INF_FK_POS_TOL    = 1e-3      # 5e-4→1e-3 metres: 1mm tolerance for raw flow filtering
                               # (refinement tightens to sub-0.01mm anyway)
INF_FK_ROT_TOL    = 1e-2      # 5e-3→1e-2 radians: slightly relaxed to let more raw
                               # candidates through for refinement
N_LATENT_SAMPLES  = 75        # hybrid sweep winner: 75 per temp (150 total) gives 94% raw @ 0.5mm
INF_TEMPERATURE   = 0.65      # primary temp; secondary temp 0.9 set in INF_TEMPERATURE_2
INF_TEMPERATURE_2 = 0.9       # hybrid: sample N_LATENT_SAMPLES at each of these two temps

# Optional post-flow refinement (recommended for deployment accuracy)
INF_ENABLE_REFINEMENT       = True
INF_MAX_REFINE_CANDIDATES   = 20    # 16→20: try more candidates before giving up
INF_REFINE_MAX_NFEV         = 60    # 50→60: more iterations for tighter convergence
INF_REFINE_ROT_WEIGHT       = 0.35
INF_REFINE_BOUND_MARGIN     = 1e-4

# Solution selection weights (weighted sum objective)
# score = w_joint * joint_disp + w_manip * (-manipulability) + w_margin * (-margin)
W_JOINT_DISP   = 0.5
W_MANIPULABILITY = 0.3
W_JOINT_MARGIN = 0.2
