"""
Quantum Neural Network for PUMA 560 IK  (v2.0 — Complete Rewrite)
==================================================================

v1.0 was fundamentally broken:
  - Gradient flow severed by .detach().numpy() in forward()
  - VQC params stored as numpy arrays, never optimised by torch
  - No FK wrist-center supervision (the key to classical ANN success)
  - No skip connection — information destroyed by random quantum layer
  - Undersized classical head (128→64→6, no residual blocks)
  - ReduceLROnPlateau instead of OneCycleLR

v2.0 fixes ALL of these:
  1. Fully differentiable quantum layer via PennyLane TorchLayer + backprop
  2. Data re-uploading circuit for high expressivity
  3. Skip connection: concat(input, quantum_features) → classical head
  4. Deep residual classical head (256-wide, 4 ResBlocks)
  5. DecoupledIKLoss with differentiable FK wrist-center supervision
  6. OneCycleLR scheduler (matches classical training pipeline)

Requirements:
    pip install torch numpy pennylane tqdm
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
from pathlib import Path
import warnings
from tqdm import tqdm

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("WARNING: PennyLane not installed. Install with: pip install pennylane")

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
#  PUMA 560 DH PARAMETERS  (matching fPUMA.m / iPUMA.m / v4 exactly)
# ═══════════════════════════════════════════════════════════════════════

DH = [
    (0.0,    671.83, -90.0),   # Link 1
    (431.8,  139.70,   0.0),   # Link 2
    (-20.32,   0.0,   90.0),   # Link 3
    (0.0,    431.8,  -90.0),   # Link 4
    (0.0,      0.0,   90.0),   # Link 5
    (0.0,     56.5,    0.0),   # Link 6
]

D1, D2, D4, D6 = DH[0][1], DH[1][1], DH[3][1], DH[5][1]
A2, A3 = DH[1][0], DH[2][0]
R_WORKSPACE = 900.0

# Backward-compatible dict alias
DH_PARAMS = {
    'a':     [dh[0] for dh in DH],
    'd':     [dh[1] for dh in DH],
    'alpha': [dh[2] for dh in DH],
}

JOINT_LIMITS = np.array([
    [-160,  160],
    [-225,   45],
    [ -45,  225],
    [-110,  170],
    [-100,  100],
    [-266,  266],
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
#  DIFFERENTIABLE FK  (PyTorch, ported from train_puma560_v4_FINAL.py)
# ═══════════════════════════════════════════════════════════════════════

def _dh_torch(a, d, alpha_deg, theta_batch):
    """DH transform. theta_batch: [B] tensor in degrees → [B,4,4]."""
    al = math.radians(alpha_deg)
    ca, sa = math.cos(al), math.sin(al)
    ca_t = torch.tensor(ca, dtype=theta_batch.dtype, device=theta_batch.device)
    sa_t = torch.tensor(sa, dtype=theta_batch.dtype, device=theta_batch.device)
    a_t = torch.tensor(float(a), dtype=theta_batch.dtype, device=theta_batch.device)
    d_t = torch.tensor(float(d), dtype=theta_batch.dtype, device=theta_batch.device)

    th = theta_batch * (math.pi / 180.0)
    ct, st = torch.cos(th), torch.sin(th)
    B = theta_batch.shape[0]

    T = torch.zeros(B, 4, 4, dtype=theta_batch.dtype, device=theta_batch.device)
    T[:, 0, 0] = ct;          T[:, 0, 1] = -st * ca_t
    T[:, 0, 2] = st * sa_t;   T[:, 0, 3] = a_t * ct
    T[:, 1, 0] = st;          T[:, 1, 1] = ct * ca_t
    T[:, 1, 2] = -ct * sa_t;  T[:, 1, 3] = a_t * st
    T[:, 2, 1] = sa_t;        T[:, 2, 2] = ca_t
    T[:, 2, 3] = d_t
    T[:, 3, 3] = 1.0
    return T


def fPUMA_torch_3(theta123):
    """Differentiable FK for joints 1-3.  theta123: [B,3] degrees → [B,4,4]."""
    B = theta123.shape[0]
    T = torch.eye(4, dtype=theta123.dtype, device=theta123.device) \
             .unsqueeze(0).expand(B, -1, -1).clone()
    for i in range(3):
        Ti = _dh_torch(DH[i][0], DH[i][1], DH[i][2], theta123[:, i])
        T = torch.bmm(T, Ti)
    return T


def sc_to_angles_torch(sc):
    """Sin/cos [B,2K] → angles [B,K] in degrees (differentiable)."""
    K = sc.shape[-1] // 2
    a = torch.zeros(*sc.shape[:-1], K, dtype=sc.dtype, device=sc.device)
    for j in range(K):
        a[..., j] = torch.atan2(sc[..., 2*j], sc[..., 2*j+1]) * (180.0 / math.pi)
    return a


# ═══════════════════════════════════════════════════════════════════════
#  QUANTUM LAYER  (differentiable, data re-uploading)
# ═══════════════════════════════════════════════════════════════════════

if HAS_PENNYLANE:

    def _make_quantum_layer(n_qubits, n_qlayers, n_inputs=3):
        """
        Build a PennyLane TorchLayer with data re-uploading VQC.

        Each layer:
          1. RX data encoding (inputs distributed across qubits)
          2. Trainable Rot(φ, θ, ω) on every qubit
          3. CNOT ring entanglement

        Returns an nn.Module with shape [B, n_inputs] → [B, n_qubits].
        Gradients flow through BOTH quantum weights AND inputs.
        """
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for layer in range(n_qlayers):
                # Data encoding (re-uploaded every layer)
                for i in range(n_qubits):
                    qml.RX(inputs[..., i % n_inputs], wires=i)
                # Trainable variational rotations
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0],
                            weights[layer, i, 1],
                            weights[layer, i, 2], wires=i)
                # Entangling: CNOT ring
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)


# ═══════════════════════════════════════════════════════════════════════
#  HYBRID QUANTUM-CLASSICAL MODEL
# ═══════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Residual block with LayerNorm + GELU (matches classical ShoulderNet)."""
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical NN for PUMA 560 IK (v3 — additive correction).

    Architecture:
      input [B,3]  ──┬──  Classical backbone (= ShoulderNet ANN)  ──→ [B, 6] classical_out
                      │
                      └──  Quantum branch                         ──→ [B, 6] q_correction
                           │  input_scaling → VQC → n_qubits expvals
                           │  → projection(n_qubits → 6, zero-init)
                                                                          │
                                                          output = classical_out + q_correction

    Key design:
      • Classical backbone is IDENTICAL to ShoulderNet ANN (3→256→6 ResBlocks→6)
      • Transfer learning copies ANN weights 1:1 — no shape mismatch
      • q_proj last layer is zero-initialised → at epoch 0 the model IS the ANN
      • If quantum learns nothing useful, the model cannot degrade below ANN
      • Quantum branch only needs to learn a small residual correction
    """

    def __init__(self, n_qubits=4, n_qlayers=3, hidden=256,
                 n_res_blocks=6, dropout=0.05):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers

        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required. Install: pip install pennylane")

        # ── Classical backbone (identical to ShoulderNet ANN) ──────────
        self.stem = nn.Sequential(
            nn.Linear(3, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden, dropout) for _ in range(n_res_blocks)]
        )
        self.head = nn.Linear(hidden, 6)    # sin/cos of J1, J2, J3

        # ── Quantum correction branch ─────────────────────────────────
        self.quantum_layer = _make_quantum_layer(n_qubits, n_qlayers, n_inputs=3)
        self.input_scaling = nn.Parameter(torch.ones(3) * 0.5)
        self.q_proj = nn.Sequential(
            nn.Linear(n_qubits, 32), nn.GELU(),
            nn.Linear(32, 6),
        )

        # Zero-init the last projection layer → q_correction = 0 at init
        # This means QNN = ANN exactly at initialisation
        nn.init.zeros_(self.q_proj[-1].weight)
        nn.init.zeros_(self.q_proj[-1].bias)

        # Kaiming init for classical backbone
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'q_proj' not in name:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, 3]  normalised wrist centre
        Returns: [B, 6]  raw logits (apply tanh externally for sin/cos)
        """
        # Classical backbone (= ANN)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        classical_out = self.head(h)

        # Quantum correction (additive residual)
        q_input = x * self.input_scaling
        q_features = self.quantum_layer(q_input)    # [B, n_qubits]
        q_correction = self.q_proj(q_features)      # [B, 6]

        return classical_out + q_correction


# ═══════════════════════════════════════════════════════════════════════
#  LOSS FUNCTION  (DecoupledIKLoss — ported from train_puma560_v4_FINAL)
# ═══════════════════════════════════════════════════════════════════════

class DecoupledIKLoss(nn.Module):
    """
    Physics-informed loss identical to the classical training pipeline:

      L = w_sc  · MSE(pred_sc, true_sc)
        + w_wc  · MSE(FK_wrist(pred_J123), P5_true) / R²
        + w_circ · mean_j (sin²+cos²−1)²

    The FK term backprops through: output → tanh → atan2 → FK → wrist pos.
    """

    def __init__(self, w_sc=1.0, w_wc=2.0, w_circ=0.05):
        super().__init__()
        self.w_sc = w_sc
        self.w_wc = w_wc
        self.w_circ = w_circ

    def forward(self, pred_raw, target_sc, P5_target):
        """
        pred_raw:   [B, 6]  raw model output (before tanh)
        target_sc:  [B, 6]  ground-truth sin/cos
        P5_target:  [B, 3]  ground-truth wrist centre (mm, NOT normalised)
        """
        pred_sc = torch.tanh(pred_raw)

        # 1. sin/cos MSE
        loss_sc = F.mse_loss(pred_sc, target_sc)

        # 2. FK wrist-centre supervision
        pred_J123 = sc_to_angles_torch(pred_sc)          # [B, 3] degrees
        T03 = fPUMA_torch_3(pred_J123)                   # [B, 4, 4]
        # Wrist centre = frame-3 origin + D4 along frame-3 z-axis
        P5_pred = T03[:, :3, 3] + D4 * T03[:, :3, 2]    # [B, 3]

        Rw = torch.tensor(R_WORKSPACE, dtype=pred_raw.dtype,
                          device=pred_raw.device)
        loss_wc = F.mse_loss(P5_pred / Rw, P5_target / Rw)

        # 3. Unit-circle penalty
        loss_circ = sum(
            ((pred_sc[:, 2*j]**2 + pred_sc[:, 2*j+1]**2 - 1.0)**2).mean()
            for j in range(3)
        ) / 3.0

        return self.w_sc * loss_sc + self.w_wc * loss_wc + self.w_circ * loss_circ


# Simpler loss for use without raw P5 (e.g., inference benchmarks)
class SimpleSCLoss(nn.Module):
    """MSE on sin/cos + unit-circle penalty (no FK)."""
    def __init__(self, w_mse=1.0, w_circ=0.05):
        super().__init__()
        self.w_mse = w_mse
        self.w_circ = w_circ

    def forward(self, pred_raw, target_sc):
        pred_sc = torch.tanh(pred_raw)
        loss_mse = F.mse_loss(pred_sc, target_sc)
        loss_circ = sum(
            ((pred_sc[:, 2*j]**2 + pred_sc[:, 2*j+1]**2 - 1.0)**2).mean()
            for j in range(3)
        ) / 3.0
        return self.w_mse * loss_mse + self.w_circ * loss_circ


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def angles_to_sc(angles_deg):
    """[..., K] angles in degrees → [..., 2K] sin/cos interleaved."""
    r = np.deg2rad(angles_deg)
    sc = np.zeros((*angles_deg.shape[:-1], angles_deg.shape[-1] * 2))
    for j in range(angles_deg.shape[-1]):
        sc[..., 2*j]   = np.sin(r[..., j])
        sc[..., 2*j+1] = np.cos(r[..., j])
    return sc


def sc_to_angles(sc):
    """[..., 2K] sin/cos → [..., K] angles in degrees ∈ [-180,180]."""
    K = sc.shape[-1] // 2
    a = np.zeros((*sc.shape[:-1], K))
    for j in range(K):
        a[..., j] = np.rad2deg(np.arctan2(sc[..., 2*j], sc[..., 2*j+1]))
    return a


def load_dataset(csv_path, test_size=0.15):
    """Load PUMA 560 dataset and split into train/test."""
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    X_all = raw[:, :12]     # Pose (nx,ny,nz,...,Px,Py,Pz)
    Y_all = raw[:, 12:18]   # Joints (theta1..theta6)

    N = len(raw)
    np.random.seed(42)
    idx = np.random.permutation(N)

    n_test = int(test_size * N)
    return (X_all[idx[n_test:]], Y_all[idx[n_test:]],
            X_all[idx[:n_test]],  Y_all[idx[:n_test]])


def compute_wrist_center(X):
    """X: [N,12] → wrist centre P5: [N,3]  (mm)."""
    ax, ay, az = X[:, 6], X[:, 7], X[:, 8]
    Px, Py, Pz = X[:, 9], X[:, 10], X[:, 11]
    return np.stack([Px - D6*ax, Py - D6*ay, Pz - D6*az], axis=1)


def normalize_wrist_center(P5_train, P5_val=None, P5_test=None):
    """Z-score normalisation using training statistics."""
    P5_mean = P5_train.mean(0, keepdims=True)
    P5_std  = P5_train.std(0, keepdims=True)
    P5_std[P5_std < 1e-8] = 1.0

    result = [(P5_train - P5_mean) / P5_std]
    if P5_val is not None:
        result.append((P5_val - P5_mean) / P5_std)
    if P5_test is not None:
        result.append((P5_test - P5_mean) / P5_std)
    return result + [P5_mean, P5_std]


# ═══════════════════════════════════════════════════════════════════════
#  TRANSFER LEARNING  (init QNN classical head from trained ANN)
# ═══════════════════════════════════════════════════════════════════════

def transfer_ann_weights(qnn_model, ann_checkpoint_path):
    """
    Initialise the QNN's classical backbone from a trained ShoulderNet checkpoint.

    Since v3 uses an additive architecture where the classical backbone is IDENTICAL
    to ShoulderNet (3→256 stem, 6 ResBlocks, head→6), ALL weights copy 1:1.
    The quantum projection (q_proj) stays zero-init, so at epoch 0 the QNN outputs
    are exactly the ANN outputs.

    Args:
        qnn_model: HybridQNN instance
        ann_checkpoint_path: path to puma560_ann_v4_FINAL.pt

    Returns:
        True on success, False on failure.
    """
    if not Path(ann_checkpoint_path).exists():
        print(f"  [WARN] ANN checkpoint not found: {ann_checkpoint_path}")
        return False

    try:
        ckpt = torch.load(ann_checkpoint_path, map_location='cpu',
                          weights_only=False)
        ann_sd = ckpt.get('model_state', ckpt)

        qnn_sd = qnn_model.state_dict()
        transferred = 0

        for key, val in ann_sd.items():
            # stem, blocks, head all map 1:1 (shapes are identical)
            if key in qnn_sd and qnn_sd[key].shape == val.shape:
                qnn_sd[key] = val.clone()
                transferred += 1

        qnn_model.load_state_dict(qnn_sd)
        print(f"  [OK] Transferred {transferred} parameter tensors from ANN → QNN (1:1)")
        return True

    except Exception as e:
        print(f"  [WARN] Transfer learning failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_qnn(model, X_train_n, Y_train_sc, P5_train_raw,
              X_val_n, Y_val_sc, P5_val_raw,
              epochs=200, lr=3e-3, batch_size=128, patience=100,
              device='cpu', transfer_mode=False):
    """
    Train hybrid QNN with physics-informed loss and CosineAnnealingWarmRestarts.

    Args:
        X_train_n / X_val_n:     normalised wrist centres [N,3]
        Y_train_sc / Y_val_sc:   sin/cos targets [N,6]
        P5_train_raw / P5_val_raw: raw wrist centres in mm [N,3]  (for FK loss)
        epochs, lr, batch_size, patience: training hyper-parameters
        transfer_mode: if True, use differential LR (quantum: lr, classical: lr/100)
                       to preserve transferred ANN weights

    Returns:
        best_state:  state_dict of best model (by val loss)
        history:     dict with 'train', 'val', 'best_epoch', 'best_val'
    """
    model = model.to(device)

    f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    X_tr_t,  Y_tr_t,  P5_tr_t  = f32(X_train_n),  f32(Y_train_sc),  f32(P5_train_raw)
    X_val_t, Y_val_t, P5_val_t = f32(X_val_n),     f32(Y_val_sc),    f32(P5_val_raw)

    loader = DataLoader(
        TensorDataset(X_tr_t, Y_tr_t, P5_tr_t),
        batch_size=batch_size, shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)

    if transfer_mode:
        # Differential LR: quantum branch trains aggressively,
        # classical backbone barely moves (already good from ANN).
        quantum_names = {'quantum_layer', 'input_scaling', 'q_proj'}
        q_params, c_params = [], []
        for name, p in model.named_parameters():
            if any(qn in name for qn in quantum_names):
                q_params.append(p)
            else:
                c_params.append(p)
        optimizer = torch.optim.AdamW([
            {'params': q_params, 'lr': lr},           # quantum: full LR
            {'params': c_params, 'lr': lr * 0.01},    # classical: 100× smaller
        ], weight_decay=5e-5)
        print(f"  [Diff LR] quantum={lr:.1e}, classical={lr*0.01:.1e}")

    # CosineAnnealingWarmRestarts: period T_0=100 epochs, doubles each restart.
    # LR profile is INDEPENDENT of total epoch count — avoids the OneCycleLR
    # bug where long runs had a 150-epoch warmup and never converged.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=lr * 1e-3,
    )
    criterion = DecoupledIKLoss(w_sc=1.0, w_wc=2.0, w_circ=0.05)

    history = {"train": [], "val": [], "best_epoch": 0, "best_val": np.inf}
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    pbar = tqdm(range(1, epochs + 1), desc="Training QNN", unit="epoch",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]")

    for ep in pbar:
        model.train()
        train_loss = 0.0
        for xb, yb, p5b in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb, p5b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)

        train_loss /= len(X_tr_t)
        # Step CosineAnnealingWarmRestarts once per epoch (not per batch)
        scheduler.step(ep - 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t, P5_val_t).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < history["best_val"]:
            history["best_val"] = val_loss
            history["best_epoch"] = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        pbar.set_postfix({
            'train': f'{train_loss:.6f}',
            'val': f'{val_loss:.6f}',
            'best': history['best_epoch'],
        })

        if no_improve >= patience:
            pbar.close()
            print(f"\n[OK] Early stopping at epoch {ep}")
            break

    return best_state, history


# ═══════════════════════════════════════════════════════════════════════
#  INFERENCE & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def predict_qnn(model, X_test_n, device='cpu'):
    """Predict J1,J2,J3 (degrees) from normalised wrist centres."""
    model.eval()
    X_t = torch.tensor(X_test_n, dtype=torch.float32, device=device)
    with torch.no_grad():
        sc_pred = torch.tanh(model(X_t)).cpu().numpy()
    return sc_to_angles(sc_pred)


def evaluate_qnn(model, X_test_n, Y_test_true_sc, device='cpu'):
    """Evaluate QNN on test set.  Returns dict with mse, mae, rmse, pred_angles."""
    model.eval()
    X_t = torch.tensor(X_test_n, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_test_true_sc, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_raw = model(X_t)
        pred_sc  = torch.tanh(pred_raw)

    mse = F.mse_loss(pred_sc, Y_t).item()

    pred_angles = sc_to_angles(pred_sc.cpu().numpy())
    true_angles = sc_to_angles(Y_test_true_sc)

    wrapped_err = np.array([
        ((pred_angles[:, j] - true_angles[:, j] + 180) % 360 - 180)
        for j in range(3)
    ])

    return {
        'mse':  mse,
        'mae':  np.mean(np.abs(wrapped_err), axis=1),
        'rmse': np.sqrt(np.mean(wrapped_err ** 2, axis=1)),
        'pred_angles': pred_angles,
    }


if __name__ == "__main__":
    print("Hybrid QNN v2.0 for PUMA 560 IK")
    print("Run:  python train_qnn_and_compare.py")
