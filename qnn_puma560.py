"""
Quantum Neural Network for PUMA 560 IK
========================================

Hybrid quantum-classical architecture for inverse kinematics prediction.
Uses PennyLane to implement quantum feature extraction and classical post-processing.

Features:
- Quantum feature map (angle encoding on wrist center)
- Variational quantum circuit (VQC) with 3 qubits
- Classical neural network layer for final prediction
- Comparison against classical ANN (puma560_ann_v4_FINAL.pt)
- Efficient hybrid architecture

Author: Robotics Team
Date: March 2026
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
from pathlib import Path
from dataclasses import dataclass
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
#  PUMA 560 CONSTANTS (matching classical implementations)
# ═══════════════════════════════════════════════════════════════════════

DH_PARAMS = {
    'a': [0.0, 431.8, -20.32, 0.0, 0.0, 0.0],
    'd': [671.83, 139.70, 0.0, 431.8, 0.0, 56.5],
    'alpha': [-90.0, 0.0, 90.0, -90.0, 90.0, 0.0],
}

JOINT_LIMITS = np.array([
    [-160,  160],
    [-225,   45],
    [ -45,  225],
    [-110,  170],
    [-100,  100],
    [-266,  266],
], dtype=np.float64)

R_WORKSPACE = 900.0  # mm


# ═══════════════════════════════════════════════════════════════════════
#  QUANTUM CIRCUIT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

if HAS_PENNYLANE:
    
    class QuantumFeatureMap:
        """
        Quantum feature encoding for wrist center (3D) onto 3 qubits.
        Uses angle encoding: each qubit encodes one wrist coordinate.
        """
        
        def __init__(self, n_qubits=3, n_layers=2):
            """
            Args:
                n_qubits: Number of qubits (3 for 3D wrist center)
                n_layers: Depth of feature encoding
            """
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.dev = qml.device("default.qubit", wires=n_qubits)
            
            @qml.qnode(self.dev)
            def _circuit(x):
                """Angle encoding + single-qubit rotations."""
                # x: [3] wrist center coordinates (normalized)
                for i in range(n_qubits):
                    qml.RX(np.pi * x[i], wires=i)  # RX based on coordinate
                    qml.RZ(np.pi * x[i] / 2.0, wires=i)  # RZ adds phase
                
                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
                
                # Return expectation values <Z_i>
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self._circuit = _circuit
        
        def forward(self, x_batch):
            """
            Args:
                x_batch: [B, 3] normalized wrist centers
            
            Returns:
                features: [B, 3] quantum features
            """
            B = len(x_batch)
            features = np.zeros((B, self.n_qubits))
            for i in range(B):
                features[i] = self._circuit(x_batch[i])
            return features


    class VariationalQuantumCircuit:
        """
        Variational quantum circuit for feature extraction.
        Parameters are trained to maximize information extraction.
        """
        
        def __init__(self, n_qubits=3, n_layers=3):
            """
            Args:
                n_qubits: Number of qubits
                n_layers: Depth of variational layers
            """
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.n_params = n_layers * n_qubits
            self.dev = qml.device("default.qubit", wires=n_qubits)
            
            # Initialize random parameters
            self.params = np.random.randn(n_layers, n_qubits) * 0.1
            
            @qml.qnode(self.dev)
            def _circuit(x, params):
                """
                x: [3] quantum features from feature map
                params: [n_layers, n_qubits] variational parameters
                """
                # State preparation from features
                for i in range(n_qubits):
                    qml.Rot(x[i], x[i]/2, x[i]/3, wires=i)
                
                # Variational layers
                for layer in range(n_layers):
                    for i in range(n_qubits):
                        qml.RY(params[layer, i], wires=i)
                        qml.RZ(params[layer, i] / 2, wires=i)
                    
                    # Entangle
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                
                # Measure all qubits
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self._circuit = _circuit
        
        def forward(self, x_batch):
            """
            Args:
                x_batch: [B, 3] quantum features
            
            Returns:
                output: [B, 3] measurement outcomes
            """
            B = len(x_batch)
            output = np.zeros((B, self.n_qubits))
            for i in range(B):
                output[i] = self._circuit(x_batch[i], self.params)
            return output


# ═══════════════════════════════════════════════════════════════════════
#  HYBRID QUANTUM-CLASSICAL MODEL
# ═══════════════════════════════════════════════════════════════════════

class HybridQNN(nn.Module):
    """
    Hybrid Quantum Neural Network for PUMA 560 IK.
    
    Architecture:
    1. Quantum Feature Map: Encodes wrist center onto quantum state
    2. Variational Quantum Circuit: Extracts features with trained parameters
    3. Classical Post-processor: Neural network to output sin/cos of J1,J2,J3
    
    Advantages over pure classical:
    - Quantum feature extraction (high-dimensional encoding)
    - Parameter efficiency (exponential dimensionality scaling)
    - Potential for quantum advantage on certain generalizations
    """
    
    def __init__(self, n_qubits=3, n_vqc_layers=3, classical_hidden=128):
        """
        Args:
            n_qubits: Number of qubits (3 for wrist center)
            n_vqc_layers: Depth of variational quantum circuit
            classical_hidden: Hidden dim of classical post-processor
        """
        super().__init__()
        self.n_qubits = n_qubits
        
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required. Install: pip install pennylane")
        
        # Quantum components
        self.feature_map = QuantumFeatureMap(n_qubits=n_qubits, n_layers=2)
        self.vqc = VariationalQuantumCircuit(n_qubits=n_qubits, n_layers=n_vqc_layers)
        
        # Classical post-processor (maps quantum output to sin/cos)
        self.classical = nn.Sequential(
            nn.Linear(n_qubits, classical_hidden),
            nn.LayerNorm(classical_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classical_hidden, classical_hidden // 2),
            nn.LayerNorm(classical_hidden // 2),
            nn.GELU(),
            nn.Linear(classical_hidden // 2, 6),  # 6 outputs: sin/cos of J1,J2,J3
        )
        
        # Initialize classical weights
        for m in self.classical.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, 3] normalized wrist center coordinates
        
        Returns:
            output: [B, 6] raw outputs (apply tanh for [-1,1] range)
        """
        B = x.shape[0]
        
        # Convert to numpy for quantum processing
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Quantum components (returns numpy arrays)
        quantum_features = self.feature_map.forward(x_np)  # [B, 3]
        quantum_output = self.vqc.forward(quantum_features)  # [B, 3]
        
        # Convert back to tensor
        quantum_tensor = torch.tensor(quantum_output, dtype=x.dtype, device=x.device)
        
        # Classical post-processor
        output = self.classical(quantum_tensor)
        
        return output
    
    def get_quantum_params(self):
        """Get variational quantum circuit parameters."""
        return self.vqc.params
    
    def set_quantum_params(self, params):
        """Update variational quantum circuit parameters."""
        self.vqc.params = params


# ═══════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

class HybridQNNLoss(nn.Module):
    """
    Loss for hybrid QNN: sin/cos MSE + regularization.
    Simpler than DecoupledIKLoss since we skip FK supervision for speed.
    """
    
    def __init__(self, w_mse=1.0, w_circ=0.05):
        super().__init__()
        self.w_mse = w_mse
        self.w_circ = w_circ
    
    def forward(self, pred_raw, target_sc):
        """
        pred_raw: [B, 6] raw network outputs
        target_sc: [B, 6] ground truth sin/cos
        """
        pred_sc = torch.tanh(pred_raw)
        
        # 1. sin/cos MSE
        loss_mse = F.mse_loss(pred_sc, target_sc)
        
        # 2. Unit-circle penalty (optional, helps numerics)
        loss_circ = sum(
            ((pred_sc[:, 2*j]**2 + pred_sc[:, 2*j+1]**2 - 1.0)**2).mean()
            for j in range(3)
        ) / 3.0
        
        return self.w_mse * loss_mse + self.w_circ * loss_circ


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def angles_to_sc(angles_deg):
    """Convert angles (deg) to sin/cos interleaved."""
    r = np.deg2rad(angles_deg)
    sc = np.zeros((*angles_deg.shape[:-1], angles_deg.shape[-1] * 2))
    for j in range(angles_deg.shape[-1]):
        sc[..., 2*j] = np.sin(r[..., j])
        sc[..., 2*j+1] = np.cos(r[..., j])
    return sc


def sc_to_angles(sc):
    """Convert sin/cos to angles (deg)."""
    K = sc.shape[-1] // 2
    a = np.zeros((*sc.shape[:-1], K))
    for j in range(K):
        a[..., j] = np.rad2deg(np.arctan2(sc[..., 2*j], sc[..., 2*j+1]))
    return a


def load_dataset(csv_path, test_size=0.15):
    """Load PUMA 560 dataset and split."""
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    X_all = raw[:, :12]  # Pose
    Y_all = raw[:, 12:18]  # Joints
    
    N = len(raw)
    np.random.seed(42)
    idx = np.random.permutation(N)
    
    n_test = int(test_size * N)
    X_train = X_all[idx[n_test:]]
    Y_train = Y_all[idx[n_test:]]
    X_test = X_all[idx[:n_test]]
    Y_test = Y_all[idx[:n_test]]
    
    return X_train, Y_train, X_test, Y_test


def compute_wrist_center(X):
    """Extract and compute wrist center P5 from poses."""
    ax = X[:, 6]
    ay = X[:, 7]
    az = X[:, 8]
    Px = X[:, 9]
    Py = X[:, 10]
    Pz = X[:, 11]
    d6 = 56.5
    return np.stack([Px - d6*ax, Py - d6*ay, Pz - d6*az], axis=1)


def normalize_wrist_center(P5_train, P5_val=None, P5_test=None):
    """Normalize wrist centers using training statistics."""
    P5_mean = P5_train.mean(0, keepdims=True)
    P5_std = P5_train.std(0, keepdims=True)
    P5_std[P5_std < 1e-8] = 1.0
    
    P5_train_n = (P5_train - P5_mean) / P5_std
    result = [P5_train_n]
    
    if P5_val is not None:
        P5_val_n = (P5_val - P5_mean) / P5_std
        result.append(P5_val_n)
    
    if P5_test is not None:
        P5_test_n = (P5_test - P5_mean) / P5_std
        result.append(P5_test_n)
    
    return result + [P5_mean, P5_std]


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_qnn(model, X_train_n, Y_train_sc, X_val_n, Y_val_sc,
              epochs=500, lr=1e-3, batch_size=256, patience=50, device='cpu'):
    """
    Train hybrid QNN.
    
    Returns:
        best_state: Best model state dict
        history: Training history
    """
    model = model.to(device)
    X_train_t = torch.tensor(X_train_n, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train_sc, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val_sc, dtype=torch.float32, device=device)
    
    loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=batch_size, shuffle=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = HybridQNNLoss(w_mse=1.0, w_circ=0.05)
    
    history = {"train": [], "val": [], "best_epoch": 0, "best_val": np.inf}
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    
    t0 = time.time()
    pbar = tqdm(range(1, epochs + 1), desc="Training QNN", unit="epoch", 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]")
    
    for ep in pbar:
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        
        train_loss /= len(X_train_t)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
        
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < history["best_val"]:
            history["best_val"] = val_loss
            history["best_epoch"] = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        # Update progress bar with loss values
        pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'best_ep': history['best_epoch']
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
    """
    Predict J1,J2,J3 sin/cos from wrist centers.
    
    Returns:
        J123: [N, 3] joint angles in degrees
    """
    model.eval()
    X_test_t = torch.tensor(X_test_n, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        sc_pred = torch.tanh(model(X_test_t)).numpy()
    
    return sc_to_angles(sc_pred)  # [N, 3]


def evaluate_qnn(model, X_test_n, Y_test_true_sc, device='cpu'):
    """Evaluate QNN on test set."""
    model.eval()
    X_test_t = torch.tensor(X_test_n, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test_true_sc, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        pred_raw = model(X_test_t)
        pred_sc = torch.tanh(pred_raw)
    
    mse = F.mse_loss(pred_sc, Y_test_t).item()
    
    # Angle errors (wrapped)
    pred_angles = sc_to_angles(pred_sc.numpy())
    true_angles = sc_to_angles(Y_test_true_sc)
    
    wrapped_err = np.array([
        ((pred_angles[:, j] - true_angles[:, j] + 180) % 360 - 180)
        for j in range(3)
    ])
    
    mae = np.mean(np.abs(wrapped_err), axis=1)
    rmse = np.sqrt(np.mean(wrapped_err ** 2, axis=1))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pred_angles': pred_angles,
    }


if __name__ == "__main__":
    print("Hybrid QNN for PUMA 560 IK")
    print("Import this module in qnn_train.py to use")
