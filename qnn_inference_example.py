#!/usr/bin/env python3
"""
qnn_inference_example.py
=========================

Example: How to use the trained QNN for inference and comparison with classical ANN.

This script demonstrates:
1. Loading trained QNN model
2. Loading classical ANN for comparison
3. Inference on new test samples
4. Full 6-DOF IK (QNN J1,J2,J3 + analytical J4,J5,J6)
5. Performance metrics

Author: Robotics Team
Date: March 2026
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
from tqdm import tqdm

# Try to import QNN module
try:
    from qnn_puma560 import (
        HybridQNN, sc_to_angles, compute_wrist_center,
        HAS_PENNYLANE
    )
except ImportError as e:
    print(f"Error: Could not import QNN module: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
#  EXAMPLE 1: Load and use QNN
# ═══════════════════════════════════════════════════════════════════════

def example_qnn_inference():
    """Example: QNN inference on a sample pose."""
    
    print("=" * 70)
    print("EXAMPLE 1: QNN Inference")
    print("=" * 70)
    print()
    
    # Check if PennyLane is available
    if not HAS_PENNYLANE:
        print("ERROR: PennyLane required for QNN. Install: pip install pennylane")
        return
    
    # Load trained QNN
    model_path = 'puma560_qnn_hybrid_v1.pt'
    if not Path(model_path).exists():
        print(f"ERROR: {model_path} not found. Train the QNN first:")
        print("  python train_qnn_and_compare.py")
        return
    
    print(f"Loading QNN from {model_path}...")
    ckpt = torch.load(model_path, map_location='cpu')
    
    # Reconstruct model
    model = HybridQNN(n_qubits=3, n_vqc_layers=3, classical_hidden=128)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    P5_mean = ckpt['P5_mean']
    P5_std = ckpt['P5_std']
    
    print("[OK] QNN loaded successfully\n")
    
    # Example pose (end-effector transformation as 12D row)
    # [nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz]
    pose_example = np.array([
        -0.278, -0.840, 0.466,    # n vector
        0.955, -0.188, 0.230,     # o vector
        -0.106, 0.509, 0.854,     # a vector
        -254.3, 333.0, 1502.3,    # position (mm)
    ])
    
    print("Input pose:")
    print(f"  Position (mm): [{pose_example[9]:8.1f}, {pose_example[10]:8.1f}, {pose_example[11]:8.1f}]")
    print(f"  Normal:        [{pose_example[0]:8.4f}, {pose_example[1]:8.4f}, {pose_example[2]:8.4f}]")
    print()
    
    # Extract wrist center
    P5 = compute_wrist_center(pose_example.reshape(1, -1))[0]
    P5_norm = (P5 - P5_mean) / P5_std
    
    print(f"Wrist center P5 (mm): [{P5[0]:7.1f}, {P5[1]:7.1f}, {P5[2]:7.1f}]")
    print(f"Normalized P5:        [{P5_norm[0]:7.3f}, {P5_norm[1]:7.3f}, {P5_norm[2]:7.3f}]")
    print()
    
    # Run inference
    print("Running QNN inference...")
    with torch.no_grad():
        P5_t = torch.tensor(P5_norm, dtype=torch.float32).unsqueeze(0)
        output_raw = model(P5_t)
        output_sc = torch.tanh(output_raw)
    
    # Convert sin/cos to angles
    J123_pred = sc_to_angles(output_sc.numpy())
    
    print("Predicted J1, J2, J3:")
    for j in range(3):
        print(f"  J{j+1} = {J123_pred[0, j]:8.4f}°")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  EXAMPLE 2: Load and use Classical ANN
# ═══════════════════════════════════════════════════════════════════════

def example_classical_ann_inference():
    """Example: Classical ANN inference on the same sample."""
    
    print("=" * 70)
    print("EXAMPLE 2: Classical ANN Inference")
    print("=" * 70)
    print()
    
    model_path = 'puma560_ann_v4_FINAL.pt'
    if not Path(model_path).exists():
        print(f"Warning: {model_path} not found. Skipping classical ANN example.\n")
        return
    
    print(f"Loading classical ANN from {model_path}...")
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location='cpu')
    
    # Reconstruct ShoulderNet architecture
    class ShoulderNet(nn.Module):
        def __init__(self, n_in=3, hidden=256, n_blocks=6):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            )
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                )
                for _ in range(n_blocks)
            ])
            self.head = nn.Linear(hidden, 6)
        
        def forward(self, x):
            x = self.stem(x)
            for blk in self.blocks:
                x = blk(x) + x
            return self.head(x)
    
    model_ann = ShoulderNet(n_in=3, hidden=256, n_blocks=6)
    
    if 'model_state' in ckpt:
        model_ann.load_state_dict(ckpt['model_state'])
    else:
        model_ann.load_state_dict(ckpt)
    
    model_ann.eval()
    
    P5_mean = ckpt.get('P5_mean', np.zeros((1, 3)))
    P5_std = ckpt.get('P5_std', np.ones((1, 3)))
    
    print("[OK] Classical ANN loaded successfully\n")
    
    # Same example pose
    pose_example = np.array([
        -0.278, -0.840, 0.466,    # n
        0.955, -0.188, 0.230,     # o
        -0.106, 0.509, 0.854,     # a
        -254.3, 333.0, 1502.3,    # position
    ])
    
    # Extract and normalize wrist center
    P5 = compute_wrist_center(pose_example.reshape(1, -1))[0]
    P5_norm = (P5 - P5_mean) / P5_std
    
    # Run inference
    print("Running classical ANN inference...")
    with torch.no_grad():
        P5_t = torch.tensor(P5_norm, dtype=torch.float32).unsqueeze(0)
        output_raw = model_ann(P5_t)
        output_sc = torch.tanh(output_raw)
    
    # Convert sin/cos to angles
    from qnn_puma560 import sc_to_angles
    J123_pred_ann = sc_to_angles(output_sc.numpy())
    
    print("Predicted J1, J2, J3:")
    for j in range(3):
        print(f"  J{j+1} = {J123_pred_ann[0, j]:8.4f}°")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  EXAMPLE 3: Full IK with analytical wrist solution
# ═══════════════════════════════════════════════════════════════════════

def example_full_ik():
    """Example: Get full 6-DOF solution (J1-J6) using QNN + analytical wrist."""
    
    print("=" * 70)
    print("EXAMPLE 3: Full 6-DOF IK (QNN + Analytical Wrist)")
    print("=" * 70)
    print()
    
    if not HAS_PENNYLANE:
        print("ERROR: PennyLane required. Install: pip install pennylane")
        return
    
    # NOTE: This is a simplified example. For the full implementation,
    # you would need to:
    # 1. Import solve_wrist from train_puma560.py
    # 2. Use the analytical wrist solver to get J4, J5, J6
    # 3. Verify via FK
    
    print("To implement full 6-DOF IK:")
    print()
    print("1. Predict J1, J2, J3 using QNN (as in Example 1)")
    print()
    print("2. Solve J4, J5, J6 analytically:")
    print("   from train_puma560 import solve_wrist")
    print("   J4, J5, J6 = solve_wrist(J123_pred, T06, flip_wrist=False)")
    print()
    print("3. Assemble full joint vector:")
    print("   J_full = np.array([J1, J2, J3, J4, J5, J6])")
    print()
    print("4. Validate via forward kinematics:")
    print("   from train_puma560 import fPUMA")
    print("   T_check = fPUMA(J_full)")
    print("   position_error = np.linalg.norm(T_check[:3,3] - T_target[:3,3])")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  EXAMPLE 4: Batch inference
# ═══════════════════════════════════════════════════════════════════════

def example_batch_inference():
    """Example: Batch inference on multiple poses."""
    
    print("=" * 70)
    print("EXAMPLE 4: Batch Inference")
    print("=" * 70)
    print()
    
    if not HAS_PENNYLANE:
        print("ERROR: PennyLane required. Install: pip install pennylane")
        return
    
    model_path = 'puma560_qnn_hybrid_v1.pt'
    if not Path(model_path).exists():
        print(f"ERROR: {model_path} not found. Train the QNN first.")
        return
    
    print("Loading QNN...")
    ckpt = torch.load(model_path, map_location='cpu')
    model = HybridQNN(n_qubits=3, n_vqc_layers=3, classical_hidden=128)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    P5_mean = ckpt['P5_mean']
    P5_std = ckpt['P5_std']
    
    print("[OK] Loaded\n")
    
    # Create batch of random poses
    n_batch = 5
    print(f"Creating batch of {n_batch} random poses...")
    
    poses_batch = np.random.randn(n_batch, 12) * np.array(
        [0.1]*3 + [0.1]*3 + [0.1]*3 + [200, 200, 400]  # Rough scaling
    ).reshape(1, 12)
    
    # Normalize pose vectors to unit vectors
    for i in tqdm(range(n_batch), desc="  Normalizing poses", unit="pose"):
        for j in range(3):
            n_vec = poses_batch[i, [j, j+3, j+6]]
            n_vec = n_vec / (np.linalg.norm(n_vec) + 1e-8)
            poses_batch[i, [j, j+3, j+6]] = n_vec
    
    print()
    
    # Extract wrist centers
    P5_batch = compute_wrist_center(poses_batch)
    P5_batch_norm = (P5_batch - P5_mean) / P5_std
    
    print(f"Wrist centers (normalized):")
    for i in range(n_batch):
        print(f"  Sample {i+1}: [{P5_batch_norm[i, 0]:7.3f}, "
              f"{P5_batch_norm[i, 1]:7.3f}, {P5_batch_norm[i, 2]:7.3f}]")
    print()
    
    # Batch inference
    print("Running batch inference...")
    with torch.no_grad():
        P5_batch_t = torch.tensor(P5_batch_norm, dtype=torch.float32)
        output_batch = model(P5_batch_t)
        output_batch_sc = torch.tanh(output_batch)
    
    # Convert to angles
    J123_batch = sc_to_angles(output_batch_sc.numpy())
    
    print(f"Predictions (batch):")
    print(f"{'Sample':<10} {'J1':<12} {'J2':<12} {'J3':<12}")
    print("-" * 46)
    for i in range(n_batch):
        print(f"{i+1:<10} {J123_batch[i,0]:12.4f}° {J123_batch[i,1]:12.4f}° "
              f"{J123_batch[i,2]:12.4f}°")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  EXAMPLE 5: Performance comparison
# ═══════════════════════════════════════════════════════════════════════

def example_performance_comparison():
    """Example: Measure inference speed of QNN vs ANN."""
    
    print("=" * 70)
    print("EXAMPLE 5: Speed Comparison")
    print("=" * 70)
    print()
    
    import time
    
    if not HAS_PENNYLANE:
        print("ERROR: PennyLane required. Install: pip install pennylane")
        return
    
    # Load QNN
    qnn_path = 'puma560_qnn_hybrid_v1.pt'
    ann_path = 'puma560_ann_v4_FINAL.pt'
    
    print("Loading models...")
    
    results = {}
    
    # QNN timing
    if Path(qnn_path).exists():
        ckpt_qnn = torch.load(qnn_path, map_location='cpu')
        qnn = HybridQNN(n_qubits=3, n_vqc_layers=3, classical_hidden=128)
        qnn.load_state_dict(ckpt_qnn['model_state'])
        qnn.eval()
        
        # Generate test batch
        np.random.seed(42)
        P5_test = np.random.randn(100, 3) * 200
        P5_test_norm = (P5_test - ckpt_qnn['P5_mean']) / ckpt_qnn['P5_std']
        
        # Warmup
        with torch.no_grad():
            P5_t = torch.tensor(P5_test_norm[:1], dtype=torch.float32)
            qnn(P5_t)
        
        # Timed inference
        print("\nQNN (hybrid quantum-classical):")
        times_qnn = []
        for i in tqdm(range(10), desc="  Timing QNN", unit="run"):
            P5_t = torch.tensor(P5_test_norm, dtype=torch.float32)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = qnn(P5_t)
            t_end = time.perf_counter()
            times_qnn.append(t_end - t0)
        
        time_per_sample_qnn = np.mean(times_qnn) / 100 * 1000  # ms
        print(f"  Batch (100 samples): {np.mean(times_qnn)*1000:.2f} ms")
        print(f"  Per-sample avg:      {time_per_sample_qnn:.3f} ms")
        
        results['QNN'] = time_per_sample_qnn
    
    # Classical ANN timing
    if Path(ann_path).exists():
        ckpt_ann = torch.load(ann_path, map_location='cpu')
        
        class ShoulderNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Linear(3, 256), nn.LayerNorm(256), nn.GELU(),
                )
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
                        nn.Dropout(0.05),
                        nn.Linear(256, 256), nn.LayerNorm(256),
                    )
                    for _ in range(6)
                ])
                self.head = nn.Linear(256, 6)
            
            def forward(self, x):
                x = self.stem(x)
                for blk in self.blocks:
                    x = blk(x) + x
                return self.head(x)
        
        ann = ShoulderNet()
        ann.load_state_dict(ckpt_ann.get('model_state', ckpt_ann))
        ann.eval()
        
        # Test data
        P5_test = np.random.randn(100, 3) * 200
        P5_test_norm = (P5_test - ckpt_ann['P5_mean']) / ckpt_ann['P5_std']
        
        # Warmup
        with torch.no_grad():
            P5_t = torch.tensor(P5_test_norm[:1], dtype=torch.float32)
            ann(P5_t)
        
        # Timed inference
        print("\nClassical ANN (fully classical):")
        times_ann = []
        for i in tqdm(range(10), desc="  Timing ANN", unit="run"):
            P5_t = torch.tensor(P5_test_norm, dtype=torch.float32)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = ann(P5_t)
            t_end = time.perf_counter()
            times_ann.append(t_end - t0)
        
        time_per_sample_ann = np.mean(times_ann) / 100 * 1000  # ms
        print(f"  Batch (100 samples): {np.mean(times_ann)*1000:.2f} ms")
        print(f"  Per-sample avg:      {time_per_sample_ann:.3f} ms")
        
        results['ANN'] = time_per_sample_ann
    
    # Summary
    print()
    if 'QNN' in results and 'ANN' in results:
        speedup = results['QNN'] / results['ANN']
        slowdown_pct = (speedup - 1) * 100
        status = "⚠ slower" if slowdown_pct > 0 else "✓ faster"
        print(f"QNN is {slowdown_pct:+.1f}% {status.split()[1]} ({status.split()[0]})")
        print(f"  (Due to classical quantum simulation overhead)")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " PUMA 560 QNN Inference Examples ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Run all examples
    example_qnn_inference()
    example_classical_ann_inference()
    example_full_ik()
    example_batch_inference()
    example_performance_comparison()
    
    print("=" * 70)
    print("END OF EXAMPLES")
    print("=" * 70)
    print()
    print("For more details, see README_QNN.md")
    print()
