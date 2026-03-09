#!/usr/bin/env python3
"""
compare_qnn_vs_ann.py
=====================
Loads ALREADY-TRAINED QNN and classical ANN checkpoints and prints
a full comparison table — no retraining required.

Run:
    python compare_qnn_vs_ann.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from qnn_puma560 import (
    HybridQNN, load_dataset, compute_wrist_center,
    normalize_wrist_center, angles_to_sc, sc_to_angles, evaluate_qnn
)

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Correct ResBlock matching train_puma560_v4_FINAL.py ────────────────

class _ResBlock(nn.Module):
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


class ShoulderNet(nn.Module):
    def __init__(self, n_in=3, hidden=256, n_blocks=6, dropout=0.05):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.blocks = nn.ModuleList([_ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 6)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# ── Main ───────────────────────────────────────────────────────────────

SEP = "=" * 80

print(SEP)
print("PUMA 560 IK — QNN vs Classical ANN Comparison")
print(SEP)
print()

# ── 1. Load test data ──────────────────────────────────────────────────
print("[1] Loading dataset...")
_, _, X_test, Y_test = load_dataset('puma560_dataset.csv', test_size=0.15)
P5_test = compute_wrist_center(X_test)
Y_test_sc = angles_to_sc(Y_test[:, :3])
print(f"  Test samples: {len(X_test)}")
print()

# ── 2. Load QNN ────────────────────────────────────────────────────────
qnn_eval = None
qnn_path = 'puma560_qnn_hybrid_v1.pt'

print(f"[2] Loading QNN from {qnn_path}...")
if not Path(qnn_path).exists():
    print(f"  ERROR: {qnn_path} not found. Run train_qnn_and_compare.py first.")
else:
    ckpt_qnn = torch.load(qnn_path, map_location='cpu', weights_only=False)
    nq = ckpt_qnn.get('n_qubits', 4)
    nl = ckpt_qnn.get('n_qlayers', 3)
    qnn_model = HybridQNN(n_qubits=nq, n_qlayers=nl)
    qnn_model.load_state_dict(ckpt_qnn['model_state'])
    qnn_model.eval()

    P5_mean_qnn = ckpt_qnn['P5_mean']
    P5_std_qnn  = ckpt_qnn['P5_std']
    P5_te_n_qnn = (P5_test - P5_mean_qnn) / P5_std_qnn

    qnn_eval = evaluate_qnn(qnn_model, P5_te_n_qnn, Y_test_sc)
    arch = ckpt_qnn.get('architecture', f'HybridQNN_{nq}q_{nl}l')
    train_time = ckpt_qnn.get('train_time', 0)
    hist = ckpt_qnn.get('history', {})

    print(f"  [OK] {arch}  (trained in {train_time:.0f}s)")
    print(f"  Best epoch: {hist.get('best_epoch','?')} | "
          f"Best val MSE: {hist.get('best_val', float('nan')):.6f}")
    print()

# ── 3. Load Classical ANN ──────────────────────────────────────────────
ann_eval       = None
ann_pred_angles = None
ann_path = 'puma560_ann_v4_FINAL.pt'

print(f"[3] Loading Classical ANN from {ann_path}...")
if not Path(ann_path).exists():
    print(f"  WARNING: {ann_path} not found — skipping classical comparison.")
else:
    try:
        ckpt_ann = torch.load(ann_path, map_location='cpu', weights_only=False)
        ann_model = ShoulderNet(n_in=3, hidden=256, n_blocks=6)
        state = ckpt_ann.get('model_state', ckpt_ann)
        ann_model.load_state_dict(state)
        ann_model.eval()

        P5_mean_ann = ckpt_ann.get('P5_mean', np.zeros((1, 3)))
        P5_std_ann  = ckpt_ann.get('P5_std',  np.ones((1, 3)))
        P5_te_n_ann = (P5_test - P5_mean_ann) / P5_std_ann

        with torch.no_grad():
            P5_t = torch.tensor(P5_te_n_ann, dtype=torch.float32)
            raw  = ann_model(P5_t)
            pred_sc = torch.tanh(raw).numpy()

        ann_pred_angles = sc_to_angles(pred_sc)
        true_angles     = Y_test[:, :3]
        true_sc         = Y_test_sc

        wrapped = np.array([
            ((ann_pred_angles[:, j] - true_angles[:, j] + 180) % 360 - 180)
            for j in range(3)
        ])
        ann_eval = {
            'mae':  np.mean(np.abs(wrapped), axis=1),
            'rmse': np.sqrt(np.mean(wrapped ** 2, axis=1)),
            'mse':  np.mean((pred_sc - true_sc) ** 2),
        }
        print(f"  [OK] ShoulderNet loaded")
        print()
    except Exception as e:
        print(f"  ERROR: {e}")
        print()

# ── 4. Print comparison table ──────────────────────────────────────────
print(SEP)
print("RESULTS")
print(SEP)
print(f"{'Method':<30} {'J1 MAE':>10} {'J2 MAE':>10} {'J3 MAE':>10} {'Avg MAE':>10}")
print("-" * 72)

if qnn_eval is not None:
    qnn_avg = qnn_eval['mae'].mean()
    print(f"{'Hybrid QNN v2 (quantum)':<30} "
          f"{qnn_eval['mae'][0]:>10.4f} {qnn_eval['mae'][1]:>10.4f} "
          f"{qnn_eval['mae'][2]:>10.4f} {qnn_avg:>10.4f}")

if ann_eval is not None:
    ann_avg = ann_eval['mae'].mean()
    print(f"{'Classical ANN (ShoulderNet)':<30} "
          f"{ann_eval['mae'][0]:>10.4f} {ann_eval['mae'][1]:>10.4f} "
          f"{ann_eval['mae'][2]:>10.4f} {ann_avg:>10.4f}")

    if qnn_eval is not None:
        print("-" * 72)
        delta = ann_avg - qnn_avg
        pct   = delta / ann_avg * 100
        if qnn_avg < ann_avg:
            print(f"  QNN is BETTER by {delta:.4f}° avg MAE  ({pct:+.1f}%)")
        else:
            print(f"  Classical ANN is better by {-delta:.4f}° avg MAE  ({-pct:+.1f}%)")

print()

# ── 5. RMSE table ──────────────────────────────────────────────────────
print(f"{'Method':<30} {'J1 RMSE':>10} {'J2 RMSE':>10} {'J3 RMSE':>10} {'Avg RMSE':>10}")
print("-" * 72)
if qnn_eval is not None:
    print(f"{'Hybrid QNN v2 (quantum)':<30} "
          f"{qnn_eval['rmse'][0]:>10.4f} {qnn_eval['rmse'][1]:>10.4f} "
          f"{qnn_eval['rmse'][2]:>10.4f} {qnn_eval['rmse'].mean():>10.4f}")
if ann_eval is not None:
    print(f"{'Classical ANN (ShoulderNet)':<30} "
          f"{ann_eval['rmse'][0]:>10.4f} {ann_eval['rmse'][1]:>10.4f} "
          f"{ann_eval['rmse'][2]:>10.4f} {ann_eval['rmse'].mean():>10.4f}")
print()

# ── 6. Plots ───────────────────────────────────────────────────────────
if HAS_MPL and qnn_eval is not None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
    true_angles = Y_test[:, :3]

    for j, ax in enumerate(axes):
        err_qnn = (qnn_eval['pred_angles'][:, j] - true_angles[:, j] + 180) % 360 - 180
        ax.hist(err_qnn, bins=60, alpha=0.65, color='#3366CC',
                label='QNN', edgecolor='none')

        if ann_eval is not None and ann_pred_angles is not None:
            err_ann = (ann_pred_angles[:, j] - true_angles[:, j] + 180) % 360 - 180
            ax.hist(err_ann, bins=60, alpha=0.55, color='#CC3333',
                    label='Classical ANN', edgecolor='none')

        ax.axvline(0, color='k', lw=1.5, ls='--', alpha=0.6)
        ax.set_xlabel(f"Error {labels[j]} (deg)", fontweight='bold')
        ax.set_ylabel("Count")
        ax.set_title(f"J{j+1}: MAE={qnn_eval['mae'][j]:.3f}°", fontweight='bold')
        ax.grid(True, alpha=0.3)
        if j == 2:
            ax.legend(fontsize=10)

    fig.suptitle("QNN vs Classical: Joint Angle Error Distribution",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig('comparison_qnn_vs_ann.png', dpi=150, bbox_inches='tight')
    print("  Saved: comparison_qnn_vs_ann.png")
    plt.close()
