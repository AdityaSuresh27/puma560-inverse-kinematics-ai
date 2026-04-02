#!/usr/bin/env python3
"""
Fast comparison dashboard generation (no fancy rendering)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qnn_puma560 import (
    HybridQNN, load_dataset, compute_wrist_center,
    normalize_wrist_center, angles_to_sc, sc_to_angles, evaluate_qnn,
)

# ── Load test data ─────────────────────────────────────────────────────
_, _, X_test, Y_test = load_dataset('puma560_dataset.csv', test_size=0.15)
P5_test = compute_wrist_center(X_test)
Y_test_sc = angles_to_sc(Y_test[:, :3])
true_angles = Y_test[:, :3]
N_test = len(X_test)

# ── Load QNN ───────────────────────────────────────────────────────────
ckpt_q = torch.load('puma560_qnn_hybrid_v1.pt', map_location='cpu', weights_only=False)
nq = ckpt_q.get('n_qubits', 4)
nl = ckpt_q.get('n_qlayers', 3)
nr = ckpt_q.get('n_res_blocks', 6)
qnn_model = HybridQNN(n_qubits=nq, n_qlayers=nl, n_res_blocks=nr)
qnn_model.load_state_dict(ckpt_q['model_state'])
qnn_model.eval()

P5_mean_q = ckpt_q['P5_mean']
P5_std_q = ckpt_q['P5_std']
P5_t_q = (P5_test - P5_mean_q) / P5_std_q

qnn_eval = evaluate_qnn(qnn_model, P5_t_q, Y_test_sc)
qnn_errors = np.array([
    ((qnn_eval['pred_angles'][:, j] - true_angles[:, j] + 180) % 360 - 180)
    for j in range(3)
])

# ── Load ANN ────────────────────────────────────────────────────────────
class _ResBlock(torch.nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.LayerNorm(dim), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim), torch.nn.LayerNorm(dim),
        )
        self.act = torch.nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class ShoulderNet(torch.nn.Module):
    def __init__(self, n_in=3, hidden=256, n_blocks=6, dropout=0.05):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU(),
        )
        self.blocks = torch.nn.ModuleList([_ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = torch.nn.Linear(hidden, 6)
    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

ckpt_a = torch.load('puma560_ann_v4_FINAL.pt', map_location='cpu', weights_only=False)
ann_model = ShoulderNet()
ann_model.load_state_dict(ckpt_a.get('model_state', ckpt_a))
ann_model.eval()

P5_mean_a = ckpt_a.get('P5_mean', np.zeros((1, 3)))
P5_std_a = ckpt_a.get('P5_std', np.ones((1, 3)))
P5_t_a = (P5_test - P5_mean_a) / P5_std_a

with torch.no_grad():
    raw = ann_model(torch.tensor(P5_t_a, dtype=torch.float32))
    pred_sc = torch.tanh(raw).numpy()

ann_pred_angles = sc_to_angles(pred_sc)
ann_errors = np.array([
    ((ann_pred_angles[:, j] - true_angles[:, j] + 180) % 360 - 180)
    for j in range(3)
])

ann_eval = {
    'mae': np.mean(np.abs(ann_errors), axis=1),
    'rmse': np.sqrt(np.mean(ann_errors ** 2, axis=1)),
    'mse': float(np.mean((pred_sc - Y_test_sc) ** 2)),
}

# ── Plot (simple, fast version) ────────────────────────────────────────
print("Generating dashboard...")
C_QNN = '#2962FF'
C_ANN = '#D50000'
JOINTS = [r'$\theta_1$ (J1)', r'$\theta_2$ (J2)', r'$\theta_3$ (J3)']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('#FAFAFA')

# Row 0: Histograms
for j in range(3):
    ax = axes[0, j]
    ax.hist(qnn_errors[j], bins=60, alpha=0.7, color=C_QNN, label='QNN', edgecolor='none')
    ax.hist(ann_errors[j], bins=60, alpha=0.6, color=C_ANN, label='ANN', edgecolor='none')
    ax.axvline(0, color='#333', lw=1, ls='--')
    ax.set_xlabel('Error (°)')
    ax.set_ylabel('Count')
    ax.set_title(f'{JOINTS[j]}: QNN MAE={qnn_eval["mae"][j]:.4f}° vs ANN {ann_eval["mae"][j]:.4f}°')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F5F5F5')

# Row 1: Scatter plots
for j in range(3):
    ax = axes[1, j]
    idx = np.random.default_rng(42).choice(N_test, min(300, N_test), replace=False)
    ax.scatter(true_angles[idx, j], qnn_eval['pred_angles'][idx, j], 
               s=6, alpha=0.4, color=C_QNN, label='QNN')
    ax.scatter(true_angles[idx, j], ann_pred_angles[idx, j], 
               s=6, alpha=0.3, color=C_ANN, label='ANN', marker='^')
    lo = min(true_angles[idx, j].min(), qnn_eval['pred_angles'][idx, j].min()) - 5
    hi = max(true_angles[idx, j].max(), qnn_eval['pred_angles'][idx, j].max()) + 5
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, zorder=4)
    ax.set_xlabel('True (°)')
    ax.set_ylabel('Predicted (°)')
    ax.set_title(f'{JOINTS[j]}: Predicted vs True')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F5F5F5')

fig.suptitle(
    f'PUMA 560 IK: Hybrid QNN vs Classical ANN\n'
    f'QNN avg MAE: {qnn_eval["mae"].mean():.4f}° | ANN: {ann_eval["mae"].mean():.4f}° '
    f'(QNN better by {(ann_eval["mae"].mean() - qnn_eval["mae"].mean())/ann_eval["mae"].mean()*100:.1f}%)',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig('comparison_dashboard.png', dpi=120, bbox_inches='tight', facecolor='#FAFAFA')
print("Saved: comparison_dashboard.png")
