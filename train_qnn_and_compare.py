#!/usr/bin/env python3
"""
train_qnn_and_compare.py
========================

Quantum Neural Network Training & Comparison with Classical ANN

This script:
1. Loads PUMA 560 dataset
2. Trains hybrid QNN
3. Compares against classical ANN (puma560_ann_v4_FINAL.pt)
4. Visualizes results

Run:
    python train_qnn_and_compare.py [--epochs 500] [--no-gpu]

Author: Robotics Team
Date: March 2026
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from qnn_puma560 import (
        HybridQNN, angles_to_sc, sc_to_angles, load_dataset,
        compute_wrist_center, normalize_wrist_center,
        train_qnn, predict_qnn, evaluate_qnn, HAS_PENNYLANE,
        transfer_ann_weights,
    )
except ImportError as e:
    print(f"ERROR: Could not import QNN module: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
#  CLASSICAL ANN LOADER (for comparison)
# ═══════════════════════════════════════════════════════════════════════

class ClassicalANNLoader:
    """Load pre-trained classical ANN for comparison."""
    
    @staticmethod
    def load(model_path='puma560_ann_v4_FINAL.pt', device='cpu'):
        """Load classical ANN model."""
        if not Path(model_path).exists():
            print(f"Warning: {model_path} not found. Skipping classical comparison.")
            return None, None, None, None
        
        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)

            # ResBlock must match train_puma560_v4_FINAL.py exactly:
            # keys are  blocks.N.block.* and blocks.N.act  (not blocks.N.*)
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
                    self.blocks = nn.ModuleList([
                        _ResBlock(hidden, dropout) for _ in range(n_blocks)
                    ])
                    self.head = nn.Linear(hidden, 6)

                def forward(self, x):
                    x = self.stem(x)
                    for blk in self.blocks:
                        x = blk(x)
                    return self.head(x)

            model = ShoulderNet(n_in=3, hidden=256, n_blocks=6)
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            else:
                model.load_state_dict(ckpt)
            
            model = model.to(device).eval()
            
            P5_mean = ckpt.get('P5_mean', np.zeros((1, 3)))
            P5_std = ckpt.get('P5_std', np.ones((1, 3)))
            
            return model, P5_mean, P5_std, ckpt
        
        except Exception as e:
            print(f"Error loading classical ANN: {e}")
            return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════
#  MAIN TRAINING & COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="QNN Training & Comparison")
    parser.add_argument("--dataset", default="puma560_dataset.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--transfer", action="store_true",
                        help="Init QNN classical head from trained ANN weights "
                             "(recommended — starts near ANN accuracy)")
    parser.add_argument("--n-qubits", type=int, default=4,
                        help="Number of qubits (4=fast, 6=more expressive)")
    parser.add_argument("--n-qlayers", type=int, default=3,
                        help="Depth of quantum data re-uploading circuit")
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-classical", action="store_true",
                        help="Skip loading/comparing against classical ANN")
    args = parser.parse_args()
    
    # ── Check dependencies ─────────────────────────────────────────────
    if not HAS_PENNYLANE:
        print("ERROR: PennyLane is required for QNN.")
        print("Install with: pip install pennylane")
        sys.exit(1)
    
    # ── Setup ──────────────────────────────────────────────────────────
    SEP = "=" * 80
    device = torch.device("cpu" if (args.no_gpu or not torch.cuda.is_available()) else "cuda")
    
    print(SEP)
    print("PUMA 560 IK: Quantum vs Classical Comparison  (QNN v2.0)")
    print(SEP); print()
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Quantum: {args.n_qubits} qubits, {args.n_qlayers} layers")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch}\n")
    
    # ── [0] Load dataset ───────────────────────────────────────────────
    print("[0] Loading dataset...")
    try:
        X_train, Y_train, X_test, Y_test = load_dataset(
            args.dataset, test_size=0.15
        )
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}\n")
    except FileNotFoundError:
        print(f"ERROR: {args.dataset} not found\n")
        sys.exit(1)
    
    # ── [1] Prepare wrist centers & targets ─────────────────────────────
    print("[1] Preparing inputs...")
    
    # Compute wrist centers
    P5_train = compute_wrist_center(X_train)
    P5_test = compute_wrist_center(X_test)
    
    # Normalize
    P5_tr_n, P5_te_n, P5_mean, P5_std = normalize_wrist_center(
        P5_train, P5_test=P5_test
    )
    
    print(f"  Wrist center stats:")
    print(f"    Mean: [{P5_mean[0,0]:7.1f}, {P5_mean[0,1]:7.1f}, {P5_mean[0,2]:7.1f}] mm")
    print(f"    Std:  [{P5_std[0,0]:7.1f}, {P5_std[0,1]:7.1f}, {P5_std[0,2]:7.1f}] mm\n")
    
    # Convert targets to sin/cos
    Y_train_sc = angles_to_sc(Y_train[:, :3])
    Y_test_sc = angles_to_sc(Y_test[:, :3])
    
    # Split train/val
    n_val = max(100, len(X_train) // 10)
    perm = np.random.RandomState(42).permutation(len(X_train))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    P5_tr_n_train = P5_tr_n[train_idx]
    P5_tr_n_val = P5_tr_n[val_idx]
    Y_tr_sc_train = Y_train_sc[train_idx]
    Y_tr_sc_val = Y_train_sc[val_idx]

    # Raw (un-normalised) wrist centres for FK loss
    P5_tr_raw_train = P5_train[train_idx]
    P5_tr_raw_val = P5_train[val_idx]
    
    print(f"  Train/Val split: {len(train_idx)} / {len(val_idx)}\n")
    
    # ── [2] Build QNN ──────────────────────────────────────────────────
    print("[2] Building Hybrid QNN v2.0...")
    qnn = HybridQNN(n_qubits=args.n_qubits, n_qlayers=args.n_qlayers,
                    hidden=256, n_res_blocks=6, dropout=0.05)
    n_q_params = sum(p.numel() for n, p in qnn.named_parameters()
                     if 'quantum' in n or 'input_scaling' in n)
    n_c_params = sum(p.numel() for n, p in qnn.named_parameters()
                     if 'quantum' not in n and 'input_scaling' not in n)
    n_params = n_q_params + n_c_params
    print(f"  Architecture: Quantum({args.n_qubits}q,{args.n_qlayers}l) "
          f"+ skip + Classical(256×6 ResBlocks)")
    print(f"  Quantum params: {n_q_params:,} | Classical params: {n_c_params:,} "
          f"| Total: {n_params:,}")

    if args.transfer:
        print("  [Transfer] Initialising classical backbone from ANN checkpoint...")
        ok = transfer_ann_weights(qnn, 'puma560_ann_v4_FINAL.pt')
        if ok:
            print("  [Transfer] Differential LR will be used (quantum: lr, classical: lr/100)")
    print()
    
    # ── [3] Train QNN ──────────────────────────────────────────────────
    print(f"[3] Training QNN ({args.epochs} epochs)...")
    t_start = time.time()
    
    best_state, hist = train_qnn(
        qnn, P5_tr_n_train, Y_tr_sc_train, P5_tr_raw_train,
        P5_tr_n_val, Y_tr_sc_val, P5_tr_raw_val,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch,
        patience=args.patience, device=device,
        transfer_mode=args.transfer,
    )
    
    train_time = time.time() - t_start
    qnn.load_state_dict(best_state)
    
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best epoch: {hist['best_epoch']} | Best val MSE: {hist['best_val']:.6f}\n")
    
    # ── [4] Evaluate QNN on test set ────────────────────────────────────
    print(f"[4] Evaluating QNN on test set ({len(X_test)} samples)...")
    
    qnn_eval = evaluate_qnn(qnn, P5_te_n, Y_test_sc, device=device)
    
    print(f"  Overall MSE on sin/cos: {qnn_eval['mse']:.6f}")
    print(f"  Joint angle errors (J1, J2, J3):")
    for j in range(3):
        print(f"    J{j+1}: MAE={qnn_eval['mae'][j]:.4f}° | "
              f"RMSE={qnn_eval['rmse'][j]:.4f}°")
    print()
    
    # ── [5] Load & evaluate classical ANN ──────────────────────────────
    ann_eval = None
    if not args.skip_classical:
        print("[5] Loading classical ANN for comparison...")
        ann_model, P5_ann_mean, P5_ann_std, ann_ckpt = ClassicalANNLoader.load(
            'puma560_ann_v4_FINAL.pt', device=device
        )
        
        if ann_model is not None:
            print(f"  [OK] Loaded classical ANN")
            
            # Normalize test set for ANN
            P5_te_n_ann = (P5_test - P5_ann_mean) / P5_ann_std
            
            ann_model.eval()
            with torch.no_grad():
                P5_te_t = torch.tensor(P5_te_n_ann, dtype=torch.float32, device=device)
                ann_pred_raw = ann_model(P5_te_t)
                ann_pred_sc = torch.tanh(ann_pred_raw)
            
            ann_pred_angles = sc_to_angles(ann_pred_sc.numpy())
            true_angles = Y_test[:, :3]
            
            wrapped_err_ann = np.array([
                ((ann_pred_angles[:, j] - true_angles[:, j] + 180) % 360 - 180)
                for j in range(3)
            ])
            
            ann_eval = {
                'mae': np.mean(np.abs(wrapped_err_ann), axis=1),
                'rmse': np.sqrt(np.mean(wrapped_err_ann ** 2, axis=1)),
                'mse': np.mean((ann_pred_sc.numpy() - Y_test_sc) ** 2),
            }
            
            print(f"  Classical ANN results:")
            for j in range(3):
                print(f"    J{j+1}: MAE={ann_eval['mae'][j]:.4f}° | "
                      f"RMSE={ann_eval['rmse'][j]:.4f}°")
            print()
    
    # ── [6] Comparison table ───────────────────────────────────────────
    print(SEP)
    print("QUANTUM vs CLASSICAL COMPARISON")
    print(SEP)
    print(f"{'Method':<25} {'J1 MAE':<12} {'J2 MAE':<12} {'J3 MAE':<12} {'Avg MAE':<12}")
    print("-" * 65)
    
    qnn_avg_mae = qnn_eval['mae'].mean()
    print(f"{'Hybrid QNN':<25} {qnn_eval['mae'][0]:<12.4f} "
          f"{qnn_eval['mae'][1]:<12.4f} {qnn_eval['mae'][2]:<12.4f} "
          f"{qnn_avg_mae:<12.4f}")
    
    if ann_eval is not None:
        ann_avg_mae = ann_eval['mae'].mean()
        improvement = (1 - qnn_avg_mae / ann_avg_mae) * 100
        print(f"{'Classical ANN':<25} {ann_eval['mae'][0]:<12.4f} "
              f"{ann_eval['mae'][1]:<12.4f} {ann_eval['mae'][2]:<12.4f} "
              f"{ann_avg_mae:<12.4f}")
        print("-" * 65)
        print(f"{'QNN vs ANN':<25} "
              f"{improvement:+.1f}% {'(positive = QNN better)' if improvement > 0 else ''}")
    
    print()
    
    # ── [7] Plots ──────────────────────────────────────────────────────
    if HAS_MPL and not args.no_plots:
        print("[6] Generating plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        colors = ['#3366CC', '#CC3333']
        labels_joint = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
        
        for j, ax in enumerate(axes):
            errors_qnn = (qnn_eval['pred_angles'][:, j] - Y_test[:, j] + 180) % 360 - 180
            
            ax.hist(errors_qnn, bins=50, alpha=0.6, color=colors[0], 
                   label='QNN', edgecolor='none')
            
            if ann_eval is not None:
                errors_ann = (ann_pred_angles[:, j] - Y_test[:, j] + 180) % 360 - 180
                ax.hist(errors_ann, bins=50, alpha=0.5, color=colors[1],
                       label='Classical ANN', edgecolor='none')
            
            ax.axvline(0, color='k', lw=1.5, ls='--', alpha=0.7)
            ax.set_xlabel(f"Angle Error {labels_joint[j]} (deg)", fontweight='bold')
            ax.set_ylabel("Frequency")
            ax.set_title(f"J{j+1}: MAE={qnn_eval['mae'][j]:.3f}°", fontweight='bold')
            ax.grid(True, alpha=0.3)
            if j == 2:
                ax.legend(loc='upper left', fontsize=10)
        
        fig.suptitle("Quantum vs Classical: Joint Angle Prediction Error", 
                    fontsize=13, fontweight='bold')
        fig.tight_layout()
        plt.savefig('qnn_vs_ann_comparison.png', dpi=150, bbox_inches='tight')
        print("  Saved: qnn_vs_ann_comparison.png\n")
        plt.close()
        
        # Training history
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(hist['train'], label='Train', color='#3366CC', lw=2)
        ax.semilogy(hist['val'], label='Val', color='#CC3333', lw=2, ls='--')
        be = hist['best_epoch'] - 1
        if 0 <= be < len(hist['val']):
            ax.axvline(be, color='k', ls=':', lw=1.5, 
                      label=f'Best (ep {hist["best_epoch"]})')
        ax.set_xlabel("Epoch", fontweight='bold')
        ax.set_ylabel("Loss (sin/cos MSE)")
        ax.set_title("Hybrid QNN Training History", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig('qnn_training_history.png', dpi=150, bbox_inches='tight')
        print("  Saved: qnn_training_history.png\n")
        plt.close()
    
    # ── [7] Save model ─────────────────────────────────────────────────
    print("[7] Saving QNN model...")
    
    torch.save({
        'model_state': best_state,
        'P5_mean': P5_mean,
        'P5_std': P5_std,
        'history': hist,
        'architecture': f'HybridQNN_v2_{args.n_qubits}q_{args.n_qlayers}l',
        'n_qubits': args.n_qubits,
        'n_qlayers': args.n_qlayers,
        'n_res_blocks': 6,
        'train_time': train_time,
    }, 'puma560_qnn_hybrid_v1.pt')
    
    print("  Saved: puma560_qnn_hybrid_v1.pt\n")
    
    # ── Summary ────────────────────────────────────────────────────────
    print(SEP)
    print("SUMMARY")
    print(SEP)
    print(f"[OK] QNN trained successfully in {train_time:.1f}s")
    print(f"[OK] QNN average J1,J2,J3 MAE: {qnn_avg_mae:.4f} deg")
    if ann_eval is not None:
        print(f"[OK] Classical ANN average MAE: {ann_avg_mae:.4f} deg")
        if qnn_avg_mae < ann_avg_mae:
            print(f"[OK] QNN OUTPERFORMS classical ANN by {improvement:.1f}%")
        else:
            print(f"  Classical ANN still outperforms by {-improvement:.1f}%")
    print()
    print("Next steps:")
    print("  - Use model with: torch.load('puma560_qnn_hybrid_v1.pt')")
    print("  - See README.md for inference example")
    print()


if __name__ == "__main__":
    main()
