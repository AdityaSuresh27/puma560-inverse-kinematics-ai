#!/usr/bin/env python3
"""
Full 6-DOF ANN trainer for PUMA 560 (CPU-only, robust long-run safe).

This script trains an ANN to predict all six joint angles from full pose input:
  input  : [nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz]  (12)
  target : [theta1..theta6] in degrees

Safety features for long runs:
- CPU-only execution by design
- Per-epoch checkpointing (last + best)
- Resume support via --resume
- NaN/Inf loss guard
- Gradient clipping
- Atomic checkpoint writes
- Plotting errors never invalidate training results

Usage examples:
  python train_ann_full6_cpu.py
  python train_ann_full6_cpu.py --epochs 3000 --resume
  python train_ann_full6_cpu.py --epochs 20 --no-plots
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ============================== Constants ===============================

DH = [
    (0.0,    671.83, -90.0),
    (431.8,  139.70,   0.0),
    (-20.32,   0.0,   90.0),
    (0.0,    431.8,  -90.0),
    (0.0,      0.0,   90.0),
    (0.0,     56.5,    0.0),
]

D4 = DH[3][1]
D6 = DH[5][1]

JOINT_LIMITS = np.array([
    [-160,  160],
    [-225,   45],
    [ -45,  225],
    [-110,  170],
    [-100,  100],
    [-266,  266],
], dtype=np.float32)

R_WORKSPACE = 900.0

# ============================== Utilities ===============================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def angles_to_sc_np(angles_deg: np.ndarray) -> np.ndarray:
    """[N,6] deg -> [N,12] interleaved sin/cos."""
    r = np.deg2rad(angles_deg)
    sc = np.zeros((angles_deg.shape[0], angles_deg.shape[1] * 2), dtype=np.float32)
    for j in range(angles_deg.shape[1]):
        sc[:, 2 * j] = np.sin(r[:, j])
        sc[:, 2 * j + 1] = np.cos(r[:, j])
    return sc


def sc_to_angles_np(sc: np.ndarray) -> np.ndarray:
    """[N,12] sin/cos -> [N,6] deg in [-180,180]."""
    k = sc.shape[1] // 2
    a = np.zeros((sc.shape[0], k), dtype=np.float32)
    for j in range(k):
        a[:, j] = np.rad2deg(np.arctan2(sc[:, 2 * j], sc[:, 2 * j + 1]))
    return a


def sc_to_angles_torch(sc: torch.Tensor) -> torch.Tensor:
    """Differentiable sin/cos -> deg."""
    k = sc.shape[-1] // 2
    out = torch.zeros(sc.shape[0], k, dtype=sc.dtype, device=sc.device)
    for j in range(k):
        out[:, j] = torch.atan2(sc[:, 2 * j], sc[:, 2 * j + 1]) * (180.0 / math.pi)
    return out


def wrap_angle_error(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    return (pred_deg - true_deg + 180.0) % 360.0 - 180.0


def atomic_torch_save(obj: dict, path: Path) -> None:
    """Best-effort atomic save with retries for transient Windows file locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    max_retries = 15
    delay_s = 0.03
    pid = os.getpid()

    for attempt in range(max_retries):
        tmp = path.with_name(f"{path.name}.tmp.{pid}.{attempt}")
        replaced = False
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            replaced = True
            return
        except OSError as exc:
            # WinError 5 is commonly caused by transient AV/indexer locks.
            winerr = getattr(exc, "winerror", None)
            if winerr != 5:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 1.8, 0.75)
        finally:
            if not replaced and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # Do not kill a long run if only atomic replacement is blocked.
    recovery = path.with_name(f"{path.stem}_recovery_{int(time.time())}{path.suffix}")
    torch.save(obj, recovery)
    print(f"[WARN] Atomic checkpoint replace failed for {path}; wrote recovery checkpoint: {recovery}")


def _dh_np(a: float, d: float, alpha_deg: float, theta_deg: float) -> np.ndarray:
    al = math.radians(alpha_deg)
    th = math.radians(theta_deg)
    ca, sa = math.cos(al), math.sin(al)
    ct, st = math.cos(th), math.sin(th)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def t0_3_np(theta123_deg: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    for i in range(3):
        T = T @ _dh_np(DH[i][0], DH[i][1], DH[i][2], float(theta123_deg[i]))
    return T


def row_pose12_to_T06_np(row: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [row[0], row[3], row[6], row[9]],
            [row[1], row[4], row[7], row[10]],
            [row[2], row[5], row[8], row[11]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def solve_wrist(theta123_deg: np.ndarray, T06: np.ndarray, flip_wrist: bool = False) -> Tuple[float, float, float]:
    """Recover J4..J6 analytically from predicted J1..J3 and target pose."""
    T0_3 = t0_3_np(theta123_deg)
    T3_6 = np.linalg.inv(T0_3) @ T06

    sin5_sq = max(0.0, 1.0 - float(T3_6[2, 2]) ** 2)
    sin5 = math.sqrt(sin5_sq)

    if sin5 < 1e-8:
        theta5 = 0.0
        theta4 = 0.0
        theta6 = math.degrees(math.atan2(float(T3_6[1, 0]), float(T3_6[0, 0])))
        return theta4, theta5, theta6

    if flip_wrist:
        sin5 = -sin5

    theta5 = math.degrees(math.atan2(sin5, float(T3_6[2, 2])))
    theta4 = math.degrees(math.atan2(float(T3_6[1, 2]), float(T3_6[0, 2])))
    theta6 = math.degrees(math.atan2(float(T3_6[2, 1]), -float(T3_6[2, 0])))
    return theta4, theta5, theta6


def compute_wrist_center_np(pose12: np.ndarray) -> np.ndarray:
    ax = pose12[:, 6]
    ay = pose12[:, 7]
    az = pose12[:, 8]
    px = pose12[:, 9]
    py = pose12[:, 10]
    pz = pose12[:, 11]
    return np.stack([px - D6 * ax, py - D6 * ay, pz - D6 * az], axis=1).astype(np.float32)


# ========================== Differentiable FK ===========================


def _dh_torch(a: float, d: float, alpha_deg: float, theta_batch: torch.Tensor) -> torch.Tensor:
    al = math.radians(alpha_deg)
    ca, sa = math.cos(al), math.sin(al)
    ca_t = torch.tensor(ca, dtype=theta_batch.dtype, device=theta_batch.device)
    sa_t = torch.tensor(sa, dtype=theta_batch.dtype, device=theta_batch.device)
    a_t = torch.tensor(float(a), dtype=theta_batch.dtype, device=theta_batch.device)
    d_t = torch.tensor(float(d), dtype=theta_batch.dtype, device=theta_batch.device)

    th = theta_batch * (math.pi / 180.0)
    ct, st = torch.cos(th), torch.sin(th)
    b = theta_batch.shape[0]

    T = torch.zeros(b, 4, 4, dtype=theta_batch.dtype, device=theta_batch.device)
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca_t
    T[:, 0, 2] = st * sa_t
    T[:, 0, 3] = a_t * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca_t
    T[:, 1, 2] = -ct * sa_t
    T[:, 1, 3] = a_t * st
    T[:, 2, 1] = sa_t
    T[:, 2, 2] = ca_t
    T[:, 2, 3] = d_t
    T[:, 3, 3] = 1.0
    return T


def fPUMA_torch_3(theta3_deg: torch.Tensor) -> torch.Tensor:
    """theta3_deg: [B,3] -> T03 [B,4,4]."""
    b = theta3_deg.shape[0]
    T = torch.eye(4, dtype=theta3_deg.dtype, device=theta3_deg.device).unsqueeze(0).expand(b, -1, -1).clone()
    for i in range(3):
        Ti = _dh_torch(DH[i][0], DH[i][1], DH[i][2], theta3_deg[:, i])
        T = torch.bmm(T, Ti)
    return T


def normalize_sc_pairs_torch(sc_raw: torch.Tensor) -> torch.Tensor:
    """Normalize each sin/cos pair to unit norm for stable angle decoding."""
    if sc_raw.shape[-1] % 2 != 0:
        raise ValueError("Expected even-sized sin/cos output")
    pairs = sc_raw.view(sc_raw.shape[0], -1, 2)
    norms = torch.linalg.norm(pairs, dim=-1, keepdim=True).clamp_min(1e-6)
    return (pairs / norms).reshape(sc_raw.shape[0], -1)


def pose12_to_targets_torch(pose12: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pose12 layout: [nx,ny,nz, ox,oy,oz, ax,ay,az, Px,Py,Pz]
    Returns:
      R_target [B,3,3] with columns n,o,a
      P_target [B,3]
    """
    n = pose12[:, 0:3]
    o = pose12[:, 3:6]
    a = pose12[:, 6:9]
    R_target = torch.stack([n, o, a], dim=2)
    P_target = pose12[:, 9:12]
    return R_target, P_target


# ============================== Model ==================================


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FullIKANN(nn.Module):
    def __init__(self, n_in: int = 3, hidden: int = 512, n_blocks: int = 8, dropout: float = 0.05, n_out: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, n_out)  # sin/cos for J1..J3

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


class FullIKLoss(nn.Module):
    def __init__(self, w_sc: float = 1.0, w_wc: float = 2.0, w_circ: float = 0.02):
        super().__init__()
        self.w_sc = w_sc
        self.w_wc = w_wc
        self.w_circ = w_circ

    def forward(self, pred_raw: torch.Tensor, target_sc: torch.Tensor, target_p5: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_sc = normalize_sc_pairs_torch(pred_raw)

        # Smooth L1 is less brittle than plain MSE when angles are initially far off.
        loss_sc = F.smooth_l1_loss(pred_sc, target_sc, beta=0.05)

        pred_j123_deg = sc_to_angles_torch(pred_sc)
        T03 = fPUMA_torch_3(pred_j123_deg)
        p5_pred = T03[:, :3, 3] + D4 * T03[:, :3, 2]

        loss_wc = F.mse_loss(p5_pred / R_WORKSPACE, target_p5 / R_WORKSPACE)

        loss_circ = 0.0
        for j in range(3):
            sj = pred_sc[:, 2 * j]
            cj = pred_sc[:, 2 * j + 1]
            loss_circ = loss_circ + ((sj * sj + cj * cj - 1.0) ** 2).mean()
        loss_circ = loss_circ / 3.0

        total = self.w_sc * loss_sc + self.w_wc * loss_wc + self.w_circ * loss_circ

        components = {
            "sc": loss_sc.detach(),
            "wc": loss_wc.detach(),
            "circ": loss_circ.detach(),
            "total": total.detach(),
        }
        return total, components


# =========================== Data pipeline ==============================


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if raw.shape[1] < 18:
        raise ValueError(f"Expected >=18 columns, got {raw.shape[1]}")
    X = raw[:, :12].astype(np.float32)
    Y = raw[:, 12:18].astype(np.float32)
    return X, Y


def split_indices(n: int, test_frac: float, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if test_frac <= 0 or val_frac <= 0 or test_frac + val_frac >= 0.9:
        raise ValueError("Use sensible split fractions, e.g. test=0.15 and val=0.10")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_test - n_val
    if n_train < 1000:
        raise ValueError("Training split too small. Reduce test/val fractions.")

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def normalize_X(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd, mu, sd


# ============================ Evaluation ================================


def evaluate_model(model: nn.Module, Xn: np.ndarray, X_pose12: np.ndarray, Y_deg: np.ndarray, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    X_t = torch.tensor(Xn, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_raw = model(X_t)
        pred_sc = normalize_sc_pairs_torch(pred_raw).cpu().numpy()

    pred_j123 = sc_to_angles_np(pred_sc)
    pred_deg = np.zeros((pred_j123.shape[0], 6), dtype=np.float32)
    pred_deg[:, :3] = pred_j123.astype(np.float32)

    for i in range(pred_deg.shape[0]):
        T06 = row_pose12_to_T06_np(X_pose12[i])
        t4, t5, t6 = solve_wrist(pred_j123[i], T06, flip_wrist=False)
        pred_deg[i, 3] = t4
        pred_deg[i, 4] = t5
        pred_deg[i, 5] = t6

    err = wrap_angle_error(pred_deg, Y_deg)
    abs_err = np.abs(err)

    mae = abs_err.mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))

    tol_metrics = {}
    for tol in [0.1, 0.25, 0.5, 1.0, 2.0]:
        tol_metrics[f"joint_within_{tol}deg"] = (abs_err <= tol).mean(axis=0)
        tol_metrics[f"all_joints_within_{tol}deg"] = float((abs_err <= tol).all(axis=1).mean())

    return {
        "pred_deg": pred_deg,
        "true_deg": Y_deg,
        "err": err,
        "abs_err": abs_err,
        "mae": mae,
        "rmse": rmse,
        "avg_mae": float(mae.mean()),
        "avg_rmse": float(rmse.mean()),
        "tol": tol_metrics,
    }


# ============================= Plotting =================================


def make_plots(history: Dict[str, list], metrics: Dict[str, np.ndarray], out_dir: Path) -> None:
    if not HAS_MPL:
        print("[WARN] matplotlib not available; skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Training history
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(history["train_total"], label="Train total", lw=2)
    ax1.plot(history["val_total"], label="Val total", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history["lr"], label="LR", color="tab:orange", alpha=0.6)
    ax2.set_ylabel("Learning rate")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.suptitle("Full 6-DOF ANN Training History")
    fig.tight_layout()
    fig.savefig(out_dir / "training_history_full6.png", dpi=150)
    plt.close(fig)

    # 2) Per-joint MAE/RMSE bars
    joints = [f"J{i}" for i in range(1, 7)] + ["Avg"]
    mae_vals = list(metrics["mae"]) + [metrics["avg_mae"]]
    rmse_vals = list(metrics["rmse"]) + [metrics["avg_rmse"]]

    x = np.arange(len(joints))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, mae_vals, w, label="MAE")
    b2 = ax.bar(x + w / 2, rmse_vals, w, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(joints)
    ax.set_ylabel("Degrees")
    ax.set_title("Per-Joint Error Metrics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_metrics_full6.png", dpi=150)
    plt.close(fig)

    # 3) Histograms by joint
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, ax in enumerate(axes.flat):
        ax.hist(metrics["err"][:, j], bins=60, color="#1976D2", alpha=0.8, edgecolor="none")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"J{j+1} error | MAE={metrics['mae'][j]:.3f}°")
        ax.set_xlabel("Error (deg)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Error Distribution by Joint")
    fig.tight_layout()
    fig.savefig(out_dir / "error_histograms_full6.png", dpi=150)
    plt.close(fig)

    # 4) Predicted vs true scatter
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    n = metrics["true_deg"].shape[0]
    idx = np.random.default_rng(42).choice(n, min(600, n), replace=False)
    for j, ax in enumerate(axes.flat):
        t = metrics["true_deg"][idx, j]
        p = metrics["pred_deg"][idx, j]
        lo = min(t.min(), p.min()) - 5
        hi = max(t.max(), p.max()) + 5
        ax.scatter(t, p, s=7, alpha=0.4, color="#D32F2F")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(f"J{j+1} predicted vs true")
        ax.set_xlabel("True (deg)")
        ax.set_ylabel("Predicted (deg)")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Predicted vs True Angles")
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_true_full6.png", dpi=150)
    plt.close(fig)


# ============================== Training ================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only full 6-DOF ANN trainer for PUMA 560")
    parser.add_argument("--dataset", type=str, default=str(Path(__file__).resolve().parents[1] / "puma560_dataset.csv"))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent))

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--min-epochs", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--w-sc", type=float, default=1.0)
    parser.add_argument("--w-pos", type=float, default=2.0, help="Wrist-center physics loss weight")
    parser.add_argument("--w-ori", type=float, default=0.0, help="Unused in decoupled trainer; kept for CLI compatibility")
    parser.add_argument("--w-circ", type=float, default=0.02)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps PyTorch default")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    print("=" * 90)
    print("PUMA 560 Full IK ANN (6-angle) | CPU-only robust trainer")
    print("=" * 90)
    print(f"Dataset: {args.dataset}")
    print(f"Output : {args.output_dir}")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs} | Batch: {args.batch} | LR: {args.lr}")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / "ann6_best.pt"
    last_path = ckpt_dir / "ann6_last.pt"
    final_path = ckpt_dir / "ann6_final.pt"

    # ---------------------------- Data ---------------------------------
    X_all, Y_all = load_dataset(Path(args.dataset))
    P5_all = compute_wrist_center_np(X_all)
    train_idx, val_idx, test_idx = split_indices(len(X_all), args.test_frac, args.val_frac, args.seed)

    X_train_pose, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val_pose, Y_val = X_all[val_idx], Y_all[val_idx]
    X_test_pose, Y_test = X_all[test_idx], Y_all[test_idx]

    X_train_wc, X_val_wc, X_test_wc = P5_all[train_idx], P5_all[val_idx], P5_all[test_idx]
    X_train_n, X_val_n, X_test_n, x_mean, x_std = normalize_X(X_train_wc, X_val_wc, X_test_wc)

    # Only shoulder joints are learned by ANN; wrist joints are solved analytically.
    Y_train_sc = angles_to_sc_np(Y_train[:, :3])
    Y_val_sc = angles_to_sc_np(Y_val[:, :3])

    print(f"Split  : train={len(X_train_n)} | val={len(X_val_n)} | test={len(X_test_n)}")

    train_ds = TensorDataset(
        torch.tensor(X_train_n, dtype=torch.float32),
        torch.tensor(Y_train_sc, dtype=torch.float32),
        torch.tensor(X_train_wc, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_n, dtype=torch.float32),
        torch.tensor(Y_val_sc, dtype=torch.float32),
        torch.tensor(X_val_wc, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch, 512), shuffle=False, num_workers=args.num_workers)

    # --------------------------- Model ---------------------------------
    model = FullIKANN(n_in=3, hidden=args.hidden, n_blocks=args.blocks, dropout=args.dropout, n_out=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : FullIKANN(input=3, output=6, hidden={args.hidden}, blocks={args.blocks}, dropout={args.dropout})")
    print(f"Params : {n_params:,}")
    if abs(float(args.w_ori)) > 1e-12:
        print("[WARN] --w-ori is ignored in this decoupled trainer.")

    criterion = FullIKLoss(w_sc=args.w_sc, w_wc=args.w_pos, w_circ=args.w_circ)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=args.lr * 1e-3
    )

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    history = {
        "train_total": [],
        "val_total": [],
        "train_sc": [],
        "val_sc": [],
        "train_pos": [],
        "val_pos": [],
        "train_ori": [],
        "val_ori": [],
        "train_circ": [],
        "val_circ": [],
        "lr": [],
    }

    if args.resume and last_path.exists():
        try:
            ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val = float(ckpt["best_val"])
            best_epoch = int(ckpt["best_epoch"])
            no_improve = int(ckpt.get("no_improve", 0))
            history = ckpt.get("history", history)
            print(f"[RESUME] epoch={start_epoch} | best_monitor={best_val:.6f} @ epoch {best_epoch}")
        except Exception as exc:
            print(f"[WARN] Resume checkpoint incompatible with current trainer: {exc}")
            print("[WARN] Starting from scratch (fresh architecture/objective).")

    # --------------------------- Train ---------------------------------
    t0 = time.time()

    try:
        pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Training ANN6", unit="epoch")
        for epoch in pbar:
            model.train()
            tr_sum = {"total": 0.0, "sc": 0.0, "wc": 0.0, "circ": 0.0}
            tr_n = 0

            for xb, yb, p5b in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                p5b = p5b.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred_raw = model(xb)
                loss, parts = criterion(pred_raw, yb, p5b)

                if not torch.isfinite(loss):
                    print("[WARN] Non-finite loss encountered; skipping batch.")
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                bsz = xb.shape[0]
                tr_n += bsz
                tr_sum["total"] += float(parts["total"]) * bsz
                tr_sum["sc"] += float(parts["sc"]) * bsz
                tr_sum["wc"] += float(parts["wc"]) * bsz
                tr_sum["circ"] += float(parts["circ"]) * bsz

            scheduler.step(epoch - 1)

            model.eval()
            va_sum = {"total": 0.0, "sc": 0.0, "wc": 0.0, "circ": 0.0}
            va_n = 0
            with torch.no_grad():
                for xb, yb, p5b in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    p5b = p5b.to(device)
                    pred_raw = model(xb)
                    _, parts = criterion(pred_raw, yb, p5b)
                    bsz = xb.shape[0]
                    va_n += bsz
                    va_sum["total"] += float(parts["total"]) * bsz
                    va_sum["sc"] += float(parts["sc"]) * bsz
                    va_sum["wc"] += float(parts["wc"]) * bsz
                    va_sum["circ"] += float(parts["circ"]) * bsz

            tr = {k: v / max(1, tr_n) for k, v in tr_sum.items()}
            va = {k: v / max(1, va_n) for k, v in va_sum.items()}

            history["train_total"].append(tr["total"])
            history["val_total"].append(va["total"])
            history["train_sc"].append(tr["sc"])
            history["val_sc"].append(va["sc"])
            history["train_pos"].append(tr["wc"])
            history["val_pos"].append(va["wc"])
            history["train_ori"].append(0.0)
            history["val_ori"].append(0.0)
            history["train_circ"].append(tr["circ"])
            history["val_circ"].append(va["circ"])
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            monitor = va["sc"]
            improved = monitor < best_val
            if improved:
                best_val = monitor
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            pbar.set_postfix({
                "tr": f"{tr['total']:.6f}",
                "va_sc": f"{va['sc']:.6f}",
                "best": best_epoch,
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            # Save last checkpoint frequently so long runs are safe.
            if (epoch % args.save_every == 0) or improved:
                ckpt_last = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "no_improve": no_improve,
                    "history": history,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "model_input_dim": 3,
                    "model_output_dim": 6,
                    "args": vars(args),
                }
                atomic_torch_save(ckpt_last, last_path)

            if improved:
                ckpt_best = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "history": history,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "model_input_dim": 3,
                    "model_output_dim": 6,
                    "args": vars(args),
                }
                atomic_torch_save(ckpt_best, best_path)

            if epoch >= args.min_epochs and no_improve >= args.patience:
                print(f"\n[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}, best_monitor={best_val:.6f}")
                break

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving emergency checkpoint...")
        ckpt_last = {
            "epoch": max(1, len(history["val_total"])),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "history": history,
            "x_mean": x_mean,
            "x_std": x_std,
            "model_input_dim": 3,
            "model_output_dim": 6,
            "args": vars(args),
        }
        atomic_torch_save(ckpt_last, last_path)
        raise

    train_time = time.time() - t0

    # ------------------------- Evaluate best ---------------------------
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])

    metrics = evaluate_model(model, X_test_n, X_test_pose, Y_test, device)

    print("=" * 90)
    print("FINAL TEST RESULTS (ALL 6 JOINTS)")
    print("=" * 90)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val shoulder sc loss: {best_val:.6f}")
    print(f"Train time: {train_time:.1f} s")
    print()

    print(f"{'Joint':<8}{'MAE (deg)':>12}{'RMSE (deg)':>14}{'Within 1 deg':>14}")
    print("-" * 48)
    for j in range(6):
        within1 = metrics["tol"]["joint_within_1.0deg"][j] * 100.0
        print(f"J{j+1:<7}{metrics['mae'][j]:>12.4f}{metrics['rmse'][j]:>14.4f}{within1:>13.2f}%")

    print("-" * 48)
    print(f"{'AVG':<8}{metrics['avg_mae']:>12.4f}{metrics['avg_rmse']:>14.4f}")
    print()
    print(f"All-joint accuracy within 0.5 deg : {metrics['tol']['all_joints_within_0.5deg'] * 100:.2f}%")
    print(f"All-joint accuracy within 1.0 deg : {metrics['tol']['all_joints_within_1.0deg'] * 100:.2f}%")
    print(f"All-joint accuracy within 2.0 deg : {metrics['tol']['all_joints_within_2.0deg'] * 100:.2f}%")

    # --------------------------- Save final ----------------------------
    final_ckpt = {
        "model_state": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "train_time": train_time,
        "history": history,
        "x_mean": x_mean,
        "x_std": x_std,
        "model_input_dim": 3,
        "model_output_dim": 6,
        "metrics": {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "avg_mae": metrics["avg_mae"],
            "avg_rmse": metrics["avg_rmse"],
            "all_joints_within_0.5deg": metrics["tol"]["all_joints_within_0.5deg"],
            "all_joints_within_1.0deg": metrics["tol"]["all_joints_within_1.0deg"],
            "all_joints_within_2.0deg": metrics["tol"]["all_joints_within_2.0deg"],
        },
        "args": vars(args),
    }
    atomic_torch_save(final_ckpt, final_path)

    np.savez(
        output_dir / "ann6_eval_results.npz",
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        avg_mae=np.array([metrics["avg_mae"]], dtype=np.float32),
        avg_rmse=np.array([metrics["avg_rmse"]], dtype=np.float32),
        pred_deg=metrics["pred_deg"],
        true_deg=metrics["true_deg"],
        err=metrics["err"],
        abs_err=metrics["abs_err"],
    )

    print(f"\nSaved: {final_path}")
    print(f"Saved: {output_dir / 'ann6_eval_results.npz'}")

    # --------------------------- Plots ---------------------------------
    if not args.no_plots:
        try:
            make_plots(history, metrics, output_dir)
            print("Saved plots:")
            print(f"  - {output_dir / 'training_history_full6.png'}")
            print(f"  - {output_dir / 'joint_metrics_full6.png'}")
            print(f"  - {output_dir / 'error_histograms_full6.png'}")
            print(f"  - {output_dir / 'pred_vs_true_full6.png'}")
        except Exception as e:
            print(f"[WARN] Plot generation failed: {e}")
            print("Training results/checkpoints are safe and already saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()
