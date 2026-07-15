#!/usr/bin/env python3
"""
Pure direct-ML full 6-DOF IK trainer (CPU-only, no analytical IK in inference).

This trainer predicts all six joints directly from full pose input:
  input  : [nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz] (12)
  output : [sin/cos(J1..J6)] (12)

Key property:
- Inference path is fully neural. No analytical wrist reconstruction is used.
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


DH = [
    (0.0, 671.83, -90.0),
    (431.8, 139.70, 0.0),
    (-20.32, 0.0, 90.0),
    (0.0, 431.8, -90.0),
    (0.0, 0.0, 90.0),
    (0.0, 56.5, 0.0),
]

WORKSPACE_SCALE_MM = 1200.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_retries = 12
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
            if getattr(exc, "winerror", None) != 5:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 1.8, 0.75)
        finally:
            if not replaced and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    recovery = path.with_name(f"{path.stem}_recovery_{int(time.time())}{path.suffix}")
    torch.save(obj, recovery)
    print(f"[WARN] Atomic replace failed for {path}; wrote {recovery}")


def angles_to_sc_np(angles_deg: np.ndarray) -> np.ndarray:
    r = np.deg2rad(angles_deg)
    sc = np.zeros((angles_deg.shape[0], 12), dtype=np.float32)
    for j in range(6):
        sc[:, 2 * j] = np.sin(r[:, j])
        sc[:, 2 * j + 1] = np.cos(r[:, j])
    return sc


def sc_to_angles_np(sc: np.ndarray) -> np.ndarray:
    out = np.zeros((sc.shape[0], 6), dtype=np.float32)
    for j in range(6):
        out[:, j] = np.rad2deg(np.arctan2(sc[:, 2 * j], sc[:, 2 * j + 1]))
    return out


def sc_to_angles_torch(sc: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(sc.shape[0], 6, dtype=sc.dtype, device=sc.device)
    for j in range(6):
        out[:, j] = torch.atan2(sc[:, 2 * j], sc[:, 2 * j + 1]) * (180.0 / math.pi)
    return out


def normalize_sc_pairs_torch(sc_raw: torch.Tensor) -> torch.Tensor:
    pairs = sc_raw.view(sc_raw.shape[0], 6, 2)
    norms = torch.linalg.norm(pairs, dim=-1, keepdim=True).clamp_min(1e-6)
    return (pairs / norms).reshape(sc_raw.shape[0], 12)


def wrap_angle_error(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    return (pred_deg - true_deg + 180.0) % 360.0 - 180.0


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if raw.shape[1] < 18:
        raise ValueError(f"Expected >=18 columns, got {raw.shape[1]}")
    X = raw[:, :12].astype(np.float32)
    Y = raw[:, 12:18].astype(np.float32)
    return X, Y


def split_indices(n: int, test_frac: float, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("Invalid split fractions")
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def normalize_X(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd, mu, sd


def pose12_to_targets_torch(pose12: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = pose12[:, 0:3]
    o = pose12[:, 3:6]
    a = pose12[:, 6:9]
    R_target = torch.stack([n, o, a], dim=2)
    p_target = pose12[:, 9:12]
    return R_target, p_target


def _dh_torch(a: float, d: float, alpha_deg: float, theta_batch_deg: torch.Tensor) -> torch.Tensor:
    al = math.radians(alpha_deg)
    ca = math.cos(al)
    sa = math.sin(al)

    ct = torch.cos(theta_batch_deg * (math.pi / 180.0))
    st = torch.sin(theta_batch_deg * (math.pi / 180.0))
    b = theta_batch_deg.shape[0]

    T = torch.zeros(b, 4, 4, dtype=theta_batch_deg.dtype, device=theta_batch_deg.device)
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a * st
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d
    T[:, 3, 3] = 1.0
    return T


def fPUMA_torch_6(theta6_deg: torch.Tensor) -> torch.Tensor:
    b = theta6_deg.shape[0]
    T = torch.eye(4, dtype=theta6_deg.dtype, device=theta6_deg.device).unsqueeze(0).expand(b, -1, -1).clone()
    for i in range(6):
        Ti = _dh_torch(DH[i][0], DH[i][1], DH[i][2], theta6_deg[:, i])
        T = torch.bmm(T, Ti)
    return T


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class DirectIKNet(nn.Module):
    def __init__(self, n_in: int = 12, hidden: int = 256, n_blocks: int = 5, dropout: float = 0.04) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


class DirectIKLoss(nn.Module):
    def __init__(self, w_sc: float, w_pos: float, w_ori: float, w_circ: float, w_ang: float) -> None:
        super().__init__()
        self.w_sc = w_sc
        self.w_pos = w_pos
        self.w_ori = w_ori
        self.w_circ = w_circ
        self.w_ang = w_ang

    def forward(
        self,
        pred_raw: torch.Tensor,
        target_sc: torch.Tensor,
        target_deg: torch.Tensor,
        pose12: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_sc = normalize_sc_pairs_torch(pred_raw)
        pred_deg = sc_to_angles_torch(pred_sc)
        T_pred = fPUMA_torch_6(pred_deg)
        R_pred = T_pred[:, :3, :3]
        p_pred = T_pred[:, :3, 3]

        R_tgt, p_tgt = pose12_to_targets_torch(pose12)

        loss_sc = F.smooth_l1_loss(pred_sc, target_sc, beta=0.05)
        loss_pos = F.mse_loss(p_pred / WORKSPACE_SCALE_MM, p_tgt / WORKSPACE_SCALE_MM)
        loss_ori = F.mse_loss(R_pred, R_tgt)

        # Periodic angular loss in joint space for direct MAE/RMSE improvement.
        delta_rad = (pred_deg - target_deg) * (math.pi / 180.0)
        loss_ang = (1.0 - torch.cos(delta_rad)).mean()

        wrapped_deg = torch.remainder(pred_deg - target_deg + 180.0, 360.0) - 180.0
        mae_deg = torch.mean(torch.abs(wrapped_deg))

        circ = 0.0
        for j in range(6):
            s = pred_sc[:, 2 * j]
            c = pred_sc[:, 2 * j + 1]
            circ = circ + ((s * s + c * c - 1.0) ** 2).mean()
        loss_circ = circ / 6.0

        total = (
            self.w_sc * loss_sc
            + self.w_pos * loss_pos
            + self.w_ori * loss_ori
            + self.w_circ * loss_circ
            + self.w_ang * loss_ang
        )
        parts = {
            "sc": loss_sc.detach(),
            "pos": loss_pos.detach(),
            "ori": loss_ori.detach(),
            "circ": loss_circ.detach(),
            "ang": loss_ang.detach(),
            "mae_deg": mae_deg.detach(),
            "total": total.detach(),
        }
        return total, parts


def evaluate(model: nn.Module, Xn: np.ndarray, Y_deg: np.ndarray, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    X_t = torch.tensor(Xn, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_raw = model(X_t)
        pred_sc = normalize_sc_pairs_torch(pred_raw).cpu().numpy()
    pred_deg = sc_to_angles_np(pred_sc)
    err = wrap_angle_error(pred_deg, Y_deg)
    abs_err = np.abs(err)
    mae = abs_err.mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    return {
        "pred_deg": pred_deg,
        "true_deg": Y_deg,
        "err": err,
        "abs_err": abs_err,
        "mae": mae,
        "rmse": rmse,
        "avg_mae": float(mae.mean()),
        "avg_rmse": float(rmse.mean()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train direct full-6 IK model with no analytical inference")
    p.add_argument("--dataset", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "puma560_dataset.csv"))
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=220)
    p.add_argument("--min-epochs", type=int, default=250)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w-sc", type=float, default=1.0)
    p.add_argument("--w-pos", type=float, default=1.2)
    p.add_argument("--w-ori", type=float, default=0.10)
    p.add_argument("--w-circ", type=float, default=0.02)
    p.add_argument("--w-ang", type=float, default=0.8)
    p.add_argument("--monitor", type=str, default="mae", choices=["mae", "sc", "total"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / "direct6_best.pt"
    last_path = ckpt_dir / "direct6_last.pt"
    final_path = ckpt_dir / "direct6_final.pt"

    X_all, Y_all = load_dataset(Path(args.dataset))
    tr_idx, va_idx, te_idx = split_indices(len(X_all), args.test_frac, args.val_frac, args.seed)

    X_tr, Y_tr = X_all[tr_idx], Y_all[tr_idx]
    X_va, Y_va = X_all[va_idx], Y_all[va_idx]
    X_te, Y_te = X_all[te_idx], Y_all[te_idx]

    X_tr_n, X_va_n, X_te_n, x_mean, x_std = normalize_X(X_tr, X_va, X_te)
    Y_tr_sc = angles_to_sc_np(Y_tr)
    Y_va_sc = angles_to_sc_np(Y_va)

    tr_ds = TensorDataset(
        torch.tensor(X_tr_n, dtype=torch.float32),
        torch.tensor(Y_tr_sc, dtype=torch.float32),
        torch.tensor(Y_tr, dtype=torch.float32),
        torch.tensor(X_tr, dtype=torch.float32),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va_n, dtype=torch.float32),
        torch.tensor(Y_va_sc, dtype=torch.float32),
        torch.tensor(Y_va, dtype=torch.float32),
        torch.tensor(X_va, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    va_loader = DataLoader(va_ds, batch_size=max(args.batch, 512), shuffle=False, num_workers=args.num_workers)

    model = DirectIKNet(hidden=args.hidden, n_blocks=args.blocks, dropout=args.dropout).to(device)
    criterion = DirectIKLoss(args.w_sc, args.w_pos, args.w_ori, args.w_circ, args.w_ang)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=120, T_mult=2)

    history: Dict[str, list] = {
        "train_total": [],
        "val_total": [],
        "train_sc": [],
        "val_sc": [],
        "train_mae_deg": [],
        "val_mae_deg": [],
        "lr": [],
    }

    best_metric = float("inf")
    best_epoch = 0
    start_epoch = 0
    stale = 0

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt.get("history", history)
        best_metric = float(ckpt.get("best_metric", ckpt.get("best_val", best_metric)))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        stale = int(ckpt.get("stale", 0))

    print(f"Training direct 6-DOF model | train={len(X_tr_n)} val={len(X_va_n)} test={len(X_te_n)}")
    print(
        f"Objective weights | w_sc={args.w_sc} w_pos={args.w_pos} w_ori={args.w_ori} "
        f"w_circ={args.w_circ} w_ang={args.w_ang} | monitor={args.monitor}"
    )

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tr_sum = {"total": 0.0, "sc": 0.0, "mae_deg": 0.0, "n": 0}
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for xb, yb_sc, yb_deg, poseb in pbar:
            xb = xb.to(device)
            yb_sc = yb_sc.to(device)
            yb_deg = yb_deg.to(device)
            poseb = poseb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, parts = criterion(pred, yb_sc, yb_deg, poseb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = xb.shape[0]
            tr_sum["total"] += float(loss.detach()) * bs
            tr_sum["sc"] += float(parts["sc"]) * bs
            tr_sum["mae_deg"] += float(parts["mae_deg"]) * bs
            tr_sum["n"] += bs
            pbar.set_postfix(loss=f"{float(loss.detach()):.5f}")

        scheduler.step(epoch + 1)

        model.eval()
        va_sum = {"total": 0.0, "sc": 0.0, "mae_deg": 0.0, "n": 0}
        with torch.no_grad():
            for xb, yb_sc, yb_deg, poseb in va_loader:
                xb = xb.to(device)
                yb_sc = yb_sc.to(device)
                yb_deg = yb_deg.to(device)
                poseb = poseb.to(device)
                pred = model(xb)
                loss, parts = criterion(pred, yb_sc, yb_deg, poseb)
                bs = xb.shape[0]
                va_sum["total"] += float(loss.detach()) * bs
                va_sum["sc"] += float(parts["sc"]) * bs
                va_sum["mae_deg"] += float(parts["mae_deg"]) * bs
                va_sum["n"] += bs

        train_total = tr_sum["total"] / max(1, tr_sum["n"])
        val_total = va_sum["total"] / max(1, va_sum["n"])
        train_sc = tr_sum["sc"] / max(1, tr_sum["n"])
        val_sc = va_sum["sc"] / max(1, va_sum["n"])
        train_mae_deg = tr_sum["mae_deg"] / max(1, tr_sum["n"])
        val_mae_deg = va_sum["mae_deg"] / max(1, va_sum["n"])
        lr = float(optimizer.param_groups[0]["lr"])

        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["train_sc"].append(train_sc)
        history["val_sc"].append(val_sc)
        history["train_mae_deg"].append(train_mae_deg)
        history["val_mae_deg"].append(val_mae_deg)
        history["lr"].append(lr)

        if args.monitor == "mae":
            metric_now = val_mae_deg
        elif args.monitor == "total":
            metric_now = val_total
        else:
            metric_now = val_sc

        improved = metric_now < best_metric
        if improved:
            best_metric = metric_now
            best_epoch = epoch + 1
            stale = 0
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "history": history,
                    "best_metric": best_metric,
                    "best_monitor": args.monitor,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "architecture": "DirectIKNet_12to12",
                    "hidden": args.hidden,
                    "blocks": args.blocks,
                    "dropout": args.dropout,
                },
                best_path,
            )
        else:
            stale += 1

        if (epoch + 1) % args.save_every == 0:
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "history": history,
                    "best_metric": best_metric,
                    "best_monitor": args.monitor,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "architecture": "DirectIKNet_12to12",
                    "hidden": args.hidden,
                    "blocks": args.blocks,
                    "dropout": args.dropout,
                },
                last_path,
            )

        print(
            f"Epoch {epoch+1:4d} | train={train_total:.6f} val={val_total:.6f} "
            f"train_sc={train_sc:.6f} val_sc={val_sc:.6f} "
            f"train_mae={train_mae_deg:.4f}deg val_mae={val_mae_deg:.4f}deg "
            f"best_{args.monitor}={best_metric:.6f} lr={lr:.2e}"
        )

        if (epoch + 1) >= args.min_epochs and stale >= args.patience:
            print(f"Early stopping at epoch {epoch+1}; best epoch {best_epoch}")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model_state"])

    metrics = evaluate(model, X_te_n, Y_te, device)
    train_time = time.time() - t0

    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "history": history,
            "metrics": metrics,
            "best_metric": best_metric,
            "best_monitor": args.monitor,
            "best_epoch": best_epoch,
            "train_time": train_time,
            "architecture": "DirectIKNet_12to12",
            "hidden": args.hidden,
            "blocks": args.blocks,
            "dropout": args.dropout,
        },
        final_path,
    )

    np.savez(
        out_dir / "direct6_eval_results.npz",
        pred_deg=metrics["pred_deg"],
        true_deg=metrics["true_deg"],
        err=metrics["err"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        avg_mae=metrics["avg_mae"],
        avg_rmse=metrics["avg_rmse"],
    )

    print("\nFinal test metrics (direct, no analytical inference):")
    print("  MAE per joint (deg):", np.array2string(metrics["mae"], precision=4))
    print("  RMSE per joint (deg):", np.array2string(metrics["rmse"], precision=4))
    print(f"  Avg MAE: {metrics['avg_mae']:.5f} deg")
    print(f"  Avg RMSE: {metrics['avg_rmse']:.5f} deg")
    print(f"Saved: {final_path}")
    print(f"Saved: {out_dir / 'direct6_eval_results.npz'}")


if __name__ == "__main__":
    main()
