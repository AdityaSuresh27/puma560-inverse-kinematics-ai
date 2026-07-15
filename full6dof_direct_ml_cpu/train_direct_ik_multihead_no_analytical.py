#!/usr/bin/env python3
"""
Multi-head pure direct-ML full 6-DOF IK trainer (CPU-only, no analytical IK in inference).

Purpose:
- Handle multi-solution IK ambiguity (especially wrist joints J4/J6) by predicting K hypotheses.
- Select one head at inference via learned confidence logits.

Input : [nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz] (12)
Output: K x [sin/cos(J1..J6)] + K confidence logits
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


def normalize_sc_pairs_torch(sc_raw: torch.Tensor) -> torch.Tensor:
    # [B,K,12] -> normalized [B,K,12]
    b, k, _ = sc_raw.shape
    pairs = sc_raw.view(b, k, 6, 2)
    norms = torch.linalg.norm(pairs, dim=-1, keepdim=True).clamp_min(1e-6)
    return (pairs / norms).reshape(b, k, 12)


def sc_to_angles_np(sc: np.ndarray) -> np.ndarray:
    # [N,12] -> [N,6]
    out = np.zeros((sc.shape[0], 6), dtype=np.float32)
    for j in range(6):
        out[:, j] = np.rad2deg(np.arctan2(sc[:, 2 * j], sc[:, 2 * j + 1]))
    return out


def sc_to_angles_torch(sc: torch.Tensor) -> torch.Tensor:
    # [B,K,12] -> [B,K,6]
    b, k, _ = sc.shape
    out = torch.zeros(b, k, 6, dtype=sc.dtype, device=sc.device)
    for j in range(6):
        out[:, :, j] = torch.atan2(sc[:, :, 2 * j], sc[:, :, 2 * j + 1]) * (180.0 / math.pi)
    return out


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


class MultiHeadDirectIKNet(nn.Module):
    def __init__(self, n_heads: int = 4, n_in: int = 12, hidden: int = 256, n_blocks: int = 6, dropout: float = 0.03) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.stem = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.sc_head = nn.Linear(hidden, n_heads * 12)
        self.conf_head = nn.Linear(hidden, n_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        sc_raw = self.sc_head(h).view(x.shape[0], self.n_heads, 12)
        logits = self.conf_head(h)
        return sc_raw, logits


class MultiHeadIKLoss(nn.Module):
    def __init__(
        self,
        w_sc: float,
        w_ang: float,
        w_pos: float,
        w_ori: float,
        w_conf: float,
        w_circ: float,
        joint_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.w_sc = w_sc
        self.w_ang = w_ang
        self.w_pos = w_pos
        self.w_ori = w_ori
        self.w_conf = w_conf
        self.w_circ = w_circ
        self.register_buffer("joint_weights", joint_weights)

    def forward(
        self,
        pred_sc_raw: torch.Tensor,
        pred_logits: torch.Tensor,
        target_sc: torch.Tensor,
        target_deg: torch.Tensor,
        pose12: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Normalize all heads.
        pred_sc = normalize_sc_pairs_torch(pred_sc_raw)
        pred_deg = sc_to_angles_torch(pred_sc)  # [B,K,6]

        b, k, _ = pred_deg.shape
        target_deg_k = target_deg.unsqueeze(1).expand(b, k, 6)
        target_sc_k = target_sc.unsqueeze(1).expand(b, k, 12)

        # Match cost per head: periodic joint loss + sc reconstruction.
        delta_rad = (pred_deg - target_deg_k) * (math.pi / 180.0)
        per_joint = (1.0 - torch.cos(delta_rad)) * self.joint_weights.view(1, 1, 6)
        loss_ang_h = per_joint.mean(dim=2)  # [B,K]

        loss_sc_h = F.smooth_l1_loss(pred_sc, target_sc_k, beta=0.05, reduction="none").mean(dim=2)  # [B,K]
        match_cost = self.w_ang * loss_ang_h + self.w_sc * loss_sc_h

        # Choose best head per sample (best-of-K assignment).
        assign = torch.argmin(match_cost, dim=1)  # [B]
        idx = torch.arange(b, device=pred_sc.device)
        sel_sc = pred_sc[idx, assign, :]          # [B,12]
        sel_deg = pred_deg[idx, assign, :]        # [B,6]

        # Confidence supervision: predict assignment.
        loss_conf = F.cross_entropy(pred_logits, assign)

        # Physical consistency on selected head.
        T_pred = fPUMA_torch_6(sel_deg)
        R_pred = T_pred[:, :3, :3]
        p_pred = T_pred[:, :3, 3]
        R_tgt, p_tgt = pose12_to_targets_torch(pose12)

        loss_pos = F.mse_loss(p_pred / WORKSPACE_SCALE_MM, p_tgt / WORKSPACE_SCALE_MM)
        loss_ori = F.mse_loss(R_pred, R_tgt)

        # Selected-head periodic MAE for logging.
        wrapped_deg = torch.remainder(sel_deg - target_deg + 180.0, 360.0) - 180.0
        mae_deg = torch.mean(torch.abs(wrapped_deg))

        # Unit-circle regularizer on selected head.
        circ = 0.0
        for j in range(6):
            s = sel_sc[:, 2 * j]
            c = sel_sc[:, 2 * j + 1]
            circ = circ + ((s * s + c * c - 1.0) ** 2).mean()
        loss_circ = circ / 6.0

        loss_match = match_cost[idx, assign].mean()

        total = loss_match + self.w_pos * loss_pos + self.w_ori * loss_ori + self.w_conf * loss_conf + self.w_circ * loss_circ

        parts = {
            "match": loss_match.detach(),
            "pos": loss_pos.detach(),
            "ori": loss_ori.detach(),
            "conf": loss_conf.detach(),
            "circ": loss_circ.detach(),
            "mae_deg": mae_deg.detach(),
            "total": total.detach(),
        }
        return total, parts


def evaluate(model: nn.Module, Xn: np.ndarray, Y_deg: np.ndarray, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    X_t = torch.tensor(Xn, dtype=torch.float32, device=device)
    with torch.no_grad():
        sc_raw, logits = model(X_t)
        pred_sc_all = normalize_sc_pairs_torch(sc_raw)
        pick = torch.argmax(logits, dim=1)
        idx = torch.arange(X_t.shape[0], device=device)
        pred_sc = pred_sc_all[idx, pick, :].cpu().numpy()

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
    p = argparse.ArgumentParser(description="Train multi-head direct full-6 IK model with no analytical inference")
    p.add_argument("--dataset", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "puma560_dataset.csv"))
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--epochs", type=int, default=1400)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=260)
    p.add_argument("--min-epochs", type=int, default=300)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.03)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--w-sc", type=float, default=1.0)
    p.add_argument("--w-ang", type=float, default=1.4)
    p.add_argument("--w-pos", type=float, default=1.0)
    p.add_argument("--w-ori", type=float, default=0.08)
    p.add_argument("--w-conf", type=float, default=0.20)
    p.add_argument("--w-circ", type=float, default=0.01)
    p.add_argument("--joint-weights", type=str, default="1.0,1.0,1.1,1.8,1.1,1.8")

    p.add_argument("--monitor", type=str, default="mae", choices=["mae", "total"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    return p


def parse_joint_weights(s: str) -> torch.Tensor:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("--joint-weights must provide 6 comma-separated values")
    return torch.tensor(vals, dtype=torch.float32)


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / "direct6_mh_best.pt"
    last_path = ckpt_dir / "direct6_mh_last.pt"
    final_path = ckpt_dir / "direct6_mh_final.pt"

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

    model = MultiHeadDirectIKNet(
        n_heads=args.heads,
        hidden=args.hidden,
        n_blocks=args.blocks,
        dropout=args.dropout,
    ).to(device)

    joint_weights = parse_joint_weights(args.joint_weights).to(device)
    criterion = MultiHeadIKLoss(
        w_sc=args.w_sc,
        w_ang=args.w_ang,
        w_pos=args.w_pos,
        w_ori=args.w_ori,
        w_conf=args.w_conf,
        w_circ=args.w_circ,
        joint_weights=joint_weights,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=120, T_mult=2)

    history: Dict[str, list] = {
        "train_total": [],
        "val_total": [],
        "train_mae_deg": [],
        "val_mae_deg": [],
        "train_conf": [],
        "val_conf": [],
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
        best_metric = float(ckpt.get("best_metric", best_metric))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        stale = int(ckpt.get("stale", 0))

    print(f"Training multi-head direct 6-DOF model | train={len(X_tr_n)} val={len(X_va_n)} test={len(X_te_n)}")
    print(
        f"heads={args.heads} hidden={args.hidden} blocks={args.blocks} dropout={args.dropout} "
        f"monitor={args.monitor} joint_weights={args.joint_weights}"
    )

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tr_sum = {"total": 0.0, "mae_deg": 0.0, "conf": 0.0, "n": 0}
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")

        for xb, yb_sc, yb_deg, poseb in pbar:
            xb = xb.to(device)
            yb_sc = yb_sc.to(device)
            yb_deg = yb_deg.to(device)
            poseb = poseb.to(device)

            optimizer.zero_grad(set_to_none=True)
            sc_raw, logits = model(xb)
            loss, parts = criterion(sc_raw, logits, yb_sc, yb_deg, poseb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = xb.shape[0]
            tr_sum["total"] += float(loss.detach()) * bs
            tr_sum["mae_deg"] += float(parts["mae_deg"]) * bs
            tr_sum["conf"] += float(parts["conf"]) * bs
            tr_sum["n"] += bs
            pbar.set_postfix(loss=f"{float(loss.detach()):.5f}")

        scheduler.step(epoch + 1)

        model.eval()
        va_sum = {"total": 0.0, "mae_deg": 0.0, "conf": 0.0, "n": 0}
        with torch.no_grad():
            for xb, yb_sc, yb_deg, poseb in va_loader:
                xb = xb.to(device)
                yb_sc = yb_sc.to(device)
                yb_deg = yb_deg.to(device)
                poseb = poseb.to(device)
                sc_raw, logits = model(xb)
                loss, parts = criterion(sc_raw, logits, yb_sc, yb_deg, poseb)
                bs = xb.shape[0]
                va_sum["total"] += float(loss.detach()) * bs
                va_sum["mae_deg"] += float(parts["mae_deg"]) * bs
                va_sum["conf"] += float(parts["conf"]) * bs
                va_sum["n"] += bs

        train_total = tr_sum["total"] / max(1, tr_sum["n"])
        val_total = va_sum["total"] / max(1, va_sum["n"])
        train_mae_deg = tr_sum["mae_deg"] / max(1, tr_sum["n"])
        val_mae_deg = va_sum["mae_deg"] / max(1, va_sum["n"])
        train_conf = tr_sum["conf"] / max(1, tr_sum["n"])
        val_conf = va_sum["conf"] / max(1, va_sum["n"])
        lr = float(optimizer.param_groups[0]["lr"])

        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["train_mae_deg"].append(train_mae_deg)
        history["val_mae_deg"].append(val_mae_deg)
        history["train_conf"].append(train_conf)
        history["val_conf"].append(val_conf)
        history["lr"].append(lr)

        metric_now = val_mae_deg if args.monitor == "mae" else val_total
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
                    "architecture": "MultiHeadDirectIKNet_12to12",
                    "hidden": args.hidden,
                    "blocks": args.blocks,
                    "dropout": args.dropout,
                    "n_heads": args.heads,
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
                    "architecture": "MultiHeadDirectIKNet_12to12",
                    "hidden": args.hidden,
                    "blocks": args.blocks,
                    "dropout": args.dropout,
                    "n_heads": args.heads,
                },
                last_path,
            )

        print(
            f"Epoch {epoch+1:4d} | train={train_total:.6f} val={val_total:.6f} "
            f"train_mae={train_mae_deg:.4f}deg val_mae={val_mae_deg:.4f}deg "
            f"train_conf={train_conf:.4f} val_conf={val_conf:.4f} "
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
            "architecture": "MultiHeadDirectIKNet_12to12",
            "hidden": args.hidden,
            "blocks": args.blocks,
            "dropout": args.dropout,
            "n_heads": args.heads,
        },
        final_path,
    )

    np.savez(
        out_dir / "direct6_mh_eval_results.npz",
        pred_deg=metrics["pred_deg"],
        true_deg=metrics["true_deg"],
        err=metrics["err"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        avg_mae=metrics["avg_mae"],
        avg_rmse=metrics["avg_rmse"],
    )

    print("\nFinal test metrics (multi-head direct, no analytical inference):")
    print("  MAE per joint (deg):", np.array2string(metrics["mae"], precision=4))
    print("  RMSE per joint (deg):", np.array2string(metrics["rmse"], precision=4))
    print(f"  Avg MAE: {metrics['avg_mae']:.5f} deg")
    print(f"  Avg RMSE: {metrics['avg_rmse']:.5f} deg")
    print(f"Saved: {final_path}")
    print(f"Saved: {out_dir / 'direct6_mh_eval_results.npz'}")


if __name__ == "__main__":
    main()
