#!/usr/bin/env python3
"""
Hybrid QNN residual trainer for full PUMA 560 IK (CPU-only).

Design:
- Backbone: pretrained ANN (from full6dof_ann_cpu) predicting shoulder sin/cos (J1..J3).
- Quantum branch: tiny differentiable 4-qubit simulator that learns residual shoulder corrections.
- Wrist: J4..J6 recovered analytically in evaluation via shared utilities.

Safety:
- CPU-only execution
- Per-epoch checkpointing (last + best)
- Resume support
- Robust atomic checkpoint write (imported from ANN trainer)
- Early stopping with minimum epochs
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Reuse proven utilities from ANN trainer.
ANN_DIR = Path(__file__).resolve().parents[1] / "full6dof_ann_cpu"
if str(ANN_DIR) not in sys.path:
    sys.path.insert(0, str(ANN_DIR))

from train_ann_full6_cpu import (  # noqa: E402
    FullIKANN,
    FullIKLoss,
    angles_to_sc_np,
    atomic_torch_save,
    compute_wrist_center_np,
    evaluate_model,
    load_dataset,
    make_plots,
    normalize_X,
    set_seed,
    split_indices,
)


class TinyQuantumLayer(nn.Module):
    """Small differentiable statevector simulator (4 qubits by default)."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits

        self.encoder = nn.Linear(3, n_qubits)
        self.theta = nn.Parameter(torch.zeros(n_layers, n_qubits, 3))
        nn.init.normal_(self.theta, mean=0.0, std=0.04)

        # Precompute permutation metadata for gate application.
        self.perms: List[List[int]] = []
        self.inv_perms: List[List[int]] = []
        for q in range(n_qubits):
            perm = [0] + [i + 1 for i in range(n_qubits) if i != q] + [q + 1]
            inv = [0] * (n_qubits + 1)
            for i, p in enumerate(perm):
                inv[p] = i
            self.perms.append(perm)
            self.inv_perms.append(inv)

        # CNOT ring topology: 0->1->2->3->0
        self.cnot_pairs: List[Tuple[int, int]] = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        for idx, (control, target) in enumerate(self.cnot_pairs):
            idx0, idx1 = self._build_cnot_swap_indices(control, target, n_qubits)
            self.register_buffer(f"cnot_idx0_{idx}", idx0)
            self.register_buffer(f"cnot_idx1_{idx}", idx1)

        # Z expectation masks per qubit.
        for q in range(n_qubits):
            z = self._build_z_mask(q, n_qubits)
            self.register_buffer(f"z_mask_{q}", z)

    @staticmethod
    def _bit(value: int, qubit: int, n_qubits: int) -> int:
        # qubit 0 corresponds to the most-significant bit of basis index.
        shift = n_qubits - 1 - qubit
        return (value >> shift) & 1

    @classmethod
    def _build_cnot_swap_indices(cls, control: int, target: int, n_qubits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx0: List[int] = []
        idx1: List[int] = []
        for i in range(2 ** n_qubits):
            c = cls._bit(i, control, n_qubits)
            t = cls._bit(i, target, n_qubits)
            if c == 1 and t == 0:
                j = i | (1 << (n_qubits - 1 - target))
                idx0.append(i)
                idx1.append(j)
        return torch.tensor(idx0, dtype=torch.long), torch.tensor(idx1, dtype=torch.long)

    @classmethod
    def _build_z_mask(cls, qubit: int, n_qubits: int) -> torch.Tensor:
        mask = torch.ones(2 ** n_qubits, dtype=torch.float32)
        for i in range(2 ** n_qubits):
            if cls._bit(i, qubit, n_qubits) == 1:
                mask[i] = -1.0
        return mask

    @staticmethod
    def _rx(theta: torch.Tensor) -> torch.Tensor:
        b = theta.shape[0]
        c = torch.cos(theta / 2.0)
        s = torch.sin(theta / 2.0)
        g = torch.zeros(b, 2, 2, dtype=torch.cfloat, device=theta.device)
        g[:, 0, 0] = c
        g[:, 1, 1] = c
        g[:, 0, 1] = -1j * s
        g[:, 1, 0] = -1j * s
        return g

    @staticmethod
    def _ry(theta: torch.Tensor) -> torch.Tensor:
        b = theta.shape[0]
        c = torch.cos(theta / 2.0)
        s = torch.sin(theta / 2.0)
        g = torch.zeros(b, 2, 2, dtype=torch.cfloat, device=theta.device)
        g[:, 0, 0] = c
        g[:, 0, 1] = -s
        g[:, 1, 0] = s
        g[:, 1, 1] = c
        return g

    @staticmethod
    def _rz(theta: torch.Tensor) -> torch.Tensor:
        b = theta.shape[0]
        g = torch.zeros(b, 2, 2, dtype=torch.cfloat, device=theta.device)
        one = torch.ones_like(theta)
        g[:, 0, 0] = torch.polar(one, -theta / 2.0)
        g[:, 1, 1] = torch.polar(one, theta / 2.0)
        return g

    def _apply_single_qubit_gate(self, state: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
        b = state.shape[0]
        n = self.n_qubits
        psi = state.view(b, *([2] * n))

        perm = self.perms[qubit]
        inv = self.inv_perms[qubit]

        psi_perm = psi.permute(perm).contiguous().view(b, -1, 2)
        out_perm = torch.einsum("bmi,bij->bmj", psi_perm, gate)
        out = out_perm.view(b, *([2] * n)).permute(inv).contiguous().view(b, self.dim)
        return out

    @staticmethod
    def _apply_cnot_swaps(state: torch.Tensor, idx0: torch.Tensor, idx1: torch.Tensor) -> torch.Tensor:
        out = state.clone()
        tmp = out[:, idx0].clone()
        out[:, idx0] = out[:, idx1]
        out[:, idx1] = tmp
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        enc = math.pi * torch.tanh(self.encoder(x))

        state = torch.zeros(b, self.dim, dtype=torch.cfloat, device=x.device)
        state[:, 0] = 1.0 + 0.0j

        # Data encoding layer.
        for q in range(self.n_qubits):
            state = self._apply_single_qubit_gate(state, self._ry(enc[:, q]), q)

        # Variational layers.
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                thx = self.theta[layer, q, 0].expand(b)
                thy = self.theta[layer, q, 1].expand(b)
                thz = self.theta[layer, q, 2].expand(b)
                state = self._apply_single_qubit_gate(state, self._rx(thx), q)
                state = self._apply_single_qubit_gate(state, self._ry(thy), q)
                state = self._apply_single_qubit_gate(state, self._rz(thz), q)

            for idx in range(len(self.cnot_pairs)):
                idx0 = getattr(self, f"cnot_idx0_{idx}")
                idx1 = getattr(self, f"cnot_idx1_{idx}")
                state = self._apply_cnot_swaps(state, idx0, idx1)

        probs = state.real.pow(2) + state.imag.pow(2)
        exps = []
        for q in range(self.n_qubits):
            z = getattr(self, f"z_mask_{q}")
            exps.append((probs * z).sum(dim=1, keepdim=True))
        return torch.cat(exps, dim=1)


class HybridQNNIK(nn.Module):
    """Frozen ANN backbone + trainable quantum residual branch."""

    def __init__(self, backbone: FullIKANN, n_qubits: int = 4, n_q_layers: int = 2):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.quantum = TinyQuantumLayer(n_qubits=n_qubits, n_layers=n_q_layers)
        self.projector = nn.Linear(n_qubits, 6)
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

        self.residual_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.alpha_eval = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.backbone(x)
        q_feat = self.quantum(x)
        delta = self.projector(q_feat)
        gain = torch.tanh(self.residual_gain)
        return base + self.alpha_eval * gain * delta


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CPU-only hybrid QNN residual trainer for full PUMA 560 IK")
    p.add_argument("--dataset", type=str, default=str(Path(__file__).resolve().parents[1] / "puma560_dataset.csv"))
    p.add_argument("--ann-checkpoint", type=str, default=str(Path(__file__).resolve().parents[1] / "full6dof_ann_cpu" / "checkpoints" / "ann6_best.pt"))
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent))

    p.add_argument("--epochs", type=int, default=1800)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr-quantum", type=float, default=3e-3)
    p.add_argument("--lr-projector", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=350)
    p.add_argument("--min-epochs", type=int, default=300)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--q-qubits", type=int, default=4)
    p.add_argument("--q-layers", type=int, default=2)

    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--w-sc", type=float, default=1.0)
    p.add_argument("--w-pos", type=float, default=2.0)
    p.add_argument("--w-circ", type=float, default=0.02)

    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--deploy-alpha", type=float, default=None)
    return p


def _select_best_alpha(
    model: HybridQNNIK,
    X_val_n: np.ndarray,
    X_val_pose: np.ndarray,
    Y_val: np.ndarray,
    device: torch.device,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Validation calibration for residual blend.
    alpha=0 is pure backbone (non-regression anchor).
    """
    grid = np.linspace(0.0, 1.5, 31)
    best_alpha = 0.0
    best_key = (float("inf"), float("inf"))
    best_metrics: Dict[str, np.ndarray] | None = None

    for alpha in grid:
        model.alpha_eval = float(alpha)
        m = evaluate_model(model, X_val_n, X_val_pose, Y_val, device)
        key = (-float(m["tol"]["all_joints_within_1.0deg"]), float(m["avg_mae"]))
        if key < best_key:
            best_key = key
            best_alpha = float(alpha)
            best_metrics = m

    assert best_metrics is not None
    model.alpha_eval = best_alpha
    return best_alpha, best_metrics


def _build_ann_baseline_from_checkpoint(ann_ckpt: dict) -> FullIKANN:
    ann = FullIKANN(
        n_in=int(ann_ckpt.get("model_input_dim", 3)),
        hidden=int(ann_ckpt["args"].get("hidden", 384)),
        n_blocks=int(ann_ckpt["args"].get("blocks", 6)),
        dropout=float(ann_ckpt["args"].get("dropout", 0.05)),
        n_out=int(ann_ckpt.get("model_output_dim", 6)),
    )
    ann.load_state_dict(ann_ckpt["model_state"])
    ann.eval()
    return ann


def main() -> None:
    args = build_arg_parser().parse_args()

    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    print("=" * 90)
    print("PUMA 560 Hybrid QNN Residual (CPU-only)")
    print("=" * 90)
    print(f"Dataset: {args.dataset}")
    print(f"ANN ckpt: {args.ann_checkpoint}")
    print(f"Output : {args.output_dir}")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs} | Batch: {args.batch}")

    ann_ckpt_path = Path(args.ann_checkpoint)
    if not ann_ckpt_path.exists():
        raise FileNotFoundError(f"ANN checkpoint not found: {ann_ckpt_path}")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / "hybrid_best.pt"
    last_path = ckpt_dir / "hybrid_last.pt"
    final_path = ckpt_dir / "hybrid_final.pt"

    # ---------------------------- Data ---------------------------------
    X_all, Y_all = load_dataset(Path(args.dataset))
    P5_all = compute_wrist_center_np(X_all)
    train_idx, val_idx, test_idx = split_indices(len(X_all), args.test_frac, args.val_frac, args.seed)

    X_train_pose, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val_pose, Y_val = X_all[val_idx], Y_all[val_idx]
    X_test_pose, Y_test = X_all[test_idx], Y_all[test_idx]

    X_train_wc, X_val_wc, X_test_wc = P5_all[train_idx], P5_all[val_idx], P5_all[test_idx]
    X_train_n, X_val_n, X_test_n, x_mean, x_std = normalize_X(X_train_wc, X_val_wc, X_test_wc)

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

    # -------------------------- Models ---------------------------------
    ann_ckpt = torch.load(ann_ckpt_path, map_location="cpu", weights_only=False)
    ann_backbone = _build_ann_baseline_from_checkpoint(ann_ckpt)
    model = HybridQNNIK(
        backbone=ann_backbone,
        n_qubits=args.q_qubits,
        n_q_layers=args.q_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (hybrid residual): {n_params:,}")

    criterion = FullIKLoss(w_sc=args.w_sc, w_wc=args.w_pos, w_circ=args.w_circ)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.quantum.parameters(), "lr": args.lr_quantum},
            {"params": list(model.projector.parameters()) + [model.residual_gain], "lr": args.lr_projector},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=150,
        T_mult=2,
        eta_min=min(args.lr_quantum, args.lr_projector) * 1e-2,
    )

    start_epoch = 1
    best_monitor = float("inf")
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
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_monitor = float(ckpt["best_monitor"])
        best_epoch = int(ckpt["best_epoch"])
        no_improve = int(ckpt.get("no_improve", 0))
        history = ckpt.get("history", history)
        model.alpha_eval = float(ckpt.get("alpha_eval", 1.0))
        print(f"[RESUME] epoch={start_epoch} | best_monitor={best_monitor:.6f} @ epoch {best_epoch}")

    # --------------------------- Train ---------------------------------
    t0 = time.time()

    pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Training HYBRID", unit="epoch")
    for epoch in pbar:
        model.train()
        model.alpha_eval = 1.0

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
        model.alpha_eval = 1.0
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
        improved = monitor < best_monitor
        if improved:
            best_monitor = monitor
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        pbar.set_postfix(
            {
                "tr": f"{tr['total']:.6f}",
                "va_sc": f"{va['sc']:.6f}",
                "best": best_epoch,
                "lr_q": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        if (epoch % args.save_every == 0) or improved:
            ckpt_last = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_monitor": best_monitor,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "history": history,
                "x_mean": x_mean,
                "x_std": x_std,
                "alpha_eval": float(model.alpha_eval),
                "args": vars(args),
            }
            atomic_torch_save(ckpt_last, last_path)

        if improved:
            ckpt_best = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_monitor": best_monitor,
                "best_epoch": best_epoch,
                "history": history,
                "x_mean": x_mean,
                "x_std": x_std,
                "alpha_eval": float(model.alpha_eval),
                "args": vars(args),
            }
            atomic_torch_save(ckpt_best, best_path)

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"\n[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}, best_monitor={best_monitor:.6f}")
            break

    train_time = time.time() - t0

    # ------------------------- Evaluate best ---------------------------
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])

    # Validation alpha calibration; optionally allow an explicit deploy alpha.
    if args.deploy_alpha is None:
        best_alpha, val_alpha_metrics = _select_best_alpha(model, X_val_n, X_val_pose, Y_val, device)
        model.alpha_eval = best_alpha
    else:
        best_alpha = float(args.deploy_alpha)
        model.alpha_eval = best_alpha
        val_alpha_metrics = evaluate_model(model, X_val_n, X_val_pose, Y_val, device)

    hybrid_metrics = evaluate_model(model, X_test_n, X_test_pose, Y_test, device)

    # ANN baseline on same split.
    ann_model = _build_ann_baseline_from_checkpoint(ann_ckpt)
    ann_x_mean = np.asarray(ann_ckpt.get("x_mean", x_mean))
    ann_x_std = np.asarray(ann_ckpt.get("x_std", x_std))
    if ann_x_mean.shape[-1] == X_test_wc.shape[-1] and ann_x_std.shape[-1] == X_test_wc.shape[-1]:
        X_test_n_ann = (X_test_wc - ann_x_mean) / ann_x_std
    else:
        X_test_n_ann = X_test_n
    ann_metrics = evaluate_model(ann_model, X_test_n_ann, X_test_pose, Y_test, device)

    print("=" * 90)
    print("FINAL TEST RESULTS (HYBRID QNN)")
    print("=" * 90)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val shoulder sc loss: {best_monitor:.6f}")
    print(f"Selected alpha (val-calibrated): {best_alpha:.3f}")
    print(f"Train time: {train_time:.1f} s")
    print()

    print(f"{'Joint':<8}{'MAE (deg)':>12}{'RMSE (deg)':>14}{'Within 1 deg':>14}")
    print("-" * 48)
    for j in range(6):
        within1 = hybrid_metrics["tol"]["joint_within_1.0deg"][j] * 100.0
        print(f"J{j+1:<7}{hybrid_metrics['mae'][j]:>12.4f}{hybrid_metrics['rmse'][j]:>14.4f}{within1:>13.2f}%")

    print("-" * 48)
    print(f"{'AVG':<8}{hybrid_metrics['avg_mae']:>12.4f}{hybrid_metrics['avg_rmse']:>14.4f}")
    print()
    print(f"All-joint accuracy within 0.5 deg : {hybrid_metrics['tol']['all_joints_within_0.5deg'] * 100:.2f}%")
    print(f"All-joint accuracy within 1.0 deg : {hybrid_metrics['tol']['all_joints_within_1.0deg'] * 100:.2f}%")
    print(f"All-joint accuracy within 2.0 deg : {hybrid_metrics['tol']['all_joints_within_2.0deg'] * 100:.2f}%")

    print("\nANN baseline on same split:")
    print(f"ANN avg MAE: {ann_metrics['avg_mae']:.4f} deg")
    print(f"ANN all joints <= 1.0 deg: {ann_metrics['tol']['all_joints_within_1.0deg'] * 100:.2f}%")
    print(f"Delta avg MAE (ANN - Hybrid): {ann_metrics['avg_mae'] - hybrid_metrics['avg_mae']:.4f} deg")
    print(
        f"Delta all joints <=1.0 deg: "
        f"{(hybrid_metrics['tol']['all_joints_within_1.0deg'] - ann_metrics['tol']['all_joints_within_1.0deg']) * 100:.2f}%"
    )

    final_ckpt = {
        "model_state": model.state_dict(),
        "best_epoch": best_epoch,
        "best_monitor": best_monitor,
        "best_alpha": best_alpha,
        "train_time": train_time,
        "history": history,
        "x_mean": x_mean,
        "x_std": x_std,
        "metrics_hybrid": {
            "mae": hybrid_metrics["mae"],
            "rmse": hybrid_metrics["rmse"],
            "avg_mae": hybrid_metrics["avg_mae"],
            "avg_rmse": hybrid_metrics["avg_rmse"],
            "all_joints_within_0.5deg": hybrid_metrics["tol"]["all_joints_within_0.5deg"],
            "all_joints_within_1.0deg": hybrid_metrics["tol"]["all_joints_within_1.0deg"],
            "all_joints_within_2.0deg": hybrid_metrics["tol"]["all_joints_within_2.0deg"],
        },
        "metrics_ann": {
            "avg_mae": ann_metrics["avg_mae"],
            "avg_rmse": ann_metrics["avg_rmse"],
            "all_joints_within_1.0deg": ann_metrics["tol"]["all_joints_within_1.0deg"],
        },
        "q_qubits": args.q_qubits,
        "q_layers": args.q_layers,
        "args": vars(args),
    }
    atomic_torch_save(final_ckpt, final_path)

    np.savez(
        output_dir / "hybrid_eval_results.npz",
        mae=hybrid_metrics["mae"],
        rmse=hybrid_metrics["rmse"],
        avg_mae=np.array([hybrid_metrics["avg_mae"]], dtype=np.float32),
        avg_rmse=np.array([hybrid_metrics["avg_rmse"]], dtype=np.float32),
        all_joints_within_0_5=np.array([hybrid_metrics["tol"]["all_joints_within_0.5deg"]], dtype=np.float32),
        all_joints_within_1_0=np.array([hybrid_metrics["tol"]["all_joints_within_1.0deg"]], dtype=np.float32),
        all_joints_within_2_0=np.array([hybrid_metrics["tol"]["all_joints_within_2.0deg"]], dtype=np.float32),
        best_alpha=np.array([best_alpha], dtype=np.float32),
        ann_avg_mae=np.array([ann_metrics["avg_mae"]], dtype=np.float32),
        ann_all_joints_within_1_0=np.array([ann_metrics["tol"]["all_joints_within_1.0deg"]], dtype=np.float32),
    )

    print(f"\nSaved: {final_path}")
    print(f"Saved: {output_dir / 'hybrid_eval_results.npz'}")

    if not args.no_plots:
        try:
            make_plots(history, hybrid_metrics, output_dir)
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
