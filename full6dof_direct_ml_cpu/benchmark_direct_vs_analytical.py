#!/usr/bin/env python3
"""
Latency + accuracy benchmark: direct full-ML IK vs analytical IK.

This script does not train models. It benchmarks:
1) Direct neural inference (no analytical wrist solve)
2) Analytical IK solver from puma560_3dof/train_puma560.py

Use --require-faster to hard-fail if direct model is not faster.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


# PUMA 560 geometric constants (mm / deg)
D1 = 671.83
A2 = 431.80
A3 = -20.32
D2 = 139.70
D4 = 431.80
D6 = 56.50

JOINT_LIMITS = np.array(
    [
        [-160, 160],
        [-225, 45],
        [-45, 225],
        [-110, 170],
        [-100, 100],
        [-266, 266],
    ],
    dtype=np.float64,
)


def wrap_angle_error(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    return (pred_deg - true_deg + 180.0) % 360.0 - 180.0


def normalize_sc_pairs_torch(sc_raw: torch.Tensor) -> torch.Tensor:
    pairs = sc_raw.view(sc_raw.shape[0], 6, 2)
    norms = torch.linalg.norm(pairs, dim=-1, keepdim=True).clamp_min(1e-6)
    return (pairs / norms).reshape(sc_raw.shape[0], 12)


def sc_to_angles_np(sc: np.ndarray) -> np.ndarray:
    out = np.zeros((sc.shape[0], 6), dtype=np.float32)
    for j in range(6):
        out[:, j] = np.rad2deg(np.arctan2(sc[:, 2 * j], sc[:, 2 * j + 1]))
    return out


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


def infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, float]:
    hidden = int(state_dict["stem.0.weight"].shape[0])
    block_indices = set()
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    n_blocks = (max(block_indices) + 1) if block_indices else 0
    dropout = 0.04
    return hidden, n_blocks, dropout


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


def _dh_np(a: float, d: float, alpha_deg: float, theta_deg: float) -> np.ndarray:
    al = np.deg2rad(alpha_deg)
    th = np.deg2rad(theta_deg)
    ca, sa = np.cos(al), np.sin(al)
    ct, st = np.cos(th), np.sin(th)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _t0_3_np(theta123_deg: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    dh = [
        (0.0, D1, -90.0),
        (A2, D2, 0.0),
        (A3, 0.0, 90.0),
    ]
    for i in range(3):
        T = T @ _dh_np(dh[i][0], dh[i][1], dh[i][2], float(theta123_deg[i]))
    return T


def _fPUMA_np(theta6_deg: np.ndarray) -> np.ndarray:
    dh = [
        (0.0, D1, -90.0),
        (A2, D2, 0.0),
        (A3, 0.0, 90.0),
        (0.0, D4, -90.0),
        (0.0, 0.0, 90.0),
        (0.0, D6, 0.0),
    ]
    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        T = T @ _dh_np(dh[i][0], dh[i][1], dh[i][2], float(theta6_deg[i]))
    return T


def _solve_wrist(theta123_deg: np.ndarray, T06: np.ndarray, flip_wrist: bool = False) -> Tuple[float, float, float]:
    T0_3 = _t0_3_np(theta123_deg)
    T3_6 = np.linalg.inv(T0_3) @ T06

    sin5_sq = max(0.0, 1.0 - float(T3_6[2, 2]) ** 2)
    sin5 = np.sqrt(sin5_sq)

    if sin5 < 1e-8:
        theta5 = 0.0
        theta4 = 0.0
        theta6 = np.rad2deg(np.arctan2(float(T3_6[1, 0]), float(T3_6[0, 0])))
        return float(theta4), float(theta5), float(theta6)

    if flip_wrist:
        sin5 = -sin5

    theta5 = np.rad2deg(np.arctan2(sin5, float(T3_6[2, 2])))
    theta4 = np.rad2deg(np.arctan2(float(T3_6[1, 2]), float(T3_6[0, 2])))
    theta6 = np.rad2deg(np.arctan2(float(T3_6[2, 1]), -float(T3_6[2, 0])))
    return float(theta4), float(theta5), float(theta6)


def _wrap180(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def analytical_ik(T06: np.ndarray, configs: Tuple[int, ...] = (1, 2, 3, 4)) -> Tuple[np.ndarray | None, float]:
    ax, ay, az = T06[0, 2], T06[1, 2], T06[2, 2]
    px, py, pz = T06[0, 3], T06[1, 3], T06[2, 3]

    p5x = px - D6 * ax
    p5y = py - D6 * ay
    p5z = pz - D6 * az

    c1_rad = np.sqrt(p5x**2 + p5y**2)
    if c1_rad < 1e-8:
        return None, np.inf

    ratio = D2 / c1_rad
    if abs(ratio) > 1.0:
        return None, np.inf

    alpha1 = np.rad2deg(np.arctan2(ratio, np.sqrt(max(0.0, 1.0 - ratio**2))))
    phi1 = np.rad2deg(np.arctan2(p5y, p5x))
    phi3 = np.rad2deg(np.arctan2(A3, D4))

    best_J = None
    best_err = np.inf

    for cfg in configs:
        shoulder_idx = (cfg - 1) // 4
        elbow_idx = ((cfg - 1) % 4) // 2
        wrist_idx = (cfg - 1) % 2

        if shoulder_idx == 0:
            theta1 = phi1 - alpha1
        else:
            theta1 = phi1 + alpha1 - 180.0
        theta1 = _wrap180(theta1)

        t1r = np.deg2rad(theta1)
        c1 = p5x * np.cos(t1r) + p5y * np.sin(t1r)
        c2 = p5z - D1
        c3 = np.sqrt(c1**2 + c2**2)
        c4 = np.sqrt(A3**2 + D4**2)

        da = np.clip((c3**2 + A2**2 - c4**2) / (2 * A2 * c3), -1.0, 1.0)
        db = np.clip((A2**2 + c4**2 - c3**2) / (2 * A2 * c4), -1.0, 1.0)

        phi2 = np.rad2deg(np.arctan2(c2, c1))
        if elbow_idx == 0:
            alpha2 = np.rad2deg(np.arctan2(np.sqrt(max(0.0, 1 - da**2)), da))
            beta = np.rad2deg(np.arctan2(np.sqrt(max(0.0, 1 - db**2)), db))
        else:
            alpha2 = np.rad2deg(np.arctan2(-np.sqrt(max(0.0, 1 - da**2)), da))
            beta = np.rad2deg(np.arctan2(-np.sqrt(max(0.0, 1 - db**2)), db))

        theta2 = alpha2 - phi2
        theta3 = beta - 90.0 - phi3

        t4, t5, t6 = _solve_wrist(np.array([theta1, theta2, theta3]), T06, flip_wrist=(wrist_idx == 1))
        J = np.array([theta1, theta2, theta3, t4, t5, t6], dtype=np.float64)

        if not np.all((J >= JOINT_LIMITS[:, 0]) & (J <= JOINT_LIMITS[:, 1])):
            continue

        T_check = _fPUMA_np(J)
        err = np.linalg.norm(T_check[:3, 3] - T06[:3, 3])
        if err < best_err:
            best_err = err
            best_J = J

    return best_J, float(best_err)


def latency_stats_ms(samples_ms: np.ndarray) -> Dict[str, float]:
    return {
        "mean_ms": float(np.mean(samples_ms)),
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
        "min_ms": float(np.min(samples_ms)),
        "max_ms": float(np.max(samples_ms)),
    }


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return raw[:, :12].astype(np.float32), raw[:, 12:18].astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark direct IK latency against analytical IK")
    p.add_argument("--dataset", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "puma560_dataset.csv"))
    p.add_argument("--checkpoint", type=str, default=str(Path(__file__).resolve().parent / "checkpoints" / "direct6_final.pt"))
    p.add_argument("--samples", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quantize", action="store_true", help="Use dynamic INT8 quantization for direct model")
    p.add_argument("--require-faster", action="store_true", help="Exit non-zero if direct model is not faster")
    p.add_argument("--speed-margin", type=float, default=0.98, help="Require direct_mean <= margin * analytical_mean")
    p.add_argument("--report-json", type=str, default=str(Path(__file__).resolve().parent / "benchmark_direct_vs_analytical.json"))
    return p.parse_args()


def resolve_checkpoint(path_str: str) -> Path:
    p_in = Path(path_str)
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    search = []
    if p_in.is_absolute():
        search.append(p_in)
    else:
        search.append(Path.cwd() / p_in)
        search.append(script_dir / p_in)
        search.append(root_dir / p_in)

    default_candidates = [
        script_dir / "checkpoints" / "direct6_final.pt",
        script_dir / "checkpoints" / "direct6_best.pt",
        script_dir / "checkpoints" / "direct6_last.pt",
        root_dir / "full6dof_direct_ml_cpu" / "checkpoints" / "direct6_final.pt",
        root_dir / "full6dof_direct_ml_cpu" / "checkpoints" / "direct6_best.pt",
        root_dir / "full6dof_direct_ml_cpu" / "checkpoints" / "direct6_last.pt",
    ]
    search.extend(default_candidates)

    checked = []
    seen = set()
    for candidate in search:
        c = candidate.resolve()
        if str(c) in seen:
            continue
        seen.add(str(c))
        checked.append(str(c))
        if c.exists():
            if c != p_in:
                print(f"[INFO] Using checkpoint: {c}")
            return c

    raise FileNotFoundError(f"No checkpoint found. Checked: {checked}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    X_all, Y_all = load_dataset(Path(args.dataset))
    n = min(args.samples, len(X_all))
    idx = np.random.default_rng(args.seed).choice(len(X_all), size=n, replace=False)
    X = X_all[idx]
    Y = Y_all[idx]

    ckpt_path = resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("hidden", 0))
    blocks = int(ckpt.get("blocks", 0))
    dropout = float(ckpt.get("dropout", 0.04))
    if hidden <= 0 or blocks <= 0:
        hidden, blocks, dropout = infer_arch_from_state_dict(ckpt["model_state"])
    model = DirectIKNet(hidden=hidden, n_blocks=blocks, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    if args.quantize:
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        print("[INFO] Dynamic quantization enabled for direct model")
    model.eval()

    x_mean = ckpt["x_mean"].astype(np.float32)
    x_std = ckpt["x_std"].astype(np.float32)
    Xn = (X - x_mean) / x_std

    # Warmup
    with torch.no_grad():
        xb = torch.tensor(Xn[: max(1, min(args.warmup, len(Xn)))], dtype=torch.float32)
        _ = model(xb)

    # Direct model latency and predictions
    direct_ms = []
    pred_direct = np.zeros((n, 6), dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            xb = torch.tensor(Xn[i : i + 1], dtype=torch.float32)
            t0 = time.perf_counter()
            raw = model(xb)
            sc = normalize_sc_pairs_torch(raw).cpu().numpy()
            t1 = time.perf_counter()
            pred_direct[i] = sc_to_angles_np(sc)[0]
            direct_ms.append((t1 - t0) * 1000.0)
    direct_ms = np.asarray(direct_ms, dtype=np.float64)

    # Analytical latency and predictions
    analytical_ms = []
    pred_analytic = np.full((n, 6), np.nan, dtype=np.float32)
    for i in range(n):
        T06 = row_pose12_to_T06_np(X[i])
        t0 = time.perf_counter()
        j_sol, _ = analytical_ik(T06, configs=(1, 2, 3, 4))
        t1 = time.perf_counter()
        analytical_ms.append((t1 - t0) * 1000.0)
        if j_sol is not None:
            pred_analytic[i] = j_sol.astype(np.float32)
    analytical_ms = np.asarray(analytical_ms, dtype=np.float64)

    # Accuracy summary
    err_direct = wrap_angle_error(pred_direct, Y)
    mae_direct = np.abs(err_direct).mean(axis=0)
    rmse_direct = np.sqrt((err_direct ** 2).mean(axis=0))

    valid = np.isfinite(pred_analytic).all(axis=1)
    if valid.any():
        err_analytic = wrap_angle_error(pred_analytic[valid], Y[valid])
        mae_analytic = np.abs(err_analytic).mean(axis=0)
        rmse_analytic = np.sqrt((err_analytic ** 2).mean(axis=0))
    else:
        mae_analytic = np.full(6, np.nan, dtype=np.float32)
        rmse_analytic = np.full(6, np.nan, dtype=np.float32)

    d_stats = latency_stats_ms(direct_ms)
    a_stats = latency_stats_ms(analytical_ms)
    speed_ratio = d_stats["mean_ms"] / max(1e-9, a_stats["mean_ms"])

    report = {
        "samples": int(n),
        "direct_latency_ms": d_stats,
        "analytical_latency_ms": a_stats,
        "direct_over_analytical_ratio": float(speed_ratio),
        "direct_accuracy": {
            "mae_deg": mae_direct.tolist(),
            "rmse_deg": rmse_direct.tolist(),
            "avg_mae_deg": float(np.mean(mae_direct)),
            "avg_rmse_deg": float(np.mean(rmse_direct)),
        },
        "analytical_accuracy": {
            "valid_fraction": float(valid.mean()),
            "mae_deg": mae_analytic.tolist(),
            "rmse_deg": rmse_analytic.tolist(),
            "avg_mae_deg": float(np.nanmean(mae_analytic)),
            "avg_rmse_deg": float(np.nanmean(rmse_analytic)),
        },
        "checkpoint": str(ckpt_path),
        "dataset": str(Path(args.dataset)),
        "threads": int(args.threads),
        "quantize": bool(args.quantize),
    }

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Direct model latency (ms):", d_stats)
    print("Analytical IK latency (ms):", a_stats)
    print(f"Direct/Analytical mean latency ratio: {speed_ratio:.4f}")
    print("Direct avg MAE (deg):", report["direct_accuracy"]["avg_mae_deg"])
    print("Analytical avg MAE (deg):", report["analytical_accuracy"]["avg_mae_deg"])
    print(f"Saved benchmark report: {args.report_json}")

    if args.require_faster:
        threshold = args.speed_margin * a_stats["mean_ms"]
        if d_stats["mean_ms"] > threshold:
            raise SystemExit(
                f"FAIL: direct mean latency {d_stats['mean_ms']:.6f}ms exceeds required threshold {threshold:.6f}ms"
            )


if __name__ == "__main__":
    main()
