#!/usr/bin/env python3
"""
Benchmark multi-head direct IK model against analytical IK.
No training is performed here.
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
    b, k, _ = sc_raw.shape
    pairs = sc_raw.view(b, k, 6, 2)
    norms = torch.linalg.norm(pairs, dim=-1, keepdim=True).clamp_min(1e-6)
    return (pairs / norms).reshape(b, k, 12)


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


def infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, float, int]:
    hidden = int(state_dict["stem.0.weight"].shape[0])
    blocks = set()
    for k in state_dict.keys():
        if k.startswith("blocks."):
            p = k.split(".")
            if len(p) > 1 and p[1].isdigit():
                blocks.add(int(p[1]))
    n_blocks = (max(blocks) + 1) if blocks else 0
    n_heads = int(state_dict["conf_head.bias"].shape[0])
    dropout = 0.03
    return hidden, n_blocks, dropout, n_heads


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
    dh = [(0.0, D1, -90.0), (A2, D2, 0.0), (A3, 0.0, 90.0)]
    for i in range(3):
        T = T @ _dh_np(dh[i][0], dh[i][1], dh[i][2], float(theta123_deg[i]))
    return T


def _fPUMA_np(theta6_deg: np.ndarray) -> np.ndarray:
    dh = [(0.0, D1, -90.0), (A2, D2, 0.0), (A3, 0.0, 90.0), (0.0, D4, -90.0), (0.0, 0.0, 90.0), (0.0, D6, 0.0)]
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
        return 0.0, 0.0, float(np.rad2deg(np.arctan2(float(T3_6[1, 0]), float(T3_6[0, 0]))))
    if flip_wrist:
        sin5 = -sin5
    t5 = np.rad2deg(np.arctan2(sin5, float(T3_6[2, 2])))
    t4 = np.rad2deg(np.arctan2(float(T3_6[1, 2]), float(T3_6[0, 2])))
    t6 = np.rad2deg(np.arctan2(float(T3_6[2, 1]), -float(T3_6[2, 0])))
    return float(t4), float(t5), float(t6)


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

        theta1 = phi1 - alpha1 if shoulder_idx == 0 else phi1 + alpha1 - 180.0
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


def pose_errors_from_joints(pred_deg: np.ndarray, pose12: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = pred_deg.shape[0]
    pos_mm = np.zeros(n, dtype=np.float64)
    ori_deg = np.zeros(n, dtype=np.float64)
    for i in range(n):
        T_pred = _fPUMA_np(pred_deg[i])
        T_tgt = row_pose12_to_T06_np(pose12[i])
        pos_mm[i] = float(np.linalg.norm(T_pred[:3, 3] - T_tgt[:3, 3]))

        R_pred = T_pred[:3, :3]
        R_tgt = T_tgt[:3, :3]
        c = (np.trace(R_pred.T @ R_tgt) - 1.0) * 0.5
        c = float(np.clip(c, -1.0, 1.0))
        ori_deg[i] = float(np.rad2deg(np.arccos(c)))
    return pos_mm, ori_deg


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

    defaults = [
        script_dir / "checkpoints" / "direct6_mh_final.pt",
        script_dir / "checkpoints" / "direct6_mh_best.pt",
        script_dir / "checkpoints" / "direct6_mh_last.pt",
    ]
    search.extend(defaults)

    seen = set()
    checked = []
    for c in search:
        r = c.resolve()
        if str(r) in seen:
            continue
        seen.add(str(r))
        checked.append(str(r))
        if r.exists():
            print(f"[INFO] Using checkpoint: {r}")
            return r
    train_hint = (
        "No multi-head checkpoint found. First run:\n"
        "python full6dof_direct_ml_cpu\\train_direct_ik_multihead_no_analytical.py "
        "--epochs 1600 --batch 256 --threads 4 --hidden 256 --blocks 6 --heads 4 "
        "--dropout 0.03 --w-ang 1.4 --w-pos 1.0 --w-ori 0.08 --w-sc 1.0 --w-conf 0.2 "
        "--w-circ 0.01 --joint-weights 1.0,1.0,1.1,1.8,1.1,1.8 --monitor mae"
    )
    raise FileNotFoundError(f"No checkpoint found. Checked: {checked}\n{train_hint}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark multi-head direct IK latency against analytical IK")
    p.add_argument("--dataset", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "puma560_dataset.csv"))
    p.add_argument("--checkpoint", type=str, default=str(Path(__file__).resolve().parent / "checkpoints" / "direct6_mh_final.pt"))
    p.add_argument("--samples", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-faster", action="store_true")
    p.add_argument("--speed-margin", type=float, default=0.98)
    p.add_argument("--valid-pos-mm", type=float, default=5.0)
    p.add_argument("--valid-ori-deg", type=float, default=2.0)
    p.add_argument("--report-json", type=str, default=str(Path(__file__).resolve().parent / "benchmark_direct_multihead_vs_analytical.json"))
    return p.parse_args()


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
    dropout = float(ckpt.get("dropout", 0.03))
    heads = int(ckpt.get("n_heads", 0))
    if hidden <= 0 or blocks <= 0 or heads <= 0:
        hidden, blocks, dropout, heads = infer_arch_from_state_dict(ckpt["model_state"])

    model = MultiHeadDirectIKNet(n_heads=heads, hidden=hidden, n_blocks=blocks, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_mean = ckpt["x_mean"].astype(np.float32)
    x_std = ckpt["x_std"].astype(np.float32)
    Xn = (X - x_mean) / x_std

    with torch.no_grad():
        xb = torch.tensor(Xn[: max(1, min(args.warmup, len(Xn)))], dtype=torch.float32)
        _ = model(xb)

    direct_ms = []
    pred_direct = np.zeros((n, 6), dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            xb = torch.tensor(Xn[i : i + 1], dtype=torch.float32)
            t0 = time.perf_counter()
            sc_raw, logits = model(xb)
            sc_all = normalize_sc_pairs_torch(sc_raw)
            pick = int(torch.argmax(logits, dim=1).item())
            sc = sc_all[0, pick, :].cpu().numpy()[None, :]
            t1 = time.perf_counter()
            pred_direct[i] = sc_to_angles_np(sc)[0]
            direct_ms.append((t1 - t0) * 1000.0)
    direct_ms = np.asarray(direct_ms, dtype=np.float64)

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

    # Pose-valid correctness (counts alternate valid IK configurations as correct).
    d_pos_mm, d_ori_deg = pose_errors_from_joints(pred_direct, X)
    direct_pose_valid = np.logical_and(d_pos_mm <= args.valid_pos_mm, d_ori_deg <= args.valid_ori_deg)

    if valid.any():
        a_pos_mm, a_ori_deg = pose_errors_from_joints(pred_analytic[valid], X[valid])
        analytic_pose_valid = np.logical_and(a_pos_mm <= args.valid_pos_mm, a_ori_deg <= args.valid_ori_deg)
    else:
        a_pos_mm = np.array([], dtype=np.float64)
        a_ori_deg = np.array([], dtype=np.float64)
        analytic_pose_valid = np.array([], dtype=bool)

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
            "pose_pos_mm_mean": float(np.mean(d_pos_mm)),
            "pose_ori_deg_mean": float(np.mean(d_ori_deg)),
            "pose_valid_fraction": float(np.mean(direct_pose_valid)),
        },
        "analytical_accuracy": {
            "valid_fraction": float(valid.mean()),
            "mae_deg": mae_analytic.tolist(),
            "rmse_deg": rmse_analytic.tolist(),
            "avg_mae_deg": float(np.nanmean(mae_analytic)),
            "avg_rmse_deg": float(np.nanmean(rmse_analytic)),
            "pose_pos_mm_mean": float(np.mean(a_pos_mm)) if a_pos_mm.size > 0 else float("nan"),
            "pose_ori_deg_mean": float(np.mean(a_ori_deg)) if a_ori_deg.size > 0 else float("nan"),
            "pose_valid_fraction": float(np.mean(analytic_pose_valid)) if analytic_pose_valid.size > 0 else float("nan"),
        },
        "checkpoint": str(ckpt_path),
        "dataset": str(Path(args.dataset)),
        "threads": int(args.threads),
        "n_heads": int(heads),
    }

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Direct multi-head latency (ms):", d_stats)
    print("Analytical IK latency (ms):", a_stats)
    print(f"Direct/Analytical mean latency ratio: {speed_ratio:.4f}")
    print("Direct avg MAE (deg):", report["direct_accuracy"]["avg_mae_deg"])
    print("Analytical avg MAE (deg):", report["analytical_accuracy"]["avg_mae_deg"])
    print(
        "Direct pose-valid fraction "
        f"(pos<={args.valid_pos_mm:.1f}mm, ori<={args.valid_ori_deg:.1f}deg):",
        report["direct_accuracy"]["pose_valid_fraction"],
    )
    print(
        "Analytical pose-valid fraction "
        f"(pos<={args.valid_pos_mm:.1f}mm, ori<={args.valid_ori_deg:.1f}deg):",
        report["analytical_accuracy"]["pose_valid_fraction"],
    )
    print(f"Saved benchmark report: {args.report_json}")

    if args.require_faster:
        threshold = args.speed_margin * a_stats["mean_ms"]
        if d_stats["mean_ms"] > threshold:
            raise SystemExit(
                f"FAIL: direct mean latency {d_stats['mean_ms']:.6f}ms exceeds required threshold {threshold:.6f}ms"
            )


if __name__ == "__main__":
    main()
