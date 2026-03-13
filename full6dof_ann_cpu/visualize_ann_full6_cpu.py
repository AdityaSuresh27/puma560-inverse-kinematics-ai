#!/usr/bin/env python3
"""
Visualize/evaluate the trained full-6DOF ANN checkpoint.

Usage:
  python visualize_ann_full6_cpu.py
  python visualize_ann_full6_cpu.py --checkpoint checkpoints/ann6_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_ann_full6_cpu import (
    FullIKANN,
    compute_wrist_center_np,
    evaluate_model,
    load_dataset,
    make_plots,
    normalize_X,
    split_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize full-6DOF ANN results")
    parser.add_argument("--dataset", type=str, default=str(Path(__file__).resolve().parents[1] / "puma560_dataset.csv"))
    parser.add_argument("--checkpoint", type=str, default=str(Path(__file__).resolve().parent / "checkpoints" / "ann6_final.pt"))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})

    X_all, Y_all = load_dataset(Path(args.dataset))
    seed = int(train_args.get("seed", 42))
    test_frac = float(train_args.get("test_frac", 0.15))
    val_frac = float(train_args.get("val_frac", 0.10))

    train_idx, val_idx, test_idx = split_indices(len(X_all), test_frac, val_frac, seed)
    X_train_pose, X_val_pose, X_test_pose = X_all[train_idx], X_all[val_idx], X_all[test_idx]
    Y_test = Y_all[test_idx]

    X_train = compute_wrist_center_np(X_train_pose)
    X_val = compute_wrist_center_np(X_val_pose)
    X_test = compute_wrist_center_np(X_test_pose)

    # Prefer normalization from checkpoint for exact reproducibility
    if "x_mean" in ckpt and "x_std" in ckpt:
        x_mean = np.asarray(ckpt["x_mean"])
        x_std = np.asarray(ckpt["x_std"])
        if x_mean.shape[-1] == X_test.shape[-1] and x_std.shape[-1] == X_test.shape[-1]:
            X_test_n = (X_test - x_mean) / x_std
        else:
            _, _, X_test_n, _, _ = normalize_X(X_train, X_val, X_test)
    else:
        _, _, X_test_n, _, _ = normalize_X(X_train, X_val, X_test)

    model = FullIKANN(
        n_in=int(ckpt.get("model_input_dim", 3)),
        hidden=int(train_args.get("hidden", 512)),
        n_blocks=int(train_args.get("blocks", 8)),
        dropout=float(train_args.get("dropout", 0.05)),
        n_out=int(ckpt.get("model_output_dim", 6)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    metrics = evaluate_model(model, X_test_n, X_test_pose, Y_test, device=torch.device("cpu"))

    print("=" * 70)
    print("ANN6 CHECKPOINT EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Best epoch: {ckpt.get('best_epoch', 'n/a')}")
    print(f"Avg MAE   : {metrics['avg_mae']:.4f} deg")
    print(f"Avg RMSE  : {metrics['avg_rmse']:.4f} deg")
    print(f"All joints <= 1.0 deg: {metrics['tol']['all_joints_within_1.0deg'] * 100:.2f}%")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If history exists, reuse it; otherwise create placeholder curves.
    history = ckpt.get("history", {
        "train_total": [0.0],
        "val_total": [0.0],
        "train_sc": [0.0],
        "val_sc": [0.0],
        "train_pos": [0.0],
        "val_pos": [0.0],
        "train_ori": [0.0],
        "val_ori": [0.0],
        "train_circ": [0.0],
        "val_circ": [0.0],
        "lr": [0.0],
    })

    make_plots(history, metrics, out_dir)
    np.savez(
        out_dir / "ann6_eval_results_visualized.npz",
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        avg_mae=np.array([metrics["avg_mae"]], dtype=np.float32),
        avg_rmse=np.array([metrics["avg_rmse"]], dtype=np.float32),
    )

    print("Saved visualizations in output directory.")


if __name__ == "__main__":
    main()
