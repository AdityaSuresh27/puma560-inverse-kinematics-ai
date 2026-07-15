#!/usr/bin/env python3
"""
Export direct full-ML IK model to ONNX.

The exported graph includes:
- input normalization using checkpoint mean/std
- model forward pass
- sin/cos pair normalization
- angle decoding to [J1..J6] in degrees
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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


def infer_arch_from_state_dict(state_dict: dict) -> tuple[int, int, float]:
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


class DeployWrapper(nn.Module):
    def __init__(self, model: nn.Module, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("x_mean", torch.tensor(x_mean.astype(np.float32)))
        self.register_buffer("x_std", torch.tensor(x_std.astype(np.float32)))

    def forward(self, pose12: torch.Tensor) -> torch.Tensor:
        x = (pose12 - self.x_mean) / self.x_std
        raw = self.model(x)
        pairs = raw.view(raw.shape[0], 6, 2)
        norms = torch.linalg.norm(pairs, dim=-1, keepdim=True).clamp_min(1e-6)
        sc = (pairs / norms).view(raw.shape[0], 12)

        out = torch.zeros(sc.shape[0], 6, dtype=sc.dtype, device=sc.device)
        for j in range(6):
            out[:, j] = torch.atan2(sc[:, 2 * j], sc[:, 2 * j + 1]) * (180.0 / torch.pi)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export direct 6-DOF IK model to ONNX")
    p.add_argument("--checkpoint", type=str, default=str(Path(__file__).resolve().parent / "checkpoints" / "direct6_final.pt"))
    p.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "checkpoints" / "direct6_inference.onnx"))
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--batch", type=int, default=1)
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

    ckpt_path = resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("hidden", 0))
    blocks = int(ckpt.get("blocks", 0))
    dropout = float(ckpt.get("dropout", 0.04))
    if hidden <= 0 or blocks <= 0:
        hidden, blocks, dropout = infer_arch_from_state_dict(ckpt["model_state"])

    model = DirectIKNet(hidden=hidden, n_blocks=blocks, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapper = DeployWrapper(model, ckpt["x_mean"], ckpt["x_std"])
    wrapper.eval()

    dummy = torch.zeros(args.batch, 12, dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["pose12"],
        output_names=["joint_deg"],
        dynamic_axes={"pose12": {0: "batch"}, "joint_deg": {0: "batch"}},
    )

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Saved ONNX model: {out_path}")


if __name__ == "__main__":
    main()
