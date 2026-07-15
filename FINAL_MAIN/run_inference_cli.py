"""Interactive IK solver for the trained cINN model.

Prompts for a 9D pose vector and prints ranked IK solutions.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.run_inference import PumaIKSolver

MODEL_PATH = ROOT / "checkpoints" / "best_model.pt"
SELECTOR_PATH = ROOT / "checkpoints" / "mode_selector.pt"
NORM_PATH = ROOT / "normalisation_stats.npz"


def parse_floats(text: str, count: int) -> list[float]:
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != count:
        raise ValueError(f"Expected {count} values, got {len(parts)}")
    return [float(p) for p in parts]


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Missing model: {MODEL_PATH}")
        return
    if not NORM_PATH.exists():
        print(f"Missing normalisation stats: {NORM_PATH}")
        return

    solver = PumaIKSolver(
        model_path=str(MODEL_PATH),
        norm_path=str(NORM_PATH),
        selector_path=str(SELECTOR_PATH) if SELECTOR_PATH.exists() else None,
        device="cpu",
    )

    print("Enter pose as 9 floats: x y z r11 r21 r31 r12 r22 r32")
    print("Type 'quit' to exit.")

    while True:
        raw = input("Pose> ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        try:
            pose_vals = parse_floats(raw, 9)
        except ValueError as exc:
            print(exc)
            continue

        q_current = None
        raw_q = input("Current joints (6 floats) or Enter to skip> ").strip()
        if raw_q:
            try:
                q_current = np.array(parse_floats(raw_q, 6), dtype=np.float64)
            except ValueError as exc:
                print(exc)
                continue

        pose_vec = np.array(pose_vals, dtype=np.float64)
        sols = solver.solve(pose_vec=pose_vec, q_current=q_current)

        if not sols:
            print("No valid IK solutions found.")
            continue

        print(f"Found {len(sols)} solutions. Showing top 5:")
        for idx, sol in enumerate(sols[:5], start=1):
            joints = " ".join(f"{v:+.4f}" for v in sol["joints"])
            print(
                f"#{idx} mode={sol['mode_name']} pos_err={sol['pos_error']*1000.0:.4f} mm "
                f"rot_err={sol['rot_error']*180.0/np.pi:.4f} deg score={sol['score']:.4f}\n"
                f"    joints: {joints}"
            )


if __name__ == "__main__":
    main()
