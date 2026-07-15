# -*- coding: utf-8 -*-
"""
PUMA 560 IK Solver - Interactive Terminal Program
Run: python ik_solver.py

Input options:
  1. Position (x, y, z) + rotation matrix (asks row by row)
  2. Raw 9-number pose vector
  3. Joint angles → compute pose via FK, then solve IK (round-trip test)
"""

from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODEL_PATH    = ROOT / "checkpoints" / "best_model.pt"
SELECTOR_PATH = ROOT / "checkpoints" / "mode_selector.pt"
NORM_PATH     = ROOT / "normalisation_stats.npz"

# ── Helpers ───────────────────────────────────────────────────────────────────

def ask_floats(prompt: str, n: int) -> np.ndarray:
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit
        parts = raw.replace(",", " ").split()
        if len(parts) != n:
            print(f"  [!] Need exactly {n} numbers, got {len(parts)}. Try again.")
            continue
        try:
            return np.array([float(p) for p in parts], dtype=np.float64)
        except ValueError:
            print("  ✗ Invalid number. Try again.")


def ask_rotation_matrix() -> np.ndarray:
    """Ask for a 3×3 rotation matrix row by row and validate it."""
    print("  Enter each row as 3 space-separated numbers.")
    while True:
        r0 = ask_floats("    Row 0 (r00 r01 r02): ", 3)
        r1 = ask_floats("    Row 1 (r10 r11 r12): ", 3)
        r2 = ask_floats("    Row 2 (r20 r21 r22): ", 3)
        R = np.stack([r0, r1, r2])          # 3×3
        det = np.linalg.det(R)
        orth_err = np.max(np.abs(R @ R.T - np.eye(3)))
        if abs(det - 1.0) > 0.05 or orth_err > 0.05:
            print(f"  ⚠  Matrix doesn't look like a rotation (det={det:.3f}, orth_err={orth_err:.4f}).")
            print("     Common valid matrices:")
            print("       Identity:  row0=[1 0 0]  row1=[0 1 0]  row2=[0 0 1]")
            print("       90° about Z: row0=[0 -1 0]  row1=[1 0 0]  row2=[0 0 1]")
            retry = input("     Retry? [Y/n]: ").strip().lower()
            if retry == "n":
                break
        else:
            break
    return R


def rotation_to_pose_vec(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Build 9-dim pose vector [x,y,z, R_col0, R_col1]."""
    col0 = R[:, 0]
    col1 = R[:, 1]
    return np.concatenate([pos, col0, col1])


def print_separator(char="-", width=60):
    print(char * width)


def print_solution(sol: dict, rank: int) -> None:
    q_rad = sol["joints"]
    q_deg = np.degrees(q_rad)
    pos_mm = sol["pos_error"] * 1000.0
    rot_deg = np.degrees(sol["rot_error"])

    print(f"\n  #{rank}  Mode {sol['mode']} — {sol['mode_name']}")
    print(f"      Joint angles (rad): {np.round(q_rad, 5).tolist()}")
    print(f"      Joint angles (deg): {np.round(q_deg, 3).tolist()}")
    print(f"      FK position error : {pos_mm:.6f} mm")
    print(f"      FK rotation error : {rot_deg:.6f} deg")
    print(f"      Manipulability    : {sol['manipulability']:.4f}")
    print(f"      Joint limit margin: {sol['joint_margin']:.4f} rad")
    if sol.get("joint_disp") and sol["joint_disp"] > 0:
        print(f"      Joint displacement: {sol['joint_disp']:.4f} rad from current")
    print(f"      Score (lower=better): {sol['score']:.5f}")


# ── Main program ──────────────────────────────────────────────────────────────

def main() -> None:
    print_separator("=")
    print("  PUMA 560 cINN Inverse Kinematics Solver")
    print("  12 coupling blocks x 384 hidden | Epoch 199 | Val loss -23.02")
    print("  Pipeline: Mode selector -> cINN sampling -> FK filter -> Refinement -> Ranking")
    print_separator("=")

    # ── Load model ────────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"\n✗ Model not found: {MODEL_PATH}")
        print("  Make sure best_model.pt is in Final_v3/checkpoints/")
        return
    if not NORM_PATH.exists():
        print(f"\n✗ Normalisation stats not found: {NORM_PATH}")
        return

    print("\nLoading model... ", end="", flush=True)
    from scripts.run_inference import PumaIKSolver
    solver = PumaIKSolver(
        model_path=str(MODEL_PATH),
        norm_path=str(NORM_PATH),
        selector_path=str(SELECTOR_PATH) if SELECTOR_PATH.exists() else None,
        device="cpu",
        enable_refinement=True,
    )
    print("done.\n")

    from utils.fk_numpy import fk_numpy, T_to_pose_vector

    print("Type 'q' or 'quit' at any prompt to exit.\n")

    # ── Query loop ────────────────────────────────────────────────────────────
    while True:
        print_separator()
        print("  How do you want to specify the target pose?")
        print("  [1] Position (x,y,z) + rotation matrix (3×3, enter row by row)")
        print("  [2] Raw 9-number pose vector [x y z r11 r21 r31 r12 r22 r32]")
        print("  [3] Joint angles → compute FK pose → solve IK (round-trip test)")
        print("  [q] Quit")
        print_separator()

        choice = input("  Choice: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\nBye.")
            break

        pose_vec: np.ndarray | None = None
        q_ground_truth: np.ndarray | None = None

        # ── Input mode 1: position + rotation matrix ─────────────────────────
        if choice == "1":
            print("\nEnter position (metres):")
            pos = ask_floats("  x y z: ", 3)
            print("\nEnter rotation matrix:")
            print("  (For identity/no rotation: row0=[1 0 0]  row1=[0 1 0]  row2=[0 0 1])")
            R = ask_rotation_matrix()
            pose_vec = rotation_to_pose_vec(pos, R)

        # ── Input mode 2: raw 9-vector ────────────────────────────────────────
        elif choice == "2":
            print("\nEnter 9 numbers: x y z r11 r21 r31 r12 r22 r32")
            print("  (position in metres, then first two columns of rotation matrix)")
            print("  Example (home pose): 0.4521 0.1498 0.4327 0 1 0 -1 0 0")
            pose_vec = ask_floats("  Pose: ", 9)

        # ── Input mode 3: joint angles → FK → solve ───────────────────────────
        elif choice == "3":
            print("\nEnter 6 joint angles in DEGREES (space-separated):")
            print("  Joint limits (deg): ±160, -225 to +45, -45 to +225, ±270, ±100, ±270")
            print("  Example (zero pose):  0 0 0 0 0 0")
            q_deg = ask_floats("  Joints (deg): ", 6)
            q_rad = np.radians(q_deg)
            T = fk_numpy(q_rad)
            pose_vec = T_to_pose_vector(T)
            q_ground_truth = q_rad
            pos = pose_vec[:3]
            R_col0 = pose_vec[3:6]
            R_col1 = pose_vec[6:9]
            R_col2 = np.cross(R_col0, R_col1)
            R = np.stack([R_col0, R_col1, R_col2], axis=1)
            print(f"\n  FK result:")
            print(f"    Position : {np.round(pos*100, 4)} cm")
            print(f"    Rotation :\n{np.round(R, 4)}")
            print(f"\n  Now solving IK for this exact pose...")
        else:
            print("  Invalid choice. Enter 1, 2, 3, or q.")
            continue

        # ── Current joint angles (optional) ───────────────────────────────────
        if q_ground_truth is None:
            print("\nCurrent joint configuration for displacement scoring (optional).")
            print("  Press Enter to skip.")
            raw_q = input("  Current joints (deg, 6 values): ").strip()
            q_current = None
            if raw_q and raw_q.lower() not in ("q", "quit"):
                try:
                    q_deg_cur = np.array(
                        [float(p) for p in raw_q.replace(",", " ").split()],
                        dtype=np.float64,
                    )
                    if len(q_deg_cur) == 6:
                        q_current = np.radians(q_deg_cur)
                    else:
                        print("  ⚠  Skipping (need exactly 6 values).")
                except ValueError:
                    print("  ⚠  Could not parse — skipping.")
        else:
            q_current = q_ground_truth   # use ground truth for displacement scoring

        # ── Solve ─────────────────────────────────────────────────────────────
        print("\nSolving... ", end="", flush=True)
        import time
        t0 = time.perf_counter()
        solutions = solver.solve(pose_vec=pose_vec, q_current=q_current)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"done. ({elapsed_ms:.0f} ms)\n")

        pos_display = pose_vec[:3]
        print(f"  Target position: x={pos_display[0]:.4f}  y={pos_display[1]:.4f}  z={pos_display[2]:.4f}  (metres)")

        if q_ground_truth is not None:
            print(f"  Ground truth joints (deg): {np.round(np.degrees(q_ground_truth), 3).tolist()}")

        if not solutions:
            print("\n  ✗ No valid IK solutions found.")
            print("    Possible reasons:")
            print("    • Pose is outside the PUMA 560 workspace")
            print("    • Position too far from the base (~0.9 m max reach)")
            print("    • Orientation conflicts with joint limits")
            print("    • Try a different position or orientation.")
            continue

        print(f"\n  ✓ Found {len(solutions)} valid solution(s):")

        # Show top 5 solutions
        for i, sol in enumerate(solutions[:5]):
            print_solution(sol, rank=i + 1)

        # If round-trip test: check if any solution matches ground truth
        if q_ground_truth is not None:
            print("\n  Round-trip check:")
            best = solutions[0]
            diff_rad = np.abs(best["joints"] - q_ground_truth)
            diff_deg = np.degrees(diff_rad)
            print(f"    Ground truth (deg): {np.round(np.degrees(q_ground_truth), 3).tolist()}")
            print(f"    Best solution (deg): {np.round(np.degrees(best['joints']), 3).tolist()}")
            print(f"    Difference (deg)  : {np.round(diff_deg, 3).tolist()}")
            max_diff = diff_deg.max()
            if max_diff < 1.0:
                print(f"    [PASS] Excellent match (max diff {max_diff:.3f} deg)")
            elif max_diff < 5.0:
                print(f"    [GOOD] Good match (max diff {max_diff:.3f} deg) - different valid branch possible")
            else:
                print(f"    [INFO] Different branch selected (max diff {max_diff:.1f} deg) - FK error is what matters")
                print(f"      FK position error is {best['pos_error']*1000:.6f} mm - still valid")

        print()


if __name__ == "__main__":
    main()
