"""
scripts/run_inference.py — Inference pipeline for PUMA 560 cINN IK solver.

Usage:
    # Single query (pose as 9 space-separated floats):
    python scripts/run_inference.py --checkpoint output/checkpoints/best_model.pt \\
        --pose "0.5 0.0 0.4 1 0 0 0 1 0"

    # Batch query from CSV:
    python scripts/run_inference.py --checkpoint output/checkpoints/best_model.pt \\
        --csv_input poses.csv

    # Interactive mode:
    python scripts/run_inference.py --checkpoint output/checkpoints/best_model.pt \\
        --interactive

Pipeline per query:
  1. [Optional] Mode selector predicts likely modes → reduces cINN queries
  2. For each active mode, sample N_LATENT_SAMPLES from cINN
  3. FK-filter: keep only solutions with ||FK(θ̂) − x_target|| < threshold
  4. Score surviving solutions by weighted objective:
       score = w1 * min_joint_displacement + w2 * (-manipulability) + w3 * (-joint_margin)
  5. Return ranked list of solutions
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    N_MODES, POSE_DIM, JOINT_DIM, COND_DIM, JOINT_LIMITS, MODE_NAMES,
    INF_FK_POS_TOL, INF_FK_ROT_TOL, N_LATENT_SAMPLES,
    W_JOINT_DISP, W_MANIPULABILITY, W_JOINT_MARGIN,
    INF_ENABLE_REFINEMENT, INF_MAX_REFINE_CANDIDATES,
    INF_REFINE_MAX_NFEV, INF_REFINE_ROT_WEIGHT, INF_REFINE_BOUND_MARGIN,
    INF_TEMPERATURE,
)

# INF_TEMPERATURE_2 is optional (added in v3 for hybrid sampling)
try:
    from config import INF_TEMPERATURE_2
except ImportError:
    INF_TEMPERATURE_2 = None
from models.cinn import PumacINN, build_condition, load_checkpoint
from models.mode_selector import ModeSelectorMLP
from utils.fk_numpy import (
    fk_numpy, T_to_pose_vector, validate_ik_solution,
    manipulability, joint_limit_margin,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)


# ── Normalisation helper (loaded from checkpoint or stats file) ───────────────

class Normaliser:
    """Holds normalisation statistics and applies/reverses them."""

    def __init__(self, norm_path: Optional[Path] = None):
        if norm_path and Path(norm_path).exists():
            data = np.load(norm_path)
            self.pose_mean  = data['pose_mean'].astype(np.float32)
            self.pose_std   = data['pose_std'].astype(np.float32)
            self.joint_mean = data['joint_mean'].astype(np.float32)
            self.joint_std  = data['joint_std'].astype(np.float32)
        else:
            # Identity normalisation (no-op) — fallback
            logger.warning("Normalisation stats not found — using identity (no normalisation).")
            self.pose_mean  = np.zeros(POSE_DIM,  dtype=np.float32)
            self.pose_std   = np.ones(POSE_DIM,   dtype=np.float32)
            self.joint_mean = np.zeros(JOINT_DIM, dtype=np.float32)
            self.joint_std  = np.ones(JOINT_DIM,  dtype=np.float32)

    def norm_pose(self, p: np.ndarray) -> np.ndarray:
        return ((p - self.pose_mean) / self.pose_std).astype(np.float32)

    def norm_joints(self, q: np.ndarray) -> np.ndarray:
        return ((q - self.joint_mean) / self.joint_std).astype(np.float32)

    def denorm_joints(self, q_norm: np.ndarray) -> np.ndarray:
        return (q_norm * self.joint_std + self.joint_mean).astype(np.float64)


# ── IK Solver ────────────────────────────────────────────────────────────────

class PumaIKSolver:
    """Full IK solver wrapping cINN + FK filter + solution ranking.

    Args:
        model_path   : path to trained cINN checkpoint (.pt)
        norm_path    : path to normalisation_stats.npz
        selector_path: path to mode selector checkpoint (optional)
        device       : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model_path:    str,
        norm_path:     Optional[str] = None,
        selector_path: Optional[str] = None,
        device:        str = 'cpu',
        enable_refinement: bool = INF_ENABLE_REFINEMENT,
        max_refine_candidates: int = INF_MAX_REFINE_CANDIDATES,
        refine_max_nfev: int = INF_REFINE_MAX_NFEV,
        refine_rot_weight: float = INF_REFINE_ROT_WEIGHT,
        refine_bound_margin: float = INF_REFINE_BOUND_MARGIN,
    ):
        self.device = device
        self.enable_refinement = enable_refinement
        self.max_refine_candidates = max_refine_candidates
        self.refine_max_nfev = refine_max_nfev
        self.refine_rot_weight = refine_rot_weight
        self.refine_bound_margin = refine_bound_margin
        self._least_squares = None
        self._least_squares_unavailable = False
        self._last_pose_vec: Optional[np.ndarray] = None
        self._last_solutions: Optional[list[dict]] = None

        # ── Load cINN ────────────────────────────────────────────────────
        logger.info(f"Loading cINN from {model_path} ...")
        self.model, self.ckpt = load_checkpoint(model_path, device=device)
        self.model.eval()
        logger.info(f"  Loaded epoch {self.ckpt.get('epoch', '?')}, "
                    f"val_loss={self.ckpt.get('val_loss', '?'):.5f}")

        # ── Load normalisation stats ──────────────────────────────────────
        self.norm = Normaliser(norm_path)

        # ── Load mode selector (optional) ─────────────────────────────────
        self.selector: Optional[ModeSelectorMLP] = None
        if selector_path and Path(selector_path).exists():
            logger.info(f"Loading mode selector from {selector_path} ...")
            sel_ckpt = torch.load(selector_path, map_location=device)
            self.selector = ModeSelectorMLP().to(device)
            self.selector.load_state_dict(sel_ckpt['model_state'])
            self.selector.eval()

    # ── Core solve method ─────────────────────────────────────────────────

    def solve(
        self,
        pose_vec: np.ndarray,
        q_current: Optional[np.ndarray] = None,
        n_samples: int = N_LATENT_SAMPLES,
        pos_tol: float = INF_FK_POS_TOL,
        rot_tol: float = INF_FK_ROT_TOL,
        selector_threshold: float = 0.3,
        weights: Optional[tuple] = None,
        temperature: float = INF_TEMPERATURE,  # 0.8: tighter than Gaussian tails
        enable_refinement: Optional[bool] = None,
        max_refine_candidates: Optional[int] = None,
        refine_max_nfev: Optional[int] = None,
        refine_rot_weight: Optional[float] = None,
        refine_bound_margin: Optional[float] = None,
    ) -> list[dict]:
        """Solve IK for a target end-effector pose.

        Args:
            pose_vec            : (9,) target pose [x,y,z, R_col0, R_col1]
            q_current           : (6,) current joint angles (for Δθ scoring)
            n_samples           : latent samples per mode
            pos_tol             : FK position filter tolerance (m)
            rot_tol             : FK rotation filter tolerance (rad)
            selector_threshold  : mode selector probability threshold
            weights             : (w_disp, w_manip, w_margin) objective weights
            temperature         : latent sampling temperature

        Returns:
            solutions : list of dicts, sorted best-first, each containing:
                {
                  'joints'       : (6,) joint angles (rad),
                  'mode'         : int (0-7),
                  'mode_name'    : str,
                  'pos_error'    : float (m),
                  'rot_error'    : float (rad),
                  'manipulability': float,
                  'joint_margin' : float,
                  'joint_disp'   : float (or None if q_current not given),
                  'score'        : float (lower = better),
                }
        """
        if weights is None:
            weights = (W_JOINT_DISP, W_MANIPULABILITY, W_JOINT_MARGIN)
        w_disp, w_manip, w_margin = weights

        if enable_refinement is None:
            enable_refinement = self.enable_refinement
        if max_refine_candidates is None:
            max_refine_candidates = self.max_refine_candidates
        if refine_max_nfev is None:
            refine_max_nfev = self.refine_max_nfev
        if refine_rot_weight is None:
            refine_rot_weight = self.refine_rot_weight
        if refine_bound_margin is None:
            refine_bound_margin = self.refine_bound_margin

        T_target = self._pose_to_T(pose_vec)

        # ── Normalise pose ────────────────────────────────────────────────
        pose_norm = self.norm.norm_pose(pose_vec.astype(np.float32))
        pose_t    = torch.tensor(pose_norm, dtype=torch.float32, device=self.device)

        # ── Determine which modes to query ────────────────────────────────
        if self.selector is not None:
            active_modes = self.selector.predict_active_modes(
                pose_t, threshold=selector_threshold, min_modes=2
            )
            logger.debug(f"Mode selector active modes: {active_modes}")
        else:
            active_modes = list(range(N_MODES))   # evaluate all 8

        # ── Query cINN for each active mode ───────────────────────────────
        solutions = []
        seen = set()
        refined_count = 0
        with torch.no_grad():
            for mode_int in active_modes:
                mode_oh = torch.zeros(N_MODES, dtype=torch.float32, device=self.device)
                mode_oh[mode_int] = 1.0

                condition = build_condition(pose_t, mode_oh)   # (1, 17)

                # Hybrid two-temperature sampling: sample at both temps for
                # better mode coverage (tight centre + wider tails).
                # Falls back to single temperature when INF_TEMPERATURE_2 is None.
                temps_to_sample = [temperature]
                if INF_TEMPERATURE_2 is not None and INF_TEMPERATURE_2 != temperature:
                    temps_to_sample = [temperature, INF_TEMPERATURE_2]

                all_joints_norm = []
                for t_val in temps_to_sample:
                    batch = self.model.sample(
                        condition.squeeze(0), n_samples=n_samples, temperature=t_val
                    )  # (n_samples, 6)
                    all_joints_norm.append(batch)

                joints_norm_samples = torch.cat(all_joints_norm, dim=0)  # (n_samples * n_temps, 6)
                joints_np = joints_norm_samples.cpu().numpy()

                total_samples = len(joints_np)
                for s in range(total_samples):
                    q_denorm = self.norm.denorm_joints(joints_np[s])  # (6,)

                    # FK filter
                    is_valid, pos_err, rot_err = validate_ik_solution(
                        q_denorm, T_target,
                        pos_tol=pos_tol, rot_tol=rot_tol,
                    )

                    if (not is_valid) and enable_refinement and refined_count < max_refine_candidates:
                        q_refined = self._refine_candidate(
                            q_denorm,
                            pose_vec,
                            rot_weight=refine_rot_weight,
                            bound_margin=refine_bound_margin,
                            max_nfev=refine_max_nfev,
                        )
                        if q_refined is not None:
                            refined_count += 1
                            is_valid, pos_err, rot_err = validate_ik_solution(
                                q_refined, T_target,
                                pos_tol=pos_tol, rot_tol=rot_tol,
                            )
                            if is_valid:
                                q_denorm = q_refined

                    if not is_valid:
                        continue

                    # Remove near-duplicate solutions from repeated latent samples.
                    sol_key = (mode_int, tuple(np.round(q_denorm, 4)))
                    if sol_key in seen:
                        continue
                    seen.add(sol_key)

                    # Compute scoring metrics
                    manip  = manipulability(q_denorm)
                    margin = joint_limit_margin(q_denorm)
                    disp   = (float(np.linalg.norm(q_denorm - q_current))
                              if q_current is not None else 0.0)

                    # Score: lower = better
                    # Normalise each term to ~[0, 1] range heuristically
                    score = (
                        w_disp   * disp
                      + w_manip  * (1.0 / (manip + 1e-6))   # reciprocal: low manip = high penalty
                      + w_margin * (1.0 - 2.0 * margin)     # 0 at centre, 1 at limit
                    )

                    solutions.append({
                        'joints'        : q_denorm,
                        'mode'          : mode_int,
                        'mode_name'     : MODE_NAMES[mode_int],
                        'pos_error'     : pos_err,
                        'rot_error'     : rot_err,
                        'manipulability': manip,
                        'joint_margin'  : margin,
                        'joint_disp'    : disp,
                        'score'         : score,
                    })

        # If mode selector was used but returned no solutions, retry all modes
        if len(solutions) == 0 and len(active_modes) < N_MODES:
            logger.debug("Mode selector fallback: trying all 8 modes")
            return self.solve(
                pose_vec, q_current, n_samples, pos_tol, rot_tol,
                selector_threshold=0.0,   # force all modes
                weights=weights, temperature=temperature * 1.2,  # slightly wider
            )

        # Sort by score (lower = better)
        solutions.sort(key=lambda x: x['score'])
        self._last_pose_vec = np.array(pose_vec, dtype=np.float64, copy=True)
        self._last_solutions = solutions
        return solutions

    def _get_least_squares(self):
        if self._least_squares_unavailable:
            return None
        if self._least_squares is not None:
            return self._least_squares
        try:
            from scipy.optimize import least_squares
            self._least_squares = least_squares
            return least_squares
        except Exception:
            self._least_squares_unavailable = True
            logger.warning("SciPy not available; disabling refinement stage.")
            return None

    def _refine_candidate(
        self,
        q_init: np.ndarray,
        pose_target_vec: np.ndarray,
        rot_weight: float,
        bound_margin: float,
        max_nfev: int,
    ) -> Optional[np.ndarray]:
        least_squares = self._get_least_squares()
        if least_squares is None:
            return None

        target = pose_target_vec.astype(np.float64)

        lo = JOINT_LIMITS[0] + bound_margin
        hi = JOINT_LIMITS[1] - bound_margin
        x0 = np.clip(np.asarray(q_init, dtype=np.float64), lo, hi)

        def residual_fn(q: np.ndarray) -> np.ndarray:
            pv = T_to_pose_vector(fk_numpy(q))
            pos_res = pv[:3] - target[:3]
            rot_res = (pv[3:] - target[3:]) * rot_weight
            return np.concatenate([pos_res, rot_res])

        try:
            res = least_squares(
                residual_fn,
                x0,
                bounds=(lo, hi),
                method='trf',
                max_nfev=max_nfev,
            )
        except Exception:
            return None

        if not np.all(np.isfinite(res.x)):
            return None
        return res.x.astype(np.float64)

    def _pose_to_T(self, pose_vec: np.ndarray) -> np.ndarray:
        """Reconstruct 4x4 transform from 9-dim pose vector for FK validation."""
        from utils.fk_numpy import pose_vector_to_T
        return pose_vector_to_T(pose_vec)

    def best_solution(
        self,
        pose_vec: np.ndarray,
        **kwargs,
    ) -> Optional[dict]:
        """Return the single best IK solution, or None if none found."""
        # Reuse immediately previous solve() result for identical pose when possible.
        if (
            len(kwargs) == 0
            and self._last_pose_vec is not None
            and np.allclose(np.asarray(pose_vec, dtype=np.float64), self._last_pose_vec)
        ):
            return self._last_solutions[0] if self._last_solutions else None

        solutions = self.solve(pose_vec, **kwargs)
        return solutions[0] if solutions else None


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_solution(sol: dict, rank: int = 1) -> None:
    """Pretty-print a single IK solution."""
    q_deg = np.degrees(sol['joints'])
    print(f"\n  Solution #{rank}  —  Mode {sol['mode']} ({sol['mode_name']})")
    print(f"    Joint angles (rad): {np.round(sol['joints'], 4)}")
    print(f"    Joint angles (deg): {np.round(q_deg, 2)}")
    print(f"    FK pos error : {sol['pos_error']:.2e} m")
    print(f"    FK rot error : {sol['rot_error']:.2e} rad")
    print(f"    Manipulability : {sol['manipulability']:.4f}")
    print(f"    Joint margin   : {sol['joint_margin']:.4f}")
    if sol['joint_disp'] is not None:
        print(f"    Joint disp     : {sol['joint_disp']:.4f} rad")
    print(f"    Score (lower=better) : {sol['score']:.5f}")


def main():
    parser = argparse.ArgumentParser(
        description='PUMA 560 cINN IK solver — inference pipeline'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--norm_stats', type=str, default=None,
                        help='Path to normalisation_stats.npz')
    parser.add_argument('--selector', type=str, default=None,
                        help='Path to mode selector checkpoint (optional)')
    parser.add_argument('--pose', type=str, default=None,
                        help='9 space-separated floats: x y z r11 r21 r31 r12 r22 r32')
    parser.add_argument('--q_current', type=str, default=None,
                        help='Current joint angles (6 floats, radians)')
    parser.add_argument('--n_solutions', type=int, default=8,
                        help='Max solutions to show (default: 8)')
    parser.add_argument('--n_samples', type=int, default=N_LATENT_SAMPLES,
                        help='Latent samples per mode')
    parser.add_argument('--pos_tol', type=float, default=INF_FK_POS_TOL,
                        help='FK position tolerance in metres')
    parser.add_argument('--rot_tol', type=float, default=INF_FK_ROT_TOL,
                        help='FK rotation tolerance in radians')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Latent sampling temperature')
    parser.add_argument('--selector_threshold', type=float, default=0.3,
                        help='Mode selector probability threshold')
    parser.add_argument('--disable_refinement', action='store_true',
                        help='Disable least-squares post-refinement')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive query loop')
    args = parser.parse_args()

    # ── Load solver ───────────────────────────────────────────────────────
    solver = PumaIKSolver(
        model_path=args.checkpoint,
        norm_path=args.norm_stats,
        selector_path=args.selector,
        device=args.device,
        enable_refinement=not args.disable_refinement,
    )

    def run_query(pose_str: str, q_str: Optional[str] = None):
        pose_vec = np.array([float(x) for x in pose_str.split()], dtype=np.float64)
        assert len(pose_vec) == 9, f"Expected 9 pose values, got {len(pose_vec)}"

        q_current = None
        if q_str:
            q_current = np.array([float(x) for x in q_str.split()], dtype=np.float64)
            assert len(q_current) == 6

        print(f"\nTarget pose (xyz + 2 rotation columns):")
        print(f"  Position : {np.round(pose_vec[:3], 4)} m")
        print(f"  Rot col0 : {np.round(pose_vec[3:6], 4)}")
        print(f"  Rot col1 : {np.round(pose_vec[6:9], 4)}")

        solutions = solver.solve(
            pose_vec,
            q_current=q_current,
            n_samples=args.n_samples,
            pos_tol=args.pos_tol,
            rot_tol=args.rot_tol,
            temperature=args.temperature,
            selector_threshold=args.selector_threshold,
        )

        if not solutions:
            print("\n  No valid IK solutions found for this pose.")
            print("    (The pose may be outside the PUMA 560 workspace.)")
            return

        print(f"\n  Found {len(solutions)} valid solution(s):")
        for i, sol in enumerate(solutions[:args.n_solutions]):
            print_solution(sol, rank=i+1)

    # ── Run mode ──────────────────────────────────────────────────────────
    if args.interactive:
        print("\nPUMA 560 cINN IK Solver — interactive mode")
        print("Enter 9 pose values: x y z r11 r21 r31 r12 r22 r32")
        print("(or 'quit' to exit)\n")
        while True:
            try:
                pose_str = input("Pose > ").strip()
                if pose_str.lower() in ('quit', 'exit', 'q'):
                    break
                q_str = input("Current joints (optional, press Enter to skip) > ").strip()
                run_query(pose_str, q_str if q_str else None)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  Error: {e}")

    elif args.pose:
        run_query(args.pose, args.q_current)

    else:
        # Demo: solve for the PUMA 560 nominal pose using FK
        from utils.fk_numpy import fk_numpy, T_to_pose_vector
        try:
            import roboticstoolbox as rtb
            robot = rtb.models.DH.Puma560()
            q_nominal = np.array(robot.qn)
        except Exception:
            # Fallback if roboticstoolbox not available (e.g. NumPy 2.x incompatibility)
            q_nominal = np.zeros(6)  # all-zero pose (PUMA 560 home)
        T_nominal = fk_numpy(q_nominal)
        pose_vec  = T_to_pose_vector(T_nominal)
        print(f"\nDemo: solving IK for PUMA 560 nominal pose")
        print(f"  Ground-truth joints (deg): {np.round(np.degrees(q_nominal), 2)}")
        run_query(' '.join(str(x) for x in pose_vec), ' '.join(str(x) for x in q_nominal))


if __name__ == '__main__':
    main()
