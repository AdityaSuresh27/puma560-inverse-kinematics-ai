# Review 3A - PUMA 560 Full 6-DOF ANN Pipeline (Code-Verified)

**Date:** March 16, 2026
**Task:** Verify whether the folder implements pure neural prediction or hybrid analytical IK, then document the complete technical process and model details.
**Scope boundary:** Only artifacts and code under `full6dof_ann_cpu` (and its own generated checkpoints/results) are used; parent-folder models are excluded.

---

## 1. Verification Outcome

This folder does **not** perform end-to-end 6-joint prediction by ANN only.

- The ANN learns only shoulder joints J1-J3 in sin/cos form (6 outputs total).
- J4-J6 are reconstructed analytically from $T_{36} = T_{03}^{-1} T_{06}$ during evaluation.
- Analytical geometry is used in both training loss (wrist-center FK consistency) and final reconstruction.

Verified implementation points:

- Wrist-center feature extraction: $P_5 = [P_x - d_6 a_x,\; P_y - d_6 a_y,\; P_z - d_6 a_z]$.
- Shoulder-only target encoding: $[\sin J_1, \cos J_1, \sin J_2, \cos J_2, \sin J_3, \cos J_3]$.
- Wrist solve uses closed-form angle extraction with singular-case branch for $\sin(J_5) \approx 0$.

---

## 2. Problem Setup

| Item | Value |
|------|-------|
| Dataset file | `../puma560_dataset.csv` |
| Input pose columns | `[nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz]` |
| Ground-truth joints | `[J1, J2, J3, J4, J5, J6]` in degrees |
| Learned input to model | Wrist center $P_5 \in \mathbb{R}^3$ |
| Learned output | 6 values = sin/cos for J1-J3 |
| Split | Train 75%, Val 10%, Test 15% |
| Seed | 42 |

---

## 3. ANN Architecture (FullIKANN)

### 3.1 Checkpoint-verified training configuration

| Hyperparameter | Value |
|---------------|-------|
| Input dim | 3 |
| Hidden width | 384 |
| Residual blocks | 6 |
| Dropout | 0.05 |
| Output dim | 6 |
| Trainable parameters | 1,787,910 |

### 3.2 Layer structure

1. Stem: `Linear(3 -> 384)` + `LayerNorm` + `GELU`
2. Trunk: 6 `ResBlock(384)` modules
3. Head: `Linear(384 -> 6)`

Each residual block contains:

- `Linear(384 -> 384)`
- `LayerNorm(384)`
- `GELU`
- `Dropout(0.05)`
- `Linear(384 -> 384)`
- `LayerNorm(384)`
- Residual add + `GELU`

Layer counts for the trained ANN:

- Linear layers: 14
- LayerNorm layers: 13

---

## 4. Loss Formulation and Analytical Coupling

The objective is:

$$
L = w_{sc} L_{sc} + w_{wc} L_{wc} + w_{circ} L_{circ}
$$

with defaults $w_{sc}=1.0$, $w_{wc}=2.0$, $w_{circ}=0.02$.

1. $L_{sc}$: SmoothL1 on normalized sin/cos pairs for J1-J3.
2. $L_{wc}$: MSE wrist-center physics constraint from differentiable FK of first 3 joints:
   $p_{5,pred} = p_{03} + d_4 z_{03}$.
3. $L_{circ}$: unit-circle regularizer on each sin/cos pair.

This confirms analytical kinematics is embedded in the training objective, not only in post-processing.

---

## 5. End-to-End Process (Step-by-Step)

1. Load CSV and validate column count.
2. Compute wrist center from pose and $d_6$.
3. Split with deterministic permutation.
4. Normalize wrist-center inputs using training mean/std.
5. Encode J1-J3 to sin/cos targets.
6. Train ANN on CPU with AdamW and cosine warm restarts.
7. Apply gradient clipping and non-finite loss guard.
8. Save atomic `last` and `best` checkpoints.
9. Early-stop by validation shoulder sin/cos loss monitor.
10. Evaluate best model on test set:
    - Decode J1-J3 from sin/cos.
    - Build $T_{06}$ from pose columns.
    - Solve J4-J6 analytically from $T_{36}$.
11. Compute wrapped angle errors and aggregate metrics.
12. Save checkpoint, NPZ metrics, and plots.

---

## 6. Test Results (Saved Artifact: ann6_eval_results.npz)

### 6.1 Per-joint MAE/RMSE (degrees)

| Joint | MAE | RMSE |
|------|----:|-----:|
| J1 | 0.1603 | 0.5370 |
| J2 | 0.1319 | 0.2395 |
| J3 | 0.2227 | 0.4504 |
| J4 | 0.4304 | 3.5820 |
| J5 | 0.1323 | 0.2936 |
| J6 | 0.4277 | 3.5791 |
| **Average** | **0.2509** | **1.4469** |

### 6.2 All-joint threshold accuracy

| Criterion | Accuracy |
|----------|---------:|
| All joints <= 0.5 deg | 81.40% |
| All joints <= 1.0 deg | 90.40% |
| All joints <= 2.0 deg | 95.87% |

---

## 7. Technical Takeaway

This ANN folder is a **decoupled IK pipeline**, not a pure 6-output direct regressor.

- Learned part: J1-J3 shoulder mapping from wrist center.
- Analytical part: J4-J6 wrist extraction from rigid-body transforms.
- Therefore, analytical kinematics is actively used and is required for full 6-DOF output generation.
