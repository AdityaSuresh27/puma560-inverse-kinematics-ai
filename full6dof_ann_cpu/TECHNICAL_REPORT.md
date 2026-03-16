# full6dof_ann_cpu Technical Report

## 1. Scope
This folder implements a CPU-only inverse kinematics pipeline for PUMA 560 where:
- J1-J3 are learned by a neural network from wrist-center Cartesian coordinates.
- J4-J6 are solved analytically from rotation decomposition.

The trainer is `train_ann_full6_cpu.py` and the evaluator/plotter is `visualize_ann_full6_cpu.py`.

## 2. Data Contract and Preprocessing
Source dataset: `../puma560_dataset.csv`

Column contract used by code:
- Input pose (12): `[nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz]`
- Target joints (6): `[theta1, theta2, theta3, theta4, theta5, theta6]` in degrees

Preprocessing sequence:
1. Load CSV and validate at least 18 columns.
2. Compute wrist center from pose:
   - `P5 = [Px - D6*ax, Py - D6*ay, Pz - D6*az]`
   - `D6 = 56.5` from DH constants.
3. Deterministic split with seed:
   - train: 75%
   - val: 10%
   - test: 15%
4. Standardize wrist-center input using train statistics only:
   - `Xn = (X - mean_train) / std_train`
5. Convert only J1-J3 target angles to sin/cos pairs:
   - output target dimension becomes 6: `[sin(t1), cos(t1), sin(t2), cos(t2), sin(t3), cos(t3)]`

## 3. Model Architecture (FullIKANN)
Runtime model build in training uses CLI defaults:
- `n_in = 3`
- `hidden = 384`
- `n_blocks = 6`
- `dropout = 0.05`
- `n_out = 6`

Architecture graph:
1. Stem:
   - `Linear(3 -> 384)`
   - `LayerNorm(384)`
   - `GELU`
2. Residual trunk: 6 identical `ResBlock(384)` modules
   - Block internal path:
     - `Linear(384 -> 384)`
     - `LayerNorm(384)`
     - `GELU`
     - `Dropout(0.05)`
     - `Linear(384 -> 384)`
     - `LayerNorm(384)`
   - Residual combine and activation:
     - `output = GELU(input + block(input))`
3. Head:
   - `Linear(384 -> 6)` for J1-J3 sin/cos prediction.

Layer counts with training defaults (`hidden=384`, `blocks=6`):
- Trainable linear layers: 14 (1 stem + 12 in residual blocks + 1 head)
- LayerNorm layers: 13 (1 stem + 12 in residual blocks)
- Trainable parameters: 1,787,910

Note on class constructor defaults:
- Class-level defaults are `hidden=512`, `n_blocks=8`, which would give 4,225,030 parameters.
- Actual training defaults come from CLI (`hidden=384`, `blocks=6`).

## 4. Objective Function and Physics Coupling
The loss module is `FullIKLoss` and combines three terms.

1. Sin/cos regression loss:
- Normalize each predicted pair `(sin, cos)` to unit norm.
- `L_sc = SmoothL1(pred_sc, target_sc, beta=0.05)`

2. Wrist-center physics loss:
- Decode J1-J3 from predicted sin/cos via `atan2`.
- Forward-kinematics `T03` from DH chain for first three joints.
- Predict wrist center using FK geometry:
  - `p5_pred = T03[:3,3] + D4 * T03[:3,2]`
  - `D4 = 431.8`
- `L_wc = MSE(p5_pred / R_WORKSPACE, p5_true / R_WORKSPACE)`
- `R_WORKSPACE = 900`

3. Unit-circle regularization:
- For each joint pair: `(sin^2 + cos^2 - 1)^2`
- `L_circ = average over J1..J3`

Total:
- `L_total = w_sc*L_sc + w_pos*L_wc + w_circ*L_circ`
- Default weights: `w_sc=1.0`, `w_pos=2.0`, `w_circ=0.02`

## 5. Training Process (Step-by-Step)
1. Seed Python, NumPy, and Torch.
2. Force CPU device.
3. Build dataloaders from normalized wrist-center features and shoulder sin/cos targets.
4. Build `FullIKANN`.
5. Optimizer: `AdamW(lr=1e-3, weight_decay=5e-5)`.
6. Scheduler: `CosineAnnealingWarmRestarts(T_0=200, T_mult=2, eta_min=lr*1e-3)`.
7. Per mini-batch:
   - Forward model.
   - Compute composite loss.
   - Skip batch if loss is non-finite.
   - Backprop + gradient clipping (`clip_norm=1.0`).
8. End-of-epoch validation on the same objective terms.
9. Monitor value is `val_sc` (shoulder sin/cos term).
10. Save checkpoints atomically:
   - `ann6_last.pt` every save interval
   - `ann6_best.pt` when monitor improves
11. Early stop after `min_epochs` and `patience` without monitor improvement.
12. Reload best checkpoint for final test evaluation.
13. Save:
   - `checkpoints/ann6_final.pt`
   - `ann6_eval_results.npz`
   - plots (if matplotlib available)

Safety controls implemented:
- Resume support.
- Atomic checkpoint writes with retry and recovery fallback.
- Non-finite loss skip.
- KeyboardInterrupt emergency checkpoint save.

## 6. Inference and Full 6-DOF Reconstruction
At evaluation time:
1. ANN predicts J1-J3 as sin/cos.
2. Convert to angles with `atan2(sin, cos)`.
3. Build target `T06` from the 12 pose columns.
4. Compute `T36 = inv(T03) @ T06`.
5. Solve wrist analytically:
   - `theta5 = atan2(sin5, T36[2,2])`
   - `theta4 = atan2(T36[1,2], T36[0,2])`
   - `theta6 = atan2(T36[2,1], -T36[2,0])`
6. Handle near-singular wrist when `sin5` is near zero.
7. Build full prediction `[J1..J6]`.
8. Compute wrapped angular error:
   - `err = (pred - true + 180) % 360 - 180`

Metrics computed:
- Per-joint MAE and RMSE
- Average MAE and RMSE
- Joint-wise and all-joint threshold accuracies for `{0.1, 0.25, 0.5, 1.0, 2.0}` degrees

## 7. Artifacts Produced in This Folder
- Training script: `train_ann_full6_cpu.py`
- Evaluation script: `visualize_ann_full6_cpu.py`
- Checkpoints: `checkpoints/ann6_best.pt`, `checkpoints/ann6_last.pt`, `checkpoints/ann6_final.pt`
- Metrics archive: `ann6_eval_results.npz`, `ann6_eval_results_visualized.npz`
- Plots: `training_history_full6.png`, `joint_metrics_full6.png`, `error_histograms_full6.png`, `pred_vs_true_full6.png`
