# full6dof_hybrid_qnn_cpu Technical Report

## 1. Scope
This folder implements a CPU-only hybrid residual model for PUMA 560 inverse kinematics:
- A pretrained ANN backbone predicts shoulder sin/cos (J1-J3 representation).
- A trainable quantum-inspired residual branch learns corrections.
- Wrist angles J4-J6 are still recovered analytically in evaluation.

Main scripts:
- `train_hybrid_qnn_full6_cpu.py`
- `visualize_hybrid_qnn_full6_cpu.py`

## 2. Data Pipeline
Input dataset and split protocol match the ANN baseline.

Steps:
1. Load `../puma560_dataset.csv`.
2. Parse pose columns `[nx..Pz]` and target angles `[theta1..theta6]`.
3. Compute wrist-center feature `P5` with `D6=56.5`.
4. Split deterministic train/val/test using same fractions and seed.
5. Normalize wrist-center input with train mean/std.
6. Convert only J1-J3 targets to sin/cos pairs (6 outputs).

This ensures direct comparability against ANN results on the same partition.

## 3. Hybrid Model Definition
Top-level model class: `HybridQNNIK`

Forward equation:
- `y_hat = y_backbone + alpha_eval * tanh(residual_gain) * projector(quantum(x))`

Where:
- `y_backbone` is frozen ANN output (6 values for J1-J3 sin/cos).
- `quantum(x)` is a differentiable 4-qubit statevector simulation outputting 4 expectation values.
- `projector` maps 4 -> 6 residual correction values.

### 3.1 Frozen Classical Backbone
- Loaded from ANN checkpoint via `_build_ann_baseline_from_checkpoint`.
- Backbone parameters are set `requires_grad=False`.
- Baseline behavior is preserved as non-regression anchor.

### 3.2 Quantum Residual Branch (TinyQuantumLayer)
Default quantum config:
- `q_qubits = 4`
- `q_layers = 2`
- Hilbert space dimension: `2^4 = 16`

Components:
1. Classical encoder:
   - `Linear(3 -> 4)` produces per-qubit encoded values.
   - Encoding angles: `pi * tanh(encoder(x))`.
2. Data encoding gates:
   - Apply `RY` on each qubit (4 gates).
3. Variational layers (2 blocks):
   - For each qubit in each layer: `RX`, `RY`, `RZ` (3 gates per qubit).
   - One variational layer gate counts:
     - 12 single-qubit rotations (4 qubits x 3)
     - 4 CNOT gates in ring topology
   - Two layers total:
     - 24 single-qubit rotations
     - 8 CNOT gates
4. Measurement:
   - Compute Z expectation for each qubit -> 4-dimensional feature vector.

Entanglement topology:
- CNOT ring: `0->1`, `1->2`, `2->3`, `3->0`

### 3.3 Residual Projection Head
- `Linear(4 -> 6)` maps measured expectations to sin/cos correction.
- Scalar `residual_gain` controls residual amplitude via `tanh`.

## 4. Trainable Parameter Count
With defaults (`q_qubits=4`, `q_layers=2`):
- Encoder `Linear(3->4)`: `3*4 + 4 = 16`
- Variational angles `theta[2,4,3]`: `24`
- Projector `Linear(4->6)`: `4*6 + 6 = 30`
- Residual gain scalar: `1`

Total trainable parameters:
- `16 + 24 + 30 + 1 = 71`

Backbone ANN parameters are frozen and excluded from training updates.

## 5. Objective Function
The trainer reuses ANN loss class `FullIKLoss` with defaults:
- `w_sc = 1.0`
- `w_pos = 2.0`
- `w_circ = 0.02`

Loss terms:
1. `L_sc`: shoulder sin/cos regression (`SmoothL1`).
2. `L_wc`: wrist-center consistency from differentiable FK of predicted J1-J3.
3. `L_circ`: unit-circle regularization on each sin/cos pair.

Total:
- `L_total = w_sc*L_sc + w_pos*L_wc + w_circ*L_circ`

## 6. Training Process (Step-by-Step)
1. Load ANN checkpoint (`--ann-checkpoint`) and build frozen backbone.
2. Build hybrid model with tiny quantum residual branch.
3. Build dataloaders from normalized wrist-center inputs and shoulder sin/cos targets.
4. Optimizer: `AdamW` with two parameter groups:
   - quantum branch: `lr_quantum=3e-3`
   - projector + residual_gain: `lr_projector=1e-3`
5. Scheduler: `CosineAnnealingWarmRestarts(T_0=150, T_mult=2, eta_min=min(lr)*1e-2)`.
6. Train epoch loop with gradient clipping (`clip_norm=1.0`) and non-finite guard.
7. Validation each epoch on same objective terms.
8. Monitor is validation shoulder sin/cos loss (`va_sc`).
9. Save atomic checkpoints:
   - `hybrid_last.pt`
   - `hybrid_best.pt` on improvement
10. Apply early stopping with `min_epochs` and `patience`.
11. Reload best model for final calibration/evaluation.

## 7. Alpha Calibration and Deployment Logic
Before final test reporting, residual blend factor is calibrated on validation split.

Calibration method (`_select_best_alpha`):
1. Evaluate alpha grid from `0.0` to `1.5` in steps of `0.05`.
2. Selection priority tuple:
   - maximize all-joints-within-1deg (implemented as minimizing negative value)
   - break ties with lower average MAE
3. Save selected alpha as `best_alpha` in final checkpoint and metrics archive.

Override path:
- `--deploy-alpha` bypasses calibration and forces a fixed alpha.

## 8. Evaluation and Baseline Comparison
On final test split:
1. Evaluate hybrid model with selected `alpha_eval`.
2. Evaluate ANN baseline on same split and normalized features.
3. Report deltas:
   - `ANN avg MAE - Hybrid avg MAE`
   - `(Hybrid all_joints<=1deg - ANN all_joints<=1deg)`

Saved outputs:
- `checkpoints/hybrid_final.pt`
- `hybrid_eval_results.npz`
- diagnostic plots (history, error bars, histograms, scatter)

## 9. Folder Artifacts
- Training: `train_hybrid_qnn_full6_cpu.py`
- Visualization: `visualize_hybrid_qnn_full6_cpu.py`
- Checkpoints: `checkpoints/hybrid_best.pt`, `checkpoints/hybrid_last.pt`, `checkpoints/hybrid_final.pt`
- Metrics: `hybrid_eval_results.npz`
- Plots: `training_history_full6.png`, `joint_metrics_full6.png`, `error_histograms_full6.png`, `pred_vs_true_full6.png`, `ann_vs_hybrid_comparison.png`
