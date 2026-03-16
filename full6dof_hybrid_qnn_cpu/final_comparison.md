# Review 3C - Final Comparison (Folder-Local): ANN vs Hybrid Residual QNN

**Date:** March 16, 2026
**Task:** Compare only the models implemented in `full6dof_ann_cpu` and `full6dof_hybrid_qnn_cpu`.
**Scope boundary:** Parent-folder models/checkpoints are excluded from this comparison.

---

## 1. Compared Pipelines

| Pipeline | Folder |
|---------|--------|
| Classical ANN baseline | `../full6dof_ann_cpu` |
| Hybrid residual QNN | `.` |

Common setup:

- Same dataset: `../puma560_dataset.csv`
- Same split: train 75%, val 10%, test 15% (seed 42)
- Same decoupled IK strategy:
  - learn shoulder (J1-J3) as sin/cos
  - recover wrist (J4-J6) analytically from transforms

---

## 2. Architecture Comparison

| Component | ANN | Hybrid |
|----------|-----|--------|
| Learned input | Wrist center $P_5$ (3D) | Wrist center $P_5$ (3D) |
| Learned output | 6 values (sin/cos J1-J3) | 6 values (sin/cos J1-J3, residual-refined) |
| Main model | FullIKANN (trainable) | Frozen FullIKANN + TinyQuantumLayer + projector |
| Wrist J4-J6 | Analytical solve | Analytical solve |
| ANN params | 1,787,910 trainable | 1,787,910 frozen |
| Added trainable params | none | 71 |

Hybrid trainable-71 breakdown:

- Encoder `Linear(3->4)`: 16
- Variational angles: 24
- Projector `Linear(4->6)`: 30
- Residual gain scalar: 1

---

## 3. Optimization Differences

| Item | ANN | Hybrid |
|------|-----|--------|
| Optimizer | AdamW (single group) | AdamW (quantum group + projector/gain group) |
| LR | 0.001 | 0.003 (quantum), 0.001 (projector/gain) |
| Scheduler | CosineAnnealingWarmRestarts, $T_0=200$ | CosineAnnealingWarmRestarts, $T_0=150$ |
| Best-model monitor | validation shoulder sin/cos loss | same |
| Extra calibration | none | validation alpha grid-search 0.0..1.5 step 0.05 |

Selected hybrid deploy alpha (saved): 0.22.

---

## 4. Test Results (Artifact-Verified)

### 4.1 Aggregate metrics

| Metric | ANN | Hybrid | Delta (Hybrid - ANN) |
|--------|----:|-------:|---------------------:|
| Avg MAE (deg) | 0.250860 | 0.250863 | +0.000003 |
| Avg RMSE (deg) | 1.446933 | 1.450541 | +0.003609 |
| All joints <= 0.5 deg | 81.40% | 81.67% | +0.27 pp |
| All joints <= 1.0 deg | 90.40% | 90.47% | +0.07 pp |
| All joints <= 2.0 deg | 95.87% | 96.00% | +0.13 pp |

### 4.2 Per-joint MAE delta (Hybrid - ANN, degrees)

| Joint | Delta MAE |
|------|----------:|
| J1 | -0.000615 |
| J2 | -0.000162 |
| J3 | -0.000992 |
| J4 | +0.000918 |
| J5 | -0.000246 |
| J6 | +0.001115 |

### 4.3 Per-joint RMSE delta (Hybrid - ANN, degrees)

| Joint | Delta RMSE |
|------|-----------:|
| J1 | +0.000003 |
| J2 | -0.000055 |
| J3 | -0.000818 |
| J4 | +0.011427 |
| J5 | -0.000351 |
| J6 | +0.011447 |

---

## 5. Technical Interpretation

1. Both pipelines are decoupled shoulder-learning systems with analytical wrist recovery.
2. Hybrid residual branch improves strict all-joint threshold rates slightly.
3. Hybrid average MAE is effectively equal to ANN (difference at micro-degree scale).
4. Hybrid average RMSE is slightly worse overall due to larger RMSE on J4/J6.
5. In this folder-local experiment, hybrid gives a threshold-accuracy gain but not an average-error gain.

---

## 6. Final Verified Conclusion

For these two folders specifically (excluding parent models):

- Analytical kinematics is definitely used in both pipelines.
- Neither pipeline is pure direct 6-joint prediction.
- Performance trade-off is mixed:
  - Hybrid improves all-joint threshold accuracies slightly.
  - ANN is marginally better on average MAE/RMSE aggregates.
