# Final Technical Comparison: ANN vs Hybrid QNN (Full 6DOF)

## 1. Comparison Scope
This comparison uses the two folder pipelines:
- ANN baseline: `../full6dof_ann_cpu`
- Hybrid residual model: `.`

Both pipelines use:
- same dataset (`../puma560_dataset.csv`)
- same split policy (train 75%, val 10%, test 15%)
- same decoupled formulation (learn J1-J3, solve J4-J6 analytically)
- same evaluation metric definitions

## 2. Architecture-Level Comparison
| Item | ANN Baseline | Hybrid QNN Residual |
|---|---|---|
| Trainable core | FullIKANN (shoulder sin/cos regressor) | Frozen FullIKANN + trainable TinyQuantumLayer residual |
| Input features to learned module | Wrist-center Cartesian `P5` (3D) | Wrist-center Cartesian `P5` (3D) |
| Main output from learned module | 6 values (sin/cos for J1-J3) | 6 values (base + residual sin/cos for J1-J3) |
| Wrist J4-J6 | Analytical solve from `T36` | Same analytical solve from `T36` |
| Backbone training state | Trainable | Frozen |
| Additional branch | None | Quantum statevector simulator + linear projector |

## 3. Layer and Parameter Accounting
### 3.1 ANN (training defaults)
Config used by trainer defaults:
- `hidden=384`, `blocks=6`, `n_in=3`, `n_out=6`

Layer counts:
- Linear layers: 14
- LayerNorm layers: 13
- Residual blocks: 6

Trainable parameters:
- 1,787,910

### 3.2 Hybrid residual branch (default quantum config)
Config:
- `q_qubits=4`, `q_layers=2`

Trainable components:
- Encoder `Linear(3->4)`: 16 params
- Variational angles `theta[2,4,3]`: 24 params
- Projector `Linear(4->6)`: 30 params
- Scalar gain: 1 param

Total trainable residual params:
- 71

Notes:
- ANN backbone is frozen in hybrid training.
- Hybrid output equation:
  - `y_hybrid = y_ann + alpha * tanh(gain) * delta_q`

## 4. Optimization and Control Differences
| Item | ANN | Hybrid |
|---|---|---|
| Optimizer | AdamW single group | AdamW two groups (quantum vs projector/gain) |
| LR defaults | `1e-3` | `3e-3` (quantum), `1e-3` (projector/gain) |
| Scheduler | Cosine warm restarts `T0=200` | Cosine warm restarts `T0=150` |
| Monitor for best model | validation shoulder sin/cos loss | validation shoulder sin/cos loss |
| Residual blending calibration | Not applicable | validation grid-search on `alpha` in [0,1.5] step 0.05 |

## 5. Reported Test Metrics
Values documented in folder READMEs and saved result artifacts.

| Metric | ANN | Hybrid | Delta (Hybrid - ANN) |
|---|---:|---:|---:|
| Avg MAE (deg) | 0.2509 | 0.2509 | 0.0000 |
| Avg RMSE (deg) | 1.6284 | 1.4505 | -0.1779 |
| All joints <= 0.5 deg | 81.20% | 81.67% | +0.47 pp |
| All joints <= 1.0 deg | 90.40% | 90.47% | +0.07 pp |
| All joints <= 2.0 deg | 95.87% | 96.00% | +0.13 pp |
| Selected deploy alpha | n/a | 0.22 | n/a |

Derived RMSE relative reduction:
- `(1.6284 - 1.4505) / 1.6284 = 10.92%`

## 6. Technical Interpretation
1. The hybrid model preserves average absolute accuracy (MAE unchanged) while reducing squared-error sensitivity (RMSE lower).
2. Improvement pattern indicates error-tail compression: largest errors are reduced more than median errors.
3. Since wrist reconstruction is analytical in both methods, observed differences primarily come from improved shoulder sin/cos prediction after residual refinement.
4. Frozen backbone plus calibrated alpha provides a controlled residual mechanism that limits catastrophic drift from the ANN baseline.

## 7. Practical Selection Guidance
1. If minimum implementation complexity is required, ANN is simpler and already strong.
2. If lower RMSE and slightly better strict-threshold all-joint accuracy are required, Hybrid is the better deployment candidate.
3. For deterministic deployment, use `best_alpha` stored in `checkpoints/hybrid_final.pt` (reported as 0.22).
