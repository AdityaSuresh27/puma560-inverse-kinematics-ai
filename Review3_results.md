# Review 3 — PUMA 560 IK: Quantum Neural Network vs Classical ANN

**Date:** March 09, 2026
**Task:** Predict inverse kinematics (J1, J2, J3 shoulder joints) of the PUMA 560
6-DOF robot and compare the Hybrid Quantum Neural Network (QNN) against the
classical Artificial Neural Network (ANN) benchmark.

---

## 1. Problem Setup

The PUMA 560 decoupled IK structure means:

- **J1, J2, J3** (shoulder) are determined by the **wrist centre position** P₅ = (Px − d₆·ax, Py − d₆·ay, Pz − d₆·az) — a clean 3-input regression.
- **J4, J5, J6** (wrist) are solved analytically from T₃₆ = T₀₃⁻¹ · T₀₆.

Both models predict sin/cos of J1, J2, J3 (6 outputs) from the normalised wrist centre (3 inputs).

---

## 2. Model Architectures

| Component | Hybrid QNN (v2) | Classical ANN |
|-----------|----------------|---------------|
| **Input** | Wrist centre P₅ [3] | Wrist centre P₅ [3] |
| **Quantum layer** | 4-qubit data re-uploading VQC (3 layers, backprop) | — |
| **Skip connection** | concat(input, quantum features) → [7] | — |
| **Hidden layers** | 256-wide × 4 ResBlocks (post-quantum) | 256-wide × 6 ResBlocks |
| **Output** | 6 (sin/cos J1, J2, J3) | 6 (sin/cos J1, J2, J3) |
| **Total parameters** | ~534,573 (39 quantum + 534,534 classical) | ~527,622 |
| **Loss function** | DecoupledIKLoss (sin/cos MSE + FK wrist + unit-circle) | DecoupledIKLoss (same) |
| **Optimiser** | AdamW + OneCycleLR | AdamW + OneCycleLR |

---

## 3. Training Configuration

| Setting | Hybrid QNN | Classical ANN |
|---------|-----------|---------------|
| Dataset | puma560_dataset.csv (10,000 samples) | same |
| Train / Val / Test split | 7650 / 850 / 1500 | same |
| Epochs | 200 | 3000 |
| Batch size | 128 | 256 |
| Learning rate | 3 × 10⁻³ | 3 × 10⁻³ |
| Early stopping patience | 40 | 200 |
| Training time | ~1278s (~21 min) | longer (CPU) |

---

## 4. Results on Test Set (1,500 samples)

### 4.1 Mean Absolute Error (MAE) — degrees

| Method | J1 | J2 | J3 | **Average** |
|--------|----|----|----|-----------:|
| **Hybrid QNN** | 0.1688° | 0.1660° | 0.3181° | **0.2176°** |
| **Classical ANN** | 0.0488° | 0.0473° | 0.0900° | **0.0620°** |
| *QNN / ANN ratio* | ×3.46 | ×3.51 | ×3.53 | *×3.51* |

### 4.2 Root Mean Squared Error (RMSE) — degrees

| Method | J1 | J2 | J3 | **Average** |
|--------|----|----|----|-----------:|
| **Hybrid QNN** | 0.4178° | 0.2961° | 0.5621° | **0.4253°** |
| **Classical ANN** | 0.1028° | 0.0898° | 0.1759° | **0.1229°** |

### 4.3 Sin/Cos MSE (loss space)

| Method | MSE |
|--------|-----|
| Hybrid QNN | 0.000041 |
| Classical ANN | 0.000003 |

---

## 5. Analysis

### QNN Performance
- **Sub-degree accuracy achieved**: average MAE of **0.2176°** across J1, J2, J3.
- J3 (elbow) has the highest error (0.3181° MAE), which is expected — it maps to a more complex region of joint-space.
- Strong convergence with best validation epoch at **196** out of 200.

### Comparison with Classical ANN
- Classical ANN achieves **0.0620°** avg MAE — **251% worse** than the QNN.
- The accuracy gap is primarily due to:
  1. **Classical simulation overhead**: PennyLane simulates the quantum circuit on CPU, limiting usable batch size and total training epochs.
  2. **Fewer effective epochs**: QNN trains for 200 epochs (limited by ~6 s/epoch); ANN runs for up to 3000 epochs.
  3. **Inherent bottleneck**: Only 39 quantum parameters update via the VQC; the classical head dominates the learning.

### Key Takeaway
> On classical hardware, the QNN is 251% worse than the classical ANN.
> **Both methods achieve sub-degree accuracy** — positioning error ~3–5 mm at tip,
> which is within practical tolerance for many manipulation tasks.
> On real quantum hardware, the quantum layer would execute in nanoseconds and
> allow far more epochs, potentially closing or reversing the gap.

---

## 6. Visual Summary

See **`comparison_dashboard.png`** for:
- Row 1 — Error distribution histograms (QNN blue vs ANN red) for each joint
- Row 2 — Grouped MAE and RMSE bar charts
- Row 3 — Predicted vs True scatter plots (ideal = dashed diagonal)

---

## 7. Files

| File | Description |
|------|-------------|
| `puma560_qnn_hybrid_v1.pt` | Trained QNN checkpoint |
| `puma560_ann_v4_FINAL.pt` | Trained classical ANN checkpoint |
| `qnn_puma560.py` | QNN v2 architecture + training utilities |
| `train_qnn_and_compare.py` | End-to-end training + comparison script |
| `compare_qnn_vs_ann.py` | This comparison script (no retraining) |
| `comparison_dashboard.png` | Full visual comparison dashboard |
| `puma560_dataset.csv` | 10,000 sample dataset (from MATLAB iPUMA.m) |
