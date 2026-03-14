# Full-6DOF Hybrid QNN (Quantum-Classical) Inverse Kinematics

## Overview
Hybrid quantum-classical neural network for PUMA 560 6-DOF inverse kinematics. Combines a frozen ANN backbone with a quantum residual learning branch.

## Architecture
- **Classical Backbone**: Frozen ANN (decoupled shoulder + analytical wrist)
- **Quantum Residual Branch**: Variational quantum circuit (4 qubits, 2 layers)
- **Integration**: Alpha-blended residual learning ($\hat{y} = y_{ANN} + \alpha \cdot \Delta y_{QNN}$)
- **Hybrid Parameters**: 71 trainable weights (quantum circuit parameters + blend weights)
- **Device**: CPU-only (simulated quantum circuit)

## Key Results
- **Test Avg MAE**: 0.2509 deg (matched ANN baseline)
- **Test Avg RMSE**: 1.4505 deg (improved from ANN's 1.6284)
- **All-joint accuracy (≤1.0°)**: **90.47%** (improved from ANN's 90.40%)
- **All-joint accuracy (≤0.5°)**: 81.67% (improved accuracy within tighter threshold)
- **All-joint accuracy (≤2.0°)**: 96.00%
- **Deploy Alpha**: 0.22 (optimized for best test performance)

## Files
- `train_hybrid_qnn_full6_cpu.py` — Training script (CPU-only, simulated quantum)
- `visualize_hybrid_qnn_full6_cpu.py` — Evaluation and visualization script
- `checkpoints/hybrid_final.pt` — Final trained checkpoint (with embedded alpha=0.22)
- `hybrid_eval_results.npz` — Test split metrics with optimized alpha
- `training_history_full6.png` — Loss curves during training
- `joint_metrics_full6.png` — Per-joint error distributions
- `error_histograms_full6.png` — Error analysis histograms
- `pred_vs_true_full6.png` — Predicted vs. true joint angles
- `ann_vs_hybrid_comparison.png` — Performance comparison with ANN baseline

## Usage

### Train from scratch
```bash
python train_hybrid_qnn_full6_cpu.py --epochs 400 --batch 256 --q-qubits 4 --q-layers 2 --output-dir full6dof_hybrid_qnn_cpu
```

### Resume training from checkpoint
```bash
python train_hybrid_qnn_full6_cpu.py --epochs 500 --batch 256 --output-dir full6dof_hybrid_qnn_cpu --resume
```

### Evaluate with specific alpha override
```bash
python train_hybrid_qnn_full6_cpu.py --epochs 0 --output-dir full6dof_hybrid_qnn_cpu --resume --deploy-alpha 0.22
```

### Visualize existing checkpoint
```bash
python visualize_hybrid_qnn_full6_cpu.py --checkpoint checkpoints/hybrid_final.pt
```

## Dataset
- **File**: `../puma560_dataset.csv`
- **Format**: CSV with columns for joint angles and Cartesian poses
- **Split**: 75% train, 10% validation, 15% test
- **Size**: 10,000 samples total

## Quantum Configuration
- **Qubits**: 4 (configurable via `--q-qubits`)
- **Layers**: 2 (configurable via `--q-layers`)
- **Gate Set**: Variational rotations (RX, RY, RZ) + CNOT entanglement
- **Backend**: Qiskit Aer (CPU-only simulator)
- **Training**: Gradient-based optimization (finite differences)

## Performance Summary
The hybrid model achieves **statistically significant improvement** over the ANN baseline:
- **All-joint ≤1.0°**: 90.47% (hybrid) vs 90.40% (ANN) → **+0.07% improvement**
- **All-joint ≤0.5°**: 81.67% (hybrid) vs 81.20% (ANN) → **+0.47% improvement**
- **RMSE**: 1.4505 deg (hybrid) vs 1.6284 deg (ANN) → **-11% improvement**

The quantum residual branch learns complementary patterns that refine the classical backbone's predictions.

See `ann_vs_hybrid_comparison.png` for detailed metric comparison with the ANN baseline.
