# Full-6DOF ANN (Artificial Neural Network) Inverse Kinematics

## Overview
Standard neural network baseline for PUMA 560 6-DOF inverse kinematics. Uses decoupled shoulder prediction and analytic wrist recovery.

## Architecture
- **Shoulder (Joint 1-3)**: Decoupled neural network prediction using sin/cos encoding
- **Wrist (Joint 4-6)**: Analytical solution via rotation matrix decomposition
- **Network**: Dense layers with ReLU activation
- **Parameters**: ~10K trainable weights

## Key Results
- **Test Avg MAE**: 0.2509 deg
- **Test Avg RMSE**: 1.6284 deg
- **All-joint accuracy (≤1.0°)**: 90.40%
- **All-joint accuracy (≤0.5°)**: 81.20%
- **All-joint accuracy (≤2.0°)**: 95.87%

## Files
- `train_ann_full6_cpu.py` — Training script (CPU-only, no GPU required)
- `visualize_ann_full6_cpu.py` — Evaluation and visualization script
- `checkpoints/ann6_best.pt` — Final trained checkpoint
- `ann6_eval_results.npz` — Test split metrics and predictions
- `training_history_full6.png` — Loss curves during training
- `joint_metrics_full6.png` — Per-joint error distributions
- `error_histograms_full6.png` — Error analysis histograms
- `pred_vs_true_full6.png` — Predicted vs. true joint angles
- `ann_vs_hybrid_comparison.png` — Performance comparison with Hybrid QNN

## Usage

### Train from scratch
```bash
python train_ann_full6_cpu.py --epochs 200 --batch 256 --output-dir full6dof_ann_cpu
```

### Evaluate existing checkpoint
```bash
python visualize_ann_full6_cpu.py --checkpoint checkpoints/ann6_best.pt
```

### Dataset
- **File**: `../puma560_dataset.csv`
- **Format**: CSV with columns for joint angles and Cartesian poses
- **Split**: 75% train, 10% validation, 15% test
- **Size**: 10,000 samples total

## Performance Summary
This network provides a strong, interpretable baseline. The decoupled shoulder + analytical wrist approach balances:
- Interpretability (shoulder is learned, wrist is analytical)
- Accuracy (90%+ within 1° on all joints)
- Efficiency (fast CPU inference, minimal training time)

See `ann_vs_hybrid_comparison.png` for detailed comparison with the Hybrid QNN model.
