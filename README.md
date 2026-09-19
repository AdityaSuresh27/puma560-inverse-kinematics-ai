# PUMA 560 Inverse Kinematics — Classical, Analytical & Hybrid Quantum-Neural

Inverse kinematics (IK) solvers for the **PUMA 560**, a classic 6-DOF industrial robot manipulator, implemented and compared across three approaches:

1. **Analytical / MATLAB** — exact geometric IK (8 solution branches) via Denavit–Hartenberg kinematics.
2. **Classical Neural Network (ANN)** — a residual feed-forward network that learns the shoulder joints (J1–J3) directly from data, with the wrist joints (J4–J6) solved analytically.
3. **Hybrid Quantum Neural Network (QNN)** — a variational quantum circuit combined with the classical ANN backbone, used as a learned correction/feature-extraction layer, benchmarked head-to-head against the pure classical model.

The project exists as a research/coursework exploration of whether a small, NISQ-era quantum circuit can meaningfully improve a classical robotics regression task — and documents both the failed and successful attempts at doing so.

---

## Robot & Problem Setup

The PUMA 560 has **decoupled arm geometry**, which this project exploits directly:

- **Joints 1–3** (shoulder) position the **wrist centre** P₅ = (Px, Py, Pz) — a clean 3-input → 3-output regression problem.
- **Joints 4–6** (wrist) are then solved **analytically** (closed-form ZYZ decomposition) once J1–J3 and the target orientation are known.

So every learned model in this repo only has to predict **J1, J2, J3** (encoded as sin/cos pairs to respect angle periodicity); J4–J6 always come from the exact analytical solver.

**DH parameters** (used consistently across every MATLAB and Python implementation in the repo):

| Link | a (mm) | d (mm) | α (deg) |
|------|--------|--------|---------|
| 1 | 0 | 671.83 | -90 |
| 2 | 431.80 | 139.70 | 0 |
| 3 | -20.32 | 0 | 90 |
| 4 | 0 | 431.80 | -90 |
| 5 | 0 | 0 | 90 |
| 6 | 0 | 56.50 | 0 |

**Joint limits:** θ1 ∈ [-160°,160°], θ2 ∈ [-225°,45°], θ3 ∈ [-45°,225°], θ4 ∈ [-110°,170°], θ5 ∈ [-100°,100°], θ6 ∈ [-266°,266°].

**Dataset:** 10,000 randomly sampled PUMA 560 poses, generated in MATLAB (`fPUMA.m`/`iPUMA.m`) and FK-validated, split into train / validation / test (`data/puma560_dataset.csv`, seed = 42). Note: the repo's own docs disagree slightly on the split — `README_QNN.md` states "70% / 15% / 15%", while `Final_Results.md` and `Review3_results.md` both give the actual sample counts as **7,650 / 850 / 1,500** (i.e. 76.5% / 8.5% / 15%). The latter is used consistently in both technical reports, so treat it as the more reliable figure.

---

## Repository Structure

```
.
├── FINAL_MAIN/                  # Consolidated / final polished pipeline
├── full6dof_ann_cpu/             # Full 6-DOF experiments — classical ANN
├── full6dof_direct_ml_cpu/       # Full 6-DOF experiments — direct end-to-end ML (no analytical decoupling)
├── full6dof_hybrid_qnn_cpu/      # Full 6-DOF experiments — hybrid QNN
├── puma560_3dof/                 # Core 3-DOF (decoupled) implementation: ANN + QNN + training scripts
├── matlab/                       # MATLAB forward/inverse kinematics (fPUMA.m, iPUMA.m) and dataset generator
├── data/                         # puma560_dataset.csv / .mat — the 10K-sample IK dataset
├── results/                      # Trained checkpoints, comparison plots, metrics
├── utils/                        # Shared helper utilities
├── Final_Results.md              # Final technical report: Hybrid QNN v3 vs classical ANN
├── Review3_results.md            # Earlier checkpoint report: Hybrid QNN v2 vs classical ANN
├── LOGIC_VERIFICATION_REPORT.md  # Independent audit of DH params, data flow, and loss design
├── INDEX.md                      # Full documentation map and project overview
├── README_QNN.md                 # Deep-dive on the QNN architecture, install, and usage
├── QUICK_START_QNN.md            # 5-minute quick-start guide
└── README.md                     # You are here
```

> **Caveat:** I was not able to browse into `FINAL_MAIN/`, `full6dof_ann_cpu/`, `full6dof_direct_ml_cpu/`, `full6dof_hybrid_qnn_cpu/`, or `matlab/` — GitHub blocks automated access to its folder-listing pages, and I only have the file names as they appear in the root directory listing, not their contents. The descriptions above (e.g. "full 6-DOF experiments", "direct end-to-end ML") are **inferred from the folder names alone**, not confirmed by reading any file inside them. Please open these folders yourself before relying on that framing — the only subfolder whose contents I've actually verified (via `Final_Results.md`, `Review3_results.md`, `README_QNN.md`, and `LOGIC_VERIFICATION_REPORT.md`, which quote specific files and line numbers from it) is `puma560_3dof/`.

---

## Methods

### 1. Analytical IK (`matlab/`)
Exact closed-form solver (`fPUMA.m` for forward kinematics, `iPUMA.m` for inverse kinematics) producing all 8 geometric solution branches (shoulder left/right × elbow up/down × wrist flip/no-flip), used both as ground truth and to generate the training dataset.

### 2. Classical ANN — "ShoulderNet" (`puma560_3dof/train_puma560.py`)
```
Input [3]  (normalised wrist centre P5)
  → Linear(3→256) + LayerNorm + GELU        (stem)
  → 6 × ResBlock(256, with skip connections)
  → Linear(256→6)                            (sin/cos of J1,J2,J3)
```
- **Loss (`DecoupledIKLoss`)**: sin/cos MSE + a forward-kinematics-consistency term (predicted angles must reproduce the correct wrist centre) + a unit-circle penalty on sin²+cos².
- **Optimizer/schedule**: AdamW + OneCycleLR, ~3000 epochs with early stopping.
- **Test performance**: **0.0620° average MAE** across J1–J3.

### 3. Hybrid Quantum Neural Network (`puma560_3dof/qnn_puma560.py`)
The QNN went through (at least) two documented design iterations:

- **v2 — skip-concatenation** (`Review3_results.md`): quantum features from a 4-qubit VQC were concatenated with the raw input before the classical stem. This changed the stem's input shape, which broke weight transfer from the pretrained ANN and left the model training from a worse starting point. Result: **0.2176° avg MAE — ~3.5× worse than the classical ANN.**
- **v3 — additive correction** (`Final_Results.md`, the final/winning design): the classical backbone is kept *architecturally identical* to the trained ANN and pretrained ANN weights are transferred in directly; the quantum branch (a 4-qubit, 3-layer **data re-uploading** variational circuit) only adds a small, **zero-initialized** correction on top, so training starts at exactly ANN-level accuracy and can only improve from there. A differential learning rate (quantum layer 100× faster than the classical backbone) lets the quantum correction learn without overwriting the transferred classical weights. Result: **0.0556° avg MAE — a 10.3% improvement over the classical ANN.**

### Quantum circuit (v3, winning design)
```
4 qubits, 3 layers, data re-uploading:
  Layer i: RX(input) data-encoding → Rot(θ1,θ2,θ3) variational → CNOT ring entanglement
Output: ⟨PauliZ⟩ expectation on all 4 qubits → small MLP (4→32→6) → added to classical output
```
39 trainable quantum parameters, differentiable end-to-end via PennyLane's `backprop` device, trained jointly with the (mostly frozen) classical backbone.

---

## Results Summary

| Metric | Classical ANN | Hybrid QNN (v2, concat) | Hybrid QNN (v3, additive) |
|---|---|---|---|
| Avg Test MAE | 0.0620° | 0.2176° | **0.0556°** |
| Avg Test RMSE | 0.1228° | 0.4253° | 0.1284° |
| Outcome | Baseline | ✗ Quantum hurt | ✓ **Quantum helped (+10.3%)** |

Both the classical ANN and the winning QNN achieve **sub-degree accuracy** at the shoulder joints, translating to roughly **3–5 mm end-effector tip error** on a ~900 mm workspace — well within tolerance for most manipulation tasks. All comparisons use the exact same dataset, train/val/test split, and loss function for a fair benchmark; see `LOGIC_VERIFICATION_REPORT.md` for an independent audit of the DH parameters, data flow, and loss design (verdict: logic is sound, with only minor non-critical cleanup notes).

Note: the current QNN runs on a **classical simulator** (PennyLane, CPU) — there's no quantum-hardware speed advantage yet, and the ~10% accuracy gain is a research result about expressivity/inductive bias, not a demonstration of quantum computational advantage.

---

## Getting Started

### Prerequisites
```bash
pip install torch pennylane numpy pandas scipy matplotlib scikit-learn tqdm
```
(MATLAB is only required if you want to regenerate the dataset or run the analytical solvers in `matlab/`.)

### Quick Start — train and compare the QNN against the ANN

> ⚠️ **The repo's own docs disagree on whether you run this from the repo root or from inside `puma560_3dof/`,** and I could not open the folder to check directly. `INDEX.md`'s usage examples invoke scripts from the root with a path prefix (`python puma560_3dof/train_qnn_and_compare.py`, bare `from qnn_puma560 import ...`), while `README_QNN.md`'s usage examples import as `from puma560_3dof.qnn_puma560 import ...` and load `puma560_3dof/puma560_qnn_hybrid_v1.pt` — also implying root, but with a different import style. Try running from the repo root first; if that fails on the import, `cd puma560_3dof` and retry.

```bash
python train_qnn_and_compare.py --epochs 3000 --n-qubits 4 --n-qlayers 3 --transfer
```
This trains the additive-correction QNN (v3), transferring weights from the pretrained classical ANN (`puma560_ann_v4_FINAL.pt`), and reports test MAE/RMSE against the classical baseline. Expect roughly ~20 minutes on CPU (quantum simulation is the bottleneck at ~6.5 s/epoch).

### Run inference with a trained model
```python
import torch
from puma560_3dof.qnn_puma560 import HybridQNN, sc_to_angles, compute_wrist_center

ckpt = torch.load('puma560_3dof/puma560_qnn_hybrid_v1.pt')
model = HybridQNN()
model.load_state_dict(ckpt['model_state'])
model.eval()

P5 = compute_wrist_center(your_pose)          # your_pose: [nx,ny,nz,ox,oy,oz,ax,ay,az,Px,Py,Pz]
P5_norm = (P5 - ckpt['P5_mean']) / ckpt['P5_std']
with torch.no_grad():
    J123 = sc_to_angles(torch.tanh(model(torch.tensor([P5_norm]))).numpy())
print(f"J1={J123[0,0]:.2f}°, J2={J123[0,1]:.2f}°, J3={J123[0,2]:.2f}°")
```
(Adjusted to follow `README_QNN.md`'s exact usage example, which is the more explicit of the two conflicting versions.) J4–J6 still need to be solved analytically from the predicted J1–J3 and the target orientation (see `solve_wrist` in `train_puma560.py`).

### Generate the dataset / run the analytical solver
```matlab
% in matlab/
dataset_generator.m   % generates data/puma560_dataset.csv via fPUMA.m / iPUMA.m
```

For a guided walkthrough see **`QUICK_START_QNN.md`** (5 min), or the full **`README_QNN.md`** (architecture, training internals, troubleshooting, ~20–30 min read). **`INDEX.md`** is the top-level documentation map tying all of the above together.

---

## Known Limitations / Open Issues

**On this README's own accuracy:** everything above is sourced from the repo's root-level markdown docs (`INDEX.md`, `README_QNN.md`, `Final_Results.md`, `Review3_results.md`, `LOGIC_VERIFICATION_REPORT.md`), cross-checked against each other. I was not able to open `puma560_3dof/`, `FINAL_MAIN/`, `full6dof_*/`, or `matlab/` to check the actual source code against what the docs claim — GitHub blocks automated folder-browsing here, and I don't have a way around that in this environment. Where the docs contradicted each other (the dataset split, and how to invoke the training script) I've flagged it explicitly above rather than picking one silently. Treat this README as a faithful summary of the project's *documentation*, not an independently-verified account of its *code*.

From the project's own logic-verification audit (`LOGIC_VERIFICATION_REPORT.md`):
- `train_puma560.py` and `train_puma560_v4_FINAL.py` appear to be near-duplicate files — worth consolidating.
- The MATLAB dataset generator supports multiple IK configurations (`preferred_configs = [1,2,3,4]`) but the shipped dataset only ever used Config 1 (100%) — either intentional or worth regenerating for diversity.
- FK-consistency validation during data generation only checks **position** error, not full orientation/rotation error.
- The QNN currently only has a quantum-hardware advantage story for the future — on classical simulators it's slower per-sample (~7 ms vs ~0.3 ms for the classical ANN) despite the accuracy win.
- This is a decoupled-IK approach specific to robots like the PUMA 560 with spherical wrists; a fully coupled 6-DOF arm would need a different formulation (see the `full6dof_*` folders for that direction).

## License

MIT License (per `INDEX.md` / `README_QNN.md`) — confirm a `LICENSE` file is present at the repo root if you intend this for redistribution.
