# Jet tagging with classical ML and QGNNs

This project reproduces the paper's data path and builds toward its permutation-invariant quantum graph neural network. It supports quark/gluon (Q/G) and up/down (u/d) classification, plus shared data/model interfaces for later tagging tasks.

## Folder map

| Folder | Contents |
|---|---|
| `data/raw/qg/` | Bundled EnergyFlow Q/G particle arrays and source metadata |
| `data/raw/flavor/` | Upstream EIC LO DIS sample, balanced u/d pilot arrays, and source metadata |
| `data/processed/` | Model inputs, labels, fixed train/validation/test indices, preprocessing JSON |
| `data/analysis/qg/` | Jet observables and summary tables |
| `data/graphs/qg/`, `data/graphs/ud/` | Complete particle graphs and edge geometry for classical analysis |
| `data/graphs/legacy/` | Graph artifacts from the original archive, retained for reference |
| `scripts/` | Data preparation, baseline training, QGNN training, prediction, and verification commands |
| `src/jetqml/` | Reusable model implementation, including the differentiable statevector QGNN |
| `results/analysis/figures/` | Existing and regenerated physics-analysis figures |
| `results/qg/classical/`, `results/ud/classical/` | Classical checkpoints, test scores, metrics, and plots |
| `results/qg/qgnn/`, `results/ud/qgnn/` | QGNN checkpoints, test scores, metrics, and plots |
| `results/qg/qiskit/`, `results/ud/qiskit/` | Qiskit Aer shot-simulator outputs and future IBM Runtime results |
| `references/ml-eic-flavor/` | Upstream EIC flavor-analysis repository and its sample data |

## Setup

Run from this folder in Windows PowerShell. The project-local Python environment is already set up; if recreating it:

```powershell
uv venv --python 3.14 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Run project scripts from this folder using the local virtual environment. The bundled Q/G arrays and EIC text sample avoid a new large data download. `requirements-energyflow.txt` is optional for reloading data through EnergyFlow; its Windows dependency may require Microsoft C++ Build Tools. The `particle` package supplies PDG charge lookup for the EIC converter.

## Paper tasks and inputs

| Task | Input particles | Features per particle | Labels |
|---|---:|---|---|
| Q/G | 5 hardest | `(z, eta-like rapidity, phi)` | gluon `0`, quark `1` |
| u/d | 10 hardest | `(z, eta, phi, scaled PDG ID, charge)` | down `0`, up `1` |

`z = particle pT / jet pT`. Azimuth is wrapped and mapped to `[0,1)`. For flavor, the sorted PDG-ID values in the selected sample are mapped at 0.1 spacing centered on zero. Keep these paper-scale values for the QGNN; do not fit a z-score scaler on all data. Splits are stratified and saved once so each model uses the same train/validation/test indices.

## Quark/gluon data

The project includes the 5,000 raw EnergyFlow Q/G jets from the supplied archive. Prepare their five-particle model representation and graph file:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --task qg
.\.venv\Scripts\python.exe scripts\build_graphs.py --task qg
```

This writes `data/processed/qg_5particle.npz` with `X.shape=(5000,5,3)`, labels, and a 64/16/20 stratified split (the paper test fraction is 20%; validation is reserved from the remaining training portion).

## Up/down flavor data

The paper's source is the EIC PYTHIA6 LO DIS dataset. The full public [Zenodo archive](https://zenodo.org/records/7538810) is 17.4 GB; this project instead uses the 32 MB `LOjets.txt` sample distributed with the [source analysis repository](https://github.com/jdmulligan/ml-eic-flavor), retained under `references/`. Its process-99, in-jet rows contain 2,084 down and 15,475 up jets above the source study's 10 GeV jet-pT threshold. To make a balanced pilot sample, the converter selects 2,084 per class (4,168 total), computes particle charge from PDG ID, and preserves source event/jet IDs separately from model features. This makes the u/d pipeline runnable without claiming the paper's separate 5,000-jet sample or reproducing all EIC cuts from the full archive.

Prepare the balanced flavor sample, ten-particle inputs, and graphs:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_flavor_sample.py
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --task ud --input data\raw\flavor\ud_pilot.npz
.\.venv\Scripts\python.exe scripts\build_graphs.py --task ud
```

This writes `data/processed/ud_10particle.npz` with `X.shape=(4168,10,5)` and the same saved split format as Q/G. Full source data and sample citation are recorded in `data/raw/flavor/ud_pilot.json` and `docs/flavor_data.md`.

## Classical baselines

Train the MLP and Particle Flow Network (PFN) on either task:

```powershell
.\.venv\Scripts\python.exe scripts\train_baselines.py --task qg --epochs 50 --batch-size 200
.\.venv\Scripts\python.exe scripts\train_baselines.py --task ud --epochs 50 --batch-size 200
```

Each run saves checkpoints, per-epoch histories, ROC/AUC/accuracy/SIC metrics, and test scores in `results/<task>/`. The PFN uses a shared per-particle network and sum aggregation, implementing `F(sum_i Phi(x_i))` and preserving permutation invariance.

## Quantum GNN

The local simulator is a differentiable exact statevector implementation in `src/jetqml/qgnn.py`. It uses one qubit per selected particle, feature-weighted angle encoding shared across particles, shared Rx node gates, shared RZZ gates on the fully connected graph, and mean Z expectation as the invariant score. The commuting complete-graph ZZ layer is evaluated as one equivalent diagonal phase. For five encoded features, rotations cycle through Rx, Ry, and Rz; the paper specifies Pauli rotations and one trainable weight per feature but does not state a unique five-feature axis assignment, so this implementation records that choice as a reproducible design assumption.

First verify the 5- and 10-qubit circuits, then train them with the paper's six layers, 50 epochs, batch size 50, Adam, learning rate 0.01, and hinge loss:

```powershell
.\.venv\Scripts\python.exe scripts\verify_qgnn.py
.\.venv\Scripts\python.exe scripts\train_qgnn.py --task qg --layers 6 --epochs 50 --batch-size 50 --lr 0.01
.\.venv\Scripts\python.exe scripts\train_qgnn.py --task ud --layers 6 --epochs 50 --batch-size 200 --lr 0.01
```

These runs save the best validation-AUC checkpoint, held-out test scores, ROC and training plots, and metrics under `results/qg/qgnn/` and `results/ud/qgnn/`. The full graph is represented by 10 edges for 5 particles and 45 edges for 10 particles. The models have 30 and 42 trainable parameters, respectively. The u/d batch is set to 200 to reduce runtime on the 10-qubit CPU simulation; use `--batch-size 50` for the paper setting, with a longer runtime.

Generate per-jet quantum scores with:

```powershell
.\.venv\Scripts\python.exe scripts\predict_qgnn.py --task qg --split test
.\.venv\Scripts\python.exe scripts\predict_qgnn.py --task ud --split test
```

The CSV includes the signed QGNN output, its `[0,1]` class-1 score, predicted class, and true label.

## Reproduce and inspect outputs

`results/qg/classical/run_summary.md` reports the existing one-seed classical Q/G run. `results/ud/classical/` contains the flavor baseline run. QGNN results are separate from the classical outputs. The first QGNN training run should be treated as a local simulator reproduction: the paper reports ten seeds and a separate PFN setup, and its hardware demonstrations require provider access beyond this local statevector simulator.

To generate per-jet classical prediction rows from a saved checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --task qg --model pfn --split test
```

Predictions include `score_class_1` (quark for Q/G; up for u/d), predicted class, and true label when the input archive includes labels.

## Paper-aligned repeated ideal runs and Qiskit hardware models

The earlier 5- and 10-qubit results use validation selection and a 64/16/20 split. They are useful pilot runs, but do not match the paper's 80/20 evaluation and ten-seed results. The repeated ideal protocol is available as:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_ideal.py --task qg --n-seeds 10
.\.venv\Scripts\python.exe scripts\train_paper_ideal.py --task ud --n-seeds 10
```

It uses a stratified 80/20 split per seed, all training events, the final epoch, 50 epochs, batch size 50, Adam, and hinge loss. Results are saved under `results/<task>/paper_ideal/`. The u/d run remains limited to the available 4,168-jet pilot unless the paper-scale source is prepared.

The Qiskit path builds the paper's reduced hardware circuits and supports local Aer and IBM Quantum Runtime SamplerV2:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_qiskit_inputs.py
.\.venv\Scripts\python.exe scripts\train_qiskit_qgnn.py --task qg --backend aer --shots 512 --maxiter 30 --seed 308
.\.venv\Scripts\python.exe scripts\train_qiskit_qgnn.py --task ud --backend aer --shots 512 --maxiter 30 --seed 308
```

This uses 2 qubits / 3 layers / 480 training jets for Q/G and 4 qubits / 2 layers / 500 training jets for u/d, as in the paper's scaled hardware study. The u/d reduced features are `q` and `abs(q) * z`, with constituents ranked by `abs(q * pT)`. For IBM hardware, configure an IBM Quantum account and use `--backend ibm --backend-name <backend_name>`; see [docs/qiskit_hardware.md](docs/qiskit_hardware.md). Aer commands do not submit hardware jobs.

## References

- Jobilal et al., “Quantum Graph Neural Networks for Jet Tagging on Quantum Hardware,” [arXiv:2609.04367](https://arxiv.org/abs/2609.04367).
- Lee et al., “Machine learning-based jet and event classification at the Electron-Ion Collider,” [arXiv:2210.06450](https://arxiv.org/abs/2210.06450).
- EIC PYTHIA6 data: [Zenodo record 7538810](https://zenodo.org/records/7538810).
- EIC upstream analysis source: [`references/ml-eic-flavor`](references/ml-eic-flavor/README.md).




Current single-seed CPU pilot results are in esults/qg/qgnn/qgnn_metrics.json and esults/ud/qgnn/qgnn_metrics.json. The 5-qubit Q/G test AUC is 0.6864; the balanced 10-qubit u/d pilot test AUC is 0.6739. These are local pilot outcomes, not the paper's ten-seed/hardware result.

