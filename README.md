# DAQCNN — Digital-Analog Quantum Convolutional Neural Network

## Overview

This repository implements the Digital-Analog Quantum Convolutional Neural Network (DAQCNN) for medical image classification on the MedMNIST benchmark datasets (BreastMNIST, PneumoniaMNIST, TissueMNIST). Quantum kernels based on Rydberg atom Hamiltonians extract features from image patches; a classical CNN head performs the final classification. The key research question is whether entanglement-based feature extraction via ZZ correlators provides a measurable advantage over classical convolutional filters.

## Physics Background

### Rydberg Hamiltonian

Each quantum kernel places $n$ qubits at fixed atom coordinates determined by the kernel topology. The system evolves under the Rydberg Hamiltonian:

$$H(t) = \frac{\Omega}{2} \sum_i X_i \;-\; \sum_i \delta_i \, \hat{n}_i \;+\; \sum_{i < j} \frac{C_6}{r_{ij}^6} \, \hat{n}_i \hat{n}_j$$

where $\Omega$ is the global Rabi frequency, $\delta_i$ is the local detuning, $C_6$ is the van der Waals coefficient, $r_{ij}$ is the inter-atom distance, and $\hat{n}_i = (1 - Z_i)/2$ is the number operator.

### Encoding Modes

Two modes map image pixel values $p_i \in [0, 1]$ into the quantum state:

- **Digital**: each pixel is applied as an $R_Y(p_i \cdot \pi)$ rotation followed by a Hadamard gate before Hamiltonian evolution.
- **Analog**: pixel values are passed directly as the local detuning parameters $\delta_i$, so the data drives the evolution itself.

### Feature Extraction

After time evolution for duration $T$, the following observables are measured for each patch:

$$\langle Z_i \rangle \quad \text{for all qubits } i, \qquad \langle Z_i \otimes Z_j \rangle \quad \text{for all qubit pairs } i < j$$

For a $3 \times 3$ kernel ($n = 9$ qubits) with ZZ correlators enabled, this yields $9 + \binom{9}{2} = 45$ features per patch. The $\langle ZZ \rangle$ terms capture genuine multi-partite correlations arising from the van der Waals interaction that are inaccessible to single-qubit measurements alone.

### Multi-Kernel Stacking

Running $K$ topologies in parallel yields $K \times 45$ features per patch. All feature maps are concatenated channel-wise before being passed to the classical head.

### Classical CNN Head

The concatenated quantum feature maps are processed by a small classical head:

$$\text{Conv}_{2 \times 2}\text{-BN-Act} \;\to\; \text{MaxPool} \;\to\; \text{Conv}_{2 \times 2}\text{-Act} \;\to\; \text{Dropout} \;\to\; \text{Flatten} \;\to\; \text{Dropout} \;\to\; \text{Linear}(d, C)$$

where $C$ is the number of classes. Only the classical head is trained; the quantum kernels are fixed.

### Kernel Topologies

Eight $3 \times 3$ atom geometries are available, each producing a different van der Waals coupling pattern $J_{ij} = C_6 / r_{ij}^6$:

| Name | Description |
|------|-------------|
| `kings` | All 8 nearest neighbours (king moves) |
| `horizontal` | Three atoms in a row |
| `vertical` | Three atoms in a column |
| `cross` | Centre + 4 cardinal neighbours |
| `ring` | Atoms evenly spaced on a circle |
| `chain` | Linear chain |
| `star` | Centre atom connected to 8 surrounding atoms |
| `grid` | Regular 3x3 grid |

## Prerequisites

- Python 3.10+
- PyTorch
- PennyLane
- NumPy, PyYAML, scikit-learn, matplotlib
- Optuna (for hyperparameter search)
- wandb (optional, for experiment tracking)
- medmnist

Install dependencies (pinned to the versions used for all paper results):

```bash
pip install -r requirements.txt
```

## Typical Workflow

The standard workflow has three steps: cache quantum features, search for the best hyperparameters, then validate the best configuration across multiple seeds.

### Step 1 — Pre-compute quantum features

Running quantum circuits for every patch in every training batch is slow. The `create_quantum_dataset.py` script runs the full dataset through the quantum kernels once and saves the output feature tensors to disk. Subsequent training runs load directly from this cache.

Pass a cache-generation config (dataset, kernel topology, encoding mode, ZZ flag):

```bash
python experiments/create_quantum_dataset.py \
    --config configs/breast_mnist/cache_generation/digital_zz.yml
```

The cache is written to `data/quantum_datasets/` with a filename encoding all relevant parameters (dataset, kernel size, stride, topologies, evolution time, etc.). The training scripts detect and load it automatically.

### Step 2 — Hyperparameter search

The search takes a base config (fixing the architecture) and a search config (defining the search space) and runs an Optuna TPE study. Each trial evaluates a single seed for speed; the best configurations are later re-evaluated with multiple seeds.

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original_daqcnn_best.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_single_kernel_search.yml \
    --n-trials 1000
```

To resume an interrupted search (the SQLite study database is preserved):

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original_daqcnn_best.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_single_kernel_search.yml \
    --n-trials 500 \
    --resume
```

To validate the top-$k$ trials found during search against multiple held-out seeds:

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original_daqcnn_best.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_single_kernel_search.yml \
    --validate-top-k 3 \
    --validation-seeds 0 1 2 3 4 5 6 7 8 9
```

Key flags:

| Flag | Description |
|------|-------------|
| `--config` | Base YAML config file |
| `--search-config` | Search space YAML file |
| `--n-trials` | Number of Optuna trials |
| `--resume` | Resume an existing study |
| `--validate-top-k N` | Re-run top N trials with full seeds after search |
| `--validation-seeds` | Seeds to use for validation |
| `--metric` | Metric to optimise (default: `test_acc`) |
| `--wandb` | Enable W&B logging |

### Step 3 — Multi-seed validation

Once the best hyperparameters are identified, run a full multi-seed experiment using the best config:

```bash
python experiments/robust_test_original_daqcnn.py \
    --config configs/breast_mnist/original_daqcnn_best.yml
```

The config's `misc.seed` field accepts a list; the script runs one independent training per seed and reports mean ± std across all seeds. Results, checkpoints, and plots are saved under `outputs/run_<dataset>_<timestamp>/`.

## Config Structure

Every experiment is controlled by a single YAML config file. The key fields are:

```yaml
dataset:
  name: breast_mnist          # dataset identifier (breast_mnist, pneumonia_mnist, tissue_mnist)
  data_root: ./data
  batch_size: 16
  color_space: GRAYSCALE

model:
  kernel_size: 3              # quantum kernel patch size
  stride: 3                   # patch stride (non-overlapping when equal to kernel_size)
  kernel_topology_names:
    - vertical                # list of topologies; one entry = 1 kernel, multiple = multi-kernel
  encoding_mode: digital      # digital or analog
  include_correlators: true   # whether to include ZZ observables
  evolution_time: 2.5         # Hamiltonian evolution time T
  scaling_factor: 1.0         # pixel value rescaling before encoding
  dropout: 0.5
  activation: gelu

optim:
  lr: 0.001
  weight_decay: 1.0e-4
  epochs: 100
  patience: 15                # early stopping patience
  grad_clip: 5.0
  use_scheduler: true

misc:
  seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # list = multi-seed run
```

Search config files (in `configs/*/hp_search/`) override individual fields with distributions:

```yaml
settings:
  metric: test_acc
  direction: maximize
  study_name: my_search

search_space:
  model.kernel_topology_names:
    type: categorical
    choices:
      - ["vertical"]
      - ["horizontal"]
      - ["star"]
  optim.lr:
    type: loguniform
    low: 1.0e-8
    high: 5.0e-3
  model.dropout:
    type: uniform
    low: 0.0
    high: 0.7
  model.include_correlators:
    type: fixed
    value: true             # fixed = not tuned, just overrides base config
```

Distribution types: `uniform`, `loguniform`, `int`, `categorical`, `fixed`.

## Other Experiments

**Head-capacity sweep** — the paper's central experiment: trains five classifier heads of increasing capacity (linear to CNN-64) on identical frozen features from all quantum, random, and classical-nonlinear sources:

```bash
python experiments/head_capacity_sweep.py --dataset breast_mnist \
    --output-dir outputs/paper_results/capacity_sweep_std/breast_mnist
```

**Linear probing** — evaluates the representational quality of frozen features with a linear SVM, swept across all eight atom-array topologies:

```bash
python experiments/linear_probing_topology_sweep.py --scale both \
    --output-csv outputs/paper_results/linear_probing/breast_mnist_topology_sweep.csv
```

**Batch runner** — runs a list of shell commands sequentially, useful for submitting a queue of experiments:

```bash
python experiments/batch_runner.py commands.txt
```

**TUI** — browse all trained models in `outputs/`, inspect configs, and run inference interactively:

```bash
python -m src.tui.run_tui
```

## Project Structure

```
src/
├── physics/
│   ├── hamiltonian.py           # Rydberg Hamiltonian construction
│   ├── evolution.py             # Time evolution via Trotter decomposition
│   └── kernel_topologies.py    # Atom coordinate sets for each topology
├── layers/
│   ├── daqk.py                  # Digital-analog quantum kernel (single topology)
│   └── quantum_convolution.py  # Sliding-window quantum convolution over an image
├── models/
│   ├── daqcnn.py                # Full DAQCNN model (quantum conv + classical head)
│   ├── daqcnn_training.py       # Training loop, early stopping, evaluation
│   ├── classical_baseline.py    # Classical CNN with fixed or trainable conv filters
│   ├── classical_baseline_training.py
│   └── vanilla_cnn.py           # Standard CNN baseline (no quantum layer)
└── utils/
    ├── data.py                  # Dataset loading (MedMNIST)
    ├── evaluate.py              # Accuracy, AUC, F1, confusion matrix
    ├── training_utils.py        # Classification head builder, device resolution
    ├── plotting.py              # Loss curves, ROC curves, confusion matrices
    ├── quantum_dataset_cache.py # Load pre-computed quantum feature caches
    ├── model_cache_manager.py   # Checkpoint loading and output directory scanning
    ├── kernel_mapping.py        # Channel-to-kernel grouping utilities
    ├── classical_nonlinear_features.py  # poly-2 / RFF per-patch feature maps
    ├── color_conversion.py      # RGB to grayscale / HSV conversion
    ├── hp_search_plots.py       # Optuna study visualisation
    ├── view_results.py          # Print results from an output directory
    └── wab/
        └── fetch_original_baseline.py  # W&B result fetching utilities

configs/
├── breast_mnist/
│   ├── hp_search/               # Search space YAMLs for each architecture variant
│   ├── original/                # Fixed configs for direct training runs
│   ├── cache_generation/        # Configs for create_quantum_dataset.py
│   ├── original_daqcnn_best.yml # Best validated digital no-ZZ config
│   ├── digital_zz_best.yml      # Best validated digital ZZ config
│   └── analog_zz_best.yml       # Best validated analog ZZ config
├── pneumonia_mnist/                 # Same layout: hp_search/, cache_generation/,
└── tissue_mnist/                    #   endtoend/, and *_best.yml winners

experiments/
├── create_quantum_dataset.py        # Pre-compute and cache quantum features
├── merge_quantum_chunks.py          # Reassemble chunked caches (HPC arrays)
├── derive_z_from_zz.py              # Slice a Z-only cache out of a ZZ cache
├── create_classical_cache.py        # poly-2 / RFF features in cache format
├── hyperparameter_search.py         # Optuna-based end-to-end HP search
├── head_capacity_sweep.py           # Keystone experiment: 5 heads x 13 sources
├── robust_test_original_daqcnn.py   # Main multi-seed training entry point
├── linear_probing_topology_sweep.py # Linear SVM probe across topologies
├── probe_*.py, verify_*.py          # Fairness/verification suite (val-selected
│                                    #   topologies, tuned classical baselines)
├── plot_capacity_grid.py            # Paper Fig. 3 (capacity sweep grid)
├── plot_capacity_delta_1col.py      # Paper Fig. 4 (quantum-classical delta)
├── gen_atom_topologies_tikz.py      # Paper Fig. 2 (TikZ topology panels)
├── batch_runner.py                  # Sequential batch experiment runner
├── summarize_results.py             # Aggregate metrics across output directories
└── archived_experiments/            # Superseded/exploratory scripts (see its README)

data/
├── *.npz                            # MedMNIST dataset files
└── quantum_datasets/                # Pre-computed quantum feature caches

docs/
└── paper/
    └── figures/                     # TikZ figure sources; matplotlib figures
                                     #   regenerate from outputs/paper_results CSVs
```

## Results

The summary CSVs and per-cell JSONs behind every table and figure in the paper
are committed under `outputs/paper_results/` (capacity sweeps, linear probing,
and end-to-end validation summaries). The plot scripts in `experiments/`
regenerate the paper figures directly from them. The pre-computed quantum
feature caches (~80 GB) are not in the repository; they can be regenerated
with `experiments/create_quantum_dataset.py` or requested from the authors.

## License

MIT (see `LICENSE`).

## Citation

This project extends the DAQCNN framework of Simen et al.; a reference to our
accompanying paper will be added upon publication.

```bibtex
@article{simenDigitalanalogQuantumConvolutional2024a,
  title = {Digital-Analog Quantum Convolutional Neural Networks for Image Classification},
  author = {Simen, Anton and Flores-Garrigos, Carlos and Hegade, Narendra N. and
            Montalban, Iraitz and Vives-Gilabert, Yolanda and Michon, Eric and
            Zhang, Qi and Solano, Enrique and Mart{\'i}n-Guerrero, Jos{\'e} D.},
  journal = {Physical Review Research},
  volume = {6},
  number = {4},
  pages = {L042060},
  year = {2024},
  doi = {10.1103/PhysRevResearch.6.L042060}
}
```
