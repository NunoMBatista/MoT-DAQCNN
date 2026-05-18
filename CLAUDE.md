# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DAQCNN (Digital-Analog Quantum Convolutional Neural Network) applies Rydberg atom Hamiltonians as fixed quantum kernels for feature extraction from medical image patches (MedMNIST datasets). Only the classical CNN head is trained; the quantum front-end is frozen. The central research question is whether ZZ correlators (entanglement-based features) give measurable classification advantage over classical filters.

## Paper

We are writing a conference paper for **IEEE QAI 2026** (IEEE International Conference on Quantum Artificial Intelligence). The paper is in `docs/paper/paper.tex` (IEEEtran conference format). The bibliography is `docs/paper/references.bib`.

The paper describes and evaluates DAQCNN. Writing happens incrementally — the user will direct what to write section by section. Follow the scientific writing style in the global CLAUDE.md (funnel intro, stats-forward results, conservative hedging, active voice with "we"). Do not add content unless explicitly asked.

## Commands

### Running experiments

```bash
# Multi-seed validation (main entry point)
python experiments/robust_test_original_daqcnn.py --config configs/breast_mnist/original_daqcnn_best.yml

# Hyperparameter search
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original_daqcnn_best.yml \
    --search-config configs/breast_mnist/hp_search/analog_zz_single_kernel_search.yml \
    --n-trials 100

# Resume interrupted search
python experiments/hyperparameter_search.py ... --resume

# Validate top-k trials after search
python experiments/hyperparameter_search.py ... --validate-top-k 3 --validation-seeds 0 1 2 3 4

# Pre-compute and cache quantum features (edit PARAMETERS section in script or pass --config)
python experiments/create_quantum_dataset.py --config configs/breast_mnist/cache_generation/analog_zz.yml

# Feature probing ablation (RF/SVM on frozen quantum features)
python experiments/feature_probing_ablation.py --seeds 42 1 2 3 4

# CKA similarity between kernel topologies
python experiments/kernel_cka_similarity.py

# TUI to browse outputs/ and run inference
python -m src.tui.run_tui
```

### Install dependencies

```bash
pip install torch pennylane numpy pyyaml scikit-learn matplotlib optuna wandb medmnist
```

## Architecture

The pipeline has two stages:

**Stage 1 — Quantum feature extraction (frozen)**
- `src/physics/kernel_topologies.py` — Defines atom coordinate layouts for each topology (kings, horizontal, vertical, cross, ring, chain, star, grid for 3×3; kings/horizontal_chains/vertical_chains/diagonal_chains/block_2x2 for 4×4).
- `src/physics/hamiltonian.py` — Builds the Rydberg `ParametrizedHamiltonian` from atom coordinates. If `use_local_detuning=True` (analog mode), adds per-qubit detuning terms driven by pixel data.
- `src/physics/evolution.py` — Applies Hamiltonian evolution via Trotter steps (`ApproxTimeEvolution`) or the exact ODE solver (`qml.evolve`).
- `src/layers/daqk.py` — `DAQKLayer`: one or more quantum kernels (topologies) in parallel; outputs `n_kernels × measurements_per_kernel` features per patch. Measurements are `⟨Z_i⟩` for all qubits plus `⟨Z_i Z_j⟩` for all pairs when `include_correlators=True`.
- `src/layers/quantum_convolution.py` — `QuantumConv2d`: slides the kernel over the image using `nn.Unfold`, normalizes patches to `[0, π]`, feeds them to `DAQKLayer`, and reshapes outputs into a spatial feature map.

**Stage 2 — Classical CNN head (trained)**
- `src/models/daqcnn.py` — `DAQCNN`: wires `QuantumConv2d` to `build_classification_head` (Conv-BN-ReLU → MaxPool → Conv-ReLU → Dropout → Flatten → LazyLinear).
- `src/utils/training_utils.py` — `build_classification_head`, `resolve_device`.
- `src/models/daqcnn_training.py` — Training loop with early stopping, cosine LR scheduler, and optional quantum feature cache bypass.

**Cache system (critical for speed)**

Running quantum circuits per batch is prohibitively slow. The cache system pre-computes feature tensors once:
- `experiments/create_quantum_dataset.py` — Processes all splits and saves `.npz` + `.json` sidecar to `data/quantum_datasets/`.
- `src/utils/quantum_dataset_cache.py` — `find_cached_quantum_dataset` scans the directory for a parameter-matching cache; `load_cached_quantum_dataset` returns standard `DataLoader`s. The training loop calls these automatically and sets `model.bypass_quantum = True` when a cache is found.
- Cache filenames encode all relevant parameters (dataset, resolution, kernel size, stride, topologies, evolution time, ZZ flag, encoding mode).

**Classical baselines**
- `src/models/classical_baseline.py` — Fixed random or trainable classical conv filters, same CNN head.
- `src/models/vanilla_cnn.py` — Standard CNN (no quantum layer).

## Config Structure

Every experiment is a YAML file. Key fields:

```yaml
dataset:
  name: breast_mnist          # breast_mnist | pneumonia_mnist | tissue_mnist (+ _64 / _128 variants)
  color_space: GRAYSCALE      # GRAYSCALE | RGB | HSV

model:
  kernel_size: 3              # patch size (2, 3, or 4); qubits = kernel_size²
  stride: 3
  kernel_topology_names: [vertical]   # list → multi-kernel concatenation
  encoding_mode: digital      # digital (RY gates) | analog (local detuning)
  include_correlators: true   # adds ⟨ZZ⟩ features; 3×3 kernel: 9 Z + 36 ZZ = 45 features/kernel
  evolution_time: 2.5
  scaling_factor: 1.0

optim:
  lr: 0.001
  epochs: 100
  patience: 15                # early stopping

misc:
  seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # list = multi-seed run
```

Search configs (in `configs/*/hp_search/`) add a `search_space` block with `uniform`, `loguniform`, `int`, `categorical`, or `fixed` distributions for any dot-path config key.

## Key Design Decisions

- **Quantum kernels are never trained**. Only the classical head parameters have `requires_grad=True`. `diff_method=None` on every QNode enforces this.
- **Feature count formula**: for kernel_size `k` and `include_correlators=True`, each kernel produces `k² + C(k², 2)` features. With ZZ: 3×3→45, 4×4→136. Without: just `k²`.
- **Analog vs. digital encoding**: digital encodes pixels as `R_Y(pixel × π)` rotations; analog passes pixels directly as local detuning parameters `δ_i` in the Hamiltonian, so the data modulates the evolution itself.
- **Cache subset extraction**: if the requested kernel set is a proper subset of a cached superset, `load_cached_quantum_dataset` slices out the correct channels using the `channel_kernel_map` stored in the `.json` sidecar. Channel order follows the original generation order, not alphabetical.
- **Device performance**: for cache generation, `lightning.qubit` + `torch` is fastest for 3×3 kernels; `lightning.gpu` + `autograd` for 4×4. JAX is consistently slow.
- **Output directory**: each run writes to `outputs/run_<dataset>_<timestamp>/` (checkpoints, plots, metrics JSON).
