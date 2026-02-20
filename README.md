# Mixture-of-Topologies Digital-Analog Quantum Convolutional Neural Network (MoT-DAQCNN)

## Overview

MoT-DAQCNN extends the Digital-Analog Quantum Convolutional Neural Network with a **Teacher-Student Mixture-of-Topologies (TS-MoE)** pipeline that learns to route image patches to the most suitable quantum kernel topology, achieving comparable accuracy with significantly fewer quantum circuit evaluations.

## Architectures

### Original DAQCNN

The baseline architecture runs all quantum kernel topologies on every image patch and feeds the concatenated features to a classical CNN head.

```bash
python experiments/robust_test_original_daqcnn.py --config configs/pneumonia_mnist.yml
```

### TS-MoE Pipeline

The TS-MoE pipeline trains in three phases:

1. **Teacher** — Trains a Grouped SE Block on cached quantum features to learn soft routing weights (alpha) over kernel topologies, with entropy regularization pushing toward decisive (bimodal) routing.
2. **Student Gatekeeper** — A lightweight MLP (<5k params) distilled from the Teacher, predicting per-patch kernel assignments from raw image pixels alone.
3. **Final Classifier** — A new CNN head trained on sparse routed features (only the selected kernel's channels per patch).

```bash
# Step 1: Create cached quantum features (run once per dataset)
python experiments/create_quantum_dataset.py --config configs/pneumonia_mnist_ts_moe.yml

# Step 2: Run the full TS-MoE pipeline
python -m src.models.train_ts_moe --config configs/pneumonia_mnist_ts_moe.yml

# Or use the experiment runner (handles both architectures via config)
python experiments/robust_test_original_daqcnn.py --config configs/pneumonia_mnist_ts_moe.yml
```

### Switching Architectures

Set `model.architecture` in your config file:

```yaml
model:
  architecture: "TS-MoE"   # or "original" for baseline DAQCNN
```

See `configs/pneumonia_mnist_ts_moe.yml` for a complete TS-MoE config example with all tunable parameters (`ts_moe.lambda_max`, `ts_moe.student_hidden_dims`, etc.).

## Output Structure

| Architecture    | Output directory                            | Contents                                                                                                                                                       |
| --------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Original DAQCNN | `outputs/default_daqcnn_run_<timestamp>/` | Checkpoints, loss curves, ROC, confusion matrix                                                                                                                |
| TS-MoE          | `outputs/moe_run_<timestamp>/seed_N/`     | `teacher/`, `student/`, `final_classifier/` subdirs with checkpoints, plots, alpha histograms, routing confusion matrices, and `pipeline_summary.json` |

## TUI

Browse trained models (both DAQCNN and TS-MoE) and run inference interactively:

```bash
python -m src.tui.run_tui
```

TS-MoE runs are labeled with 🔀 in the model tree and display pipeline summary metrics (Teacher accuracy, Student agreement, speedup factor) in the info panel.

## Project Structure

```
src/
├── layers/
│   ├── grouped_se_block.py      # Patch-wise SE routing (Teacher core)
│   ├── daqk.py                  # Digital-analog quantum kernel
│   └── quantum_convolution.py   # Quantum convolution layer
├── models/
│   ├── daqcnn.py                # Original DAQCNN model
│   ├── teacher_moe.py           # Teacher with Grouped SE routing
│   ├── student_gatekeeper.py    # Lightweight routing MLP
│   ├── ts_moe_classification_head.py  # Final classifier on sparse features
│   ├── teacher_moe_training.py  # Teacher training loop
│   ├── student_training.py      # Student distillation loop
│   ├── ts_moe_classification_head_training.py  # Final classifier training
│   └── train_ts_moe.py          # Unified pipeline orchestrator
├── utils/
│   ├── kernel_mapping.py        # Kernel-to-channel grouping
│   ├── sparse_reconstruction.py # Sparse tensor building from routing
│   ├── losses.py                # Entropy loss + lambda annealing
│   ├── evaluate.py              # Metrics (accuracy, AUC, F1, recall)
│   ├── plotting.py              # All plot functions (histograms, ROC, etc.)
│   ├── quantum_dataset_cache.py # Cached quantum feature loading
│   └── model_cache_manager.py   # Checkpoint loading & output scanning
├── tui/
│   └── run_tui.py               # Interactive model testing interface
└── config.py                    # Project paths and constants
```


## Citations


```bibtex
@article{simenDigitalanalogQuantumConvolutional2024a,
  title = {Digital-Analog Quantum Convolutional Neural Networks for Image Classification},
  author = {Simen, Anton and Flores-Garrigos, Carlos and Hegade, Narendra N. and Montalban, Iraitz and Vives-Gilabert, Yolanda and Michon, Eric and Zhang, Qi and Solano, Enrique and Martín-Guerrero, José D.},
  journal = {Physical Review Research},
  volume = {6},
  number = {4},
  pages = {L042060},
  year = {2024},
  doi = {10.1103/PhysRevResearch.6.L042060}
}
