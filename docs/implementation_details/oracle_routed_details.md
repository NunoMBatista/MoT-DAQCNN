# Oracle-Routed Mixture-of-Topologies

## Overview

The oracle-routed pipeline learns a lightweight image-level router that predicts which quantum kernel topology will perform best for a given input image. At inference, only that single topology's quantum kernel is evaluated, achieving multi-kernel accuracy at single-kernel computational cost.

This pipeline replaces the TS-MoE approach (which suffered from Teacher mode collapse) by using per-topology classifier performance as ground truth for routing, rather than trying to discover routing through entropy regularization.

## Architecture

```
model:
    architecture: "oracle-routed"
```

## Pipeline Phases

### Phase 1: Oracle Label Generation

For each topology, a separate classification head is trained on that topology's cached quantum features. These per-topology classifiers are then used to generate oracle routing labels:

- For each sample, the topology whose classifier assigns the highest P(true_class) is selected as the oracle label.
- This is done for train, val, and test splits.
- Train labels are mildly overfit (classifiers trained on same data), but this is acceptable for an upper-bound training signal.

### Phase 2: Router Training

A lightweight CNN (`OracleRouter`) is trained to predict the oracle topology label from the raw input image:

- Input: raw image (e.g., 28x28 grayscale)
- Output: M-way classification (which topology to use)
- Architecture: 2 conv blocks (Conv-BN-ReLU-Pool) + FC head, ~15-20k parameters
- Trained with standard cross-entropy loss

### Phase 3: Final Classifier

The trained router predicts a topology for each sample. Sparse feature tensors are built (only the predicted topology's channels are non-zero), and a classification head is trained on these sparse features.

## Inference Flow

```
Raw Image --> OracleRouter --> topology_index (no quantum compute)
Raw Image --> Quantum Kernel (topology_index only) --> features (1/M cost)
features --> Sparse Tensor --> Final Classifier --> prediction
```

## Config Parameters

```yaml
oracle_router:
    # Phase 1 (per-topology classifiers)
    phase1_lr: 0.0001          # Override base optim.lr for Phase 1
    phase1_epochs: 100         # Override base optim.epochs for Phase 1

    # Phase 2 (image-level router)
    router_hidden_dims: [16, 32]   # Conv block channel dimensions
    router_fc_dim: 64              # FC hidden dimension
    router_dropout: 0.3            # Dropout in FC layers
    router_lr: 0.001               # Learning rate
    router_weight_decay: 0.00001   # Weight decay
    router_epochs: 30              # Max training epochs
    router_patience: 10            # Early stopping patience

    # Phase 3 (final classifier on sparse features)
    final_lr: 0.0001           # Override base optim.lr for Phase 3
    final_epochs: 100          # Override base optim.epochs for Phase 3
```

All Phase 1 and Phase 3 parameters default to the base `optim` section values if not specified. The `model` section parameters (dropout, activation, kernel_topology_names, etc.) are shared across all phases.

## Key Differences from TS-MoE

| Aspect | TS-MoE | Oracle-Routed |
|--------|--------|---------------|
| Routing signal | Teacher attention (CE + entropy loss) | Per-topology classifier performance |
| Routing level | Per-patch | Per-sample (image-level) |
| Router input | Raw pixel patches (3x3) | Full raw image |
| Failure mode | Teacher mode collapse | Router accuracy limited by oracle label quality |
| Phases | Teacher -> Student -> Final | Per-topo classifiers -> Router -> Final |

## Compatibility

Fully compatible with:
- `experiments/robust_test_original_daqcnn.py` (multi-seed evaluation)
- `experiments/hyperparameter_search.py` (HP search via Optuna)
- W&B logging (same result dict keys as other architectures)

## Files

- `src/models/oracle_router.py` - Router model (OracleRouter)
- `src/models/oracle_router_training.py` - Training pipeline (run_oracle_routed)
