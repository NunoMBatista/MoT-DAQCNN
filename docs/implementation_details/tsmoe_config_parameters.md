# TS-MoE Configuration Parameters Reference

Comprehensive list of all tunable parameters in the TS-MoE (Teacher-Student Mixture-of-Experts) pipeline configuration.---

## Dataset Configuration

| Parameter                  | Type   | Default       | Description                                                                                                          |
| -------------------------- | ------ | ------------- | -------------------------------------------------------------------------------------------------------------------- |
| `dataset.name`           | string | -             | Dataset identifier. Options:`pneumonia_mnist`, `breast_mnist`, `path_mnist`, `derma_mnist`, `tissue_mnist` |
| `dataset.data_root`      | string | `./data`    | Root directory for dataset storage                                                                                   |
| `dataset.batch_size`     | int    | 32            | Training batch size for Teacher and Final Classifier                                                                 |
| `dataset.num_workers`    | int    | 2             | Number of DataLoader workers                                                                                         |
| `dataset.download`       | bool   | true          | Whether to download dataset if not present                                                                           |
| `dataset.color_space`    | string | `GRAYSCALE` | Image color space:`GRAYSCALE` or `RGB`                                                                           |
| `dataset.train_fraction` | float  | 1.0           | Fraction of training data to use (0.0-1.0)                                                                           |

---

## Model Configuration

### Architecture

| Parameter              | Type   | Default    | Description                                                                        |
| ---------------------- | ------ | ---------- | ---------------------------------------------------------------------------------- |
| `model.architecture` | string | `TS-MoE` | Pipeline type:`"original"` for baseline DAQCNN, `"TS-MoE"` for Teacher-Student |
| `model.num_classes`  | int    | 2          | Number of output classes                                                           |

### Quantum Kernel Parameters

| Parameter                       | Type   | Default                                      | Description                                                               |
| ------------------------------- | ------ | -------------------------------------------- | ------------------------------------------------------------------------- |
| `model.kernel_size`           | int    | 3                                            | Spatial size of quantum kernel (k×k patches)                             |
| `model.stride`                | int    | 3                                            | Stride for patch extraction (typically = kernel_size for non-overlapping) |
| `model.kernel_topology_names` | list   | `["kings", "horizontal", "cross", "ring"]` | Qubit connectivity topologies to use                                      |
| `model.scaling_factor`        | float  | 1.0                                          | Pixel intensity scaling before encoding                                   |
| `model.evolution_time`        | float  | 2.5                                          | Hamiltonian evolution time `t` in exp(-iHt)                             |
| `model.mode`                  | string | `trotter`                                  | Evolution mode:`"trotter"` or `"exact"`                               |

**Available Topologies:**

- `kings` - King's graph (8-connected, like chess king moves)
- `horizontal` - Horizontal nearest-neighbor connections
- `vertical` - Vertical nearest-neighbor connections
- `cross` - Cross pattern (+) connections
- `ring` - Circular ring connectivity
- `u_shape` - U-shaped connectivity

### Classical Head

| Parameter                  | Type   | Default  | Description                                                   |
| -------------------------- | ------ | -------- | ------------------------------------------------------------- |
| `model.dropout`          | float  | 0.5      | Dropout rate in classification head                           |
| `model.activation`       | string | `relu` | Activation function:`"relu"`, `"gelu"`, `"silu"`        |
| `model.classical_device` | string | `auto` | Device for classical layers:`"auto"`, `"cuda"`, `"cpu"` |

### Quantum Device (PennyLane)

| Parameter                       | Type   | Default           | Description                          |
| ------------------------------- | ------ | ----------------- | ------------------------------------ |
| `model.quantum_device`        | string | `default.qubit` | PennyLane device name                |
| `model.quantum_device_kwargs` | dict   | `{}`            | Additional kwargs for quantum device |

---

## TS-MoE Parameters

### Phase 1: Teacher Training

The Teacher uses soft attention (Grouped Squeeze-and-Excitation) to route patches to different kernels.

#### Entropy Regularization

| Parameter                       | Type  | Default | Description                                      |
| ------------------------------- | ----- | ------- | ------------------------------------------------ |
| `ts_moe.lambda_start_epoch`   | int   | 0       | Epoch to start applying entropy regularization   |
| `ts_moe.lambda_entropy_start` | float | 0.0     | Initial entropy regularization weight            |
| `ts_moe.lambda_max`           | float | 0.1     | Maximum entropy/confidence regularization weight |
| `ts_moe.lambda_warmup_epochs` | int   | 50      | Number of epochs to linearly ramp up λ          |

**Effect of λ_entropy:**

- `λ = 0`: No regularization, routing may collapse to one kernel
- `λ > 0`: Encourages more uniform/diverse routing across kernels
- Too high: Forces uniform routing, defeating the purpose of MoE

#### Attention Gate Architecture

| Parameter                           | Type     | Default | Description                                                                  |
| ----------------------------------- | -------- | ------- | ---------------------------------------------------------------------------- |
| `ts_moe.attention_hidden_dim`     | int/null | 64      | Hidden dimension of attention gate MLP.`null` = auto (total_channels // 2) |
| `ts_moe.attention_gate_zero_init` | bool     | true    | Zero-initialize last gate layer so initial α = 1/M (uniform)                |

**Zero-init rationale:** Starting with uniform routing lets the network learn which kernel is best for each patch, rather than being biased by random initialization.

---

### Phase 2: Student Gatekeeper

The Student is a lightweight MLP that distills Teacher's routing decisions for efficient inference.

#### Model Architecture

| Parameter                      | Type | Default       | Description                        |
| ------------------------------ | ---- | ------------- | ---------------------------------- |
| `ts_moe.student_hidden_dims` | list | `[128, 64]` | Hidden layer sizes for Student MLP |

**Sizing guidelines:**

- Small patches (2×2): `[32, 16]` is often sufficient
- Larger patches or many kernels: `[128, 64]` or `[256, 128]`
- If Student under-fits (low train accuracy): increase dims
- If Student over-fits: decrease dims or add dropout

#### Training Schedule

| Parameter                     | Type  | Default | Description                         |
| ----------------------------- | ----- | ------- | ----------------------------------- |
| `ts_moe.student_epochs`     | int   | 30      | Maximum training epochs             |
| `ts_moe.student_lr`         | float | 1e-3    | Adam learning rate                  |
| `ts_moe.student_batch_size` | int   | 256     | Batch size for patch-level training |

#### Early Stopping

| Parameter                              | Type  | Default | Description                                |
| -------------------------------------- | ----- | ------- | ------------------------------------------ |
| `ts_moe.student_agreement_threshold` | float | 0.90    | Target Teacher-Student agreement (0-1)     |
| `ts_moe.student_agreement_patience`  | int   | 5       | Consecutive epochs above threshold to stop |

**Agreement metric:** Fraction of patches where `argmax(Student) == argmax(Teacher)`

#### Handling Label Imbalance

When one kernel dominates, the Student may collapse to always predicting that kernel.

| Parameter                           | Type | Default | Description                                              |
| ----------------------------------- | ---- | ------- | -------------------------------------------------------- |
| `ts_moe.student_weighted_ce`      | bool | true    | Use class-weighted cross-entropy (weight ∝ 1/frequency) |
| `ts_moe.student_balanced_sampler` | bool | false   | Use WeightedRandomSampler for balanced batches           |

**Recommendations:**

- Start with `weighted_ce: true`
- Add `balanced_sampler: true` if collapse persists
- Check routing distribution diagnostic printed at training start

#### Confidence Filtering

Filter out patches where Teacher is uncertain (near-uniform α).

| Parameter                       | Type  | Default | Description                               |
| ------------------------------- | ----- | ------- | ----------------------------------------- |
| `ts_moe.confidence_threshold` | float | 0.3     | Discard patches where max(α) < threshold |

**Guidelines:**

- `0.0`: Keep all patches (no filtering)
- `0.3`: Remove patches where best kernel has <30% weight
- `0.5`: More aggressive filtering, keep only confident decisions
- Check confidence histogram printed at training start

⚠️ **Warning:** Setting too high may discard too many patches, causing training to fail.

#### Knowledge Distillation

Instead of training on hard argmax labels only, also use Teacher's soft distribution.

| Parameter                         | Type  | Default | Description                                                         |
| --------------------------------- | ----- | ------- | ------------------------------------------------------------------- |
| `ts_moe.student_kd_alpha`       | float | 0.5     | Weight of KL distillation term (0 = hard CE only, 1 = soft KL only) |
| `ts_moe.student_kd_temperature` | float | 4.0     | Temperature for softening distributions                             |

**Temperature effect:**

- `T = 1`: Sharp distributions (near one-hot)
- `T = 4-8`: Softer targets, transfers inter-kernel relationships
- Higher T: More regularization, may lose sharp decisions

**Loss formula:**

```
Loss = (1 - α) * CE(Student, argmax(Teacher)) + α * T² * KL(soft_Student || soft_Teacher)
```

#### Feature Augmentation

Append extra features to raw patch pixels to help Student make better decisions.

| Parameter                                | Type | Default | Description                                                |
| ---------------------------------------- | ---- | ------- | ---------------------------------------------------------- |
| `ts_moe.student_features_stats`        | bool | true    | Add mean, std, min, max (+4 features)                      |
| `ts_moe.student_features_range_energy` | bool | true    | Add range, L2 energy, L1 norm (+3 features)                |
| `ts_moe.student_features_gradients`    | bool | true    | Add horizontal/vertical gradients, magnitude (+3 features) |

**Legacy shorthand:**

| Parameter                           | Type | Default | Description                                                  |
| ----------------------------------- | ---- | ------- | ------------------------------------------------------------ |
| `ts_moe.student_augment_features` | bool | true    | Enable all feature augmentation (overrides individual flags) |

---

### Phase 3: Final Classifier

The Final Classifier trains on sparse routed features (only selected kernel's channels per patch).

| Parameter                   | Type  | Default | Description                                         |
| --------------------------- | ----- | ------- | --------------------------------------------------- |
| `ts_moe.final_epochs`     | int   | 100     | Training epochs for Final Classifier                |
| `ts_moe.final_lr`         | float | 1e-3    | Learning rate for Final Classifier                  |
| `ts_moe.use_mask_channel` | bool  | false   | Append routing mask as extra channel (experimental) |

**Note:** Final Classifier uses the same `model.dropout` and `model.activation` as Teacher.

---

## Optimizer Configuration

These settings apply to **Teacher training**. Student and Final have their own `lr` settings above.

| Parameter                   | Type     | Default | Description                                  |
| --------------------------- | -------- | ------- | -------------------------------------------- |
| `optim.lr`                | float    | 1e-4    | Learning rate for Teacher                    |
| `optim.weight_decay`      | float    | 1e-4    | L2 regularization weight                     |
| `optim.epochs`            | int      | 100     | Total Teacher training epochs                |
| `optim.grad_clip`         | float    | 1.0     | Gradient clipping norm (null to disable)     |
| `optim.patience`          | int/null | null    | Early stopping patience (null = disabled)    |
| `optim.use_scheduler`     | bool     | true    | Use cosine annealing LR scheduler            |
| `optim.scheduler_T_max`   | int      | 100     | Cosine annealing period (usually = epochs)   |
| `optim.scheduler_eta_min` | float    | 0.0     | Minimum learning rate at end of cosine cycle |

**Alternative (shorthand):**

| Parameter                  | Type  | Default | Description                                  |
| -------------------------- | ----- | ------- | -------------------------------------------- |
| `training.epochs`        | int   | 50      | Teacher epochs (alternative to optim.epochs) |
| `training.learning_rate` | float | 1e-3    | Teacher LR (alternative to optim.lr)         |
| `teacher.epochs`         | int   | 50      | Explicit Teacher epochs (highest priority)   |

---

## Miscellaneous

| Parameter          | Type     | Default | Description                                  |
| ------------------ | -------- | ------- | -------------------------------------------- |
| `misc.seed`      | int/list | 42      | Random seed(s). If list, runs multiple seeds |
| `misc.log_every` | int      | 50      | Log training metrics every N batches         |

---

## Example Configurations

### Minimal Config (uses defaults)

```yaml
dataset:
  name: breast_mnist

model:
  architecture: TS-MoE
  num_classes: 2
  kernel_topology_names: ["kings", "horizontal", "cross"]

optim:
  epochs: 50
```

### High-Performance Config

```yaml
dataset:
  name: pneumonia_mnist
  batch_size: 64

model:
  architecture: TS-MoE
  kernel_size: 3
  stride: 3
  kernel_topology_names: ["kings", "horizontal", "cross", "ring"]
  evolution_time: 2.5
  dropout: 0.5

ts_moe:
  # Teacher
  lambda_max: 0.1
  lambda_warmup_epochs: 30
  attention_gate_zero_init: true
  
  # Student
  student_hidden_dims: [128, 64]
  student_epochs: 50
  confidence_threshold: 0.3
  student_kd_alpha: 0.5
  student_kd_temperature: 4.0
  student_weighted_ce: true
  
  # Final
  final_epochs: 100
  final_lr: 1e-3

optim:
  lr: 1e-4
  epochs: 100
  grad_clip: 1.0

misc:
  seed: [42, 123, 456]
```

### Quick Debug Config

```yaml
dataset:
  name: breast_mnist
  batch_size: 16

model:
  architecture: TS-MoE
  kernel_topology_names: ["kings", "horizontal"]

ts_moe:
  student_epochs: 5
  final_epochs: 10

optim:
  epochs: 10

misc:
  seed: 42
```

---

## Troubleshooting Guide

### Student collapses to one kernel

- ✓ Enable `student_weighted_ce: true`
- ✓ Enable `student_balanced_sampler: true`
- ✓ Increase `confidence_threshold` to filter ambiguous patches
- ✓ Increase `student_kd_alpha` to 0.7-0.8

### Student accuracy too low

- ✓ Increase `student_hidden_dims` (e.g., `[256, 128]`)
- ✓ Decrease `confidence_threshold` to keep more training data
- ✓ Enable feature augmentation flags

### No patches remain after confidence filtering

- ✓ Decrease `confidence_threshold` (try 0.2 or 0.1)
- ✓ Train Teacher longer to develop confident routing
- ✓ Increase `lambda_max` for more decisive routing

### Final Classifier underfits

- ✓ Increase `final_epochs`
- ✓ Decrease `model.dropout`
- ✓ Check if routing decisions are sensible (visualize)

### Training too slow

- ✓ Increase `dataset.batch_size`
- ✓ Reduce `ts_moe.student_epochs`
- ✓ Use fewer kernels in `kernel_topology_names`
