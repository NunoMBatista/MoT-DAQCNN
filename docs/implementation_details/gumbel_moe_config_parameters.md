# GS-MoE Configuration Parameters Reference

**Gumbel-Softmax Mixture-of-Experts — Complete Parameter Guide**

This document provides an exhaustive reference for every configuration parameter that affects the GS-MoE pipeline. Each parameter includes its type, default value, valid range, and practical guidance.

For architectural details and implementation explanations, see [gumbel_moe_details.md](./gumbel_moe_details.md).

---

## Table of Contents

1. [Dataset Configuration](#dataset-configuration)
2. [Model Configuration](#model-configuration)
   - [Architecture](#architecture)
   - [Quantum Kernel Parameters](#quantum-kernel-parameters)
   - [Classical Head](#classical-head)
   - [Quantum Device (PennyLane)](#quantum-device-pennylane)
3. [Gumbel Parameters](#gumbel-parameters)
   - [Router](#router)
   - [Temperature Annealing](#temperature-annealing)
   - [Hard Switching](#hard-switching)
4. [Optimizer Configuration](#optimizer-configuration)
5. [Miscellaneous](#miscellaneous)
6. [Example Configurations](#example-configurations)
   - [Minimal Config (uses defaults)](#minimal-config-uses-defaults)
   - [High-Performance Config](#high-performance-config)
   - [Quick Debug Config](#quick-debug-config)
   - [Aggressive Sparsity Config](#aggressive-sparsity-config)
   - [Conservative (No Hard Switch) Config](#conservative-no-hard-switch-config)
7. [Parameter Interactions](#parameter-interactions)
8. [Comparison with TS-MoE Parameters](#comparison-with-ts-moe-parameters)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## Dataset Configuration

Located under the `dataset:` key. These parameters are shared across all architectures (original DAQCNN, TS-MoE, GS-MoE).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Dataset name. Supported: `breast_mnist`, `pneumonia_mnist`. Must match the cached quantum dataset. |
| `data_root` | `str` | `./data` | Root directory for dataset storage. |
| `batch_size` | `int` | `32` | Training and evaluation batch size. Larger batches give more stable routing statistics per epoch but use more memory. |
| `num_workers` | `int` | `2` | Number of DataLoader worker processes. Set to 0 for debugging. |
| `download` | `bool` | `true` | Whether to download the dataset if not found locally. Only relevant for the raw dataset — the GS-MoE pipeline loads from the quantum cache. |
| `color_space` | `str` | `GRAYSCALE` | Input color space. All MedMNIST datasets in this project use `GRAYSCALE`. |

**GS-MoE note:** The dataset parameters must match the config used to generate the quantum dataset cache. If `name`, `kernel_size`, `stride`, or `kernel_topology_names` differ from the cache, `find_cached_quantum_dataset` will fail to locate the `.npz` file.

---

## Model Configuration

Located under the `model:` key. Some parameters control the quantum feature computation (used during cache generation), while others control the classical model.

### Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | `str` | — | **Must be `"gumbel"`** to invoke the GS-MoE pipeline. Other values: `"original"` (baseline DAQCNN), `"TS-MoE"` (Teacher-Student). |

### Quantum Kernel Parameters

These parameters define the quantum circuit structure. They must match the parameters used to generate the quantum dataset cache — the GS-MoE pipeline does not run quantum circuits itself, but uses these values to locate and validate the cache.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel_size` | `int` | `3` | Spatial size of the quantum kernel (patch size). Determines N = kernel_size² channels per kernel. Common values: 2 (4 channels/kernel) or 3 (9 channels/kernel). |
| `stride` | `int` | `3` | Stride of the patch extraction. Usually equal to `kernel_size` for non-overlapping patches. |
| `kernel_topology_names` | `list[str]` | — | Ordered list of quantum kernel topology names. Examples: `["kings", "horizontal"]`, `["kings", "horizontal", "cross"]`. These define M (number of kernels) and must match the cache. |
| `scaling_factor` | `float` | `1.0` | Scaling factor for the quantum Hamiltonian. Affects quantum feature magnitudes. |
| `evolution_time` | `float` | `2.5` | Time evolution parameter for the quantum circuit. |
| `mode` | `str` | `trotter` | Simulation mode for the quantum circuit. |

**Derived quantities:**
- `M` (number of kernels) = `len(kernel_topology_names)`
- `N` (channels per kernel) = `kernel_size²`
- `total_channels` = `M × N`
- Spatial dimensions: `H = W = image_size // stride` (e.g., 28 // 3 = 9 for MNIST with stride 3)

### Classical Head

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | `int` | auto | Number of output classes. If omitted, inferred from the dataset. |
| `dropout` | `float` | `0.1` | Dropout probability in the classification head. Applied after the second Conv2d and after Flatten. Higher values (0.3–0.5) recommended for small datasets. |
| `activation` | `str` | `"relu"` | Activation function for the classification head. Options: `"relu"`, `"gelu"`. GELU often gives slightly better results but is marginally slower. |

### Quantum Device (PennyLane)

These are only used during quantum cache generation, not during GS-MoE training. Included for completeness since they appear in config files.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantum_device` | `str` | `default.qubit` | PennyLane device for quantum simulation. |
| `quantum_device_kwargs` | `dict` | `{}` | Additional keyword arguments for the quantum device. |
| `classical_device` | `str` | `auto` | PyTorch device for classical computation. `"auto"` selects CUDA if available, else CPU. |

---

## Gumbel Parameters

Located under the `gumbel:` key. These are **unique to the GS-MoE pipeline** and have no equivalents in the original DAQCNN or TS-MoE configs.

### Router

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `router_hidden_dim` | `int` or `null` | `null` | ≥ 4, or `null` | Hidden dimension for the RouterNetwork MLP. When `null`, automatically set to `max(M*N, 32)`. Larger values give the router more capacity but risk overfitting on small datasets. |
| `sparsity_weight` | `float` | `0.01` | ≥ 0.0 | Weight λ for the budget loss: `total = task + λ × budget`. Set to 0.0 to disable the budget loss entirely (task loss only). |

**Guidance for `sparsity_weight`:**

| Value | Effect |
|-------|--------|
| `0.0` | No sparsity pressure. Router learns routing only from task loss gradients flowing through masked features. |
| `0.001` | Very gentle sparsity pressure. Unlikely to interfere with task learning. |
| `0.01` | Moderate sparsity pressure (default). Good starting point. |
| `0.05` | Strong sparsity pressure. May compete with task loss — monitor `task_losses_history` vs `budget_losses_history`. |
| `0.1+` | Very strong. Likely to dominate task loss and cause routing to collapse. Not recommended without careful tuning. |

**Guidance for `router_hidden_dim`:**

| Config | M×N | Auto dim | Manual suggestion |
|--------|-----|----------|-------------------|
| 2 kernels, kernel_size=2 | 8 | 32 | 32–64 |
| 2 kernels, kernel_size=3 | 18 | 32 | 32–64 |
| 3 kernels, kernel_size=3 | 27 | 32 | 32–128 |
| 4 kernels, kernel_size=3 | 36 | 36 | 64–128 |

The auto value works well for most cases. Only increase manually if you observe that routing is not learning (flat routing ratios).

### Temperature Annealing

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `tau_start` | `float` | `1.0` | > 0.0 | Initial Gumbel temperature. Higher = softer (more exploratory) routing. |
| `tau_end` | `float` | `0.1` | > 0.0, ≤ `tau_start` | Final Gumbel temperature. Lower = sharper (near-discrete) routing. |
| `tau_anneal_epochs` | `int` | `80` | ≥ 1 | Number of epochs over which τ decays exponentially from `tau_start` to `tau_end`. After this many epochs, τ stays at `tau_end`. |

**The annealing formula:**
```
progress = min(epoch / tau_anneal_epochs, 1.0)
tau = tau_start * (tau_end / tau_start) ** progress
```

This is exponential (geometric) decay, giving equal log-scale resolution at each step. See [gumbel_moe_details.md § Temperature Annealing](./gumbel_moe_details.md#temperature-annealing) for the full rationale.

**Guidance for temperature parameters:**

| Scenario | `tau_start` | `tau_end` | `tau_anneal_epochs` |
|----------|-------------|-----------|---------------------|
| Standard (default) | 1.0 | 0.1 | 80 |
| Slow exploration | 2.0 | 0.1 | 100 |
| Fast convergence | 1.0 | 0.05 | 40 |
| Near-hard from start | 0.5 | 0.05 | 20 |
| Debug (constant τ) | 1.0 | 1.0 | 1 |

**Temperature at specific epochs** (for default τ_start=1.0, τ_end=0.1, anneal=80):

| Epoch | τ |
|-------|---|
| 0 | 1.000 |
| 10 | 0.750 |
| 20 | 0.562 |
| 30 | 0.422 |
| 40 | 0.316 |
| 50 | 0.237 |
| 60 | 0.178 |
| 70 | 0.133 |
| 80+ | 0.100 |

### Hard Switching

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `hard_after_epoch` | `int` | `80` | ≥ 0 | Epoch at which the router switches from soft Gumbel-Softmax to hard routing with straight-through estimator (STE). |

**Common configurations:**

| Setting | Effect |
|---------|--------|
| `hard_after_epoch: 80` (= `tau_anneal_epochs`) | Default. Smooth transition: τ reaches minimum, then switch to hard. Routing is already near-discrete when the switch happens. |
| `hard_after_epoch: 0` | Hard routing from epoch 1. The STE provides gradients from the start. More aggressive — may cause early collapse. |
| `hard_after_epoch: 200` (> `epochs`) | Never switch to hard routing. Model trains entirely in soft mode. Useful for ablation or when soft routing gives better task performance. |
| `hard_after_epoch: 40` (< `tau_anneal_epochs`) | Switch to hard while τ is still relatively high. Forces discrete routing decisions earlier but maintains gradient flow via STE. |

---

## Optimizer Configuration

Located under the `optim:` key. These parameters are shared across all architectures.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `lr` | `float` | `1e-3` | > 0 | Learning rate for Adam optimizer. GS-MoE configs typically use 1e-4, which is lower than the default. The router and head are trained jointly, so a moderate LR prevents the router from oscillating. |
| `weight_decay` | `float` | `0.0` | ≥ 0 | L2 weight decay for Adam. Small values (1e-7 to 1e-5) can help prevent overfitting. |
| `epochs` | `int` | `1` | ≥ 1 | Maximum number of training epochs. Should be ≥ `tau_anneal_epochs` for the full temperature schedule to play out. |
| `grad_clip` | `float` | `0.0` | ≥ 0 | Maximum gradient norm. 0 = no clipping. Recommended: 1.0 for stability, especially with low τ where Gumbel gradients can spike. |
| `patience` | `int` or `null` | `null` | ≥ 1, or `null` | Early stopping patience. If validation loss doesn't improve for this many consecutive epochs, training stops. `null` = disabled. |
| `use_scheduler` | `bool` | `false` | — | Whether to use a CosineAnnealingLR scheduler on top of the base LR. |
| `scheduler_T_max` | `int` | `epochs` | ≥ 1 | Period for cosine annealing (in epochs). Usually set equal to `epochs`. |
| `scheduler_eta_min` | `float` | `0.0` | ≥ 0 | Minimum learning rate for cosine annealing. |

**Guidance for `lr`:**

The GS-MoE reference configs use `lr: 0.0001` (1e-4), which is 10× lower than the typical 1e-3 default for Adam. This is because:
1. The router and classifier are trained jointly — too high a LR causes the router to oscillate before the head can adapt.
2. Gumbel-Softmax gradients at low temperature can have high variance — a lower LR smooths this.
3. The combined loss (task + budget) means the effective gradient magnitude is larger than pure CE.

| LR | Typical use case |
|----|-----------------|
| 1e-3 | Aggressive. May work for very short runs (≤10 epochs) or when using a scheduler. |
| 1e-4 | Default for GS-MoE configs. Good balance of speed and stability. |
| 5e-5 | Conservative. For longer runs (200+ epochs) or when observing oscillations. |

**Guidance for `patience`:**

If using early stopping with temperature annealing, be aware that validation loss may temporarily increase during the soft→hard transition. Set patience high enough (≥ 10) to survive this transient, or disable early stopping entirely for the first run to observe the full training dynamics.

**Guidance for `grad_clip`:**

| Value | When to use |
|-------|-------------|
| 0.0 | No clipping. Fine for high τ, but risky at low τ. |
| 1.0 | Recommended default. Prevents gradient explosions during low-τ training. |
| 5.0 | Very permissive clipping. Only catches catastrophic spikes. |

---

## Miscellaneous

Located under the `misc:` key.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` or `list[int]` | — | Random seed(s) for reproducibility. If a list, one run per seed with aggregated results. |
| `log_every` | `int` | `50` | How often to log batch-level metrics (in batches). Not currently used by GS-MoE, but respected by the experiment dispatcher. |

---

## Example Configurations

### Minimal Config (uses defaults)

The simplest valid GS-MoE config. Most parameters fall back to defaults.

```yaml
dataset:
  name: breast_mnist
  batch_size: 32

model:
  architecture: "gumbel"
  num_classes: 2
  kernel_size: 3
  stride: 3
  kernel_topology_names: ["kings", "horizontal"]

optim:
  epochs: 100

misc:
  seed: 42
```

All `gumbel:` parameters use defaults (τ: 1.0→0.1 over 80 epochs, sparsity_weight=0.01, hard after epoch 80).

### High-Performance Config

Tuned for best task performance on Pneumonia MNIST with 3 kernels.

```yaml
dataset:
  name: pneumonia_mnist
  data_root: ./data
  batch_size: 32
  num_workers: 2
  download: true
  color_space: GRAYSCALE

model:
  architecture: "gumbel"
  num_classes: 2
  kernel_size: 3
  stride: 3
  kernel_topology_names: ["kings", "horizontal", "cross"]
  scaling_factor: 1.0
  evolution_time: 2.5
  mode: trotter
  dropout: 0.5
  activation: gelu
  quantum_device: default.qubit
  quantum_device_kwargs: {}
  classical_device: auto

gumbel:
  router_hidden_dim: null
  sparsity_weight: 0.01
  tau_start: 1.0
  tau_end: 0.1
  tau_anneal_epochs: 80
  hard_after_epoch: 80

optim:
  lr: 0.0001
  weight_decay: 1e-7
  epochs: 100
  grad_clip: 1.0
  patience: 15
  use_scheduler: false
  scheduler_T_max: 100
  scheduler_eta_min: 0.0

misc:
  seed: [0, 1, 2]
  log_every: 50
```

### Quick Debug Config

For rapid iteration and smoke testing. Runs 3 epochs with minimal computation.

```yaml
dataset:
  name: breast_mnist
  batch_size: 8

model:
  architecture: "gumbel"
  num_classes: 2
  kernel_topology_names: ["kings", "horizontal"]

gumbel:
  tau_anneal_epochs: 3
  hard_after_epoch: 3

optim:
  epochs: 3

misc:
  seed: 42
```

### Aggressive Sparsity Config

Higher sparsity pressure to force the router toward single-kernel selection. Useful for studying routing collapse dynamics.

```yaml
dataset:
  name: pneumonia_mnist
  batch_size: 32

model:
  architecture: "gumbel"
  num_classes: 2
  kernel_topology_names: ["kings", "horizontal"]
  dropout: 0.3
  activation: gelu

gumbel:
  router_hidden_dim: 64
  sparsity_weight: 0.05
  tau_start: 0.5
  tau_end: 0.05
  tau_anneal_epochs: 40
  hard_after_epoch: 40

optim:
  lr: 0.0001
  epochs: 60
  grad_clip: 1.0
  patience: 20

misc:
  seed: [0, 1, 2]
```

### Conservative (No Hard Switch) Config

Trains entirely in soft routing mode. Good for ablation studies comparing soft vs hard routing.

```yaml
dataset:
  name: pneumonia_mnist
  batch_size: 32

model:
  architecture: "gumbel"
  num_classes: 2
  kernel_topology_names: ["kings", "horizontal", "cross"]
  dropout: 0.5
  activation: gelu

gumbel:
  sparsity_weight: 0.01
  tau_start: 1.0
  tau_end: 0.1
  tau_anneal_epochs: 80
  hard_after_epoch: 999     # Never switches to hard (epochs < 999)

optim:
  lr: 0.0001
  epochs: 100
  grad_clip: 1.0

misc:
  seed: [0, 1, 2]
```

---

## Parameter Interactions

Several parameters interact in non-obvious ways. Understanding these interactions is critical for effective tuning.

### τ Schedule vs Training Length

```
epochs ≥ tau_anneal_epochs       → Full temperature schedule plays out.
epochs < tau_anneal_epochs       → Training ends before τ reaches tau_end.
                                   Final τ = tau_start * (tau_end/tau_start)^(epochs/anneal)
                                   This may be fine if early stopping finds a good model.
```

**Recommendation:** Set `epochs ≥ tau_anneal_epochs + 20` to give the model time to train at the final temperature after annealing completes.

### hard_after_epoch vs tau_anneal_epochs

| Relationship | Behavior |
|-------------|----------|
| `hard_after_epoch == tau_anneal_epochs` | Default. Smoothest transition — τ is at minimum when hard activates. |
| `hard_after_epoch < tau_anneal_epochs` | Hard routing while τ is still relatively high. STE gradients + Gumbel noise = more stochastic training. |
| `hard_after_epoch > tau_anneal_epochs` | Extra soft epochs at τ_end before switching to hard. Gives the model time to stabilize at low τ. |
| `hard_after_epoch > epochs` | Never hard. Entirely soft training. |

### sparsity_weight vs lr

The gradient from the budget loss scales as `sparsity_weight × ∂budget/∂θ`. If `sparsity_weight × lr` is too large, the budget term dominates and the router ignores task performance.

**Rule of thumb:** `sparsity_weight × lr ≤ 1e-5`. With the default `lr=1e-4` and `sparsity_weight=0.01`, this gives `1e-6` — well within the safe zone.

### patience vs tau_anneal_epochs

Early stopping with small patience can terminate training before the temperature schedule completes:

| Risk | Condition | Mitigation |
|------|-----------|------------|
| Stops during annealing | `patience < 10` and validation loss noisy | Use `patience ≥ 15` or disable early stopping |
| Stops at hard switch | Hard switch causes transient val loss spike | Set `patience > 5` or align hard switch with a validation improvement period |

### batch_size vs routing statistics

Routing statistics (`routing_history`) are computed from the last training batch of each epoch. Smaller batches give noisier per-epoch routing ratios. For stable diagnostics, use `batch_size ≥ 16`. For publication-quality routing ratio plots, use `batch_size ≥ 32`.

### grad_clip vs temperature

At very low τ (< 0.1), Gumbel-Softmax gradients can spike because the softmax output approaches a step function. Without gradient clipping, these spikes can destabilize training:

| τ range | Gradient behavior | Recommended grad_clip |
|---------|------------------|----------------------|
| τ > 0.5 | Smooth, moderate magnitude | 0.0 (no clipping needed) |
| 0.1 ≤ τ ≤ 0.5 | Occasionally spiky | 1.0–5.0 |
| τ < 0.1 | Frequently spiky, high variance | 1.0 (strongly recommended) |

---

## Comparison with TS-MoE Parameters

For users familiar with the TS-MoE pipeline, this table maps TS-MoE concepts to their GS-MoE equivalents:

| TS-MoE Parameter | TS-MoE Section | GS-MoE Equivalent | GS-MoE Section | Notes |
|------------------|----------------|-------------------|-----------------|-------|
| `lambda_max` | `ts_moe` | `sparsity_weight` | `gumbel` | Both control routing regularisation strength. TS-MoE ramps λ; GS-MoE uses a fixed weight. |
| `lambda_warmup_epochs` | `ts_moe` | — | — | No equivalent. GS-MoE uses fixed `sparsity_weight` (no warmup). Temperature annealing serves a similar "start gentle" purpose. |
| `lambda_entropy_start` | `ts_moe` | — | — | No equivalent. The Gumbel budget loss has no warmup. |
| `lambda_start_epoch` | `ts_moe` | — | — | No equivalent. |
| `attention_hidden_dim` | `ts_moe` | `router_hidden_dim` | `gumbel` | Both control the routing network's capacity. |
| `attention_gate_zero_init` | `ts_moe` | — | — | No equivalent. The RouterNetwork uses standard initialization. LayerNorm at the input serves a similar purpose (scale-invariant decisions). |
| `student_*` | `ts_moe` | — | — | No equivalents. GS-MoE has no Student phase. |
| `final_*` | `ts_moe` | — | — | No equivalents. GS-MoE has no separate Final Classifier phase. |
| — | — | `tau_start` | `gumbel` | Unique to GS-MoE. Controls initial routing exploration. |
| — | — | `tau_end` | `gumbel` | Unique to GS-MoE. Controls final routing sharpness. |
| — | — | `tau_anneal_epochs` | `gumbel` | Unique to GS-MoE. Controls annealing speed. |
| — | — | `hard_after_epoch` | `gumbel` | Unique to GS-MoE. Switches to straight-through estimator. |

**Key conceptual differences:**

1. **Regularisation approach:** TS-MoE uses entropy regularisation (pushes alpha weights toward 0/1). GS-MoE uses budget loss (L1 on routing probs) + temperature annealing (mechanically sharpens distributions). The temperature is a more powerful lever than λ — even with `sparsity_weight=0`, lowering τ forces decisive routing.

2. **Warmup strategy:** TS-MoE warms up λ over epochs (starting gentle, becoming strict). GS-MoE achieves the same effect through temperature annealing: high τ = gentle (exploratory), low τ = strict (decisive). There is no need for a separate warmup schedule.

3. **Number of hyperparameters:** GS-MoE has 6 unique parameters (`router_hidden_dim`, `sparsity_weight`, `tau_start`, `tau_end`, `tau_anneal_epochs`, `hard_after_epoch`). TS-MoE has 16+ across its three phases. The single-phase design of GS-MoE significantly reduces the tuning surface.

---

## Troubleshooting Guide

### Router collapses to one kernel

**Symptom:** Routing ratio plot shows one kernel at ~100%, others at ~0%. Entropy drops to 0 early.

| Fix | Parameter | Change |
|-----|-----------|--------|
| Reduce sparsity pressure | `sparsity_weight` | 0.01 → 0.001 or 0.0 |
| More exploration early | `tau_start` | 1.0 → 2.0 or 5.0 |
| Slower annealing | `tau_anneal_epochs` | 80 → 120 or 150 |
| Larger router | `router_hidden_dim` | null → 64 or 128 |
| Lower learning rate | `lr` | 1e-4 → 5e-5 |

### Routing stays uniform (H ≈ 1.0 throughout)

**Symptom:** Routing ratio lines are flat at 1/K. Entropy never decreases. Router is not learning to discriminate.

| Fix | Parameter | Change |
|-----|-----------|--------|
| Sharper final temperature | `tau_end` | 0.1 → 0.05 or 0.01 |
| More sparsity pressure | `sparsity_weight` | 0.01 → 0.05 |
| Train longer | `epochs` | 100 → 200 |
| Higher learning rate | `lr` | 1e-4 → 5e-4 |
| Check data | — | Verify kernels produce different features in the cache |

### Budget loss dominates task loss

**Symptom:** `budget_losses_history` >> `task_losses_history`. Task loss stagnates or increases. Test accuracy is poor.

| Fix | Parameter | Change |
|-----|-----------|--------|
| Reduce budget weight | `sparsity_weight` | 0.01 → 0.001 |
| Disable budget loss | `sparsity_weight` | → 0.0 |
| Increase learning rate | `lr` | (higher LR helps task loss catch up) |

### Accuracy drops at hard switch

**Symptom:** Validation accuracy drops sharply at `hard_after_epoch`, then may or may not recover.

| Fix | Parameter | Change |
|-----|-----------|--------|
| Align with annealing end | `hard_after_epoch` | Set equal to `tau_anneal_epochs` |
| Disable hard switch | `hard_after_epoch` | Set > `epochs` |
| Lower τ at switch time | `tau_end` | 0.1 → 0.05 (routing already near-discrete before switch) |
| More patience for recovery | `patience` | 15 → 25 or null |

### Training too slow

| Fix | Parameter | Change |
|-----|-----------|--------|
| Larger batches | `batch_size` | 32 → 64 or 128 |
| Fewer epochs | `epochs` | 100 → 50 (with proportionally shorter annealing) |
| Fewer kernels | `kernel_topology_names` | Remove least useful kernel |
| Smaller router | `router_hidden_dim` | null → 16 |

### Missing cache error

```
RuntimeError: GumbelMoE requires a pre-computed quantum dataset cache...
```

| Fix | Action |
|-----|--------|
| Generate cache | `python experiments/create_quantum_dataset.py --config <your_config.yml>` |
| Check config match | Ensure `name`, `kernel_size`, `stride`, `kernel_topology_names` match the cache |
| Check paths | Ensure `data_root` points to the correct directory |

### No patches remain after confidence filtering

This error belongs to the TS-MoE pipeline, not GS-MoE. GS-MoE does not use confidence filtering. If you see this error, check that `model.architecture` is set to `"gumbel"`, not `"TS-MoE"`.