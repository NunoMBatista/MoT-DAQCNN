# Gumbel-Softmax Mixture-of-Experts (GS-MoE) — Implementation Steps

## Overview & Goal

Implement a `GumbelMoE` architecture that:
- Accepts a config with `model.architecture: "gumbel"`.
- Is invoked automatically by `experiments/robust_test_original_daqcnn.py` via `run_single_seed`.
- Loads pre-computed cached quantum features (same `.npz` cache pipeline as `original` and `TS-MoE`).
- Uses a lightweight `RouterNetwork` + `GumbelRouterBlock` to select one kernel per patch.
- Trains end-to-end in a single phase with a combined task loss + budget (sparsity) loss.
- Returns the exact same result dict shape as `run_single_seed` so all downstream aggregation,
  plotting, and JSON saving works without modification.

The implementation touches **5 files to create** and **2 files to modify**.

---

## 0. Orientation — What Already Exists (Read Before Writing Anything)

Before writing a single line, re-read these files to avoid re-inventing things:

| File | Why relevant |
|---|---|
| `src/models/daqcnn.py` | Base model shape; `bypass_quantum`, `head` pattern |
| `src/models/daqcnn_training.py` | `run_single_seed` — the function to extend |
| `src/utils/training_utils.py` | `build_classification_head`, `resolve_device`, `entropy_loss`, `compute_lambda` |
| `src/utils/quantum_dataset_cache.py` | Cache loading pipeline; `find_cached_quantum_dataset`, `load_cached_quantum_dataset` |
| `src/utils/kernel_mapping.py` | `build_kernel_to_channels_map`, `get_kernel_names` |
| `src/utils/evaluate.py` | `evaluate` function signature — used verbatim |
| `src/layers/kernel_channel_attention_block.py` | Existing per-patch routing reference; understand channel group logic |
| `src/models/teacher_moe.py` | How `TeacherMoE` wraps the attention block — same wrapper pattern to follow |
| `configs/breast_mnist_ts_moe_3kern.yml` | TS-MoE config shape; the new `gumbel` config mirrors this structure |
| `experiments/robust_test_original_daqcnn.py` | The exact branching logic on `architecture`; new branch goes here |

---

## 1. Files to Create

### 1.1 `src/models/gumbel_moe.py` — The Model

**Class: `RouterNetwork`**

- A lightweight classical CNN operating on raw quantum feature patches.
- Input shape: `(Batch * H * W, M*N)` — one feature vector per patch (M kernels × N channels).
  - NOTE: unlike the spec doc which assumed raw pixel patches, here the router sees quantum
    features from the cache, not raw pixels. There are no spatial sub-patches to convolve
    over — the input is already a flat feature vector per patch position. Use an MLP, not a CNN.
- Architecture (MLP):
  - `Linear(M*N, hidden_dim)` + `LayerNorm(hidden_dim)` + `GELU`
  - `Linear(hidden_dim, hidden_dim // 2)` + `GELU`
  - `Linear(hidden_dim // 2, K)` — output logits, shape `(Batch * H * W, K)`
- `hidden_dim` should default to `max(M*N, 32)`.
- Do NOT apply softmax inside the router. Output raw logits.

**Class: `GumbelRouterBlock`**

- Init args: `temperature: float`, `hard: bool`.
- `forward(logits)`:
  1. Apply `torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)`.
  2. Return the mask tensor of shape `(Batch * H * W, K)`.
- Expose `self.temperature` as a plain attribute so the training loop can anneal it externally
  (i.e., `model.router_block.temperature = new_tau`).

**Class: `GumbelMoE`**

- Init args (all sourced from config):
  - `num_classes: int`
  - `total_channels: int` — M * N, the full channel count from the cache
  - `num_kernels: int` — K, number of topologies
  - `channels_per_kernel: int` — N = kernel_size^2
  - `kernel_names: list[str]` — ordered list of topology names
  - `temperature: float = 1.0`
  - `hard: bool = False` — start soft, anneal to hard
  - `sparsity_weight: float = 0.01`
  - `dropout: float = 0.1`
  - `activation: str = "relu"`
  - `router_hidden_dim: int = None` — passed to RouterNetwork; None = auto
- Components:
  - `self.router = RouterNetwork(total_channels, num_kernels, hidden_dim)`
  - `self.router_block = GumbelRouterBlock(temperature, hard)`
  - `self.head = build_classification_head(total_channels, num_classes, dropout, activation)`
    - Use the existing factory from `src/utils/training_utils.py`. The head sees the full
      `M*N` channel tensor (some channels zeroed by the mask) — same input size as `TeacherMoE`.
- `forward(x: Tensor) -> tuple[Tensor, Tensor]`:
  - `x` shape: `(B, M*N, H, W)` — cached quantum features.
  - Reshape to `(B*H*W, M*N)` for the router (each patch gets its own routing decision).
  - Get logits from `RouterNetwork`: `(B*H*W, K)`.
  - Get binary/soft mask from `GumbelRouterBlock`: `(B*H*W, K)`.
  - Reshape mask back to `(B, H, W, K)`, then permute to `(B, K, H, W)`.
  - For each kernel `k`, multiply all `N` channels of kernel `k` by `mask[:, k:k+1, :, :]`
    broadcast across those N channels.
  - The masked feature map has shape `(B, M*N, H, W)` — same as input, but some kernel
    groups are zeroed out by the mask.
  - Pass masked features through `self.head` to get logits `(B, num_classes)`.
  - Compute `routing_probs`: soft probabilities for the budget loss. When `hard=True`,
    Gumbel gives a straight-through estimator — use the soft (pre-hard) probabilities
    for the budget loss, NOT the hard one-hot mask. Store them as a second return value.
  - Return `(class_logits, routing_probs)` where `routing_probs` has shape `(B*H*W, K)`.
- `compute_loss(class_logits, routing_probs, labels, criterion)`:
  - A convenience method so the training loop stays clean.
  - `loss_task = criterion(class_logits, labels)`
  - `loss_budget = torch.mean(torch.sum(routing_probs, dim=-1))` — L1 penalty on total
    kernel activations per patch (same formulation as the spec doc).
  - `total_loss = loss_task + self.sparsity_weight * loss_budget`
  - Return `(total_loss, loss_task.item(), loss_budget.item())`
- Expose `self.last_routing_probs` (detached, shape `(B*H*W, K)`) after each forward pass
  for logging routing distribution histograms.

---

### 1.2 `src/models/gumbel_moe_training.py` — Training Loop

This mirrors `daqcnn_training.py` but for the GS-MoE model.

**Function: `build_gumbel_moe_from_cfg(cfg, cache_meta, num_classes, device)`**

- Reads `cfg["model"]` and `cfg.get("gumbel", {})` for GS-MoE specific params.
- Builds and returns a `GumbelMoE` instance moved to `device`.
- Extracts `total_channels`, `num_kernels`, `channels_per_kernel`, `kernel_names`
  from `cache_meta` using the existing `build_kernel_to_channels_map` /
  `get_kernel_names` / `get_channels_per_kernel` utilities from `src/utils/kernel_mapping.py`.

**Function: `train_one_epoch_gumbel(model, loader, optim, device, grad_clip, tau_schedule_fn, epoch)`**

- Standard training loop, same structure as `daqcnn_training.train_one_epoch`.
- Calls `model(imgs)` → `(logits, routing_probs)`.
- Calls `model.compute_loss(logits, routing_probs, labels, criterion)`.
- Backprop on `total_loss`.
- Logs and returns `(avg_total_loss, avg_task_loss, avg_budget_loss, avg_acc)`.
- Does NOT update temperature here — that is the caller's job after each epoch.

**Function: `run_gumbel_moe(cfg, seed, output_dir, verbose, set_seed_fn)`**

- Top-level function called from `robust_test_original_daqcnn.py`.
- Must return the **exact same dict keys** as `run_single_seed` in `daqcnn_training.py`:
  ```
  {
      "seed", "train_losses", "val_losses", "train_accs", "val_accs",
      "test_loss", "test_acc", "test_auc", "test_f1", "test_recall",
      "test_probs", "test_labels", "test_confusion_matrix", "num_classes",
      "final_train_loss", "final_val_loss", "final_train_acc", "final_val_acc",
      "model_metadata"
  }
  ```
- Steps inside `run_gumbel_moe`:
  1. Set seed via `set_seed_fn(seed)`.
  2. Resolve device via `resolve_device(cfg)`.
  3. Call `find_cached_quantum_dataset(cfg)`. If None, raise a clear `RuntimeError`
     explaining that GS-MoE requires a pre-computed cache and giving the command to
     generate one (point to `experiments/create_quantum_dataset.py`).
  4. Call `load_cached_quantum_dataset(cached_path, batch_size, num_workers,
     requested_kernels=model_cfg.get("kernel_topology_names"))`.
  5. Build model via `build_gumbel_moe_from_cfg(cfg, cache_meta, n_classes, device)`.
  6. Build optimizer: Adam with `lr`, `weight_decay` from `cfg["optim"]`.
  7. Optionally build `CosineAnnealingLR` scheduler (same pattern as `daqcnn_training`).
  8. **Temperature annealing**: read `tau_start`, `tau_end`, `tau_anneal_epochs` from
     `cfg.get("gumbel", {})`. Compute `tau` at each epoch as:
     ```
     progress = min(epoch / tau_anneal_epochs, 1.0)
     tau = tau_start * (tau_end / tau_start) ** progress   # exponential decay
     model.router_block.temperature = tau
     ```
  9. **Hard switching**: read `hard_after_epoch` from `cfg.get("gumbel", {})`. Once
     `epoch >= hard_after_epoch`, set `model.router_block.hard = True`. Log this once.
  10. Training loop over epochs:
      - `train_one_epoch_gumbel(...)` → `(total_loss, task_loss, budget_loss, acc)`
      - `evaluate(model_wrapper, val_loader, device, split_name="Val")` — see note below.
      - Append to `train_losses` (use `total_loss`, not `task_loss`), `val_losses`, etc.
      - Early stopping on `val_loss`, save best model state.
      - Log per-epoch: tau, hard flag, task_loss, budget_loss, total_loss, val_loss.
  11. Load best model state before test evaluation.
  12. Call `evaluate(model_wrapper, test_loader, device, compute_full_metrics=True)`.
  13. Save model checkpoint, loss plots, ROC curve, confusion matrix
      (same calls as `daqcnn_training.run_single_seed`).
  14. Return the result dict.

**Important — `evaluate` compatibility wrapper:**

`evaluate` in `src/utils/evaluate.py` calls `model(imgs)` and expects a single tensor
back (class logits). `GumbelMoE.forward` returns a tuple `(logits, routing_probs)`.
You must create a thin wrapper:

```python
class _GumbelEvalWrapper(nn.Module):
    """Wraps GumbelMoE so evaluate() only sees class logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits, _ = self.model(x)
        return logits
```

Instantiate it once at the top of `run_gumbel_moe` and pass it to every `evaluate` call.
It does NOT need to be a persistent attribute of `GumbelMoE` — it is a local training utility.

---

### 1.3 `configs/breast_mnist_gumbel_2kern.yml` — Reference Config

Create one reference config for `breast_mnist` with 2 kernels as a minimal working example.
It should contain every key the training code will read, with sensible defaults and comments.

Structure mirrors `breast_mnist_ts_moe_3kern.yml` but replaces the `ts_moe:` block with
a `gumbel:` block:

```yaml
dataset:
  name: breast_mnist
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
  kernel_topology_names: ["kings", "horizontal"]
  scaling_factor: 1.0
  evolution_time: 2.5
  mode: trotter
  dropout: 0.5
  activation: gelu
  quantum_device: default.qubit
  quantum_device_kwargs: {}
  classical_device: auto

gumbel:
  # --- Router ---
  router_hidden_dim: null         # null = auto (max(M*N, 32))
  sparsity_weight: 0.01           # lambda for budget loss

  # --- Temperature annealing ---
  tau_start: 1.0                  # initial Gumbel temperature
  tau_end: 0.1                    # final Gumbel temperature (near-argmax)
  tau_anneal_epochs: 80           # epochs over which to decay tau exponentially

  # --- Hard switching ---
  hard_after_epoch: 80            # switch to hard=True (straight-through) after this epoch

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

---

### 1.4 `configs/pneumonia_mnist_gumbel_2kern.yml` — Second Reference Config

Same structure as above but for `pneumonia_mnist`. Provide a 3-kernel variant too
(`pneumonia_mnist_gumbel_3kern.yml`) so there is a direct experimental comparison
point against `pneumonia_mnist_multi_seed_3kern.yml` and `pneumonia_mnist_ts_moe_3kern.yml`.

---

### 1.5 `src/models/__init__.py` — Update Exports (Minor)

Add `GumbelMoE` and `run_gumbel_moe` to the module's public API so other scripts can
import them cleanly:
```python
from src.models.gumbel_moe import GumbelMoE
from src.models.gumbel_moe_training import run_gumbel_moe
```

---

## 2. Files to Modify

### 2.1 `experiments/robust_test_original_daqcnn.py`

**Location:** The `if/elif` block that dispatches on `architecture` (around the
`run_ts_moe_pipeline` / `run_single_seed` call, inside the `for seed in seeds` loop).

**Change:** Add a new branch **before** the `else` fallback:

```python
elif architecture == "gumbel":
    from src.models.gumbel_moe_training import run_gumbel_moe
    result = run_gumbel_moe(
        cfg, seed, output_dir, verbose=(len(seeds) == 1), set_seed_fn=set_seed
    )
```

**Also update** the `prefix` variable used for the output directory name:
```python
# Before:
prefix = "ts_moe" if architecture == "TS-MoE" else "run"

# After:
if architecture == "TS-MoE":
    prefix = "ts_moe"
elif architecture == "gumbel":
    prefix = "gumbel"
else:
    prefix = "run"
```

No other changes to this file are needed. The result dict aggregation, JSON saving,
and plotting code are already architecture-agnostic (they read keys by name).

---

### 2.2 `src/utils/training_utils.py`

No structural changes needed. However, verify that `build_classification_head` works
with `total_channels = M*N` as the input channel count — it should, because it uses
`nn.Conv2d(in_channels, 64, kernel_size=2)` and `nn.LazyLinear`, which adapt to
arbitrary spatial sizes. This is already the case for `TeacherMoE`, so no change needed.

If during testing you find that the spatial dimensions after the two Conv2d + MaxPool
layers collapse to 0 (e.g., with very small feature maps from small images / large strides),
add an `AdaptiveAvgPool2d` fallback inside `build_classification_head` — but only make
this change if actually needed, as it would affect all architectures.

---

## 3. Key Invariants to Preserve

These are things the implementation must respect to stay compatible with the rest of
the codebase. Violating any of them will cause silent failures or broken outputs.

1. **Result dict keys**: `run_gumbel_moe` must return exactly the same top-level keys as
   `run_single_seed`. The aggregation loop in `robust_test_original_daqcnn.py` iterates
   over a fixed `metric_keys` list — any missing key is silently skipped, any extra key
   is ignored. Do not rename `test_acc`, `test_auc`, `test_f1`, `test_recall`, etc.

2. **Cache dependency**: GS-MoE only trains on cached quantum features. The router
   operates on quantum-encoded patch representations, not raw pixels. This is intentional
   and physically correct. If no cache is found, `run_gumbel_moe` must raise a clear,
   actionable error. Do NOT silently fall back to computing quantum circuits at training
   time — that would make the router see the full per-batch quantum computation at each
   epoch, which is prohibitively slow and defeats the purpose.

3. **`evaluate` signature**: `src/utils/evaluate.py:evaluate` expects `model(imgs)` to
   return a single tensor. Always use the `_GumbelEvalWrapper` for validation and test
   evaluation — never call `evaluate` directly with a raw `GumbelMoE` instance.

4. **Channel ordering**: When building the masked feature map, iterate over kernels in
   the same order as `cache_meta["kernel_topology_names"]` (which is sorted by first
   channel index via `get_kernel_names`). Do NOT use alphabetical order of kernel names.
   This is the same trap documented in `quantum_dataset_cache.py` for subset extraction.

5. **`model_metadata` key in result dict**: `robust_test_original_daqcnn.py` checks
   `if "model_metadata" in all_results[0]` to attach metadata to aggregate stats.
   The `model_metadata` dict must at minimum contain `total_params` and `trainable_params`
   (same fields as `get_model_metadata` in `daqcnn_training.py`). You can reuse
   `get_model_metadata` directly by passing the `GumbelMoE` instance.

6. **Seed reproducibility**: `set_seed_fn` is called once at the top of `run_gumbel_moe`
   before any tensor operations or model construction. Do not move or skip it.

7. **`bypass_quantum` flag**: `GumbelMoE` does not have a `QuantumConv2d` layer, so there
   is no `bypass_quantum` flag to set. When loading from cache, the model always operates
   on pre-computed features. Do not add this flag; it is specific to `DAQCNN`.

---

## 4. Gumbel-Softmax Mechanics — Implementation Notes

These clarify non-obvious choices to avoid common mistakes:

### Straight-through estimator and the hard flag

`F.gumbel_softmax(..., hard=True)` returns a one-hot tensor in the forward pass but
uses the soft Gumbel sample for the backward pass (straight-through gradient estimator).
This means:
- During forward: exactly one kernel per patch is selected (discrete, interpretable).
- During backward: gradients flow through the soft distribution to the router logits.

Start with `hard=False` for the first `hard_after_epoch` epochs. This gives smooth
gradients and prevents early collapse. Switch to `hard=True` once the router has learned
a useful routing policy.

### Budget loss — use soft probs, not the hard mask

When `hard=True`, `routing_probs` in `forward()` should be the **soft** Gumbel sample
(before the `hard` rounding), not the one-hot. This is because:
- The hard one-hot always sums to exactly 1 per patch (one kernel selected).
- `sum(one_hot) = 1` for every patch, making the budget loss constant and useless.
- The soft probs reflect the router's confidence distribution and give a meaningful
  L1 signal even after switching to hard routing.

To achieve this cleanly in `GumbelMoE.forward`: compute the soft sample with
`hard=False` always, use it for the budget loss, then apply the hard mask separately:
```python
soft_mask = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
if self.router_block.hard:
    # Straight-through: forward with hard mask, backward through soft_mask
    hard_mask = (soft_mask == soft_mask.max(dim=-1, keepdim=True)[0]).float()
    mask = hard_mask - soft_mask.detach() + soft_mask  # STE
else:
    mask = soft_mask
return class_logits, soft_mask   # always return soft probs for budget loss
```
Alternatively, simply let `GumbelRouterBlock` handle this internally and return both
`(mask_for_forward, soft_probs_for_loss)`.

### Temperature annealing schedule

Use exponential decay (geometric interpolation), not linear. Linear decay spends too
many epochs near `tau=1.0` where the distribution is nearly uniform. Exponential decay
gives equal "log-probability resolution" at each temperature scale:
```python
tau = tau_start * (tau_end / tau_start) ** (epoch / tau_anneal_epochs)
```
Clamp to `[tau_end, tau_start]` to prevent float precision issues near the boundary.

### Router input normalization

Before passing the feature vector `(B*H*W, M*N)` into `RouterNetwork`, apply
`LayerNorm` or at least check that features are not wildly different in scale across
kernels. The per-group `BatchNorm2d` in `KernelChannelAttentionBlock` is not present
here (the router is an MLP, not a conv). Add a `nn.LayerNorm(total_channels)` as the
first layer of `RouterNetwork` to keep routing decisions scale-invariant across kernel
topologies.

---

## 5. Diagnostics and Logging to Add

These are essential for understanding whether the approach is working. Add them to
`train_one_epoch_gumbel` or `run_gumbel_moe`:

1. **Routing distribution per epoch**: log the fraction of patches routed to each kernel
   (i.e., mean of the hard mask over the training set). Print a one-line summary like:
   ```
   Epoch 10 | tau=0.82 | routing=[kings: 61.2%, horizontal: 38.8%]
   ```
   A distribution that collapses to 100% for one kernel is a sign of router collapse.

2. **Router entropy**: compute `H = -sum(soft_probs * log(soft_probs + eps))` mean over
   patches. Log it each epoch. Healthy training sees entropy decrease gradually as the
   router becomes more decisive.

3. **Task vs budget loss breakdown**: log both `loss_task` and `loss_budget` separately
   each epoch (not just `total_loss`). This helps detect if the budget penalty is
   overwhelming the task loss or vice versa.

4. **Hard-switch announcement**: when `epoch == hard_after_epoch`, print a clearly
   visible line like `>>> Switching to hard routing (straight-through) at epoch N <<<`.

5. **Cache info**: at startup, print the number of kernels, channels per kernel,
   kernel names, and total feature channels loaded from cache. Confirm they match
   `model.kernel_names`.

---

## 6. Testing Checklist (Manual, Before Committing)

Run through each of these checks with a minimal config (small dataset, 2 epochs,
small `batch_size=8`) before running full experiments:

- [ ] `python experiments/robust_test_original_daqcnn.py --config configs/breast_mnist_gumbel_2kern.yml`
  exits without error for 2 epochs with 2 seeds.
- [ ] Output directory is created as `outputs/gumbel_TIMESTAMP/`.
- [ ] `outputs/gumbel_TIMESTAMP/individual_results.json` contains correct keys.
- [ ] `outputs/gumbel_TIMESTAMP/aggregate_metrics.json` contains mean/std/min/max for
  all standard metric keys.
- [ ] Loss, ROC, and confusion matrix plots are saved for each seed.
- [ ] With `hard=False` for all epochs: router mask is soft (non-binary). Verify with a
  `print(mask[:3])` debug line.
- [ ] With `hard_after_epoch=0` (force hard from epoch 0): router mask is one-hot.
- [ ] `sparsity_weight=0.0`: `total_loss == task_loss` exactly (budget term is off).
- [ ] Routing distribution log shows both kernels used (not collapsed) for the first few
  epochs when `tau=1.0`.
- [ ] Changing `seed` in config produces different routing distributions (not identical),
  confirming seed is respected.
- [ ] If cache file is missing: error message is clear, no stack trace from quantum layers.

---

## 7. File Summary

```
FILES TO CREATE:
  src/models/gumbel_moe.py                     ← RouterNetwork, GumbelRouterBlock, GumbelMoE
  src/models/gumbel_moe_training.py             ← build_gumbel_moe_from_cfg, train_one_epoch_gumbel, run_gumbel_moe
  configs/breast_mnist_gumbel_2kern.yml         ← reference config (2 kernels, breast_mnist)
  configs/pneumonia_mnist_gumbel_2kern.yml      ← reference config (2 kernels, pneumonia_mnist)
  configs/pneumonia_mnist_gumbel_3kern.yml      ← reference config (3 kernels, pneumonia_mnist)

FILES TO MODIFY:
  experiments/robust_test_original_daqcnn.py   ← add "gumbel" branch + prefix update
  src/models/__init__.py                        ← export GumbelMoE, run_gumbel_moe
```
