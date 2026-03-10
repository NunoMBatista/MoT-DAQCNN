# GS-MoE Pipeline: Complete Implementation Details

**Gumbel-Softmax Mixture-of-Experts for Quantum Kernel Routing**

This document provides a comprehensive technical explanation of the GS-MoE (Gumbel-Softmax Mixture-of-Experts) pipeline implemented in the MoT-DAQCNN project. The pipeline learns to route image patches to different quantum kernel topologies using differentiable discrete sampling, training end-to-end in a single phase.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Comparison with TS-MoE](#comparison-with-ts-moe)
3. [Architecture](#architecture)
   - [Router Network](#router-network)
   - [Gumbel Router Block](#gumbel-router-block)
   - [Channel Masking](#channel-masking)
   - [Classification Head](#classification-head)
4. [Gumbel-Softmax Mechanics](#gumbel-softmax-mechanics)
   - [The Gumbel-Softmax Trick](#the-gumbel-softmax-trick)
   - [Straight-Through Estimator](#straight-through-estimator)
   - [Temperature and the Soft→Hard Transition](#temperature-and-the-softhard-transition)
5. [Loss Function](#loss-function)
   - [Task Loss](#task-loss)
   - [Budget Loss](#budget-loss)
   - [Why Budget Loss Uses Soft Probs](#why-budget-loss-uses-soft-probs)
6. [Temperature Annealing](#temperature-annealing)
   - [Exponential Decay Schedule](#exponential-decay-schedule)
   - [Hard Switching](#hard-switching)
   - [Diagnosing Annealing Speed](#diagnosing-annealing-speed)
7. [Training Loop](#training-loop)
   - [Epoch Structure](#epoch-structure)
   - [Evaluation Wrapper](#evaluation-wrapper)
   - [Early Stopping](#early-stopping)
8. [Diagnostics and Outputs](#diagnostics-and-outputs)
   - [Result Dict Fields](#result-dict-fields)
   - [Per-Epoch Histories](#per-epoch-histories)
   - [Generated Plots](#generated-plots)
   - [Interpreting Diagnostics](#interpreting-diagnostics)
9. [Configuration Reference](#configuration-reference)
10. [Data Flow Summary](#data-flow-summary)
11. [Key Implementation Files](#key-implementation-files)
12. [Example Configuration](#example-configuration)
13. [Troubleshooting Guide](#troubleshooting-guide)

---

## Pipeline Overview

The GS-MoE pipeline trains a single model end-to-end in **one phase**:

```
Single Phase: GS-MoE Training
    - Loads pre-computed quantum features from cache (ALL kernels)
    - RouterNetwork (MLP) produces per-patch routing logits over K kernels
    - GumbelRouterBlock samples soft/hard masks via Gumbel-Softmax
    - Selected kernel channels are kept, others are zeroed
    - Classification head operates on masked features
    - Joint loss = task CE + sparsity_weight × budget penalty
    - Temperature τ anneals from τ_start to τ_end over training
    - Output: class predictions + learned routing policy
```

**Key Insight:** Gumbel-Softmax provides a differentiable approximation to discrete routing. At high temperature the model explores all routing combinations with smooth gradients; as temperature decreases, routing converges to near-discrete one-hot selections. This eliminates the need for the three-phase Teacher→Student→Classifier pipeline used by TS-MoE.

---

## Comparison with TS-MoE

Understanding the differences between GS-MoE and TS-MoE clarifies the design decisions:

| Aspect | TS-MoE | GS-MoE |
|--------|--------|--------|
| **Training phases** | 3 (Teacher → Student → Classifier) | 1 (end-to-end) |
| **Routing mechanism** | SE-block soft attention + argmax distillation | Gumbel-Softmax sampling |
| **Routing input** | Teacher sees quantum features; Student sees raw pixels | Router sees quantum features |
| **Differentiability** | Soft in Teacher, hard (non-differentiable) in Student | Always differentiable (STE for hard mode) |
| **Regularisation** | λ·entropy on alpha weights (push toward 0/1) | Budget loss on routing probs (L1 sparsity) |
| **Temperature** | KD temperature in Student (fixed) | Gumbel τ annealed over training (decays) |
| **Sparsity at inference** | One-hot via Student argmax | One-hot via hard Gumbel or argmax |
| **Normalization** | Per-group BatchNorm2d before gating | LayerNorm on flat feature vector in router |
| **Classification head** | Three separate heads (Teacher, Student routing, Final) | Single shared head |
| **Complexity** | Higher (3 models, 3 training phases) | Lower (1 model, 1 training phase) |

**When to prefer GS-MoE:** Simpler pipeline, faster iteration, single-phase training. Good for ablation studies and when you want end-to-end gradient flow from classification through routing.

**When to prefer TS-MoE:** When you need the Student to route from raw pixels (no quantum features at inference), when you want explicit control over each training phase, or when the three-phase decomposition helps with debugging.

---

## Architecture

### Architecture Overview

The GS-MoE model processes pre-computed quantum features from cached datasets:

```
Input: Quantum features (B, M*N, H, W)
    where M = number of kernels, N = channels per kernel (kernel_size²)

    ↓
Reshape to per-patch vectors: (B*H*W, M*N)

    ↓
RouterNetwork (MLP)
    - LayerNorm(M*N)
    - Linear → LayerNorm → GELU
    - Linear → GELU
    - Linear → raw logits (B*H*W, K)

    ↓
GumbelRouterBlock
    - Gumbel-Softmax sampling
    - Returns (mask, soft_probs)
    - mask shape: (B*H*W, K) — soft or hard depending on mode

    ↓
Channel Masking
    - Reshape mask to (B, K, H, W)
    - Multiply each kernel's N channels by its mask value
    - Result: (B, M*N, H, W) with unselected groups zeroed

    ↓
Classification Head (shared CNN, same as DAQCNN/Teacher)

    ↓
Output: (class_logits, soft_routing_probs)
```

Located in [gumbel_moe.py](../../src/models/gumbel_moe.py).

### Router Network

The router is a lightweight MLP that produces per-patch routing logits over K kernel topologies. It operates on flat quantum feature vectors — one decision per spatial position.

```python
class RouterNetwork(nn.Module):
    def __init__(self, total_channels, num_kernels, hidden_dim=None):
        # hidden_dim defaults to max(total_channels, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(total_channels),         # Scale-invariant input
            nn.Linear(total_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_kernels),  # Raw logits, no softmax
        )
```

**Key Design Decisions:**

1. **MLP, not CNN:** The router input is a flat feature vector per patch position `(B*H*W, M*N)`. There is no spatial structure within a single patch's feature vector to convolve over — the quantum features are already computed per patch. An MLP is the natural choice.

2. **LayerNorm on input:** Different kernel topologies (e.g., Kings graph vs. Horizontal chain) produce quantum features with inherently different output magnitudes due to their different interaction strengths (1/r⁶ coupling terms). Without normalization, the router could learn to route based on raw magnitude rather than feature content. `LayerNorm(total_channels)` makes routing decisions scale-invariant.

   This serves the same purpose as the Teacher's per-group `BatchNorm2d`, but is simpler and more appropriate for an MLP operating on flat vectors.

3. **No softmax in the router:** The router outputs raw logits. The Gumbel-Softmax block handles the probability conversion with temperature scaling. Keeping them separate allows the training loop to control temperature externally.

4. **Auto hidden dim:** When `router_hidden_dim` is null in the config, it defaults to `max(M*N, 32)`. For a 2-kernel setup with `kernel_size=3` (M\*N = 18), this gives hidden_dim = 32. For larger configs it scales with the input dimension.

### Gumbel Router Block

The Gumbel router block applies Gumbel-Softmax sampling to convert raw logits into routing masks:

```python
class GumbelRouterBlock(nn.Module):
    def __init__(self, temperature=1.0, hard=False):
        self.temperature = temperature  # Mutable — annealed by training loop
        self.hard = hard                # Mutable — switched by training loop

    def forward(self, logits):
        # Always compute soft sample (needed for budget loss)
        soft_mask = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)

        if self.hard:
            # Straight-through: one-hot forward, soft backward
            hard_mask = (soft_mask == soft_mask.max(dim=-1, keepdim=True)[0]).float()
            mask = hard_mask - soft_mask.detach() + soft_mask  # STE
        else:
            mask = soft_mask

        return mask, soft_mask  # mask for forward pass, soft_mask for budget loss
```

**Two return values:** The block always returns both the routing mask (used for channel masking in the forward pass) and the soft probabilities (used for the budget loss). This separation is critical — see [Why Budget Loss Uses Soft Probs](#why-budget-loss-uses-soft-probs).

**Externally mutable attributes:** Both `temperature` and `hard` are plain Python attributes. The training loop modifies them directly:
```python
model.router_block.temperature = tau   # Each epoch
model.router_block.hard = True         # After hard_after_epoch
```

### Channel Masking

After the router produces a mask `(B*H*W, K)`, GumbelMoE applies it to the quantum features:

```python
# Reshape mask to spatial form: (B, K, H, W)
mask_spatial = mask.reshape(B, H, W, self.num_kernels).permute(0, 3, 1, 2)

# For each kernel k, multiply its N channels by the mask value
N = self.channels_per_kernel
masked_x = x.clone()
for k in range(self.num_kernels):
    ch_start = k * N
    ch_end = (k + 1) * N
    # mask[:, k:k+1, :, :] broadcasts across N channels
    masked_x[:, ch_start:ch_end, :, :] = (
        x[:, ch_start:ch_end, :, :] * mask_spatial[:, k:k+1, :, :]
    )
```

**Channel ordering:** Channels are grouped by kernel in the order returned by `get_kernel_names()` (sorted by first channel index from the cache's `channel_kernel_map`). Kernel 0 owns channels `[0, N)`, kernel 1 owns `[N, 2N)`, etc. This ordering is inherited from the quantum dataset cache and preserved throughout.

**Soft masking behavior:**
- When `hard=False`: each kernel group is scaled by a value in (0, 1). Multiple kernels contribute with different weights. The classification head sees a smooth blend.
- When `hard=True`: exactly one kernel group retains its values, all others are zeroed. The STE ensures gradients still flow through the soft distribution.

### Classification Head

The GS-MoE model uses the exact same CNN classification head as the original DAQCNN and TS-MoE Teacher, via the shared factory in [training_utils.py](../../src/utils/training_utils.py):

```python
def build_classification_head(in_channels, num_classes, dropout=0.1, activation="relu"):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=2, stride=1, padding=0),
        nn.BatchNorm2d(64),
        nn.ReLU(),  # or GELU
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.LazyLinear(num_classes),
    )
```

The head receives the full `M*N`-channel tensor with unselected kernel groups zeroed. This is the same input dimensionality as the Teacher (which receives alpha-weighted features) and the original DAQCNN (which receives all channels unmasked). Using the same architecture enables fair comparison across all three approaches.

**Contrast with TS-MoE:** The TS-MoE pipeline trains three separate heads: one for the Teacher (soft-weighted input), one implicitly for Student routing, and one for the Final Classifier (sparse input). GS-MoE trains a single head end-to-end — the head learns jointly with the router what "zeroed channels" means.

---

## Gumbel-Softmax Mechanics

### The Gumbel-Softmax Trick

The fundamental problem: we want to select exactly one kernel per patch (a discrete, non-differentiable operation), but we need gradients to flow through the selection to train the router.

The Gumbel-Softmax trick (Jang et al., 2017; Maddison et al., 2017) provides a continuous relaxation. Given logits `π₁, ..., πK` from the router:

1. **Sample Gumbel noise:** `gᵢ ~ Gumbel(0, 1)` for each kernel i, computed as `gᵢ = -log(-log(uᵢ))` where `uᵢ ~ Uniform(0, 1)`.

2. **Add noise to logits:** `ẑᵢ = (πᵢ + gᵢ) / τ` where τ is the temperature.

3. **Apply softmax:** `yᵢ = exp(ẑᵢ) / Σⱼ exp(ẑⱼ)`

The result `y` is a K-dimensional probability vector that:
- **As τ → ∞**: approaches Uniform(1/K, ..., 1/K) regardless of logits.
- **As τ → 0**: approaches a one-hot vector (the argmax of `πᵢ + gᵢ`), equivalent to sampling from `Categorical(softmax(π))`.
- **For finite τ**: provides a smooth interpolation, fully differentiable with respect to the logits π.

### Straight-Through Estimator

For inference-like behavior during training, we want one-hot routing (exactly one kernel selected per patch) while maintaining gradient flow. The straight-through estimator (STE) achieves this:

```python
# Forward pass: one-hot (discrete)
hard_mask = one_hot(argmax(soft_mask))

# Backward pass: gradients flow through soft_mask
mask = hard_mask - soft_mask.detach() + soft_mask
```

The `- soft_mask.detach() + soft_mask` trick means:
- **Forward:** `mask = hard_mask` (since `soft_mask.detach()` and `soft_mask` cancel in value).
- **Backward:** `∂mask/∂soft_mask = 1` (since `hard_mask` and `soft_mask.detach()` have no gradient).

This is activated when `model.router_block.hard = True`, typically after a warm-up period of soft routing.

### Temperature and the Soft→Hard Transition

Temperature τ controls the "sharpness" of routing decisions:

| τ value | Routing behavior | Gradient quality | Use case |
|---------|-----------------|------------------|----------|
| τ = 5.0 | Nearly uniform — all kernels contribute equally | Smooth but uninformative | Never used (too high) |
| τ = 1.0 | Moderately soft — preferences visible but not decisive | Good exploration gradients | Start of training |
| τ = 0.5 | Fairly sharp — dominant kernel clearly preferred | Reasonable gradients | Mid-training |
| τ = 0.1 | Near one-hot — almost binary selection | Sparse, potentially noisy gradients | End of training |
| τ → 0 | True one-hot — identical to argmax | Zero gradients (without STE) | Inference only |

The annealing schedule (see [Temperature Annealing](#temperature-annealing)) gradually moves from exploration (τ = 1.0) to exploitation (τ = 0.1), giving the router time to discover good routing patterns before committing to discrete decisions.

---

## Loss Function

### Task Loss

Standard cross-entropy between predicted class logits and ground truth labels:

```python
loss_task = nn.CrossEntropyLoss()(class_logits, labels)
```

This is identical to the loss used by the original DAQCNN and the TS-MoE Teacher.

### Budget Loss

An L1 penalty on the total routing activation per patch:

```python
# routing_probs shape: (B*H*W, K) — soft Gumbel probabilities
loss_budget = torch.mean(torch.sum(routing_probs, dim=-1))
```

For a soft Gumbel sample, `sum(routing_probs, dim=-1)` is close to 1.0 per patch (softmax outputs sum to 1), but the Gumbel noise introduces slight variation. The budget loss provides a gentle pressure toward sparser routing.

**Combined loss:**

```python
total_loss = loss_task + sparsity_weight * loss_budget
```

Where `sparsity_weight` (default 0.01) is a fixed hyperparameter set in the config.

**Comparison with TS-MoE's entropy loss:** The Teacher uses `λ × normalized_entropy(alpha)` to push alpha weights toward decisive 0/1 values. The Gumbel budget loss serves a similar conceptual purpose (encourage sparsity) but with a different mechanism. The Gumbel pipeline does not need an explicit entropy loss because temperature annealing inherently drives routing toward discrete decisions — lower τ mechanically produces sharper distributions.

### Why Budget Loss Uses Soft Probs

This is a subtle but critical design choice. When `hard=True`, the forward pass uses a one-hot mask, but the budget loss must always use the **soft** Gumbel probabilities:

**Problem with hard probs:** The hard one-hot mask always sums to exactly 1.0 per patch (one kernel selected). This makes `mean(sum(one_hot, dim=-1)) = 1.0` — a constant. A constant loss has zero gradient and provides no training signal.

**Solution:** The soft probabilities reflect the router's confidence distribution and vary across patches. Even after switching to hard routing, the soft probs give a meaningful L1 signal that the optimizer can act on.

This is why `GumbelRouterBlock.forward()` returns both `(mask, soft_probs)`:
- `mask` → used for channel masking in the forward pass (hard or soft depending on mode).
- `soft_probs` → always soft, always used for budget loss computation.

---

## Temperature Annealing

### Exponential Decay Schedule

Temperature follows an exponential (geometric) decay from `τ_start` to `τ_end`:

```python
progress = min(epoch / max(tau_anneal_epochs, 1), 1.0)
tau = tau_start * (tau_end / tau_start) ** progress
tau = max(min(tau, tau_start), tau_end)  # Clamp for float precision
```

**Why exponential, not linear?** Linear decay spends too many epochs near τ = 1.0 where the distribution is nearly uniform and provides weak routing signal. Exponential decay gives equal "log-probability resolution" at each temperature scale:

```
Epoch   Linear τ    Exponential τ
  0     1.000       1.000
 20     0.775       0.562
 40     0.550       0.316
 60     0.325       0.178
 80     0.100       0.100
```

With linear decay, the router spends 60% of training above τ = 0.4 where routing is largely exploratory. With exponential decay, it transitions smoothly through the critical τ ∈ [0.2, 0.5] range where routing decisions sharpen.

### Hard Switching

After `hard_after_epoch` epochs, the training loop sets `model.router_block.hard = True`:

```python
if epoch >= hard_after_epoch and not model.router_block.hard:
    model.router_block.hard = True
```

From this point forward:
- **Forward pass:** One-hot routing (exactly one kernel per patch).
- **Backward pass:** Gradients flow through the soft distribution via STE.
- **Budget loss:** Still uses soft probs (unchanged).

**Typical configuration:** Set `hard_after_epoch = tau_anneal_epochs` (both default to 80). This means hard routing activates precisely when temperature reaches `τ_end`, creating a smooth transition from soft to hard routing.

### Diagnosing Annealing Speed

The `tau_entropy_curve_seed_{seed}.png` plot (τ on log-scale left axis, normalized entropy on right axis) is the primary diagnostic:

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Entropy tracks τ downward smoothly | Healthy — router becoming more decisive as τ decreases | None needed |
| Entropy stays high despite low τ | Router confused — logits are nearly uniform, τ can't help | Increase training epochs, check data quality |
| Entropy drops to 0 very early | Premature collapse — router locked onto one kernel | Slower annealing (increase `tau_anneal_epochs`), lower `sparsity_weight` |
| Entropy oscillates wildly | Training instability | Lower learning rate, increase `grad_clip` |
| Entropy drops then rises at hard switch | Hard-switch shock — STE gradients different from soft | Set `hard_after_epoch` earlier or equal to `tau_anneal_epochs` |

---

## Training Loop

### Epoch Structure

Each epoch in `run_gumbel_moe` follows this sequence:

```
1. Temperature update
    tau = tau_start * (tau_end / tau_start) ** (epoch / tau_anneal_epochs)
    model.router_block.temperature = tau

2. Hard-switch check
    if epoch >= hard_after_epoch:
        model.router_block.hard = True

3. Train one epoch
    For each batch:
        logits, routing_probs = model(features)
        total_loss, task_loss, budget_loss = model.compute_loss(...)
        total_loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

4. Validate
    val_loss, val_acc = evaluate(eval_wrapper, val_loader, device)

5. Record histories
    train_losses, val_losses, train_accs, val_accs
    task_losses_history, budget_losses_history
    routing_history (kernel_name -> fraction)
    tau_history, entropy_history

6. Generate per-epoch routing probability histogram

7. Early stopping check
    if val_loss < best_val_loss: save state
    else: increment patience counter

8. Scheduler step (if enabled)
```

Located in [gumbel_moe_training.py](../../src/models/gumbel_moe_training.py).

### Evaluation Wrapper

The shared `evaluate()` function in [evaluate.py](../../src/utils/evaluate.py) expects `model(x)` to return a single tensor (class logits). But `GumbelMoE.forward()` returns a tuple `(logits, routing_probs)`.

A thin wrapper resolves this:

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

This wrapper is instantiated once at the top of `run_gumbel_moe` and passed to every `evaluate()` call. It is a local training utility, not a persistent attribute of the model.

**Design rationale:** Rather than modifying the shared `evaluate()` function (which would break the contract for other architectures), we adapt the model's interface. This preserves the invariant that all architectures use the same evaluation code path.

### Early Stopping

Early stopping monitors validation loss with configurable patience:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0
elif patience is not None:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        break
```

After training completes (or stops early), the best model state is loaded before test evaluation. This means test metrics always reflect the model with the lowest validation loss, not the final epoch.

**Interaction with temperature annealing:** If early stopping triggers before `tau_anneal_epochs`, the router may not have reached its target temperature. This is usually fine — it means the model found a good routing policy at a higher temperature. Check `tau_history[-1]` in the result dict to see where training stopped.

---

## Diagnostics and Outputs

### Result Dict Fields

`run_gumbel_moe` returns a dict with all standard keys (compatible with `run_single_seed` for downstream aggregation) plus Gumbel-specific diagnostics:

**Standard keys (same as `run_single_seed`):**

| Key | Type | Description |
|-----|------|-------------|
| `seed` | `int` | Random seed for this run |
| `train_losses` | `list[float]` | Per-epoch total training loss (task + budget) |
| `val_losses` | `list[float]` | Per-epoch validation loss |
| `train_accs` | `list[float]` | Per-epoch training accuracy |
| `val_accs` | `list[float]` | Per-epoch validation accuracy |
| `test_loss` | `float` | Final test loss |
| `test_acc` | `float` | Final test accuracy |
| `test_auc` | `float` | Final test AUC |
| `test_f1` | `float` | Final test F1 score |
| `test_recall` | `float` | Final test recall |
| `test_probs` | `list` | Test set predicted probabilities (JSON-serializable) |
| `test_labels` | `list` | Test set ground truth labels (JSON-serializable) |
| `test_confusion_matrix` | `list` | Confusion matrix as nested list |
| `num_classes` | `int` | Number of output classes |
| `final_train_loss` | `float` | Last epoch's training loss |
| `final_val_loss` | `float` | Last epoch's validation loss |
| `final_train_acc` | `float` | Last epoch's training accuracy |
| `final_val_acc` | `float` | Last epoch's validation accuracy |
| `model_metadata` | `dict` | Parameter counts (`total_params`, `trainable_params`) |

**Gumbel-specific keys:**

| Key | Type | Description |
|-----|------|-------------|
| `routing_history` | `list[dict]` | Per-epoch routing ratio: each dict maps `kernel_name → fraction` of patches routed to that kernel. Fractions sum to 1.0 per epoch. |
| `tau_history` | `list[float]` | Per-epoch Gumbel temperature τ. Positive, decreasing. |
| `entropy_history` | `list[float]` | Per-epoch normalized router entropy in [0, 1]. 0 = perfectly decisive, 1 = uniform. |
| `task_losses_history` | `list[float]` | Per-epoch task (cross-entropy) loss component only. |
| `budget_losses_history` | `list[float]` | Per-epoch budget (sparsity) loss component only. |
| `kernel_names` | `list[str]` | Ordered kernel topology names matching the cache. |

### Per-Epoch Histories

**`routing_history`** — The most important diagnostic. Each entry is a dict like:
```python
{"kings": 0.612, "horizontal": 0.388}
```
This tells you what fraction of patches were routed to each kernel at the end of that epoch (based on `argmax` of the soft routing probabilities from the last training batch).

**`tau_history`** — Records the actual temperature used at each epoch. Essential for correlating routing behavior with temperature. If early stopping fires, the list will be shorter than `tau_anneal_epochs`.

**`entropy_history`** — Normalized router entropy computed as:
```python
H = -mean_over_patches(sum_over_kernels(p_k * log(p_k))) / log(K)
```
Where `p_k` is the soft routing probability for kernel k. Values:
- H ≈ 1.0: router assigns equal probability to all kernels (maximally uncertain).
- H ≈ 0.0: router assigns all probability to one kernel per patch (maximally decisive).

**`task_losses_history` and `budget_losses_history`** — Separate the two components of the total training loss. This lets you detect:
- Budget loss dominating task loss → reduce `sparsity_weight`.
- Task loss not decreasing → routing may be interfering with classification.
- Budget loss constant → routing probs are degenerate (check temperature).

### Generated Plots

For each seed, the pipeline generates:

| File | Description |
|------|-------------|
| `loss_curve_seed_{seed}.png` | Training/validation loss with CE and budget components as separate lines |
| `gumbel_routing_ratio_seed_{seed}.png` | Per-kernel routing fraction over epochs (one line per kernel) |
| `tau_entropy_curve_seed_{seed}.png` | Dual-axis: τ (log scale) and normalized entropy over epochs |
| `roc_curve_seed_{seed}.png` | ROC curve from test evaluation |
| `confusion_matrix_seed_{seed}.png` | Confusion matrix from test evaluation |
| `routing_prob_histograms_seed_{seed}/epoch_{NNN}_all_kernels.png` | Per-epoch histogram of soft routing probabilities for all kernels |

**Loss curve:** Blue = total training loss (task + budget), Red = validation loss, Yellow = CE task loss only, Green = budget loss only. The green line is labeled "Budget Loss" (not "λ·Entropy Loss" — that label is specific to the TS-MoE Teacher).

**Routing ratio plot:** One line per kernel showing what fraction of patches were routed to it at each epoch. Healthy behavior: lines converge to stable values (not 0% or 100% for any kernel). This is the same plot format as the Teacher's `teacher_routing_ratio_seed_{seed}.png`.

**Tau-entropy curve:** Left axis (blue, log scale) shows temperature τ; right axis (red dashed, linear 0–1) shows normalized entropy. You want to see entropy decrease as τ decreases, confirming that lower temperature drives more decisive routing.

**Routing probability histograms:** One per epoch, showing the distribution of soft routing probabilities for all kernels overlaid. Saved to a seed-specific `routing_prob_histograms_seed_{seed}/` subdirectory so that multi-seed runs don't overwrite each other. These are the Gumbel analog of the Teacher's alpha histograms (`alpha_histograms/epoch_NNN_all_kernels.png`).

### Interpreting Diagnostics

**Routing probability histogram shapes:**

| Shape | Meaning | Temperature regime |
|-------|---------|-------------------|
| Single peak at 1/K (uniform) | Router not discriminating — all kernels equal | High τ (early training) |
| Broad bell shape | Router developing preferences but not decisive | Medium τ |
| Bimodal (peaks near 0 and 1) | Decisive routing — patches committed to specific kernels | Low τ (late training) |
| Single spike at 1.0 for one kernel | Collapsed — router always picks the same kernel | Any (pathological) |

**Routing ratio plot patterns:**

| Pattern | Diagnosis |
|---------|-----------|
| Lines converge to different stable values | Healthy specialization — different kernels serve different patches |
| One line at ~100%, others at ~0% | Router collapse — one kernel dominates |
| Lines oscillate without converging | Unstable — learning rate too high or τ annealing too aggressive |
| Lines are flat at 1/K throughout | Router not learning — check gradients, learning rate |
| Sharp transition at `hard_after_epoch` | Hard-switch shock — may need gentler transition |

---

## Configuration Reference

All GS-MoE parameters are under the `gumbel:` section in YAML configs:

### Router

| Parameter | Default | Description |
|-----------|---------|-------------|
| `router_hidden_dim` | null (auto) | Hidden dimension for RouterNetwork MLP. `null` = `max(M*N, 32)`. |
| `sparsity_weight` | 0.01 | λ for the budget loss: `total = task + λ × budget`. |

### Temperature Annealing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_start` | 1.0 | Initial Gumbel temperature. Higher = softer routing. |
| `tau_end` | 0.1 | Final Gumbel temperature. Lower = near-discrete routing. |
| `tau_anneal_epochs` | 80 | Epochs over which τ decays exponentially from `tau_start` to `tau_end`. |

### Hard Switching

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hard_after_epoch` | 80 | Switch to hard routing (STE) at this epoch. Set equal to `tau_anneal_epochs` for a smooth transition. Set to 0 for hard routing from the start. |

### Shared Parameters (from `model:` and `optim:` sections)

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `model` | `architecture` | — | Must be `"gumbel"` to invoke GS-MoE pipeline |
| `model` | `num_classes` | auto | Number of output classes (inferred from dataset if omitted) |
| `model` | `kernel_topology_names` | — | List of kernel names, e.g. `["kings", "horizontal"]` |
| `model` | `dropout` | 0.1 | Dropout probability for classification head |
| `model` | `activation` | "relu" | Activation function: `"relu"` or `"gelu"` |
| `optim` | `lr` | 1e-3 | Learning rate for Adam optimizer |
| `optim` | `weight_decay` | 0.0 | L2 weight decay |
| `optim` | `epochs` | 1 | Maximum training epochs |
| `optim` | `grad_clip` | 0.0 | Max gradient norm (0 = no clipping) |
| `optim` | `patience` | null | Early stopping patience (null = disabled) |
| `optim` | `use_scheduler` | false | Enable CosineAnnealingLR scheduler |
| `optim` | `scheduler_T_max` | epochs | Cosine annealing period |
| `optim` | `scheduler_eta_min` | 0.0 | Minimum learning rate for cosine scheduler |

---

## Data Flow Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SINGLE PHASE: GS-MoE TRAINING                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Cached Quantum Features (B, M*N, H, W)                              │
│     Loaded from .npz cache (same cache used by TS-MoE and DAQCNN)    │
│                    ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ Reshape: (B, M*N, H, W) → (B*H*W, M*N)              │            │
│  │ Each spatial position becomes an independent sample   │            │
│  └──────────────────────────────────────────────────────┘            │
│                    ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ RouterNetwork (MLP):                                  │            │
│  │   LayerNorm(M*N) → Linear → LN → GELU →              │            │
│  │   Linear → GELU → Linear                              │            │
│  │   Output: logits (B*H*W, K)                           │            │
│  └──────────────────────────────────────────────────────┘            │
│                    ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ GumbelRouterBlock:                                    │            │
│  │   Add Gumbel noise to logits                          │            │
│  │   Apply softmax with temperature τ                    │            │
│  │   If hard: apply straight-through estimator           │            │
│  │   Output: mask (B*H*W, K), soft_probs (B*H*W, K)     │            │
│  └──────────────────────────────────────────────────────┘            │
│                    ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ Channel Masking:                                      │            │
│  │   Reshape mask → (B, K, H, W)                         │            │
│  │   For each kernel k:                                  │            │
│  │     channels[k*N : (k+1)*N] *= mask[:, k:k+1, :, :]  │            │
│  │   Output: masked features (B, M*N, H, W)              │            │
│  │   (Unselected kernel groups are zeroed)               │            │
│  └──────────────────────────────────────────────────────┘            │
│                    ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ Classification Head (shared CNN architecture):        │            │
│  │   Conv2d → BN → ReLU → MaxPool →                     │            │
│  │   Conv2d → ReLU → Dropout → Flatten → LazyLinear     │            │
│  │   Output: class logits (B, num_classes)               │            │
│  └──────────────────────────────────────────────────────┘            │
│                    ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ Loss = CE(logits, labels)                             │            │
│  │       + sparsity_weight × mean(sum(soft_probs))       │            │
│  └──────────────────────────────────────────────────────┘            │
│                    ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ Temperature Annealing (per epoch):                    │            │
│  │   τ = τ_start × (τ_end / τ_start)^(epoch/anneal_ep)  │            │
│  │                                                       │            │
│  │ Hard Switch (once, at hard_after_epoch):               │            │
│  │   model.router_block.hard = True                      │            │
│  └──────────────────────────────────────────────────────┘            │
│                    ↓                                                  │
│  Outputs:                                                            │
│    - Class predictions (test metrics)                                │
│    - Routing history, tau history, entropy history                    │
│    - Per-epoch routing probability histograms                        │
│    - Trained model checkpoint                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Implementation Files

| File | Description |
|------|-------------|
| [gumbel_moe.py](../../src/models/gumbel_moe.py) | Model: `RouterNetwork`, `GumbelRouterBlock`, `GumbelMoE` |
| [gumbel_moe_training.py](../../src/models/gumbel_moe_training.py) | Training: `_GumbelEvalWrapper`, `build_gumbel_moe_from_cfg`, `train_one_epoch_gumbel`, `run_gumbel_moe` |
| [training_utils.py](../../src/utils/training_utils.py) | Shared: `build_classification_head`, `resolve_device` |
| [plotting.py](../../src/utils/plotting.py) | Plots: `plot_routing_ratio_over_epochs`, `plot_routing_prob_histogram`, `plot_tau_entropy_curve`, `plot_loss_curves` |
| [evaluate.py](../../src/utils/evaluate.py) | Shared: `evaluate()` (used via `_GumbelEvalWrapper`) |
| [kernel_mapping.py](../../src/utils/kernel_mapping.py) | Shared: `build_kernel_to_channels_map`, `get_kernel_names`, etc. |
| [quantum_dataset_cache.py](../../src/utils/quantum_dataset_cache.py) | Shared: `find_cached_quantum_dataset`, `load_cached_quantum_dataset` |
| [robust_test_original_daqcnn.py](../../experiments/robust_test_original_daqcnn.py) | Experiment dispatcher (routes `architecture: "gumbel"` to `run_gumbel_moe`) |

**Config files:**

| File | Description |
|------|-------------|
| [breast_mnist_gumbel_3kern.yml](../../configs/breast_mnist_gumbel_3kern.yml) | Breast MNIST, 3 kernels |
| [pneumonia_mnist_gumbel_2kern.yml](../../configs/pneumonia_mnist_gumbel_2kern.yml) | Pneumonia MNIST, 2 kernels |
| [pneumonia_mnist_gumbel_3kern.yml](../../configs/pneumonia_mnist_gumbel_3kern.yml) | Pneumonia MNIST, 3 kernels |

**Test file:**

| File | Description |
|------|-------------|
| [test_gumbel_moe_e2e.py](../../tests/test_gumbel_moe_e2e.py) | 37 tests: unit, integration, and E2E smoke tests |

---

## Example Configuration

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
  # Router
  router_hidden_dim: null         # null = auto (max(M*N, 32))
  sparsity_weight: 0.01           # lambda for budget loss

  # Temperature annealing
  tau_start: 1.0                  # initial Gumbel temperature
  tau_end: 0.1                    # final Gumbel temperature
  tau_anneal_epochs: 80           # exponential decay over this many epochs

  # Hard switching
  hard_after_epoch: 80            # switch to STE after this epoch

optim:
  lr: 0.0001
  weight_decay: 1e-7
  epochs: 100
  grad_clip: 1.0
  patience: 15
  use_scheduler: false

misc:
  seed: [0, 1, 2]
  log_every: 50
```

---

## Troubleshooting Guide

### Router collapses to one kernel

All patches are routed to the same kernel (one line at 100% in the routing ratio plot).

- ✓ Reduce `sparsity_weight` (try 0.001 or 0.0)
- ✓ Increase `tau_start` (try 2.0 or 5.0) to force more exploration early
- ✓ Slow down annealing: increase `tau_anneal_epochs`
- ✓ Check that `router_hidden_dim` is large enough (try explicit 64 or 128)
- ✓ Verify quantum features are not degenerate (check cache data)

### Routing stays uniform (entropy ≈ 1.0 throughout)

The router never learns to discriminate between kernels.

- ✓ Decrease `tau_end` (try 0.05 or 0.01) for sharper final routing
- ✓ Increase `sparsity_weight` to add more routing pressure
- ✓ Train for more epochs
- ✓ Check that different kernels actually produce different features (compare channel statistics in cache)

### Task loss not decreasing

The classification head is not learning, even though routing looks reasonable.

- ✓ Increase learning rate
- ✓ Reduce `sparsity_weight` — budget loss may be dominating
- ✓ Check `task_losses_history` vs `budget_losses_history` — if budget >> task, rebalance
- ✓ Reduce `dropout`

### Budget loss stays constant

All soft routing probabilities are identical across patches.

- ✓ Check temperature — if τ is too high, all probs ≈ 1/K regardless of logits
- ✓ Verify router gradients are non-zero (`model.router.net[-1].weight.grad`)
- ✓ Ensure `LayerNorm` is not collapsing input variance

### Hard-switch causes accuracy drop

Performance degrades sharply at `hard_after_epoch`.

- ✓ Set `hard_after_epoch = tau_anneal_epochs` so the switch happens when τ is already low
- ✓ Try not using hard routing at all (set `hard_after_epoch` > `epochs`)
- ✓ The soft-to-hard transition is inherently lossy — a small drop is normal

### Training too slow

- ✓ Increase `dataset.batch_size` (32 → 64 or 128)
- ✓ Reduce `optim.epochs` — GS-MoE typically converges faster than TS-MoE since it's single-phase
- ✓ Use fewer kernels in `kernel_topology_names`
- ✓ Set `router_hidden_dim` to a smaller explicit value

### Missing cache error

```
RuntimeError: GumbelMoE requires a pre-computed quantum dataset cache...
```

- ✓ Generate the cache first: `python experiments/create_quantum_dataset.py --config <your_config.yml>`
- ✓ Verify `kernel_topology_names` in config matches the cache contents
- ✓ Check that `data_root` and dataset name match