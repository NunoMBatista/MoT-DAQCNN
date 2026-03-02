# Teacher Training Details

## Overview

The Teacher is Phase 1 of the TS-MoE (Teacher-Student Mixture-of-Topologies) pipeline. It learns **which kernel topology is best for each image patch** by training on pre-computed quantum features from all available topologies simultaneously.

The Teacher never runs quantum circuits itself — it operates entirely on cached quantum feature maps. Its job is to learn a routing policy that the Student (Phase 2) will later mimic.

## Architecture

```
Cached quantum features (B, M*N, H, W)
    -> Kernel Channel Attention Block (soft routing weights per patch)
    -> Weighted features (B, M*N, H, W)
    -> Classification head (Conv-BN-ReLU -> MaxPool -> Conv-ReLU -> Dropout -> Flatten -> Linear)
    -> Logits (B, num_classes)
```

Where:
- **B** = batch size
- **M** = number of kernel topologies (e.g., 2 for `[kings, horizontal]`, 4 for `[kings, horizontal, vertical, u_shape]`)
- **N** = channels per kernel = `kernel_size²` (e.g., 4 for 2×2, 9 for 3×3)
- **H, W** = spatial dimensions of the feature map (depends on image size, kernel size, and stride)

## Kernel Channel Attention Block

This is the core routing mechanism. It replaced the original Squeeze-and-Excitation (SE) block.

### Why not SE?

The original SE block **squeezed** each kernel group's N channels down to a single scalar via `mean(dim=1)` before feeding that into the gating MLP. For a 3×3 kernel with 2 topologies, the gate saw only 2 numbers per patch — the average activation of each kernel group.

This had two problems:

1. **Information destruction**: Each of the N qubit expectation values (`⟨Z_i⟩`) captures a different spatial relationship determined by the kernel topology's atom geometry. Averaging them together destroys the activation *pattern* — the texture fingerprint — and keeps only the overall activation level. A patch where qubits 0,1,3 fire high but the rest are low looks identical (after averaging) to a patch where all qubits are medium.

2. **Scale bias**: The Rydberg Hamiltonian interaction strength goes as `1/r⁶`. The Kings graph has all atoms close together (strong interactions, larger raw output magnitudes). Sparser topologies like Horizontal or Vertical have some atom pairs at distance `FAR=100` (effectively zero interaction). The mean of a Kings group is systematically larger than the mean of a Horizontal group, so the gate was biased toward Kings before it even started learning.

### What the Kernel Channel Attention Block does differently

**No squeeze.** The gate MLP sees all M×N raw channels. For 2 kernels × 9 channels, that's 18 inputs instead of 2. The gate can now distinguish *which specific qubits* activated strongly and route based on the full quantum texture pattern.

**Per-group BatchNorm2d.** Before the channels enter the gate, each kernel group's N channels are independently normalized via `BatchNorm2d`. This puts all topologies on an equal scale so the gate makes its routing decision based on activation *patterns*, not raw magnitudes.

**The excitation still uses original (un-normalized) features.** Only the gate's *input* is normalized. After the gate produces its alpha weights, those weights multiply the original un-normalized quantum features. This way the classification head still sees the true quantum output magnitudes — the normalization only affects the routing decision, not the signal that flows to classification.

### Forward pass

```
Input x: (B, M*N, H, W)
    │
    ├─── Per-group BatchNorm2d ──→ x_normed: (B, M*N, H, W)
    │                                   │
    │                              Gate MLP (1x1 convs)
    │                                   │
    │                              softmax over M kernels
    │                                   │
    │                              alpha: (B, M, H, W)
    │                                   │
    └─── x * alpha_k (per group) ──→ out: (B, M*N, H, W)
```

### Config parameters

| Parameter | Config key | Default | Description |
|-----------|-----------|---------|-------------|
| Hidden dimension | `ts_moe.attention_hidden_dim` | `max(total_channels // 2, num_kernels * 2, 8)` | Hidden layer size of the 1×1 conv gating MLP. Controls how much capacity the gate has to learn routing patterns. |

## Loss Function

The Teacher trains with two loss terms:

```
loss = CE(logits, labels) + λ * entropy(alpha)
```

### Cross-entropy loss (CE)

Standard classification loss. This is the primary training signal — it tells the gate which routing decisions help classification accuracy. Gradients flow backward through the classification head, through the alpha-weighted features, and into the gate parameters.

### Entropy regularization

The entropy of the alpha distribution at each patch measures how "decisive" the routing is:
- **High entropy** (alpha ≈ uniform across kernels): the gate is indecisive, mixing all kernels equally. This defeats the purpose of routing.
- **Low entropy** (alpha ≈ one-hot): the gate commits to a single kernel per patch. This is what we want for the Student to have clear labels to learn from.

By adding `λ * mean_entropy(alpha)` to the loss, we penalize indecisive routing. The gate receives two gradient signals:
1. From CE loss (via weighted features): *"which routing helps classification"*
2. From entropy loss (via alpha directly): *"be more decisive"*

### Lambda annealing

Lambda starts at `lambda_entropy_start` (typically 0) and ramps linearly to `lambda_max` over `lambda_warmup_epochs`. This lets the network first explore freely (find which kernels are useful for which patches) and then gradually commit to decisive routing.

| Parameter | Config key | Default |
|-----------|-----------|---------|
| Start value | `ts_moe.lambda_entropy_start` | `0.0` |
| Max value | `ts_moe.lambda_max` | `0.1` |
| Warmup epochs | `ts_moe.lambda_warmup_epochs` | `epochs // 2` |

## Training Loop

Each epoch:

1. **Train**: For each batch of cached quantum features + class labels:
   - Forward pass through attention block + classification head
   - Compute CE loss + λ × entropy loss
   - Backward pass and optimizer step
   - Optional gradient clipping

2. **Validate**: Compute validation loss and accuracy using the standard `evaluate()` utility.

3. **Log routing stats**: On the validation set, compute what fraction of patches each kernel "wins" (has the highest alpha). This is tracked over epochs to monitor whether routing is diversifying or collapsing.

4. **Save alpha histograms**: Plot the distribution of alpha values per kernel on the validation set. This visualizes whether the gate is producing decisive (bimodal near 0/1) or mushy (clustered around 1/M) routing weights.

5. **Early stopping** (optional): If `patience` is set, stop training when validation loss hasn't improved for that many epochs.

6. **LR scheduling** (optional): Cosine annealing scheduler if `use_scheduler` is enabled.

## Outputs

After training, the Teacher saves:

- **Checkpoint** (`teacher_best_seed_{seed}.pt` and `teacher_final_seed_{seed}.pt`): Contains the model state dict, config, metadata (including `attention_hidden_dim`), kernel names, and num_classes. These checkpoints are loaded by Phase 2 (Student training) to generate routing labels.

- **Plots**: Loss curves, ROC curve, confusion matrix, routing ratio over epochs, and per-epoch alpha histograms.

- **Metadata**: Parameter counts, head structure, and attention block config are saved into the checkpoint's metadata dict so that the Student and Final Classifier phases can reconstruct the Teacher architecture exactly when loading from a checkpoint.

## What the Student receives

After Teacher training, Phase 2 loads the best Teacher checkpoint, runs it in eval mode on the cached quantum features, and extracts:

- **Hard routing labels**: `argmax(alpha, dim=1)` → shape `(N, H, W)`, values in `{0, ..., M-1}`. These are the per-patch kernel assignments the Student MLP learns to predict from raw pixels.

- **Soft labels**: The pre-softmax logits from the attention block → shape `(N, M, H, W)`. Used for knowledge distillation (KL divergence loss with temperature scaling).

- **Confidence**: `max(alpha, dim=1)` → shape `(N, H, W)`. Patches where the Teacher was uncertain (low max alpha) can be filtered out via `confidence_threshold` to avoid training the Student on noisy labels.