# TS-MoE Pipeline: Complete Implementation Details

**Teacher-Student Mixture-of-Experts for Quantum Kernel Routing**

This document provides a comprehensive technical explanation of the TS-MoE (Teacher-Student Mixture-of-Experts) pipeline implemented in the MoT-DAQCNN project. The pipeline learns to route image patches to different quantum kernel topologies efficiently.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Phase 1: Teacher Training](#phase-1-teacher-training)
   - [Kernel Channel Attention Block](#kernel-channel-attention-block)
   - [Per-Group Batch Normalization](#per-group-batch-normalization)
   - [Alpha Weights and Soft Routing](#alpha-weights-and-soft-routing)
   - [Entropy Regularization and Lambda Scheduling](#entropy-regularization-and-lambda-scheduling)
   - [Classification Head](#classification-head)
3. [Phase 2: Student Gatekeeper Training](#phase-2-student-gatekeeper-training)
   - [Routing Label Generation](#routing-label-generation)
   - [Patch Feature Augmentation](#patch-feature-augmentation)
   - [Knowledge Distillation Loss](#knowledge-distillation-loss)
   - [Class Balancing Strategies](#class-balancing-strategies)
   - [Confidence Filtering](#confidence-filtering)
4. [Phase 3: Final Classifier Training](#phase-3-final-classifier-training)
   - [Sparse Tensor Reconstruction](#sparse-tensor-reconstruction)
   - [Group Normalization at Inference](#group-normalization-at-inference)
   - [Final Classification Head](#final-classification-head)
5. [Configuration Reference](#configuration-reference)
6. [Data Flow Summary](#data-flow-summary)

---

## Pipeline Overview

The TS-MoE pipeline consists of three sequential training phases:

```
Phase 1: Teacher Training
    - Runs ALL kernel topologies on every patch (brute force)
    - Learns soft routing weights via Kernel Channel Attention
    - Trains classification head on weighted features
    - Output: Per-patch soft alpha weights (B, M, H, W)

Phase 2: Student Gatekeeper Training
    - Lightweight MLP that predicts routing from raw patches
    - Distilled from Teacher's argmax routing decisions
    - Does NOT run quantum circuits
    - Output: Hard routing decisions (one kernel per patch)

Phase 3: Final Classifier Training
    - Routes patches through Student → sparse tensor
    - Trains new classification head on sparse features
    - Learns that zeros mean "not selected", not "low signal"
    - Output: Final diagnostic predictions
```

**Key Insight:** The Teacher runs all quantum kernels to learn optimal routing, the Student mimics this routing cheaply, and the Final Classifier interprets sparse routed features for diagnosis.

---

## Phase 1: Teacher Training

### Architecture Overview

The Teacher model processes pre-computed quantum features from cached datasets:

```
Input: Quantum features (B, M*N, H, W)
    where M = number of kernels, N = channels per kernel (kernel_size²)

    ↓
Kernel Channel Attention Block
    - Per-group BatchNorm2d
    - Gating MLP (1×1 convolutions)
    - Softmax → alpha weights
    - Channel reweighting

    ↓
Weighted features (B, M*N, H, W)

    ↓
Classification Head (CNN)

    ↓
Output: Logits (B, num_classes)
```

### Kernel Channel Attention Block

The attention block is the "Judge" that decides how much each kernel topology contributes at every spatial position. Located in [kernel_channel_attention_block.py](../../src/layers/kernel_channel_attention_block.py).

**Key Design Decisions:**

1. **No Squeeze Operation:** Unlike traditional SE blocks, this block does NOT average-pool each kernel group down to a single scalar. The gating MLP sees ALL raw channels (the complete quantum texture fingerprint), enabling richer routing decisions.

2. **Patch-wise Operation:** Each spatial position (i, j) gets its own M routing weights. Different patches in the same image can route to different kernels.

3. **1×1 Convolutions:** The gating MLP uses 1×1 convolutions, which is equivalent to a per-pixel MLP but processes all spatial positions in parallel.

```python
# Gating network structure
self.gate = nn.Sequential(
    nn.Conv2d(total_channels, hidden_dim, kernel_size=1, bias=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(hidden_dim, num_kernels, kernel_size=1, bias=True),
)
```

**Gate Zero Initialization:** When `gate_zero_init=True`, the last Conv2d layer is initialized to zeros. This means initial logits are all zero, and softmax produces uniform alpha weights (1/M for each kernel). Symmetry is then broken by gradients during training rather than random initialization.

### Per-Group Batch Normalization

**Problem:** Different kernel topologies (e.g., Kings graph vs. Horizontal chain) produce quantum features with inherently different output magnitudes due to their different interaction strengths (1/r⁶ coupling terms).

**Solution:** Each kernel group is normalized independently via its own BatchNorm2d layer BEFORE the gating decision:

```python
self.group_norms = nn.ModuleList([
    nn.BatchNorm2d(channels_per_kernel) 
    for _ in range(num_kernels)
])

# During forward pass:
groups = x.split(channels_per_kernel, dim=1)  # Split by kernel
normed_groups = [self.group_norms[k](g) for k, g in enumerate(groups)]
x_normed = torch.cat(normed_groups, dim=1)  # Reassemble
```

**Critical:** The classification head receives the NORMALIZED features (`x_normed * alpha`), not raw features. This ensures that kernel contributions are purely determined by the learned alpha weights, giving the gate clean gradient feedback.

If raw features were used after normalization-based gating, a kernel with large raw magnitudes could dominate the head's input even with a low alpha weight, causing oscillating gradients.

### Alpha Weights and Soft Routing

The attention block produces alpha weights via softmax over kernel dimension:

```python
alpha_logits = self.gate(x_normed)      # (B, M, H, W) - pre-softmax
alpha = torch.softmax(alpha_logits, dim=1)  # Normalize across kernels
```

**Three versions are stored after each forward pass:**

| Variable | Description | Use Case |
|----------|-------------|----------|
| `last_alpha_live` | Gradient-attached tensor | Entropy loss backpropagation |
| `last_alpha` | Detached copy | Logging, histograms, distillation labels |
| `last_logits` | Pre-softmax logits | Knowledge distillation (temperature scaling) |

**Channel Reweighting:**

```python
N = channels_per_kernel
alpha_expanded = alpha.repeat_interleave(N, dim=1)  # (B, M*N, H, W)
out = x_normed * alpha_expanded
```

Each scalar alpha weight is repeated N times so it broadcasts over all channels in its kernel group.

### Entropy Regularization and Lambda Scheduling

**Goal:** Encourage the gate to make decisive routing decisions (one kernel per patch) rather than uniform mixing.

**Normalized Entropy Loss:**

```python
def entropy_loss(alpha):
    M = alpha.shape[1]  # Number of kernels
    eps = 1e-8
    # Entropy per patch: -sum_k alpha_k * log(alpha_k)
    h = -torch.sum(alpha * torch.log(alpha + eps), dim=1)  # (B, H, W)
    # Normalize by max entropy so value is in [0, 1]
    max_entropy = math.log(M) if M > 1 else 1.0
    return h.mean() / max_entropy
```

**Normalization Rationale:** Raw entropy ranges from 0 to log(M). Without normalization, the same `lambda_max` would produce different gradient magnitudes for 2-kernel vs 4-kernel configs. Dividing by log(M) makes the loss consistently interpretable:
- 0.0 = perfectly decisive (one-hot alpha)
- 1.0 = maximally indecisive (uniform alpha)

**Lambda Scheduling:**

```python
def compute_lambda(epoch, warmup_epochs, lambda_max, lambda_start=0.0, start_epoch=0):
```

The entropy regularization weight ramps up over training:

```
Epochs 0 to start_epoch:              lambda = lambda_start (typically 0)
Epochs start_epoch to warmup_epochs:  linear ramp from lambda_start to lambda_max
Epochs after warmup:                  lambda = lambda_max
```

**Config parameters:**
- `lambda_entropy_start`: Starting lambda value (default 0.0)
- `lambda_max`: Maximum lambda value (default 0.1)
- `lambda_warmup_epochs`: Number of epochs for linear ramp
- `lambda_start_epoch`: Epoch at which lambda starts growing

**Training Loss:**

```python
total_loss = cross_entropy_loss + lambda * entropy_loss(alpha_live)
```

The gate receives two gradient signals:
1. **From CE loss (via weighted features):** "Which routing helps classification?"
2. **From entropy loss (via alpha directly):** "Be more decisive!"

### Classification Head

The Teacher uses the same CNN classification head as the original DAQCNN, defined in [training_utils.py](../../src/utils/training_utils.py):

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
        nn.LazyLinear(num_classes),  # Infers input size on first forward
    )
```

This shared factory ensures Teacher, original DAQCNN, and Final Classifier use identical architectures for fair comparison.

---

## Phase 2: Student Gatekeeper Training

The Student is a lightweight MLP that predicts routing decisions from raw image patches WITHOUT running any quantum circuits. Located in [student_gatekeeper.py](../../src/models/student_gatekeeper.py).

### Architecture

```python
class StudentGatekeeper(nn.Module):
    def __init__(self, patch_dim, num_kernels, hidden_dims=(32, 16)):
        # patch_dim = C * kernel_size²  (e.g., 1 * 3² = 9 for grayscale 3×3)
        
        layers = []
        in_dim = patch_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_kernels))
        self.net = nn.Sequential(*layers)
```

**Why MLP instead of CNN?** For patch sizes used in this project (2×2 or 3×3), convolutional kernels would cover the entire input, making convolutions degenerate. An MLP operating on flattened pixels is more appropriate.

### Routing Label Generation

The frozen Teacher generates training labels for the Student:

```python
@torch.no_grad()
def generate_routing_labels(teacher, loader, device):
    """
    For each sample:
        1. Forward through Teacher to get alpha weights
        2. argmax(alpha, dim=1) → hard routing labels
        3. max(alpha, dim=1) → confidence scores
        4. Store raw logits for knowledge distillation
    
    Returns:
        routing_labels: (N, H, W) in {0, ..., M-1}
        routing_logits: (N, M, H, W) - PRE-SOFTMAX (for KD temperature)
        confidence: (N, H, W) - max alpha value per patch
    """
```

**Important:** The returned `routing_logits` are pre-softmax. The KD loss applies softmax with temperature scaling internally, so passing logits preserves that flexibility.

### Patch Feature Augmentation

Very small patches (e.g., 2×2 = 4 pixels) carry limited raw information. Optional feature augmentation appends extra scalars:

| Feature Group | Features Added | Description |
|---------------|----------------|-------------|
| `stats` | mean, std, min, max | Basic intensity statistics (+4) |
| `range_energy` | range, L2 energy, L1 norm | Contrast and activation levels (+3) |
| `gradients` | h_grad, v_grad, magnitude | Spatial edge information (+3) |

**Gradient Features:**

```python
# Reshape to (N, ks, ks)
spatial = flat_patches[:, :ks*ks].reshape(-1, ks, ks)

# Horizontal gradient: column differences
h_grad = (spatial[:, :, 1:] - spatial[:, :, :-1]).mean(dim=(1,2))

# Vertical gradient: row differences  
v_grad = (spatial[:, 1:, :] - spatial[:, :-1, :]).mean(dim=(1,2))

# Gradient magnitude
grad_mag = (h_grad**2 + v_grad**2).sqrt()
```

**Rationale:** Kernel topologies have different spatial connectivity (horizontal, vertical, kings, ring). A patch with a strong horizontal edge is more likely to be routed to a horizontally-sensitive kernel.

### Knowledge Distillation Loss

The Student supports three training modes controlled by `kd_alpha`:

```python
def build_student_loss_fn(..., kd_alpha=0.0, kd_temperature=2.0, ...):
```

**Mode 1: Hard CE Only (kd_alpha = 0.0)**
```python
loss = CrossEntropy(student_logits, teacher_argmax_labels)
```

**Mode 2: Soft Distillation Only (kd_alpha = 1.0)**
```python
student_log_soft = F.log_softmax(logits / T, dim=1)
teacher_soft = F.softmax(teacher_logits / T, dim=1)
loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * T²
```

**Mode 3: Mixed (0 < kd_alpha < 1)**
```python
loss = (1 - kd_alpha) * CE_loss + kd_alpha * KL_loss
```

**Temperature Scaling:** Higher temperature (T=4–8) produces softer probability distributions, transferring more inter-class relationship information. Lower temperature (T=1–2) keeps targets close to one-hot.

The T² scaling is the standard correction from Hinton et al. (2015) to ensure gradient magnitudes are comparable to hard labels.

### Class Balancing Strategies

When one kernel dominates the Teacher's routing decisions, the Student can collapse to always predicting that kernel (minimizes CE by predicting majority).

**Strategy A: Weighted Cross-Entropy**
```python
# Weight for class k = total / (num_classes * count_k)
class_weights = total_patches / (num_classes * label_counts.clamp(min=1))
ce_fn = nn.CrossEntropyLoss(weight=class_weights)
```

**Strategy B: Balanced Sampler**
```python
sample_weights = class_weights[labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)
```

**Config parameters:**
- `student_weighted_ce`: Enable weighted CE (recommended when labels are skewed)
- `student_balanced_sampler`: More aggressive balancing via sampling

### Confidence Filtering

When the Teacher's alpha weights are nearly uniform, the argmax label is essentially random noise. Training on these patches hurts more than it helps.

```python
if confidence_threshold > 0.0:
    flat_conf = confidence.reshape(-1)
    keep_mask = flat_conf >= confidence_threshold
    flat_patches = flat_patches[keep_mask]
    flat_labels = flat_labels[keep_mask]
```

**Config parameter:** `confidence_threshold` (try 0.3–0.5 to remove ambiguous patches)

A diagnostic summary of Teacher confidence distribution is printed at training start to help choose appropriate thresholds.

---

## Phase 3: Final Classifier Training

After the Student predicts routing decisions, we build sparse tensors and train a new classification head. Located in [ts_moe_classification_head_training.py](../../src/models/ts_moe_classification_head_training.py).

### Why a New Classification Head?

The Teacher's head was trained on **soft-mixed features** where all kernels contribute with weighted alpha values. The sparse routing creates a fundamentally different input distribution where:
- Selected kernel's channels contain normalized quantum features
- Non-selected kernel's channels are **exactly zero**

The Teacher's head interprets zeros as "low signal." The Final Classifier must learn that zeros mean "not selected."

### Sparse Tensor Reconstruction

Located in [sparse_reconstruction.py](../../src/utils/sparse_reconstruction.py):

```python
def build_sparse_tensor_fast(quantum_features, routing_map, channel_groups, group_norms=None):
    """
    For every spatial position (i, j):
        - routing_map[i, j] says which kernel k was selected
        - Keep that kernel's channels, zero the rest
    
    Example (M=2 kernels, N=9 channels each):
        Patch assigned to kernel 0: channels 0-8 filled, channels 9-17 = 0
        Patch assigned to kernel 1: channels 0-8 = 0, channels 9-17 filled
    """
    sparse = torch.zeros_like(quantum_features)
    
    for k, channels in enumerate(channel_groups):
        # Boolean mask: True where kernel k was selected
        mask = (routing_map == k).unsqueeze(1)  # (B, 1, H, W)
        
        if not mask.any():
            continue
            
        ch_idx = torch.tensor(channels, device=device)
        sparse[:, ch_idx, :, :] = quantum_features[:, ch_idx, :, :] * mask
    
    return sparse
```

### Group Normalization at Inference

**Critical Detail:** The Teacher applies per-group BatchNorm BEFORE computing alpha weights. For the Final Classifier to see the same feature scale, we must apply the same normalization to sparse features.

```python
if group_norms is not None:
    # Normalize ALL groups first, THEN apply sparse mask
    channels_per_kernel = len(channel_groups[0])
    groups = quantum_features.split(channels_per_kernel, dim=1)
    normed_groups = [group_norms[k](g) for k, g in enumerate(groups)]
    quantum_features = torch.cat(normed_groups, dim=1)
```

**Why normalize before masking?** The BatchNorm statistics must be computed over the full (non-sparse) features, exactly as the Teacher does. This ensures consistent feature scales between training and inference.

The `group_norms` are extracted from the Teacher checkpoint (`attention_block.group_norms`) and passed to the sparse reconstruction function.

### Final Classification Head

Uses the exact same architecture as Teacher/DAQCNN:

```python
class FinalClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.1, activation="relu"):
        self.head = build_classification_head(
            in_channels, num_classes, dropout, activation
        )
    
    def forward(self, x):
        return self.head(x)  # x is sparse tensor (B, M*N, H, W)
```

**Input:** Sparse tensor of shape `(B, M*N, H, W)` where only selected kernel channels are non-zero.

**Training:** Standard cross-entropy loss on class labels (same as DAQCNN).

### Routing Analysis

After training, we analyze which kernels are used for which classes:

```python
def compute_routing_analysis(routing_maps, class_labels, kernel_names):
    """
    Returns: Dict mapping class_idx -> {kernel_name -> fraction}
    
    Example output:
    {
        0: {"kings": 0.35, "horizontal": 0.40, "cross": 0.15, "ring": 0.10},
        1: {"kings": 0.45, "horizontal": 0.20, "cross": 0.25, "ring": 0.10}
    }
    """
```

This reveals whether the learned routing has discovered meaningful patterns (e.g., pneumonia patches prefer certain kernel topologies).

---

## Configuration Reference

All TS-MoE parameters are under the `ts_moe:` section in YAML configs:

### Phase 1: Teacher

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attention_hidden_dim` | auto | Hidden dim for gating MLP; null = total_channels // 2 |
| `attention_gate_zero_init` | false | Zero-init gate for uniform initial alphas |
| `lambda_entropy_start` | 0.0 | Starting entropy regularization weight |
| `lambda_max` | 0.1 | Maximum entropy regularization weight |
| `lambda_warmup_epochs` | epochs//2 | Epochs to ramp lambda |
| `lambda_start_epoch` | 0 | Epoch when lambda starts growing |

### Phase 2: Student

| Parameter | Default | Description |
|-----------|---------|-------------|
| `student_hidden_dims` | [32, 16] | MLP hidden layer sizes |
| `student_epochs` | 30 | Max training epochs |
| `student_lr` | 1e-3 | Learning rate |
| `student_batch_size` | 256 | Batch size for patches |
| `student_weighted_ce` | false | Use inverse-frequency class weights |
| `student_balanced_sampler` | false | Use WeightedRandomSampler |
| `confidence_threshold` | 0.0 | Filter patches below this confidence |
| `student_kd_alpha` | 0.0 | KL distillation weight (0=hard CE only) |
| `student_kd_temperature` | 2.0 | Temperature for soft targets |
| `student_features_stats` | false | Append mean/std/min/max |
| `student_features_range_energy` | false | Append range/energy/L1 |
| `student_features_gradients` | false | Append h_grad/v_grad/mag |

### Phase 3: Final Classifier

| Parameter | Default | Description |
|-----------|---------|-------------|
| `final_epochs` | 100 | Training epochs |
| `final_lr` | 1e-3 | Learning rate |
| `use_mask_channel` | false | Append routing mask as extra channel |

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: TEACHER                            │
├─────────────────────────────────────────────────────────────────────┤
│  Cached Quantum Features (B, M*N, H, W)                             │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Per-Group BatchNorm2d (normalize each kernel group) │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Gating MLP: Conv1x1 → ReLU → Conv1x1 → Softmax      │            │
│  │ Output: alpha weights (B, M, H, W)                  │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Reweight: normed_features × alpha_expanded          │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Classification Head (CNN)                           │            │
│  │ Loss = CE + λ × normalized_entropy(alpha)           │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  Outputs: class predictions, saved alpha weights                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       PHASE 2: STUDENT                              │
├─────────────────────────────────────────────────────────────────────┤
│  Raw Image Patches (B*n_patches, patch_dim)                         │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Optional: Feature Augmentation (stats/energy/grads) │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ MLP: Linear → ReLU → Linear → ReLU → Linear         │            │
│  │ Output: kernel logits (B*n_patches, M)              │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Loss = (1-α)×CE(hard_labels) + α×KL(soft_targets)   │            │
│  │ Labels from: argmax(Teacher.alpha)                  │            │
│  │ Soft targets from: softmax(Teacher.logits / T)      │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  Outputs: routing predictions (one kernel per patch)                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 3: FINAL CLASSIFIER                         │
├─────────────────────────────────────────────────────────────────────┤
│  Student Routes Patches → routing_map (B, H, W)                     │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Apply Teacher's Group BatchNorms to quantum features │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Sparse Reconstruction:                               │            │
│  │   For each patch (i,j):                              │            │
│  │     - k = routing_map[i,j]                           │            │
│  │     - Keep channels for kernel k                     │            │
│  │     - Zero all other channels                        │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  Sparse Tensor (B, M*N, H, W) — zeros are "not selected"            │
│                    ↓                                                 │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ Classification Head (same arch as Teacher)          │            │
│  │ Loss = CE(class_labels)                              │            │
│  └─────────────────────────────────────────────────────┘            │
│                    ↓                                                 │
│  Outputs: final class predictions, routing analysis                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Implementation Files

| File | Description |
|------|-------------|
| [train_ts_moe.py](../../src/models/train_ts_moe.py) | Pipeline orchestrator |
| [teacher_moe.py](../../src/models/teacher_moe.py) | Teacher model definition |
| [teacher_moe_training.py](../../src/models/teacher_moe_training.py) | Teacher training loop |
| [kernel_channel_attention_block.py](../../src/layers/kernel_channel_attention_block.py) | Attention/routing block |
| [student_gatekeeper.py](../../src/models/student_gatekeeper.py) | Student model definition |
| [student_training.py](../../src/models/student_training.py) | Student training loop |
| [ts_moe_classification_head.py](../../src/models/ts_moe_classification_head.py) | Final classifier model |
| [ts_moe_classification_head_training.py](../../src/models/ts_moe_classification_head_training.py) | Final classifier training |
| [sparse_reconstruction.py](../../src/utils/sparse_reconstruction.py) | Sparse tensor construction |
| [training_utils.py](../../src/utils/training_utils.py) | Shared utilities (loss, head factory) |

---

## Example Configuration

```yaml
ts_moe:
  # Phase 1: Teacher
  attention_hidden_dim: 64
  attention_gate_zero_init: true
  lambda_entropy_start: 0.0
  lambda_max: 0.1
  lambda_warmup_epochs: 50
  
  # Phase 2: Student
  student_hidden_dims: [128, 64]
  student_epochs: 30
  student_lr: 1e-3
  student_batch_size: 256
  student_weighted_ce: true
  confidence_threshold: 0.3
  student_kd_alpha: 0.5
  student_kd_temperature: 4.0
  student_features_stats: true
  student_features_range_energy: true
  student_features_gradients: true
  
  # Phase 3: Final Classifier
  final_epochs: 100
  final_lr: 1e-3
```

---

*Document generated: March 2026*
