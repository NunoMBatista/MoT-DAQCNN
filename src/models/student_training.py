"""
Training loop for the Student Gatekeeper (distillation from Teacher).

Follows the same structural patterns as ``teacher_moe_training.py`` but with
a different data pipeline: the Student trains on raw image patches paired with
hard routing labels extracted from the frozen Teacher's alpha weights.

Pipeline:
    1. Load trained Teacher checkpoint and cached quantum features.
    2. Run Teacher on all quantum features to produce per-patch routing labels
       (argmax of alpha weights) and optionally soft alpha distributions.
    3. Load original images and extract raw patches matching the quantum
       convolution's spatial grid (same kernel_size and stride).
    4. Optionally augment patches with cheap classical statistics
       (per-channel mean, std, min, max) to give the student more signal.
    5. Optionally filter out low-confidence patches (where the Teacher was
       unsure which kernel to pick) to reduce label noise.
    6. Train the Student MLP on (patch, routing_label) pairs using either:
         - Hard CE (standard cross-entropy on argmax labels), or
         - Soft distillation (KL divergence against Teacher's softmax with
           temperature scaling), or
         - A weighted combination of both.
       Class weights are optionally computed from label frequencies to
       counteract majority-class collapse.
    7. Evaluate agreement between Student predictions and Teacher labels.

Uses existing utilities for dataset loading, evaluation, and plotting.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.config import DATA_DIR, DATASET_CHANNELS, DATASET_REGISTRY
from src.models.student_gatekeeper import (
    build_student_from_metadata,
    count_parameters,
)
from src.models.teacher_moe import build_teacher_from_metadata
from src.utils.color_conversion import apply_color_conversion
from src.utils.data import load_medmnist_dataset
from src.utils.plotting import plot_loss_curves, plot_routing_confusion_matrix
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)

# -------------------------------------------------------------------------
# Label generation from the frozen Teacher
# -------------------------------------------------------------------------


@torch.no_grad()
def generate_routing_labels(teacher, loader, device):
    """Run frozen Teacher on quantum features and extract per-patch routing labels.

    For every sample in ``loader``, the Teacher produces alpha weights of shape
    (M, H, W). Taking argmax over M gives the routing decision for each patch.
    The raw alpha values (before argmax) are also returned so the caller can
    use them for soft-label distillation or confidence filtering.

    Args:
        teacher: Trained TeacherMoE model (will be set to eval mode).
        loader: DataLoader yielding (quantum_features, class_labels).
            The class_labels are ignored here.
        device: Device string.

    Returns:
        routing_labels: LongTensor of shape (N, H, W) with values in
            {0, ..., M-1}, where N is the total number of samples.
        soft_labels: FloatTensor of shape (N, M, H, W) — the Teacher's raw
            alpha weights (after softmax over M), one distribution per patch.
            Useful for soft-label distillation.
        confidence: FloatTensor of shape (N, H, W) — max alpha value per
            patch, in [0, 1].  Low values indicate the Teacher was uncertain.
    """
    teacher.eval()
    all_labels = []
    all_soft = []
    all_conf = []

    for features, _ in tqdm(loader, desc="Generating routing labels", leave=False):
        features = features.to(device)
        teacher(features)
        alpha = teacher.last_alpha  # (B, M, H, W) - softmax probabilities
        logits = teacher.last_logits  # (B, M, H, W) - pre-softmax logits

        # Soft labels: use PRE-SOFTMAX logits for knowledge distillation
        # The KD loss will apply softmax with temperature, so we must NOT apply it here
        soft = logits  # (B, M, H, W)

        winners = alpha.argmax(dim=1)  # (B, H, W)
        conf = alpha.max(dim=1).values  # (B, H, W)

        all_labels.append(winners.cpu())
        all_soft.append(soft.cpu())
        all_conf.append(conf.cpu())

    return (
        torch.cat(all_labels, dim=0),  # (N, H, W)
        torch.cat(all_soft, dim=0),  # (N, M, H, W)
        torch.cat(all_conf, dim=0),  # (N, H, W)
    )


# -------------------------------------------------------------------------
# Raw patch extraction from original images
# -------------------------------------------------------------------------


def extract_patches(images, kernel_size, stride):
    """Extract patches from a batch of images using the same spatial grid
    as the quantum convolution.

    Args:
        images: Tensor of shape (B, C, H, W).
        kernel_size: Patch size (square).
        stride: Stride for patch extraction.

    Returns:
        patches: Tensor of shape (B, n_patches, C * ks * ks).
            n_patches = h_out * w_out where h_out, w_out are the output
            spatial dimensions of a conv with the given kernel/stride.
    """
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
    # (B, C*ks*ks, n_patches)
    patches = unfold(images)
    # -> (B, n_patches, C*ks*ks)
    patches = patches.transpose(1, 2)
    return patches


def extract_all_patches(dataset, kernel_size, stride, color_space, batch_size=256):
    """Extract patches from an entire MedMNIST dataset.

    Loads images in order (no shuffling) and applies the same color-space
    conversion used when creating the quantum dataset.

    Args:
        dataset: MedMNIST dataset instance.
        kernel_size: Patch size.
        stride: Stride for extraction.
        color_space: One of "RGB", "HSV", or "GRAYSCALE".
        batch_size: Batch size for processing.

    Returns:
        patches: Tensor of shape (N, n_patches, patch_dim).
        class_labels: LongTensor of shape (N,) — original class labels
            (useful for analysis, not used in student training).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_patches = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting patches", leave=False):
        # Apply same color conversion as quantum dataset creation
        images = apply_color_conversion(images, color_space)

        # Handle HSV: student only sees V channel (same as quantum processing)
        if color_space == "HSV" and images.shape[1] == 3:
            images = images[:, 2:3, :, :]  # V channel only

        patches = extract_patches(images, kernel_size, stride)
        all_patches.append(patches)

        lbl = labels
        if lbl.dim() > 1:
            lbl = lbl.squeeze(-1)
        all_labels.append(lbl)

    return torch.cat(all_patches, dim=0), torch.cat(all_labels, dim=0).long()


# -------------------------------------------------------------------------
# Optional patch feature augmentation
# -------------------------------------------------------------------------


def compute_patch_features(flat_patches, kernel_size, feature_flags):
    """Append selected classical feature groups to the raw pixel vector.

    Each group is independently toggled via ``feature_flags`` (a dict of
    bool values read from the config).  Only enabled groups are computed and
    appended, so you can mix and match freely.

    All features are computed patch-by-patch (no cross-batch statistics),
    so they remain valid at inference time without any normalisation state.

    Args:
        flat_patches: FloatTensor of shape (N, patch_dim).
            patch_dim = C * kernel_size^2 (flattened raw pixels).
        kernel_size: int — spatial side length of each patch (e.g. 2 or 3).
            Needed to reshape the patch for gradient computation.
        feature_flags: dict with boolean values for each group:
            "stats"        — mean, std, min, max                    (+4)
            "range_energy" — range (max-min), L2 energy, L1 norm    (+3)
            "gradients"    — horizontal grad, vertical grad, mag     (+3)

    Returns:
        augmented: FloatTensor of shape (N, patch_dim + n_extra), where
            n_extra is the total number of appended features.
    """
    extras = []

    # ------------------------------------------------------------------
    # Group: stats
    # Basic first and second order statistics of the pixel intensities.
    # Captures average brightness, contrast, and dynamic range.
    # ------------------------------------------------------------------
    if feature_flags.get("stats", False):
        mean = flat_patches.mean(dim=1, keepdim=True)  # (N, 1)
        std = flat_patches.std(dim=1, keepdim=True)  # (N, 1)
        mn = flat_patches.min(dim=1, keepdim=True).values  # (N, 1)
        mx = flat_patches.max(dim=1, keepdim=True).values  # (N, 1)
        extras.extend([mean, std, mn, mx])

    # ------------------------------------------------------------------
    # Group: range_energy
    # Range = max - min: how much contrast exists in the patch.
    # L2 energy = sum of squared pixels: overall activation / power.
    # L1 norm  = sum of |pixels|: similar but less sensitive to outliers.
    # ------------------------------------------------------------------
    if feature_flags.get("range_energy", False):
        mn = flat_patches.min(dim=1, keepdim=True).values
        mx = flat_patches.max(dim=1, keepdim=True).values
        rng = mx - mn  # (N, 1)
        energy = (flat_patches**2).sum(dim=1, keepdim=True)  # (N, 1)
        l1 = flat_patches.abs().sum(dim=1, keepdim=True)  # (N, 1)
        extras.extend([rng, energy, l1])

    # ------------------------------------------------------------------
    # Group: gradients
    # Approximate horizontal and vertical intensity gradients by taking
    # finite differences across the patch columns / rows.
    #
    # Why this matters: the kernel topologies (horizontal, vertical,
    # kings, u_shape) have different spatial connectivity patterns.
    # A patch with a strong horizontal edge is more likely to be routed
    # to a horizontally-sensitive kernel, and vice versa.
    #
    # For a patch of shape (ks, ks) we compute:
    #   h_grad = mean of (right_col - left_col) differences  -> scalar
    #   v_grad = mean of (bottom_row - top_row) differences  -> scalar
    #   grad_mag = sqrt(h_grad^2 + v_grad^2)                 -> scalar
    #
    # For ks=2: one column pair and one row pair each.
    # For ks=3: two column pairs and two row pairs, averaged.
    # ------------------------------------------------------------------
    if feature_flags.get("gradients", False):
        # Reshape to (N, ks, ks); assumes single channel (C=1) or that
        # flat_patches already contains only 1 channel (grayscale / V).
        # For multi-channel patches the first channel is used.
        ks = kernel_size
        spatial = flat_patches[:, : ks * ks].reshape(-1, ks, ks)  # (N, ks, ks)

        # Horizontal gradient: difference between adjacent columns, averaged
        h_grad = (spatial[:, :, 1:] - spatial[:, :, :-1]).mean(
            dim=(1, 2), keepdim=False
        )
        # Vertical gradient: difference between adjacent rows, averaged
        v_grad = (spatial[:, 1:, :] - spatial[:, :-1, :]).mean(
            dim=(1, 2), keepdim=False
        )
        grad_mag = (h_grad**2 + v_grad**2).sqrt()

        extras.extend(
            [
                h_grad.unsqueeze(1),  # (N, 1)
                v_grad.unsqueeze(1),  # (N, 1)
                grad_mag.unsqueeze(1),  # (N, 1)
            ]
        )

    if not extras:
        return flat_patches

    return torch.cat([flat_patches] + extras, dim=1)


# -------------------------------------------------------------------------
# Student dataset creation (pairing patches with routing labels)
# -------------------------------------------------------------------------


def create_student_dataloaders(
    patches,
    routing_labels,
    batch_size=64,
    num_workers=0,
    val_patches=None,
    val_routing_labels=None,
    soft_labels=None,
    val_soft_labels=None,
    confidence=None,
    val_confidence=None,
    confidence_threshold=0.0,
    feature_flags=None,
    use_balanced_sampler=False,
    kernel_size=2,
    num_kernels=None,
):
    """Create DataLoaders for student training from paired patches and labels.

    Flattens the spatial dimensions so each patch becomes an independent sample:
        patches (N, n_patches, patch_dim) -> (N * n_patches, patch_dim)
        routing_labels (N, H, W) -> (N * H * W,)

    Optionally:
      - Filters out patches where the Teacher confidence < confidence_threshold.
      - Appends classical statistics to patch features (augment_patch_features).
      - Uses a WeightedRandomSampler to balance kernel classes during training.

    Args:
        patches: Training patches, shape (N, n_patches, patch_dim).
        routing_labels: Training routing labels, shape (N, H, W).
        batch_size: Batch size for DataLoaders.
        num_workers: Number of DataLoader workers.
        val_patches: Optional validation patches.
        val_routing_labels: Optional validation routing labels.
        soft_labels: Optional soft (Teacher alpha) labels for training,
            shape (N, M, H, W).  When provided, included in the dataset
            so the training loop can use them for KL distillation.
        val_soft_labels: Same as soft_labels but for validation.
        confidence: Optional per-patch confidence tensor, shape (N, H, W).
            Used for filtering when confidence_threshold > 0.
        val_confidence: Same for validation set.
        confidence_threshold: Discard patches whose Teacher confidence
            (max alpha) is below this value.  0.0 = keep everything.
        feature_flags: Dict of booleans controlling which classical feature
            groups are appended to the raw patch pixels.  Keys: "stats",
            "range_energy", "gradients".  None or empty = no augmentation.
        use_balanced_sampler: If True, use WeightedRandomSampler so that
            each kernel class is sampled equally during training.  Useful
            when the routing label distribution is heavily skewed.

    Returns:
        train_loader: DataLoader yielding (flat_patch, routing_label) or
            (flat_patch, routing_label, soft_label) when soft_labels given.
        val_loader: DataLoader for validation, or None if not provided.
        effective_patch_dim: int — the actual feature dimension of the
            patches in the loaders (may be larger than raw patch_dim when
            use_augmented_features=True).
        class_weights: FloatTensor of shape (M,) — inverse-frequency weights
            for each kernel class, computed on the (possibly filtered)
            training set.  Always returned so the caller can decide whether
            to use them.
    """
    # --- Flatten spatial dimensions ---
    N, n_patches, patch_dim = patches.shape
    flat_patches = patches.reshape(N * n_patches, patch_dim)
    flat_labels = routing_labels.reshape(-1).long()

    assert flat_patches.shape[0] == flat_labels.shape[0], (
        f"Patch count {flat_patches.shape[0]} != label count {flat_labels.shape[0]}"
    )

    # Flatten soft labels if provided: (N, M, H, W) -> (N*H*W, M)
    if soft_labels is not None:
        M = soft_labels.shape[1]
        flat_soft = soft_labels.permute(0, 2, 3, 1).reshape(-1, M)  # (N*H*W, M)
    else:
        flat_soft = None

    # --- Confidence filtering ---
    if confidence is not None and confidence_threshold > 0.0:
        flat_conf = confidence.reshape(-1)  # (N*H*W,)
        keep_mask = flat_conf >= confidence_threshold
        n_before = flat_patches.shape[0]
        flat_patches = flat_patches[keep_mask]
        flat_labels = flat_labels[keep_mask]
        if flat_soft is not None:
            flat_soft = flat_soft[keep_mask]
        n_after = flat_patches.shape[0]
        frac_kept = n_after / max(n_before, 1)
        print(
            f"  Confidence filtering (threshold={confidence_threshold:.2f}): "
            f"kept {n_after}/{n_before} patches ({frac_kept:.1%})"
        )
    else:
        print(
            f"  Confidence filtering: disabled (threshold={confidence_threshold:.2f})"
        )

    # --- Optional feature augmentation ---
    active_groups = [k for k, v in (feature_flags or {}).items() if v]
    if active_groups:
        flat_patches = compute_patch_features(flat_patches, kernel_size, feature_flags)
        print(
            f"  Feature augmentation: raw pixels + {active_groups} "
            f"-> feature dim {flat_patches.shape[1]}"
        )

    effective_patch_dim = flat_patches.shape[1]

    # --- Per-class label distribution (always printed for diagnostics) ---
    # Use num_kernels as minlength so class_weights always has shape (M,) even
    # when some kernels never appear in the (possibly filtered) training labels.
    minlength = num_kernels if num_kernels is not None else flat_labels.max().item() + 1
    label_counts = torch.bincount(flat_labels, minlength=minlength)
    total_patches = flat_labels.shape[0]
    print("  Training label distribution (kernel -> count / fraction):")
    for k, cnt in enumerate(label_counts.tolist()):
        print(f"    kernel {k}: {cnt:>8d}  ({cnt / max(total_patches, 1):.1%})")

    # --- Inverse-frequency class weights ---
    # Weight for class k = total / (num_classes * count_k).
    # These can be passed to CrossEntropyLoss(weight=...) to counteract
    # majority-class dominance.
    num_classes = label_counts.shape[0]
    class_weights = total_patches / (num_classes * label_counts.float().clamp(min=1))
    class_weights = class_weights / class_weights.sum() * num_classes  # re-scale

    # --- Build dataset ---
    if flat_soft is not None:
        train_ds = TensorDataset(flat_patches, flat_labels, flat_soft)
    else:
        train_ds = TensorDataset(flat_patches, flat_labels)

    # --- Optional balanced sampler ---
    if use_balanced_sampler:
        sample_weights = class_weights[flat_labels]  # (N_train,)
        sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
        print("  Balanced sampler: enabled (WeightedRandomSampler)")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    # --- Validation loader ---
    val_loader = None
    if val_patches is not None and val_routing_labels is not None:
        Nv, npv, pdv = val_patches.shape
        vflat_patches = val_patches.reshape(Nv * npv, pdv)
        vflat_labels = val_routing_labels.reshape(-1).long()

        # Filter val by confidence as well (same threshold)
        if val_confidence is not None and confidence_threshold > 0.0:
            vflat_conf = val_confidence.reshape(-1)
            vkeep = vflat_conf >= confidence_threshold
            vflat_patches = vflat_patches[vkeep]
            vflat_labels = vflat_labels[vkeep]
            if val_soft_labels is not None:
                M = val_soft_labels.shape[1]
                vflat_soft = val_soft_labels.permute(0, 2, 3, 1).reshape(-1, M)
                vflat_soft = vflat_soft[vkeep]
            else:
                vflat_soft = None
        else:
            if val_soft_labels is not None:
                M = val_soft_labels.shape[1]
                vflat_soft = val_soft_labels.permute(0, 2, 3, 1).reshape(-1, M)
            else:
                vflat_soft = None

        if active_groups:
            vflat_patches = compute_patch_features(
                vflat_patches, kernel_size, feature_flags
            )

        if vflat_soft is not None:
            val_ds = TensorDataset(vflat_patches, vflat_labels, vflat_soft)
        else:
            val_ds = TensorDataset(vflat_patches, vflat_labels)

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader, effective_patch_dim, class_weights


# -------------------------------------------------------------------------
# Loss function factory
# -------------------------------------------------------------------------


def build_student_loss_fn(
    num_kernels,
    class_weights=None,
    use_weighted_ce=False,
    kd_alpha=0.0,
    kd_temperature=2.0,
    device="cpu",
):
    """Build the student loss function based on config flags.

    Supports three modes (controlled by kd_alpha):

    1. Hard CE only (kd_alpha = 0.0):
        loss = CrossEntropy(logits, hard_labels)
        Standard supervised learning on Teacher argmax labels.

    2. Soft distillation only (kd_alpha = 1.0):
        loss = KL(student_soft || teacher_soft)  [with temperature T]
        Trains purely against the Teacher's probability distribution,
        preserving soft inter-class relationships.

    3. Mixed (0 < kd_alpha < 1):
        loss = (1 - kd_alpha) * CE + kd_alpha * KL
        Balances hard-label supervision with soft distillation.

    When use_weighted_ce=True, the CE term uses inverse-frequency class
    weights so that minority kernels are penalised more strongly, preventing
    the student from collapsing to the majority class.

    Args:
        num_kernels: Number of kernel classes (M).
        class_weights: FloatTensor of shape (M,) from create_student_dataloaders.
            Only used when use_weighted_ce=True.
        use_weighted_ce: Whether to weight CE by inverse label frequency.
        kd_alpha: Weight for the KL distillation term.  0 = hard CE only,
            1 = soft KL only, values in between blend both.
        kd_temperature: Temperature T for softening the Teacher/Student
            distributions in the KL term.  Higher T -> softer distributions,
            more information transfer.  Typical values: 2–8.
        device: Device to place class_weights on.

    Returns:
        loss_fn: callable(logits, hard_labels, soft_labels) -> scalar loss.
            soft_labels may be None if kd_alpha == 0.
    """
    if use_weighted_ce and class_weights is not None:
        weights = class_weights.to(device)
    else:
        weights = None

    ce_fn = nn.CrossEntropyLoss(weight=weights)

    def loss_fn(logits, hard_labels, soft_labels=None):
        """Compute student loss.

        Args:
            logits: (B, M) student output logits.
            hard_labels: (B,) integer class indices from Teacher argmax.
            soft_labels: (B, M) Teacher softmax probabilities, or None.

        Returns:
            Scalar loss tensor.
        """
        # Hard cross-entropy term
        hard_loss = ce_fn(logits, hard_labels)

        if kd_alpha == 0.0 or soft_labels is None:
            return hard_loss

        # Soft KL divergence term (knowledge distillation)
        # Scale both distributions by temperature before computing KL.
        student_log_soft = F.log_softmax(logits / kd_temperature, dim=1)
        teacher_soft = F.softmax(soft_labels / kd_temperature, dim=1)
        # T^2 scaling is the standard KD correction (Hinton et al., 2015)
        kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (
            kd_temperature**2
        )

        return (1.0 - kd_alpha) * hard_loss + kd_alpha * kl_loss

    return loss_fn


# -------------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------------


def train_student_one_epoch(
    model, loader, optimizer, loss_fn, device, grad_clip, epoch_pbar
):
    """Train the Student for one epoch on routing labels.

    Args:
        model: StudentGatekeeper instance.
        loader: DataLoader yielding (flat_patch, routing_label) or
            (flat_patch, routing_label, soft_label).
        optimizer: Torch optimizer.
        loss_fn: Callable returned by build_student_loss_fn.
        device: Device string.
        grad_clip: Max gradient norm (0 to disable).
        epoch_pbar: Outer progress bar for description updates.

    Returns:
        Tuple of (avg_loss, avg_accuracy).
    """
    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    batch_pbar = tqdm(loader, desc="Student training", leave=False)
    for batch_idx, batch in enumerate(batch_pbar):
        if len(batch) == 3:
            patches, labels, soft = batch
            soft = soft.to(device)
        else:
            patches, labels = batch
            soft = None

        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = loss_fn(logits, labels, soft)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = labels.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs

        batch_pbar.set_description(
            f"Student {batch_idx + 1}/{len(loader)} | loss={loss.item():.4f}"
        )

    batch_pbar.close()
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_student(model, loader, loss_fn, device):
    """Evaluate Student on routing labels (loss + accuracy).

    Args:
        model: StudentGatekeeper instance.
        loader: DataLoader yielding (flat_patch, routing_label) or
            (flat_patch, routing_label, soft_label).
        loss_fn: Callable returned by build_student_loss_fn.
        device: Device string.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Student eval", leave=False):
        if len(batch) == 3:
            patches, labels, soft = batch
            soft = soft.to(device)
        else:
            patches, labels = batch
            soft = None

        patches = patches.to(device)
        labels = labels.to(device)

        logits = model(patches)
        loss = loss_fn(logits, labels, soft)

        bs = labels.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs

    return loss_sum / max(total, 1), correct / max(total, 1)


# -------------------------------------------------------------------------
# Agreement and routing analysis
# -------------------------------------------------------------------------


@torch.no_grad()
def compute_routing_confusion(model, loader, device, num_kernels):
    """Compute the M x M routing confusion matrix: teacher labels vs student preds.

    Element (i, j) counts how many patches had teacher label i but student
    predicted j. Useful for detecting lazy bias (student always predicts
    one kernel).

    Args:
        model: StudentGatekeeper instance.
        loader: DataLoader yielding (flat_patch, routing_label[, soft_label]).
        device: Device string.
        num_kernels: Number of kernel topologies (M).

    Returns:
        confusion: numpy array of shape (M, M), rows=teacher, cols=student.
        agreement: float, fraction of patches where student == teacher.
        per_kernel_agreement: dict mapping kernel_index -> agreement for that kernel.
    """
    model.eval()
    cm = np.zeros((num_kernels, num_kernels), dtype=np.int64)

    for batch in loader:
        patches, labels = batch[0], batch[1]
        patches = patches.to(device)
        preds = model(patches).argmax(dim=1).cpu().numpy()
        labels_np = labels.numpy()

        for true_k, pred_k in zip(labels_np, preds):
            cm[true_k, pred_k] += 1

    total = cm.sum()
    agreement = cm.trace() / max(total, 1)

    per_kernel_agreement = {}
    for k in range(num_kernels):
        row_total = cm[k, :].sum()
        if row_total > 0:
            per_kernel_agreement[k] = cm[k, k] / row_total
        else:
            per_kernel_agreement[k] = 0.0

    return cm, agreement, per_kernel_agreement


@torch.no_grad()
def compute_prediction_distribution(model, loader, device, num_kernels):
    """Compute how often the student predicts each kernel.

    Returns:
        dict mapping kernel_index -> fraction of all patches predicted as
        that kernel.
    """
    model.eval()
    counts = np.zeros(num_kernels, dtype=np.int64)
    total = 0

    for batch in loader:
        patches = batch[0]
        patches = patches.to(device)
        preds = model(patches).argmax(dim=1).cpu().numpy()
        for k in range(num_kernels):
            counts[k] += (preds == k).sum()
        total += len(preds)

    return {k: counts[k] / max(total, 1) for k in range(num_kernels)}


# -------------------------------------------------------------------------
# Full training pipeline
# -------------------------------------------------------------------------


def _load_teacher_from_checkpoint(ckpt_path, device):
    """Load a trained Teacher model from a checkpoint file.

    Args:
        ckpt_path: Path to the .pt checkpoint saved by ``run_teacher_training()``.
        device: Device to place the model on.

    Returns:
        Tuple of (teacher_model, metadata, kernel_names, num_classes).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    metadata = ckpt["metadata"]
    num_classes = ckpt["num_classes"]
    kernel_names = ckpt["kernel_names"]

    # Get SE block config from metadata (saved during teacher training)
    se_hidden_dim = metadata.get("se_hidden_dim", None)
    se_use_std = metadata.get("se_use_std", False)

    teacher = build_teacher_from_metadata(
        metadata=metadata,
        num_classes=num_classes,
        dropout=0.0,  # no dropout needed in eval mode
        se_hidden_dim=se_hidden_dim,
        se_use_std=se_use_std,
    )
    teacher.to(device)

    # Initialize lazy layers with a dummy forward pass
    total_channels = metadata["out_channels"]
    kernel_size = metadata["kernel_size"]
    stride = metadata["stride"]
    # Spatial dim from a 28x28 image (standard MedMNIST size)
    spatial = (28 - kernel_size) // stride + 1
    dummy = torch.randn(1, total_channels, spatial, spatial, device=device)
    with torch.no_grad():
        teacher(dummy)

    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()

    return teacher, metadata, kernel_names, num_classes


def run_student_training(
    cfg,
    seed,
    output_dir,
    teacher_ckpt_path,
    verbose=True,
    set_seed_fn=None,
    datasets_dir=None,
    data_root=None,
    raw_image_datasets=None,
):
    """Train the Student Gatekeeper via distillation from a trained Teacher.

    Mirrors ``run_teacher_training()`` in structure so the experiment runner
    can orchestrate Teacher -> Student -> Final Classifier in sequence.

    Args:
        cfg: Loaded YAML config dict.
        seed: Random seed for this run.
        output_dir: Directory to save outputs (models, plots, logs).
        teacher_ckpt_path: Path to the trained Teacher checkpoint (.pt).
        verbose: Whether to print progress.
        set_seed_fn: Optional callable to set global random seeds.
        datasets_dir: Optional path to search for cached quantum datasets.
        data_root: Optional root directory for MedMNIST data download.
            If None, uses the project default ``data/``.
        raw_image_datasets: Optional dict ``{"train": dataset, "val": dataset}``
            providing pre-built PyTorch datasets of raw images. When supplied,
            these are used instead of downloading MedMNIST. Useful for testing
            with synthetic data where the sample count must match the cached
            quantum dataset.

    Returns:
        dict with all metrics, matching the project's result dict conventions.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Student Gatekeeper Training — seed={seed}")
        print(f"{'=' * 60}\n")

    if set_seed_fn:
        set_seed_fn(seed)

    # --- Device ---
    requested = cfg.get("model", {}).get("classical_device", "auto")
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = requested
    if verbose:
        print(f"Device: {device}")

    # --- Load trained Teacher ---
    if verbose:
        print(f"Loading teacher from: {teacher_ckpt_path}")
    teacher, metadata, kernel_names, _ = _load_teacher_from_checkpoint(
        teacher_ckpt_path, device
    )
    num_kernels = len(kernel_names)
    if verbose:
        print(f"Teacher loaded: {num_kernels} kernels ({kernel_names})")

    # --- Load cached quantum dataset (for Teacher routing label generation) ---
    cached_path = find_cached_quantum_dataset(cfg, datasets_dir=datasets_dir)
    if cached_path is None:
        raise FileNotFoundError(
            "No cached quantum dataset found matching the config.\n"
            "Run `python experiments/create_quantum_dataset.py` first."
        )

    batch_size = cfg.get("dataset", {}).get("batch_size", 32)
    model_cfg = cfg.get("model", {})
    requested_kernels = model_cfg.get("kernel_topology_names", None)
    q_train_loader, q_val_loader, _, _, q_metadata = load_cached_quantum_dataset(
        cached_path, batch_size, num_workers=0, requested_kernels=requested_kernels
    )
    if verbose:
        print(
            f"Quantum dataset: {len(q_train_loader.dataset)} train, "
            f"{len(q_val_loader.dataset)} val"
        )

    # --- Generate routing labels from Teacher ---
    # This now also returns soft alpha distributions and per-patch confidence.
    if verbose:
        print("Generating routing labels from Teacher...")
    t0 = time.time()
    train_routing_labels, train_soft_labels, train_confidence = generate_routing_labels(
        teacher, q_train_loader, device
    )
    val_routing_labels, val_soft_labels, val_confidence = generate_routing_labels(
        teacher, q_val_loader, device
    )
    label_gen_time = time.time() - t0
    if verbose:
        print(
            f"Labels generated in {label_gen_time:.1f}s — "
            f"train: {train_routing_labels.shape}, val: {val_routing_labels.shape}"
        )
        # Print Teacher confidence statistics for diagnostics
        flat_conf_train = train_confidence.reshape(-1)
        print(
            f"Teacher confidence (train) — "
            f"mean={flat_conf_train.mean():.3f}, "
            f"min={flat_conf_train.min():.3f}, "
            f"max={flat_conf_train.max():.3f}, "
            f"frac<0.5={(flat_conf_train < 0.5).float().mean():.1%}"
        )

    # --- Load original images and extract patches ---
    dataset_name = cfg.get("dataset", {}).get("name", "pneumonia_mnist")
    color_space = cfg.get("dataset", {}).get("color_space", "RGB")
    kernel_size = metadata["kernel_size"]
    stride = metadata["stride"]

    # Determine input channels for the student
    raw_channels = DATASET_CHANNELS.get(dataset_name, 1)
    if color_space == "GRAYSCALE" and raw_channels == 3:
        student_in_channels = 1
    elif color_space == "HSV" and raw_channels == 3:
        student_in_channels = 1  # only V channel
    else:
        student_in_channels = raw_channels

    effective_data_root = data_root or cfg.get("dataset", {}).get(
        "data_root", str(DATA_DIR)
    )

    if raw_image_datasets is not None:
        train_dataset = raw_image_datasets["train"]
        val_dataset = raw_image_datasets["val"]
        if verbose:
            print(
                f"Using provided raw image datasets "
                f"(train={len(train_dataset)}, val={len(val_dataset)})"
            )
    else:
        if verbose:
            print(
                f"Loading original images ({dataset_name}, {color_space}) "
                f"for patch extraction..."
            )
        train_dataset = load_medmnist_dataset(
            dataset_name, "train", effective_data_root
        )
        val_dataset = load_medmnist_dataset(dataset_name, "val", effective_data_root)

    train_patches, _ = extract_all_patches(
        train_dataset, kernel_size, stride, color_space
    )
    val_patches, _ = extract_all_patches(val_dataset, kernel_size, stride, color_space)

    if verbose:
        print(
            f"Patches extracted — train: {train_patches.shape}, "
            f"val: {val_patches.shape}"
        )

    # Validate alignment: number of samples and spatial dimensions must match
    N_train, n_patches_train, patch_dim = train_patches.shape
    H_W_train = train_routing_labels.shape[1] * train_routing_labels.shape[2]
    if N_train != train_routing_labels.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {N_train} images vs "
            f"{train_routing_labels.shape[0]} routing label sets. "
            "The cached quantum dataset and MedMNIST splits may not align."
        )
    if n_patches_train != H_W_train:
        raise ValueError(
            f"Spatial mismatch: {n_patches_train} patches per image vs "
            f"{H_W_train} routing labels per image (H*W)."
        )

    # --- Read student config ---
    ts_moe_cfg = cfg.get("ts_moe", {})
    student_batch_size = ts_moe_cfg.get("student_batch_size", 256)
    confidence_threshold = float(ts_moe_cfg.get("confidence_threshold", 0.0))
    feature_flags = {
        "stats": bool(ts_moe_cfg.get("student_features_stats", False)),
        "range_energy": bool(ts_moe_cfg.get("student_features_range_energy", False)),
        "gradients": bool(ts_moe_cfg.get("student_features_gradients", False)),
    }
    use_weighted_ce = bool(ts_moe_cfg.get("student_weighted_ce", False))
    use_balanced_sampler = bool(ts_moe_cfg.get("student_balanced_sampler", False))
    kd_alpha = float(ts_moe_cfg.get("student_kd_alpha", 0.0))
    kd_temperature = float(ts_moe_cfg.get("student_kd_temperature", 4.0))

    # Only pass soft labels to the dataloader when KD is actually used
    use_soft = kd_alpha > 0.0
    if verbose:
        print(
            f"\nStudent dataset config:\n"
            f"  confidence_threshold    = {confidence_threshold}\n"
            f"  feature_flags           = {feature_flags}\n"
            f"  use_weighted_ce         = {use_weighted_ce}\n"
            f"  use_balanced_sampler    = {use_balanced_sampler}\n"
            f"  kd_alpha                = {kd_alpha} "
            f"({'soft distillation' if kd_alpha > 0 else 'hard CE only'})\n"
            f"  kd_temperature          = {kd_temperature}\n"
        )

    # --- Create Student DataLoaders ---
    train_loader, val_loader, effective_patch_dim, class_weights = (
        create_student_dataloaders(
            patches=train_patches,
            routing_labels=train_routing_labels,
            batch_size=student_batch_size,
            num_workers=0,
            val_patches=val_patches,
            val_routing_labels=val_routing_labels,
            soft_labels=train_soft_labels if use_soft else None,
            val_soft_labels=val_soft_labels if use_soft else None,
            confidence=train_confidence,
            val_confidence=val_confidence,
            confidence_threshold=confidence_threshold,
            feature_flags=feature_flags,
            use_balanced_sampler=use_balanced_sampler,
            kernel_size=kernel_size,
            num_kernels=num_kernels,
        )
    )
    if verbose:
        total_train = len(train_loader.dataset)
        total_val = len(val_loader.dataset) if val_loader else 0
        print(
            f"Student datasets: {total_train} train patches, {total_val} val patches\n"
            f"Effective patch feature dim: {effective_patch_dim}"
        )

    # --- Build Student model ---
    # effective_patch_dim may differ from raw patch_dim when augmented features are on
    hidden_dims = tuple(ts_moe_cfg.get("student_hidden_dims", [32, 16]))

    student, _ = build_student_from_metadata(
        metadata=metadata,
        hidden_dims=hidden_dims,
        in_channels=student_in_channels,
    )

    # If feature augmentation enlarged the input, rebuild with the correct dim
    if effective_patch_dim != student.patch_dim:
        from src.models.student_gatekeeper import StudentGatekeeper

        student = StudentGatekeeper(
            patch_dim=effective_patch_dim,
            num_kernels=num_kernels,
            hidden_dims=hidden_dims,
        )

    student.to(device)

    total_params, trainable_params = count_parameters(student)
    if verbose:
        print(
            f"Student built: {total_params:,} params "
            f"({trainable_params:,} trainable), "
            f"hidden_dims={hidden_dims}, patch_dim={effective_patch_dim}"
        )
        if total_params > 5000:
            print(
                f"WARNING: Student has {total_params} params (target is <5k). "
                "Consider reducing hidden_dims."
            )

    # --- Loss function ---
    loss_fn = build_student_loss_fn(
        num_kernels=num_kernels,
        class_weights=class_weights,
        use_weighted_ce=use_weighted_ce,
        kd_alpha=kd_alpha,
        kd_temperature=kd_temperature,
        device=device,
    )
    if verbose:
        print(
            f"Loss: "
            f"{'weighted ' if use_weighted_ce else ''}CE"
            f"{f' + {kd_alpha:.2f} * KL(T={kd_temperature})' if kd_alpha > 0 else ''}"
        )

    # --- Optimizer ---
    optim_cfg = cfg.get("optim", {})
    lr = float(ts_moe_cfg.get("student_lr", optim_cfg.get("lr", 1e-3)))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    student_epochs = int(ts_moe_cfg.get("student_epochs", optim_cfg.get("epochs", 20)))
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = ts_moe_cfg.get("student_patience", optim_cfg.get("patience", None))
    agreement_threshold = ts_moe_cfg.get("student_agreement_threshold", None)
    agreement_patience = ts_moe_cfg.get("student_agreement_patience", 3)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if ts_moe_cfg.get("student_use_scheduler", False):
        T_max = ts_moe_cfg.get("student_scheduler_T_max", student_epochs)
        eta_min = float(ts_moe_cfg.get("student_scheduler_eta_min", 0.0))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    if verbose:
        print(
            f"Optimizer: Adam lr={lr}, epochs={student_epochs}, patience={patience}, "
            f"agreement_threshold={agreement_threshold}, "
            f"agreement_patience={agreement_patience}"
        )

    # --- Training loop ---
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    agreement_history = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    epochs_above_threshold = 0

    epoch_pbar = tqdm(
        range(1, student_epochs + 1), desc=f"Student seed {seed}", leave=True
    )
    for epoch in epoch_pbar:
        t0 = time.time()

        train_loss, train_acc = train_student_one_epoch(
            student, train_loader, optimizer, loss_fn, device, grad_clip, epoch_pbar
        )

        val_loss, val_acc = evaluate_student(student, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        dt = time.time() - t0

        # Agreement is the same as val_acc for the student (it's predicting
        # teacher labels), but we track it explicitly for clarity.
        agreement_history.append(val_acc)

        epoch_pbar.set_description(
            f"Student seed {seed} | E{epoch}/{student_epochs} | "
            f"loss={train_loss:.4f} val={val_loss:.4f} "
            f"agree={val_acc:.1%} | {dt:.1f}s"
        )

        if verbose:
            tqdm.write(
                f"Epoch {epoch}/{student_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc(agreement)={val_acc:.4f} | "
                f"{dt:.1f}s"
            )

        # Agreement-threshold patience: stop once student reliably mirrors teacher
        if agreement_threshold is not None and val_acc >= agreement_threshold:
            epochs_above_threshold += 1
            if epochs_above_threshold >= agreement_patience:
                if verbose:
                    print(
                        f"Agreement threshold reached: {val_acc:.1%} >= "
                        f"{agreement_threshold:.1%} for {agreement_patience} "
                        f"consecutive epoch(s). Stopping at epoch {epoch}."
                    )
                # Keep the best model seen so far
                if best_model_state is None:
                    best_model_state = {
                        k: v.clone() for k, v in student.state_dict().items()
                    }
                break
        else:
            epochs_above_threshold = 0

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in student.state_dict().items()}
            if patience is not None:
                epochs_without_improvement = 0
        elif patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping after {epoch} epochs (patience={patience})")
                break

        if scheduler is not None:
            scheduler.step()

    epoch_pbar.close()

    # --- Restore best model for final evaluation ---
    if best_model_state is not None:
        student.load_state_dict(best_model_state)

    # --- Final evaluation: routing confusion matrix + agreement ---
    if verbose:
        print("\nComputing final routing analysis...")

    cm, final_agreement, per_kernel_agree = compute_routing_confusion(
        student, val_loader, device, num_kernels
    )
    pred_dist = compute_prediction_distribution(
        student, val_loader, device, num_kernels
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Student Results (seed={seed})")
        print(f"{'=' * 60}")
        print(f"Final Agreement:  {final_agreement:.4f} ({final_agreement:.1%})")
        print("Target:           >90%")
        status = "PASS" if final_agreement > 0.9 else "NEEDS IMPROVEMENT"
        print(f"Status:           {status}")
        print("\nPer-kernel agreement:")
        for k, name in enumerate(kernel_names):
            print(
                f"  {name}: {per_kernel_agree[k]:.1%} agreement, "
                f"{pred_dist[k]:.1%} of predictions"
            )
        print("\nRouting confusion matrix (rows=teacher, cols=student):")
        # Header
        header = "         " + "".join(f"{name:>10s}" for name in kernel_names)
        print(header)
        for k, name in enumerate(kernel_names):
            row = f"{name:>8s} " + "".join(
                f"{cm[k, j]:>10d}" for j in range(num_kernels)
            )
            print(row)
        print(f"{'=' * 60}\n")

    # --- Save models ---
    os.makedirs(output_dir, exist_ok=True)

    final_model_path = os.path.join(output_dir, f"student_final_seed_{seed}.pt")
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "seed": seed,
            "num_kernels": num_kernels,
            "kernel_names": kernel_names,
            "metadata": metadata,
            "patch_dim": student.patch_dim,
            "hidden_dims": hidden_dims,
            "student_in_channels": student_in_channels,
            "agreement": final_agreement,
            # Needed by the classification head to apply the same feature
            # augmentation at inference time as was used during training.
            "feature_flags": feature_flags,
            "kernel_size": kernel_size,
        },
        final_model_path,
    )
    if verbose:
        print(f"Final student saved to: {final_model_path}")

    if best_model_state is not None:
        best_model_path = os.path.join(output_dir, f"student_best_seed_{seed}.pt")
        torch.save(
            {
                "model_state_dict": best_model_state,
                "config": cfg,
                "seed": seed,
                "num_kernels": num_kernels,
                "kernel_names": kernel_names,
                "metadata": metadata,
                "patch_dim": student.patch_dim,
                "hidden_dims": hidden_dims,
                "student_in_channels": student_in_channels,
                "best_val_loss": best_val_loss,
                "agreement": final_agreement,
                # Needed by the classification head to apply the same feature
                # augmentation at inference time as was used during training.
                "feature_flags": feature_flags,
                "kernel_size": kernel_size,
            },
            best_model_path,
        )
        if verbose:
            print(f"Best student saved to: {best_model_path}")

    # --- Plots ---
    # Loss curves
    loss_path = os.path.join(output_dir, f"student_loss_curve_seed_{seed}.png")
    plot_loss_curves(train_losses, val_losses, loss_path)

    # Routing confusion matrix
    cm_path = os.path.join(output_dir, f"student_routing_confusion_seed_{seed}.png")
    plot_routing_confusion_matrix(cm, kernel_names, cm_path)

    if verbose:
        print(f"Plots saved to: {output_dir}")

    # --- Return results dict ---
    return {
        "seed": seed,
        "architecture": "TS-MoE-Student",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "agreement": final_agreement,
        "agreement_history": agreement_history,
        "per_kernel_agreement": {
            kernel_names[k]: v for k, v in per_kernel_agree.items()
        },
        "prediction_distribution": {kernel_names[k]: v for k, v in pred_dist.items()},
        "routing_confusion_matrix": cm.tolist(),
        "kernel_names": kernel_names,
        "num_kernels": num_kernels,
        "student_params": total_params,
        "label_generation_time_s": label_gen_time,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
    }
