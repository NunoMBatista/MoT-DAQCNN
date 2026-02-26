"""
Training loop for the Student Gatekeeper (distillation from Teacher).

Follows the same structural patterns as ``teacher_moe_training.py`` but with
a different data pipeline: the Student trains on raw image patches paired with
hard routing labels extracted from the frozen Teacher's alpha weights.

Pipeline:
    1. Load trained Teacher checkpoint and cached quantum features.
    2. Run Teacher on all quantum features to produce per-patch routing labels
       (argmax of alpha weights).
    3. Load original images and extract raw patches matching the quantum
       convolution's spatial grid (same kernel_size and stride).
    4. Train the Student MLP on (patch, routing_label) pairs using CE loss.
    5. Evaluate agreement between Student predictions and Teacher labels.

Uses existing utilities for dataset loading, evaluation, and plotting.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from src.config import DATA_DIR, DATASET_CHANNELS, DATASET_REGISTRY
from src.models.student_gatekeeper import (
    build_student_from_metadata,
    count_parameters,
)
from src.models.teacher_moe import build_teacher_from_metadata
from src.utils.color_conversion import apply_color_conversion
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

    Args:
        teacher: Trained TeacherMoE model (will be set to eval mode).
        loader: DataLoader yielding (quantum_features, class_labels).
            The class_labels are ignored here.
        device: Device string.

    Returns:
        routing_labels: LongTensor of shape (N, H, W) with values in
            {0, ..., M-1}, where N is the total number of samples.
    """
    teacher.eval()
    all_labels = []

    for features, _ in tqdm(loader, desc="Generating routing labels", leave=False):
        features = features.to(device)
        teacher(features)
        alpha = teacher.last_alpha  # (B, M, H, W)
        winners = alpha.argmax(dim=1)  # (B, H, W)
        all_labels.append(winners.cpu())

    return torch.cat(all_labels, dim=0)  # (N, H, W)


# -------------------------------------------------------------------------
# Raw patch extraction from original images
# -------------------------------------------------------------------------


def _load_medmnist_dataset(dataset_name, split, data_root=None):
    """Load a MedMNIST dataset for a given split, no shuffling.

    Uses the same loading pattern as ``experiments/create_quantum_dataset.py``
    to guarantee index alignment with cached quantum features.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    _, class_name = DATASET_REGISTRY[dataset_name]

    try:
        import medmnist

        dataset_class = getattr(medmnist, class_name)
    except ImportError as exc:
        raise ImportError(
            f"medmnist is required for {dataset_name}. "
            "Install with `pip install medmnist`."
        ) from exc

    if data_root is None:
        data_root = str(DATA_DIR)

    transform = transforms.ToTensor()
    ds = dataset_class(
        split=split,
        download=True,
        root=str(data_root),
        transform=transform,
    )
    return ds


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
# Student dataset creation (pairing patches with routing labels)
# -------------------------------------------------------------------------


def create_student_dataloaders(
    patches,
    routing_labels,
    batch_size=64,
    num_workers=0,
    val_patches=None,
    val_routing_labels=None,
):
    """Create DataLoaders for student training from paired patches and labels.

    Flattens the spatial dimensions so each patch becomes an independent sample:
        patches (N, n_patches, patch_dim) -> (N * n_patches, patch_dim)
        routing_labels (N, H, W) -> (N * H * W,)

    Args:
        patches: Training patches, shape (N, n_patches, patch_dim).
        routing_labels: Training routing labels, shape (N, H, W).
        batch_size: Batch size for DataLoaders.
        num_workers: Number of DataLoader workers.
        val_patches: Optional validation patches.
        val_routing_labels: Optional validation routing labels.

    Returns:
        train_loader: DataLoader yielding (flat_patch, routing_label).
        val_loader: DataLoader for validation, or None if not provided.
    """
    # Flatten spatial dimensions
    N, n_patches, patch_dim = patches.shape
    flat_patches = patches.reshape(N * n_patches, patch_dim)
    flat_labels = routing_labels.reshape(-1).long()

    assert flat_patches.shape[0] == flat_labels.shape[0], (
        f"Patch count {flat_patches.shape[0]} != label count {flat_labels.shape[0]}"
    )

    train_ds = TensorDataset(flat_patches, flat_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = None
    if val_patches is not None and val_routing_labels is not None:
        Nv, npv, pdv = val_patches.shape
        vflat_patches = val_patches.reshape(Nv * npv, pdv)
        vflat_labels = val_routing_labels.reshape(-1).long()
        val_ds = TensorDataset(vflat_patches, vflat_labels)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader


# -------------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------------


def train_student_one_epoch(model, loader, optimizer, device, grad_clip, epoch_pbar):
    """Train the Student for one epoch on routing labels.

    Args:
        model: StudentGatekeeper instance.
        loader: DataLoader yielding (flat_patch, routing_label).
        optimizer: Torch optimizer.
        device: Device string.
        grad_clip: Max gradient norm (0 to disable).
        epoch_pbar: Outer progress bar for description updates.

    Returns:
        Tuple of (avg_loss, avg_accuracy).
    """
    model.train()
    ce_fn = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0

    batch_pbar = tqdm(loader, desc="Student training", leave=False)
    for batch_idx, (patches, labels) in enumerate(batch_pbar):
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = ce_fn(logits, labels)
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
def evaluate_student(model, loader, device):
    """Evaluate Student on routing labels (CE loss + accuracy).

    Args:
        model: StudentGatekeeper instance.
        loader: DataLoader yielding (flat_patch, routing_label).
        device: Device string.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    ce_fn = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0

    for patches, labels in tqdm(loader, desc="Student eval", leave=False):
        patches = patches.to(device)
        labels = labels.to(device)

        logits = model(patches)
        loss = ce_fn(logits, labels)

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
        loader: DataLoader yielding (flat_patch, routing_label).
        device: Device string.
        num_kernels: Number of kernel topologies (M).

    Returns:
        confusion: numpy array of shape (M, M), rows=teacher, cols=student.
        agreement: float, fraction of patches where student == teacher.
        per_kernel_agreement: dict mapping kernel_index -> agreement for that kernel.
    """
    model.eval()
    cm = np.zeros((num_kernels, num_kernels), dtype=np.int64)

    for patches, labels in loader:
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

    for patches, _ in loader:
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

    teacher = build_teacher_from_metadata(
        metadata=metadata,
        num_classes=num_classes,
        dropout=0.0,  # no dropout needed in eval mode
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
    q_train_loader, q_val_loader, _, _, q_metadata = load_cached_quantum_dataset(
        cached_path, batch_size, num_workers=0
    )
    if verbose:
        print(
            f"Quantum dataset: {len(q_train_loader.dataset)} train, "
            f"{len(q_val_loader.dataset)} val"
        )

    # --- Generate routing labels from Teacher ---
    if verbose:
        print("Generating routing labels from Teacher...")
    t0 = time.time()
    train_routing_labels = generate_routing_labels(teacher, q_train_loader, device)
    val_routing_labels = generate_routing_labels(teacher, q_val_loader, device)
    label_gen_time = time.time() - t0
    if verbose:
        print(
            f"Labels generated in {label_gen_time:.1f}s — "
            f"train: {train_routing_labels.shape}, val: {val_routing_labels.shape}"
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
        train_dataset = _load_medmnist_dataset(
            dataset_name, "train", effective_data_root
        )
        val_dataset = _load_medmnist_dataset(dataset_name, "val", effective_data_root)

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

    # --- Create Student DataLoaders ---
    ts_moe_cfg = cfg.get("ts_moe", {})
    student_batch_size = ts_moe_cfg.get("student_batch_size", 256)

    train_loader, val_loader = create_student_dataloaders(
        patches=train_patches,
        routing_labels=train_routing_labels,
        batch_size=student_batch_size,
        num_workers=0,
        val_patches=val_patches,
        val_routing_labels=val_routing_labels,
    )
    if verbose:
        total_train = len(train_loader.dataset)
        total_val = len(val_loader.dataset) if val_loader else 0
        print(f"Student datasets: {total_train} train patches, {total_val} val patches")

    # --- Build Student model ---
    hidden_dims = tuple(ts_moe_cfg.get("student_hidden_dims", [32, 16]))

    student, _ = build_student_from_metadata(
        metadata=metadata,
        hidden_dims=hidden_dims,
        in_channels=student_in_channels,
    )
    student.to(device)

    total_params, trainable_params = count_parameters(student)
    if verbose:
        print(
            f"Student built: {total_params:,} params "
            f"({trainable_params:,} trainable), "
            f"hidden_dims={hidden_dims}"
        )
        if total_params > 5000:
            print(
                f"WARNING: Student has {total_params} params (target is <5k). "
                "Consider reducing hidden_dims."
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
            student, train_loader, optimizer, device, grad_clip, epoch_pbar
        )

        val_loss, val_acc = evaluate_student(student, val_loader, device)

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
