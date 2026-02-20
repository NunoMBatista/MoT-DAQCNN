"""
Training loop for the Final Classifier (Phase 3 of TS-MoE).

Takes a trained Student Gatekeeper, routes quantum features through it to
build sparse tensors, then trains a new classification head on those sparse
tensors from scratch.

The Teacher's head can't be reused because it learned on soft-mixed features
(all kernels weighted). The Final Classifier must learn that zeros mean
"not selected" rather than "low signal".

Pipeline:
    1. Load trained Student checkpoint and cached quantum features.
    2. Load original images for patch extraction (Student needs raw patches).
    3. Route each patch through the Student to get per-patch kernel decisions.
    4. Build sparse tensors: keep selected kernel's channels, zero others.
    5. Train a new classification head on the sparse tensors.
    6. Evaluate and compare with Teacher oracle and original DAQCNN baseline.

Uses existing utilities for evaluation, plotting, and dataset loading.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import DATA_DIR, DATASET_REGISTRY
from src.models.final_classifier import build_final_classifier_from_metadata
from src.models.student_gatekeeper import StudentGatekeeper
from src.utils.evaluate import accuracy, evaluate
from src.utils.kernel_mapping import build_kernel_to_channels_map, get_kernel_names
from src.utils.plotting import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_roc_curve,
)
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)
from src.utils.sparse_reconstruction import build_sparse_tensor_fast

# -------------------------------------------------------------------------
# Sparse dataset construction
# -------------------------------------------------------------------------


def _load_medmnist_dataset(dataset_name, split, data_root=None):
    """Load a MedMNIST dataset for a given split, no shuffling."""
    from torchvision import transforms

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
    return dataset_class(
        split=split, download=True, root=str(data_root), transform=transform
    )


def _extract_patches(images, kernel_size, stride):
    """Extract patches from a batch of images, same grid as quantum conv."""
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
    patches = unfold(images)  # (B, C*ks*ks, n_patches)
    return patches.transpose(1, 2)  # (B, n_patches, C*ks*ks)


@torch.no_grad()
def build_sparse_dataset(
    quantum_loader,
    student,
    raw_dataset,
    kernel_size,
    stride,
    color_space,
    channel_groups,
    device,
    use_mask_channel=False,
):
    """Route an entire split through the Student and produce sparse tensors.

    Iterates over the quantum DataLoader (unshuffled), extracts raw patches
    from the corresponding original images, runs the Student to get routing
    decisions, and assembles sparse tensors.

    Args:
        quantum_loader: DataLoader yielding (quantum_features, class_labels),
            must be unshuffled so indices align with raw_dataset.
        student: Trained StudentGatekeeper in eval mode.
        raw_dataset: Raw image dataset (indexable, yields (image, label)).
        kernel_size: Patch size.
        stride: Patch stride.
        color_space: Color space ("RGB", "HSV", "GRAYSCALE").
        channel_groups: List of lists — channel_groups[k] is the channel
            indices for kernel k.
        device: Device for student inference.
        use_mask_channel: Whether to append an extra mask channel.

    Returns:
        sparse_features: Tensor (N, C_out, H, W).
        class_labels: LongTensor (N,).
        routing_maps: LongTensor (N, H, W).
    """
    from src.utils.color_conversion import apply_color_conversion

    student.eval()
    all_sparse = []
    all_labels = []
    all_routing = []
    sample_idx = 0

    for q_features, q_labels in tqdm(
        quantum_loader, desc="Building sparse tensors", leave=False
    ):
        B = q_features.shape[0]
        _, _, H, W = q_features.shape

        # Gather corresponding raw images
        raw_images = []
        for i in range(B):
            img, _ = raw_dataset[sample_idx + i]
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img, dtype=torch.float32)
            raw_images.append(img)
        raw_images = torch.stack(raw_images)
        sample_idx += B

        # Same color conversion as quantum pipeline
        raw_images = apply_color_conversion(raw_images, color_space)
        if color_space == "HSV" and raw_images.shape[1] == 3:
            raw_images = raw_images[:, 2:3, :, :]

        # Extract patches and run student
        patches = _extract_patches(
            raw_images, kernel_size, stride
        )  # (B, n_patches, dim)
        flat_patches = patches.reshape(-1, patches.shape[-1]).to(device)
        routing_flat = student.predict(flat_patches).cpu()  # (B*n_patches,)
        routing_map = routing_flat.reshape(B, H, W)

        # Build sparse tensor for this batch
        sparse = build_sparse_tensor_fast(
            q_features, routing_map, channel_groups, use_mask_channel
        )

        all_sparse.append(sparse)
        all_routing.append(routing_map)

        labels = q_labels.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        all_labels.append(labels)

    return (
        torch.cat(all_sparse, dim=0),
        torch.cat(all_labels, dim=0).long(),
        torch.cat(all_routing, dim=0),
    )


# -------------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------------


def train_final_one_epoch(model, loader, optimizer, device, grad_clip, epoch_pbar):
    """Train the final classifier for one epoch.

    Args:
        model: FinalClassifier instance.
        loader: DataLoader yielding (sparse_features, class_labels).
        optimizer: Torch optimizer.
        device: Device string.
        grad_clip: Max gradient norm (0 to disable).
        epoch_pbar: Outer progress bar.

    Returns:
        Tuple of (avg_loss, avg_accuracy).
    """
    model.train()
    ce_fn = nn.CrossEntropyLoss()

    loss_sum = 0.0
    acc_sum = 0.0
    count = 0

    batch_pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (features, labels) in enumerate(batch_pbar):
        features = features.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = ce_fn(logits, labels)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = labels.size(0)
        loss_sum += loss.item() * bs
        acc_sum += accuracy(logits, labels) * bs
        count += bs

        batch_pbar.set_description(
            f"Train {batch_idx + 1}/{len(loader)} | loss={loss.item():.4f}"
        )

    batch_pbar.close()
    n = max(count, 1)
    return loss_sum / n, acc_sum / n


# -------------------------------------------------------------------------
# Routing analysis
# -------------------------------------------------------------------------


def compute_routing_analysis(routing_maps, class_labels, kernel_names):
    """Compute which kernels are used for which classes.

    Args:
        routing_maps: LongTensor (N, H, W) with routing decisions.
        class_labels: LongTensor (N,) with class labels.
        kernel_names: List of kernel names.

    Returns:
        Dict mapping class_idx -> dict mapping kernel_name -> fraction.
    """
    classes = sorted(class_labels.unique().tolist())

    analysis = {}
    for c in classes:
        mask = class_labels == c
        routes = routing_maps[mask]  # (Nc, H, W)
        total = routes.numel()
        if total == 0:
            analysis[c] = {name: 0.0 for name in kernel_names}
            continue
        per_kernel = {}
        for k, name in enumerate(kernel_names):
            per_kernel[name] = (routes == k).sum().item() / total
        analysis[c] = per_kernel

    return analysis


# -------------------------------------------------------------------------
# Student checkpoint loading
# -------------------------------------------------------------------------


def _load_student_from_checkpoint(ckpt_path, device):
    """Load a trained Student from checkpoint.

    Args:
        ckpt_path: Path to the .pt checkpoint saved by ``run_student_training()``.
        device: Device to place the model on.

    Returns:
        Tuple of (student, metadata, kernel_names, num_kernels).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    metadata = ckpt["metadata"]
    kernel_names = ckpt["kernel_names"]
    num_kernels = ckpt["num_kernels"]
    patch_dim = ckpt["patch_dim"]
    hidden_dims = tuple(ckpt["hidden_dims"])

    student = StudentGatekeeper(
        patch_dim=patch_dim,
        num_kernels=num_kernels,
        hidden_dims=hidden_dims,
    )
    student.load_state_dict(ckpt["model_state_dict"])
    student.to(device)
    student.eval()

    return student, metadata, kernel_names, num_kernels


# -------------------------------------------------------------------------
# Main training orchestrator
# -------------------------------------------------------------------------


def run_final_classifier_training(
    cfg,
    seed,
    output_dir,
    student_ckpt_path,
    verbose=True,
    set_seed_fn=None,
    datasets_dir=None,
    data_root=None,
    raw_image_datasets=None,
    teacher_test_acc=None,
    baseline_test_acc=None,
):
    """Train the Final Classifier on sparse routed tensors.

    Mirrors ``run_teacher_training()`` and ``run_student_training()`` in
    structure so the experiment runner can chain all three phases.

    Args:
        cfg: Loaded YAML config dict.
        seed: Random seed.
        output_dir: Directory to save outputs.
        student_ckpt_path: Path to the trained Student checkpoint (.pt).
        verbose: Whether to print progress.
        set_seed_fn: Optional callable to set global random seeds.
        datasets_dir: Optional path for cached quantum datasets.
        data_root: Optional MedMNIST data root.
        raw_image_datasets: Optional dict ``{"train": ds, "val": ds, "test": ds}``
            of raw image datasets. When supplied, skips MedMNIST download.
        teacher_test_acc: Teacher oracle accuracy for comparison logging.
        baseline_test_acc: Original DAQCNN baseline accuracy for comparison.

    Returns:
        dict with all metrics.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Final Classifier Training — seed={seed}")
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

    # --- Load trained Student ---
    if verbose:
        print(f"Loading student from: {student_ckpt_path}")
    student, metadata, kernel_names, num_kernels = _load_student_from_checkpoint(
        student_ckpt_path, device
    )
    if verbose:
        print(f"Student loaded: {num_kernels} kernels ({kernel_names})")

    # --- Load cached quantum dataset ---
    cached_path = find_cached_quantum_dataset(cfg, datasets_dir=datasets_dir)
    if cached_path is None:
        raise FileNotFoundError(
            "No cached quantum dataset found matching the config.\n"
            "Run `python experiments/create_quantum_dataset.py` first."
        )

    batch_size = cfg.get("dataset", {}).get("batch_size", 32)
    # Use shuffle=False loaders for sparse dataset construction (index alignment!)
    q_train_loader, q_val_loader, q_test_loader, n_classes, q_metadata = (
        load_cached_quantum_dataset(cached_path, batch_size, num_workers=0)
    )
    # Rebuild unshuffled train loader for sparse construction
    data = np.load(cached_path, allow_pickle=True)
    train_features = torch.from_numpy(data["train_features"]).float()
    train_labels_t = torch.from_numpy(data["train_labels"]).long()
    q_train_unshuffled = DataLoader(
        TensorDataset(train_features, train_labels_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if verbose:
        print(
            f"Quantum dataset: {len(q_train_unshuffled.dataset)} train, "
            f"{len(q_val_loader.dataset)} val, {len(q_test_loader.dataset)} test"
        )

    # --- Kernel mapping ---
    channel_kernel_map = metadata["channel_kernel_map"]
    kernel_to_channels = build_kernel_to_channels_map(channel_kernel_map)
    ordered_kernel_names = get_kernel_names(kernel_to_channels)
    channel_groups = [kernel_to_channels[name] for name in ordered_kernel_names]

    # --- Load raw images ---
    dataset_name = cfg.get("dataset", {}).get("name", "pneumonia_mnist")
    color_space = cfg.get("dataset", {}).get("color_space", "RGB")
    kernel_size = metadata["kernel_size"]
    stride = metadata["stride"]

    ts_moe_cfg = cfg.get("ts_moe", {})
    use_mask_channel = ts_moe_cfg.get("use_mask_channel", False)

    effective_data_root = data_root or cfg.get("dataset", {}).get(
        "data_root", str(DATA_DIR)
    )

    if raw_image_datasets is not None:
        raw_train = raw_image_datasets["train"]
        raw_val = raw_image_datasets["val"]
        raw_test = raw_image_datasets.get("test", raw_image_datasets["val"])
        if verbose:
            print("Using provided raw image datasets")
    else:
        if verbose:
            print(f"Loading original images ({dataset_name}, {color_space})...")
        raw_train = _load_medmnist_dataset(dataset_name, "train", effective_data_root)
        raw_val = _load_medmnist_dataset(dataset_name, "val", effective_data_root)
        raw_test = _load_medmnist_dataset(dataset_name, "test", effective_data_root)

    # --- Build sparse datasets ---
    if verbose:
        print("Building sparse training set...")
    t0_route = time.time()

    sparse_train, labels_train, routing_train = build_sparse_dataset(
        q_train_unshuffled,
        student,
        raw_train,
        kernel_size,
        stride,
        color_space,
        channel_groups,
        device,
        use_mask_channel,
    )
    sparse_val, labels_val, routing_val = build_sparse_dataset(
        q_val_loader,
        student,
        raw_val,
        kernel_size,
        stride,
        color_space,
        channel_groups,
        device,
        use_mask_channel,
    )
    sparse_test, labels_test, routing_test = build_sparse_dataset(
        q_test_loader,
        student,
        raw_test,
        kernel_size,
        stride,
        color_space,
        channel_groups,
        device,
        use_mask_channel,
    )

    routing_time = time.time() - t0_route
    if verbose:
        print(
            f"Sparse datasets built in {routing_time:.1f}s — "
            f"train: {sparse_train.shape}, val: {sparse_val.shape}, test: {sparse_test.shape}"
        )

    # Create DataLoaders for training
    train_loader = DataLoader(
        TensorDataset(sparse_train, labels_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(sparse_val, labels_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(sparse_test, labels_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --- Build Final Classifier ---
    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", n_classes)

    model = build_final_classifier_from_metadata(
        metadata=metadata,
        num_classes=num_classes,
        use_mask_channel=use_mask_channel,
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
    )
    model.to(device)

    # Initialize LazyLinear
    with torch.no_grad():
        sample, _ = next(iter(train_loader))
        model(sample.to(device))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(
            f"FinalClassifier built: {total_params:,} params "
            f"({trainable_params:,} trainable)"
        )

    # --- Optimizer ---
    optim_cfg = cfg.get("optim", {})
    lr = float(ts_moe_cfg.get("final_lr", optim_cfg.get("lr", 1e-3)))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    final_epochs = int(ts_moe_cfg.get("final_epochs", optim_cfg.get("epochs", 20)))
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = ts_moe_cfg.get("final_patience", optim_cfg.get("patience", None))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if ts_moe_cfg.get("final_use_scheduler", optim_cfg.get("use_scheduler", False)):
        T_max = ts_moe_cfg.get("final_scheduler_T_max", final_epochs)
        eta_min = float(ts_moe_cfg.get("final_scheduler_eta_min", 0.0))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    if verbose:
        print(f"Optimizer: Adam lr={lr}, epochs={final_epochs}, patience={patience}")

    # --- Training loop ---
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    epoch_pbar = tqdm(range(1, final_epochs + 1), desc=f"Final seed {seed}", leave=True)
    for epoch in epoch_pbar:
        t0 = time.time()

        train_loss, train_acc = train_final_one_epoch(
            model, train_loader, optimizer, device, grad_clip, epoch_pbar
        )

        val_loss, val_acc = evaluate(model, val_loader, device, split_name="Val")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        dt = time.time() - t0

        epoch_pbar.set_description(
            f"Final seed {seed} | E{epoch}/{final_epochs} | "
            f"loss={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s"
        )

        if verbose:
            tqdm.write(
                f"Epoch {epoch}/{final_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s"
            )

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
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

    # --- Restore best model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # --- Test evaluation with full metrics ---
    if verbose:
        print("\nEvaluating final classifier on test set...")

    t0_infer = time.time()
    test_metrics = evaluate(
        model, test_loader, device, split_name="Test", compute_full_metrics=True
    )
    inference_time = time.time() - t0_infer

    # Full pipeline time = routing + inference
    pipeline_time = routing_time + inference_time

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Final Classifier Test Results (seed={seed})")
        print(f"{'=' * 60}")
        print(f"Test Loss:     {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUC:      {test_metrics['auc']:.4f}")
        print(f"Test F1:       {test_metrics['f1']:.4f}")
        print(f"Test Recall:   {test_metrics['recall']:.4f}")

    # --- Comparison with Teacher and Baseline ---
    if verbose:
        print("\n--- Performance Comparison ---")
        if teacher_test_acc is not None:
            delta_t = test_metrics["accuracy"] - teacher_test_acc
            print(f"Teacher Oracle Acc:   {teacher_test_acc:.4f}")
            print(
                f"Final Classifier Acc: {test_metrics['accuracy']:.4f}  (Δ = {delta_t:+.4f})"
            )
        if baseline_test_acc is not None:
            delta_b = test_metrics["accuracy"] - baseline_test_acc
            print(f"DAQCNN Baseline Acc:  {baseline_test_acc:.4f}")
            print(
                f"Final Classifier Acc: {test_metrics['accuracy']:.4f}  (Δ = {delta_b:+.4f})"
            )

        # Speedup factor: only 1/M of kernels run per patch
        speedup = num_kernels  # ideally M× fewer quantum circuits
        print(
            f"\nSpeedup factor:       {speedup}× (routing 1/{num_kernels} kernels per patch)"
        )
        print(f"Routing time:         {routing_time:.2f}s")
        print(f"Inference time:       {inference_time:.2f}s")
        print(f"Full pipeline time:   {pipeline_time:.2f}s")
        print(f"{'=' * 60}\n")

    # --- Routing analysis ---
    routing_analysis = compute_routing_analysis(
        routing_test, labels_test, ordered_kernel_names
    )
    if verbose:
        print("Routing analysis (test set — fraction of patches per kernel, by class):")
        header = "  Class  | " + " | ".join(f"{n:>10s}" for n in ordered_kernel_names)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for c in sorted(routing_analysis.keys()):
            vals = " | ".join(
                f"{routing_analysis[c][n]:>10.1%}" for n in ordered_kernel_names
            )
            print(f"  {c:>5d}  | {vals}")
        print()

    # --- Save models ---
    os.makedirs(output_dir, exist_ok=True)

    final_model_path = os.path.join(output_dir, f"final_classifier_seed_{seed}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "seed": seed,
            "num_classes": num_classes,
            "kernel_names": ordered_kernel_names,
            "metadata": metadata,
            "use_mask_channel": use_mask_channel,
            "test_accuracy": test_metrics["accuracy"],
        },
        final_model_path,
    )
    if verbose:
        print(f"Final classifier saved to: {final_model_path}")

    if best_model_state is not None:
        best_path = os.path.join(output_dir, f"final_classifier_best_seed_{seed}.pt")
        torch.save(
            {
                "model_state_dict": best_model_state,
                "config": cfg,
                "seed": seed,
                "num_classes": num_classes,
                "kernel_names": ordered_kernel_names,
                "metadata": metadata,
                "use_mask_channel": use_mask_channel,
                "best_val_loss": best_val_loss,
                "test_accuracy": test_metrics["accuracy"],
            },
            best_path,
        )
        if verbose:
            print(f"Best final classifier saved to: {best_path}")

    # --- Plots ---
    loss_path = os.path.join(output_dir, f"final_loss_curve_seed_{seed}.png")
    plot_loss_curves(train_losses, val_losses, loss_path)

    roc_path = os.path.join(output_dir, f"final_roc_curve_seed_{seed}.png")
    plot_roc_curve(
        test_metrics["labels"], test_metrics["probs"], roc_path, num_classes=num_classes
    )

    cm_path = os.path.join(output_dir, f"final_confusion_matrix_seed_{seed}.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], cm_path, num_classes=num_classes
    )

    if verbose:
        print(f"Plots saved to: {output_dir}")

    # --- Build comparison summary ---
    comparison = {
        "final_classifier_acc": test_metrics["accuracy"],
        "teacher_oracle_acc": teacher_test_acc,
        "baseline_acc": baseline_test_acc,
        "speedup_factor": num_kernels,
        "routing_time_s": routing_time,
        "inference_time_s": inference_time,
        "pipeline_time_s": pipeline_time,
    }

    # --- Return results dict ---
    return {
        "seed": seed,
        "architecture": "TS-MoE-FinalClassifier",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_recall": test_metrics["recall"],
        "test_probs": test_metrics["probs"].tolist(),
        "test_labels": test_metrics["labels"].tolist(),
        "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "num_classes": num_classes,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "kernel_names": ordered_kernel_names,
        "num_kernels": num_kernels,
        "use_mask_channel": use_mask_channel,
        "comparison": comparison,
        "routing_analysis": {
            str(c): {name: frac for name, frac in per_kernel.items()}
            for c, per_kernel in routing_analysis.items()
        },
        "routing_time_s": routing_time,
        "classifier_params": total_params,
    }
