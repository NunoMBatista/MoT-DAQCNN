"""
Training loop for the Teacher MoE model.

Follows the same structure as ``daqcnn_training.py`` but adds:
    - Entropy regularization loss with lambda annealing
    - Alpha weight collection and histogram plotting per epoch
    - Global routing ratio tracking
    - Separate logging of CE loss and entropy loss components

Uses existing utilities for evaluation, plotting, and dataset loading.
"""

import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.teacher_moe import build_teacher_from_metadata
from src.utils.evaluate import accuracy, evaluate
from src.utils.losses import compute_lambda, entropy_loss
from src.utils.plotting import (
    plot_alpha_histogram,
    plot_alpha_histogram_combined,
    plot_confusion_matrix,
    plot_loss_curves,
    plot_roc_curve,
    plot_routing_ratio_over_epochs,
)
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)

# -------------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------------


def train_teacher_one_epoch(
    model, loader, optimizer, device, grad_clip, lam, epoch_pbar
):
    """Train the teacher for one epoch with entropy regularization.

    Args:
        model: TeacherMoE instance.
        loader: Training DataLoader (yields quantum features + labels).
        optimizer: Torch optimizer.
        device: Device string.
        grad_clip: Max gradient norm (0 to disable).
        lam: Current entropy regularization weight (lambda).
        epoch_pbar: Outer progress bar for description updates.

    Returns:
        Tuple of (avg_total_loss, avg_ce_loss, avg_ent_loss, avg_acc).
    """
    model.train()
    ce_fn = nn.CrossEntropyLoss()

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    ent_loss_sum = 0.0
    acc_sum = 0.0
    count = 0

    batch_pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (features, labels) in enumerate(batch_pbar):
        features = features.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()

        logits = model(features)
        ce = ce_fn(logits, labels)

        # Entropy regularization on the routing weights.
        # The SE block stores a live (gradient-attached) alpha during forward,
        # so the entropy loss can backprop through the gate parameters.
        # This gives the gate two gradient signals:
        #   1. From CE loss (via weighted features) — "which routing helps classification"
        #   2. From entropy loss (via alpha directly) — "be more decisive"
        alpha_live = model.se_block.last_alpha_live  # (B, M, H, W)
        ent = entropy_loss(alpha_live)

        loss = ce + lam * ent

        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = labels.size(0)
        total_loss_sum += loss.item() * bs
        ce_loss_sum += ce.item() * bs
        ent_loss_sum += ent.item() * bs
        acc_sum += accuracy(logits, labels) * bs
        count += bs

        batch_pbar.set_description(
            f"Train {batch_idx + 1}/{len(loader)} | "
            f"loss={loss.item():.4f} ce={ce.item():.4f} ent={ent.item():.4f}"
        )

    batch_pbar.close()
    n = max(count, 1)
    return total_loss_sum / n, ce_loss_sum / n, ent_loss_sum / n, acc_sum / n


@torch.no_grad()
def collect_alpha_weights(model, loader, device):
    """Collect alpha routing weights for the entire dataset.

    Runs the model in eval mode over every batch in ``loader`` and
    concatenates the alpha weights.

    Args:
        model: TeacherMoE instance.
        loader: DataLoader (typically the validation set).
        device: Device string.

    Returns:
        dict mapping kernel_name -> 1-D CPU tensor of alpha values
        across all patches in the dataset.
    """
    model.eval()
    all_alphas = {name: [] for name in model.kernel_names}

    for features, _ in loader:
        features = features.to(device)
        model(features)
        stats = model.get_routing_stats()
        for name in model.kernel_names:
            all_alphas[name].append(stats["alpha_flat"][name])

    # Concatenate
    for name in model.kernel_names:
        all_alphas[name] = torch.cat(all_alphas[name], dim=0)

    return all_alphas


@torch.no_grad()
def compute_routing_ratio(model, loader, device):
    """Compute global routing ratio across a full dataset.

    Returns:
        dict mapping kernel_name -> fraction of patches where that kernel won.
    """
    model.eval()
    wins = {name: 0 for name in model.kernel_names}
    total = 0

    for features, _ in loader:
        features = features.to(device)
        model(features)
        alpha = model.last_alpha  # (B, M, H, W)
        winners = alpha.argmax(dim=1)  # (B, H, W)
        for k, name in enumerate(model.kernel_names):
            wins[name] += (winners == k).sum().item()
        total += winners.numel()

    return {name: wins[name] / max(total, 1) for name in model.kernel_names}


# -------------------------------------------------------------------------
# Main training orchestrator
# -------------------------------------------------------------------------


def run_teacher_training(
    cfg, seed, output_dir, verbose=True, set_seed_fn=None, datasets_dir=None
):
    """Train the Teacher MoE model for one seed.

    Mirrors ``daqcnn_training.run_single_seed()`` in structure so the
    experiment runner can call either one interchangeably.

    Args:
        cfg: Loaded YAML config dict.
        seed: Random seed for this run.
        output_dir: Directory to save outputs (models, plots, logs).
        verbose: Whether to print progress.
        set_seed_fn: Optional callable to set global random seeds.
        datasets_dir: Optional path to search for cached quantum datasets.
            If None, uses the default ``data/quantum_datasets/`` directory.

    Returns:
        dict with all metrics, matching the format of ``run_single_seed()``.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Teacher MoE Training — seed={seed}")
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

    # --- Load cached quantum dataset (REQUIRED) ---
    cached_path = find_cached_quantum_dataset(cfg, datasets_dir=datasets_dir)
    if cached_path is None:
        raise FileNotFoundError(
            "No cached quantum dataset found matching the config.\n"
            "Run `python experiments/create_quantum_dataset.py` first to generate one."
        )

    if verbose:
        print(f"Using cached quantum dataset: {cached_path}")

    batch_size = cfg.get("dataset", {}).get("batch_size", 32)
    num_workers = cfg.get("dataset", {}).get("num_workers", 2)
    train_loader, val_loader, test_loader, n_classes, metadata = (
        load_cached_quantum_dataset(cached_path, batch_size, num_workers)
    )

    if verbose:
        print(
            f"Dataset: {len(train_loader.dataset)} train, "
            f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test"
        )

    # --- Build model ---
    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", n_classes)
    ts_moe_cfg = cfg.get("ts_moe", {})

    model = build_teacher_from_metadata(
        metadata=metadata,
        num_classes=num_classes,
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
        se_hidden_dim=ts_moe_cfg.get("se_hidden_dim", None),
    )
    model.to(device)

    if verbose:
        print(
            f"Kernels: {model.kernel_names} (M={model.num_kernels}, N={model.channels_per_kernel})"
        )

    # --- Optimizer ---
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    epochs = optim_cfg.get("epochs", 1)
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = optim_cfg.get("patience", None)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR scheduler (optional)
    scheduler = None
    if optim_cfg.get("use_scheduler", False):
        T_max = optim_cfg.get("scheduler_T_max", epochs)
        eta_min = float(optim_cfg.get("scheduler_eta_min", 0.0))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # --- Entropy regularization ---
    lambda_max = float(ts_moe_cfg.get("lambda_max", 0.1))
    lambda_warmup = int(ts_moe_cfg.get("lambda_warmup_epochs", epochs // 2))

    if verbose:
        print(
            f"Entropy regularization: lambda_max={lambda_max}, warmup={lambda_warmup} epochs"
        )

    # --- Training loop ---
    train_losses, val_losses = [], []
    train_ce_losses, train_ent_losses = [], []
    train_accs, val_accs = [], []
    lambda_history = []
    routing_history = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    # Create output subdirectory for alpha histograms
    alpha_hist_dir = os.path.join(output_dir, "alpha_histograms")
    os.makedirs(alpha_hist_dir, exist_ok=True)

    # Initialize LazyLinear by running one batch through the model
    with torch.no_grad():
        sample_features, _ = next(iter(train_loader))
        model(sample_features.to(device))

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"TeacherMoE built: {total_params:,} params ({trainable_params:,} trainable)"
        )

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Teacher seed {seed}", leave=True)
    for epoch in epoch_pbar:
        t0 = time.time()

        # Current lambda
        lam = compute_lambda(epoch - 1, lambda_warmup, lambda_max)
        lambda_history.append(lam)

        # Train
        total_loss, ce_loss, ent_loss, train_acc = train_teacher_one_epoch(
            model, train_loader, optimizer, device, grad_clip, lam, epoch_pbar
        )

        # Validate (uses existing evaluate() — returns (loss, acc))
        val_loss, val_acc = evaluate(model, val_loader, device, split_name="Val")

        # Record
        train_losses.append(total_loss)
        train_ce_losses.append(ce_loss)
        train_ent_losses.append(ent_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        dt = time.time() - t0

        # Routing stats on validation set
        routing_ratio = compute_routing_ratio(model, val_loader, device)
        routing_history.append(routing_ratio)

        # Alpha histograms — combined plot with all kernels in different colors
        alpha_vals = collect_alpha_weights(model, val_loader, device)
        hist_path = os.path.join(alpha_hist_dir, f"epoch_{epoch:03d}_all_kernels.png")
        plot_alpha_histogram_combined(alpha_vals, epoch, hist_path)

        epoch_pbar.set_description(
            f"Teacher seed {seed} | E{epoch}/{epochs} | "
            f"loss={total_loss:.4f} val={val_loss:.4f} | {dt:.1f}s"
        )

        if verbose:
            routing_str = " ".join(
                f"{name}={ratio:.1%}" for name, ratio in routing_ratio.items()
            )
            tqdm.write(
                f"Epoch {epoch}/{epochs} | "
                f"total={total_loss:.4f} ce={ce_loss:.4f} ent={ent_loss:.4f} "
                f"lambda={lam:.4f} | "
                f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"routing: {routing_str} | {dt:.1f}s"
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

    # --- Save models ---
    final_model_path = os.path.join(output_dir, f"teacher_final_seed_{seed}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "seed": seed,
            "num_classes": num_classes,
            "kernel_names": model.kernel_names,
            "metadata": metadata,
        },
        final_model_path,
    )
    if verbose:
        print(f"Final teacher saved to: {final_model_path}")

    if best_model_state is not None:
        best_model_path = os.path.join(output_dir, f"teacher_best_seed_{seed}.pt")
        torch.save(
            {
                "model_state_dict": best_model_state,
                "config": cfg,
                "seed": seed,
                "num_classes": num_classes,
                "best_val_loss": best_val_loss,
                "kernel_names": model.kernel_names,
                "metadata": metadata,
            },
            best_model_path,
        )
        if verbose:
            print(f"Best teacher saved to: {best_model_path}")

    # --- Test evaluation with full metrics ---
    if verbose:
        print("\nEvaluating teacher on test set...")

    test_metrics = evaluate(
        model, test_loader, device, split_name="Test", compute_full_metrics=True
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Teacher Test Results (seed={seed})")
        print(f"{'=' * 60}")
        print(f"Test Loss:     {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUC:      {test_metrics['auc']:.4f}")
        print(f"Test F1:       {test_metrics['f1']:.4f}")
        print(f"Test Recall:   {test_metrics['recall']:.4f}")
        print(f"{'=' * 60}\n")

    # --- Plots ---
    # Loss curves
    loss_path = os.path.join(output_dir, f"teacher_loss_curve_seed_{seed}.png")
    plot_loss_curves(train_losses, val_losses, loss_path)

    # ROC curve
    roc_path = os.path.join(output_dir, f"teacher_roc_curve_seed_{seed}.png")
    plot_roc_curve(
        test_metrics["labels"], test_metrics["probs"], roc_path, num_classes=num_classes
    )

    # Confusion matrix
    cm_path = os.path.join(output_dir, f"teacher_confusion_matrix_seed_{seed}.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], cm_path, num_classes=num_classes
    )

    # Routing ratio over epochs
    routing_path = os.path.join(output_dir, f"teacher_routing_ratio_seed_{seed}.png")
    plot_routing_ratio_over_epochs(routing_history, model.kernel_names, routing_path)

    if verbose:
        print(f"Plots saved to: {output_dir}")

    # --- Return results dict (same format as daqcnn_training.run_single_seed) ---
    return {
        "seed": seed,
        "architecture": "TS-MoE-Teacher",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ce_losses": train_ce_losses,
        "train_ent_losses": train_ent_losses,
        "lambda_history": lambda_history,
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
        "routing_history": routing_history,
        "kernel_names": model.kernel_names,
    }
