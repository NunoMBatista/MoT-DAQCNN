import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.daqcnn import DAQCNN
from src.utils.data import get_dataloaders
from src.utils.evaluate import evaluate
from src.utils.training_utils import resolve_device
from src.utils.plotting import plot_confusion_matrix, plot_loss_curves, plot_roc_curve
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)


def get_model_metadata(model, dummy_input=None):
    """Extract CNN head structure and parameter count from model.

    Args:
        model: The DAQCNN model
        dummy_input: Optional dummy input to initialize LazyLinear layers
    """
    metadata = {
        "total_params": 0,
        "trainable_params": 0,
        "head_structure": [],
        "quantum_kernel_size": None,
        "quantum_kernel_topologies": [],
    }

    # Extract quantum layer information if available
    if hasattr(model, "quantum_convolutional_layer"):
        q_layer = model.quantum_convolutional_layer
        if hasattr(q_layer, "kernel_size"):
            metadata["quantum_kernel_size"] = q_layer.kernel_size
        if hasattr(q_layer, "quantum_kernel") and hasattr(
            q_layer.quantum_kernel, "kernel_topology_names"
        ):
            metadata["quantum_kernel_topologies"] = list(
                q_layer.quantum_kernel.kernel_topology_names
            )

    # Count total parameters (skip uninitialized parameters)
    for param in model.parameters():
        if param.is_meta:
            # Skip uninitialized parameters
            continue
        try:
            metadata["total_params"] += param.numel()
            if param.requires_grad:
                metadata["trainable_params"] += param.numel()
        except ValueError:
            # Skip uninitialized lazy parameters
            continue

    # Extract head structure
    if hasattr(model, "head"):
        for i, layer in enumerate(model.head):
            layer_info = {
                "index": i,
                "type": layer.__class__.__name__,
            }

            # Add layer-specific details
            if isinstance(layer, nn.Conv2d):
                layer_info.update(
                    {
                        "in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": layer.kernel_size,
                        "stride": layer.stride,
                        "padding": layer.padding,
                    }
                )
            elif isinstance(layer, nn.Linear):
                if hasattr(layer, "in_features"):
                    layer_info["in_features"] = layer.in_features
                if hasattr(layer, "out_features"):
                    layer_info["out_features"] = layer.out_features
            elif isinstance(layer, nn.LazyLinear):
                # For LazyLinear, only out_features is known before initialization
                if hasattr(layer, "out_features"):
                    layer_info["out_features"] = layer.out_features
                if hasattr(layer, "in_features") and layer.in_features is not None:
                    layer_info["in_features"] = layer.in_features
            elif isinstance(layer, nn.BatchNorm2d):
                layer_info["num_features"] = layer.num_features
            elif isinstance(layer, nn.Dropout):
                layer_info["p"] = layer.p
            elif isinstance(layer, nn.MaxPool2d):
                layer_info["kernel_size"] = layer.kernel_size
                layer_info["stride"] = layer.stride

            metadata["head_structure"].append(layer_info)

    return metadata


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, optim, device, grad_clip, epoch_pbar):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    batch_pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (imgs, labels) in enumerate(batch_pbar):
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optim.zero_grad()
        batch_pbar.set_description("Forward pass")
        logits = model(imgs)

        batch_pbar.set_description("Computing loss")
        loss = loss_fn(logits, labels)

        batch_pbar.set_description("Backward pass")
        loss.backward()

        if grad_clip:
            batch_pbar.set_description("Clipping gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        batch_pbar.set_description("Optimizer step")
        optim.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, labels) * bs
        total_count += bs

        batch_pbar.set_description(
            f"Train batch {batch_idx + 1}/{len(loader)} | loss={loss.item():.4f}"
        )

    batch_pbar.close()
    return total_loss / total_count, total_acc / total_count


def run_single_seed(cfg, seed, output_dir, verbose=True, set_seed_fn=None):
    """Run experiment for a single seed."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running experiment with seed={seed}")
        print(f"{'=' * 60}\n")

    if set_seed_fn:
        set_seed_fn(seed)

    device = resolve_device(cfg, verbose=verbose)

    # Check for a cached quantum dataset before doing anything expensive
    using_cache = False
    cached_path = find_cached_quantum_dataset(cfg)

    # Get model config early (needed for requested_kernels)
    model_cfg = cfg.get("model", {})

    if cached_path is not None:
        if verbose:
            print(f"Found cached quantum dataset: {cached_path}")
            print("Skipping quantum computation, loading pre-computed features...\n")

        batch_size = cfg.get("dataset", {}).get("batch_size", 32)
        num_workers = cfg.get("dataset", {}).get("num_workers", 2)
        requested_kernels = model_cfg.get("kernel_topology_names", None)
        train_loader, val_loader, test_loader, n_classes, cache_meta = (
            load_cached_quantum_dataset(
                cached_path,
                batch_size,
                num_workers,
                requested_kernels=requested_kernels,
            )
        )
        using_cache = True
    else:
        if verbose:
            print("No cached quantum dataset found, will compute from scratch.\n")
            print("Loading data...")
        train_loader, val_loader, test_loader, n_classes = get_dataloaders(cfg)

    if verbose:
        print(
            f"Dataset loaded: {len(train_loader.dataset)} train, "
            f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples\n"
        )

    # Model
    num_classes = model_cfg.get("num_classes", n_classes)

    # When using cache, override the expected quantum output channels
    # This is needed for HSV datasets where output structure differs
    override_quantum_out_channels = None
    if using_cache and cache_meta:
        override_quantum_out_channels = cache_meta.get("out_channels")

    if verbose:
        print("Building model...")

    # Build model kwargs
    model_kwargs = {
        "num_classes": num_classes,
        "kernel_size": model_cfg.get("kernel_size", 2),
        "stride": model_cfg.get("stride", 1),
        "kernel_topology_names": model_cfg.get("kernel_topology_names", None),
        "scaling_factor": model_cfg.get("scaling_factor", 1.0),
        "evolution_time": model_cfg.get("evolution_time", 0.2),
        "mode": model_cfg.get("mode", "trotter"),
        "dropout": model_cfg.get("dropout", 0.1),
        "activation": model_cfg.get("activation", "relu"),
        "quantum_device": model_cfg.get("quantum_device", "default.qubit"),
        "quantum_device_kwargs": model_cfg.get("quantum_device_kwargs", None),
        "classical_device": device,
        "in_channels": model_cfg.get("in_channels", 1),
        "interface": model_cfg.get("interface", "torch"),
        "include_correlators": model_cfg.get("include_correlators", False),
        "head_hidden_channels": model_cfg.get("head_hidden_channels", 64),
    }

    # Add override if using cached data
    if override_quantum_out_channels is not None:
        model_kwargs["override_quantum_out_channels"] = override_quantum_out_channels

    model = DAQCNN(**model_kwargs)

    if using_cache:
        model.bypass_quantum = True
        if verbose:
            print("Quantum layer bypassed (using cached features)")

    model.to(device)

    if verbose:
        print(f"Model built and moved to {device}\n")

    # Optimizer
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    epochs = optim_cfg.get("epochs", 1)
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = optim_cfg.get("patience", None)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler (optional)
    use_scheduler = optim_cfg.get("use_scheduler", False)
    scheduler = None
    if use_scheduler:
        T_max = optim_cfg.get("scheduler_T_max", epochs)
        eta_min = float(optim_cfg.get("scheduler_eta_min", 0.0))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Early stopping variables
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Seed {seed}", leave=True)
    for epoch in epoch_pbar:
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, grad_clip, epoch_pbar
        )
        val_loss, val_acc = evaluate(model, val_loader, device, split_name="Val")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        dt = time.time() - t0

        epoch_pbar.set_description(
            f"Seed {seed} | Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"{dt:.1f}s"
        )

        if verbose:
            tqdm.write(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"{dt:.1f}s"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            if patience is not None:
                epochs_without_improvement = 0
        elif patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(
                        f"Early stopping triggered after {epoch} epochs (patience={patience})"
                    )
                break

        # Step the scheduler if enabled
        if scheduler is not None:
            scheduler.step()

    epoch_pbar.close()

    # Extract model metadata (after training, LazyLinear is initialized)
    model_metadata = get_model_metadata(model)

    if verbose:
        print(
            f"\nModel parameters: {model_metadata['total_params']:,} total, {model_metadata['trainable_params']:,} trainable"
        )

    # Save final model
    final_model_path = os.path.join(output_dir, f"final_model_seed_{seed}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "seed": seed,
            "num_classes": num_classes,
        },
        final_model_path,
    )
    if verbose:
        print(f"Final model saved to: {final_model_path}")

    # Save best model
    if best_model_state is not None:
        best_model_path = os.path.join(output_dir, f"best_model_seed_{seed}.pt")
        torch.save(
            {
                "model_state_dict": best_model_state,
                "config": cfg,
                "seed": seed,
                "num_classes": num_classes,
                "best_val_loss": best_val_loss,
            },
            best_model_path,
        )
        if verbose:
            print(f"Best model saved to: {best_model_path}")

    # Test evaluation with full metrics
    if verbose:
        print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, device, split_name="Test", compute_full_metrics=True
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Test Results (seed={seed})")
        print(f"{'=' * 60}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"{'=' * 60}\n")

    # Save loss plot for this seed
    seed_plot_path = os.path.join(output_dir, f"loss_curve_seed_{seed}.png")
    plot_loss_curves(train_losses, val_losses, seed_plot_path)
    if verbose:
        print(f"Loss curve saved to: {seed_plot_path}")

    # Save ROC curve for this seed
    roc_plot_path = os.path.join(output_dir, f"roc_curve_seed_{seed}.png")
    plot_roc_curve(
        test_metrics["labels"],
        test_metrics["probs"],
        roc_plot_path,
        num_classes=num_classes,
    )
    if verbose:
        print(f"ROC curve saved to: {roc_plot_path}")

    # Save confusion matrix for this seed
    cm_plot_path = os.path.join(output_dir, f"confusion_matrix_seed_{seed}.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        cm_plot_path,
        num_classes=num_classes,
    )
    if verbose:
        print(f"Confusion matrix saved to: {cm_plot_path}")

    return {
        "seed": seed,
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
        "model_metadata": model_metadata,
    }
