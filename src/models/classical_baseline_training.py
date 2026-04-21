import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.classical_baseline import ClassicalBaselineCNN
from src.utils.data import get_dataloaders
from src.utils.evaluate import evaluate
from src.utils.training_utils import resolve_device
from src.utils.plotting import plot_confusion_matrix, plot_loss_curves, plot_roc_curve
from src.models.daqcnn_training import train_one_epoch, get_model_metadata


def run_classical_baseline(cfg, seed, output_dir, verbose=True, set_seed_fn=None):
    """Run experiment for the Classical Baseline CNN."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running classical baseline experiment with seed={seed}")
        print(f"{'=' * 60}\n")

    if set_seed_fn:
        set_seed_fn(seed)

    device = resolve_device(cfg, verbose=verbose)

    if verbose:
        print("Loading data...")
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(cfg)

    if verbose:
        print(
            f"Dataset loaded: {len(train_loader.dataset)} train, "
            f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples\n"
        )

    # Model
    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", n_classes)

    if verbose:
        print("Building Classical Baseline model...")

    # Determine out_channels based on config. If missing, assume 45 for 1-kernel ZZ, 180 for 4-kernel ZZ
    out_channels = model_cfg.get("out_channels", 180)

    model = ClassicalBaselineCNN(
        num_classes=num_classes,
        in_channels=model_cfg.get("in_channels", 1),
        kernel_size=model_cfg.get("kernel_size", 3),
        stride=model_cfg.get("stride", 3),
        out_channels=out_channels,
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
        head_hidden_channels=model_cfg.get("head_hidden_channels", 64),
        fixed_random_filters=model_cfg.get("fixed_random_filters", False),
    )

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

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=weight_decay
    )

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

    # Extract model metadata
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
