import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.daqcnn import DAQCNN  # noqa: E402
from src.utils.data import get_dataloaders  # noqa: E402
from src.utils.plotting import (  # noqa: E402
    plot_confusion_matrix,
    plot_loss_curves,
    plot_multi_seed_loss_curves,
    plot_multi_seed_roc_curves,
    plot_roc_curve,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute accuracy, AUC, F1, and Recall metrics."""
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    acc = (preds == labels_np).mean()

    # For binary classification
    if probs.shape[1] == 2:
        auc = roc_auc_score(labels_np, probs[:, 1])
    else:
        # Multi-class: use one-vs-rest
        auc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")

    f1 = f1_score(labels_np, preds, average="macro", zero_division="warn")
    recall = recall_score(labels_np, preds, average="macro", zero_division="warn")
    cm = confusion_matrix(labels_np, preds)

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "recall": float(recall),
        "confusion_matrix": cm,
    }


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


def evaluate(model, loader, device, split_name="Val", compute_full_metrics=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    all_logits = []
    all_labels = []

    batch_pbar = tqdm(loader, desc=f"{split_name} evaluation", leave=False)
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(batch_pbar):
            imgs = imgs.to(device)
            labels = labels.squeeze().long().to(device)

            batch_pbar.set_description(f"{split_name} forward pass")
            logits = model(imgs)
            loss = loss_fn(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, labels) * bs
            total_count += bs

            if compute_full_metrics:
                all_logits.append(logits)
                all_labels.append(labels)

            batch_pbar.set_description(
                f"{split_name} batch {batch_idx + 1}/{len(loader)} | loss={loss.item():.4f}"
            )

    batch_pbar.close()

    if compute_full_metrics:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_logits, all_labels)
        metrics["loss"] = total_loss / total_count
        # Add probabilities and labels for ROC plotting
        probs = torch.softmax(all_logits, dim=1).cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        metrics["probs"] = probs
        metrics["labels"] = labels_np
        return metrics
    else:
        return total_loss / total_count, total_acc / total_count


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_seed(cfg, seed, output_dir, verbose=True):
    """Run experiment for a single seed."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running experiment with seed={seed}")
        print(f"{'=' * 60}\n")

    set_seed(seed)

    # Determine device
    requested_device = cfg.get("model", {}).get("classical_device", "auto")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = requested_device

    if verbose:
        print(f"Device: {device}")

    # Data
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
        print("Building model...")
    model = DAQCNN(
        num_classes=num_classes,
        kernel_size=model_cfg.get("kernel_size", 2),
        stride=model_cfg.get("stride", 1),
        kernel_topology_names=model_cfg.get("kernel_topology_names", None),
        scaling_factor=model_cfg.get("scaling_factor", 1.0),
        mode=model_cfg.get("mode", "trotter"),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
        quantum_device=model_cfg.get("quantum_device", "default.qubit"),
        quantum_device_kwargs=model_cfg.get("quantum_device_kwargs", None),
        classical_device=device,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

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

    epoch_pbar.close()

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
    }


def main():
    parser = argparse.ArgumentParser(description="Robust DAQCNN experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "pneumonia_mnist.yml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Get seeds - can be single value or list
    seed_spec = cfg.get("misc", {}).get("seed", 0)
    if isinstance(seed_spec, list):
        seeds = seed_spec
    else:
        seeds = [seed_spec]

    print(f"\n{'=' * 60}")
    print(f"Robust DAQCNN Experiment")
    print(f"{'=' * 60}")
    print(f"Config: {args.config}")
    print(f"Seeds: {seeds}")
    print(f"Number of runs: {len(seeds)}")
    print(f"{'=' * 60}\n")

    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Save config to output directory
    config_save_path = os.path.join(output_dir, "config.yml")
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}\n")

    # Run experiments for each seed
    all_results = []
    for seed in seeds:
        result = run_single_seed(cfg, seed, output_dir, verbose=(len(seeds) == 1))
        all_results.append(result)

    # Aggregate results
    print(f"\n{'=' * 60}")
    print(f"Summary of All Runs")
    print(f"{'=' * 60}\n")

    # Save individual results
    individual_results_path = os.path.join(output_dir, "individual_results.json")
    with open(individual_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Individual results saved to: {individual_results_path}")

    # Compute and save aggregate metrics
    metrics = {
        "test_loss": [r["test_loss"] for r in all_results],
        "test_acc": [r["test_acc"] for r in all_results],
        "test_auc": [r["test_auc"] for r in all_results],
        "test_f1": [r["test_f1"] for r in all_results],
        "test_recall": [r["test_recall"] for r in all_results],
        "final_train_loss": [r["final_train_loss"] for r in all_results],
        "final_val_loss": [r["final_val_loss"] for r in all_results],
        "final_train_acc": [r["final_train_acc"] for r in all_results],
        "final_val_acc": [r["final_val_acc"] for r in all_results],
    }

    aggregate_stats = {}
    for metric_name, values in metrics.items():
        aggregate_stats[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    aggregate_stats_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(aggregate_stats_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Aggregate metrics saved to: {aggregate_stats_path}\n")

    # Print summary
    print(f"{'=' * 60}")
    print(f"Performance Summary (n={len(seeds)} runs)")
    print(f"{'=' * 60}")
    for metric_name, stats in aggregate_stats.items():
        print(f"{metric_name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"{'=' * 60}\n")

    # Plot multi-seed loss curves if multiple seeds
    if len(seeds) > 1:
        all_train_losses = [r["train_losses"] for r in all_results]
        all_val_losses = [r["val_losses"] for r in all_results]
        multi_seed_plot_path = os.path.join(output_dir, "loss_curve_multi_seed.png")
        plot_multi_seed_loss_curves(
            all_train_losses, all_val_losses, multi_seed_plot_path
        )
        print(f"Multi-seed loss curve saved to: {multi_seed_plot_path}\n")

        # Plot multi-seed ROC curves
        all_test_labels = [np.array(r["test_labels"]) for r in all_results]
        all_test_probs = [np.array(r["test_probs"]) for r in all_results]
        num_classes = all_results[0]["num_classes"]
        multi_roc_plot_path = os.path.join(output_dir, "roc_curve_multi_seed.png")
        plot_multi_seed_roc_curves(
            all_test_labels,
            all_test_probs,
            multi_roc_plot_path,
            num_classes=num_classes,
        )
        print(f"Multi-seed ROC curve saved to: {multi_roc_plot_path}\n")

    print(f"All outputs saved to: {output_dir}")
    print(f"\nExperiment complete!")


if __name__ == "__main__":
    main()
