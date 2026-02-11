import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tqdm import tqdm

from src.utils.data import get_dataloaders
from src.utils.model_cache_manager import load_model_from_checkpoint
from src.utils.plotting import plot_confusion_matrix, plot_roc_curve


def evaluate_model(model, loader, device):
    """Evaluate model on a dataset and return metrics."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            labels = labels.squeeze().long().to(device)

            logits = model(imgs)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    preds = all_logits.argmax(dim=1).cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    probs = torch.softmax(all_logits, dim=1).cpu().numpy()

    acc = (preds == labels_np).mean()

    # AUC
    if probs.shape[1] == 2:
        auc = roc_auc_score(labels_np, probs[:, 1])
    else:
        auc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")

    f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    recall = recall_score(labels_np, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels_np, preds)

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "recall": float(recall),
        "confusion_matrix": cm,
        "probs": probs,
        "labels": labels_np,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference with pre-trained DAQCNN")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "pneumonia_mnist",
            "breast_mnist",
            "path_mnist",
            "derma_mnist",
            "tissue_mnist",
        ],
        required=True,
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to checkpoint directory)",
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    print("Model loaded successfully")

    # Get dataset config
    cfg = checkpoint.get("config", {})
    cfg["dataset"]["name"] = args.dataset  # Override dataset

    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(cfg)

    # Select the appropriate loader
    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    print(f"Evaluating on {args.split} split ({len(loader.dataset)} samples)...")

    # Evaluate
    metrics = evaluate_model(model, loader, device)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results on {args.dataset} ({args.split} split)")
    print(f"{'=' * 60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC:      {metrics['auc']:.4f}")
    print(f"F1:       {metrics['f1']:.4f}")
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"{'=' * 60}\n")

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.checkpoint)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    # Save plots
    checkpoint_name = os.path.basename(args.checkpoint).replace(".pt", "")

    cm_path = os.path.join(
        output_dir, f"{checkpoint_name}_inference_{args.dataset}_{args.split}_cm.png"
    )
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path, num_classes=n_classes)
    print(f"Confusion matrix saved to: {cm_path}")

    roc_path = os.path.join(
        output_dir, f"{checkpoint_name}_inference_{args.dataset}_{args.split}_roc.png"
    )
    plot_roc_curve(metrics["labels"], metrics["probs"], roc_path, num_classes=n_classes)
    print(f"ROC curve saved to: {roc_path}")


if __name__ == "__main__":
    main()
