"""
quick_plot_samples.py

Quick experiment script to load and plot random samples from any MedMNIST dataset.

Usage:
    python experiments/quick_plot_samples.py
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DATASET_CHANNELS, DATASET_REGISTRY


def load_dataset(dataset_name, split="train"):
    """Load a MedMNIST dataset."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    data_flag, class_name = DATASET_REGISTRY[dataset_name]

    try:
        from medmnist import INFO
        from torchvision import transforms

        dataset_class = getattr(
            __import__("medmnist", fromlist=[class_name]), class_name
        )
    except ImportError as exc:
        raise ImportError(
            f"medmnist is required. Install with `pip install medmnist`."
        ) from exc

    info = INFO[data_flag]
    n_classes = len(info["label"])

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = dataset_class(
        split=split,
        download=True,
        root=str(DATA_DIR),
        transform=transform,
    )

    return dataset, info, n_classes


def plot_random_samples(dataset, info, n_samples=16, seed=42):
    """Plot random samples from the dataset."""
    random.seed(seed)
    np.random.seed(seed)

    # Select random indices
    n_total = len(dataset)
    indices = random.sample(range(n_total), min(n_samples, n_total))

    # Calculate grid dimensions
    n_cols = 4
    n_rows = (len(indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    label_dict = info.get("label", {})

    for idx, sample_idx in enumerate(indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get image and label
        image, label = dataset[sample_idx]

        # Convert tensor to numpy and handle channel dimension
        img_np = image.numpy()

        # Handle different channel formats
        if img_np.shape[0] == 1:  # Grayscale (C, H, W) -> (H, W)
            img_np = img_np.squeeze(0)
            ax.imshow(img_np, cmap="gray")
        elif img_np.shape[0] == 3:  # RGB (C, H, W) -> (H, W, C)
            img_np = np.transpose(img_np, (1, 2, 0))
            ax.imshow(img_np)
        else:
            img_np = img_np.squeeze()
            ax.imshow(img_np, cmap="gray")

        # Get label name if available
        label_val = label.item() if torch.is_tensor(label) else int(label)
        label_name = label_dict.get(str(label_val), str(label_val))
        ax.set_title(f"Label: {label_name}", fontsize=10)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(len(indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot random samples from MedMNIST datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_mnist",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to load",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
        help="Number of samples to plot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the plot (optional)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("QUICK DATASET SAMPLE PLOTTER")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Seed: {args.seed}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset, info, n_classes = load_dataset(args.dataset, args.split)
    print(f"Loaded {len(dataset)} samples with {n_classes} classes")
    print(f"Dataset info: {info['description']}")
    print()

    # Plot samples
    print("Plotting random samples...")
    fig = plot_random_samples(dataset, info, args.n_samples, args.seed)

    dataset_type = "RGB" if DATASET_CHANNELS.get(args.dataset, 1) == 3 else "Grayscale"
    fig.suptitle(
        f"{args.dataset} - {args.split} split ({dataset_type})",
        fontsize=14,
        fontweight="bold",
    )

    # Save or show
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        print("Displaying plot...")
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
