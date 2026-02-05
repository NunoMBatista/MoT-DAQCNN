#!/usr/bin/env python3
"""
augment_classical_dimensionality.py

Create a "fair fight" baseline by projecting classical image data to the same
dimensionality as the quantum-transformed data using random feature expansion.

This helps verify whether any improvements from quantum features are due to
the quantum transformation itself, or just from having more features.

Three methods are available:
    1. Random Fourier Features (RFF): Approximates an RBF kernel
    2. Random Matrix Projection: Simple random linear projection
    3. Random CNN: Random convolutional kernels (non-trained)

Usage:
    python experiments/augment_classical_dimensionality.py

Modify the parameters below to match your quantum dataset's output shape.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# =============================================================================
# PARAMETERS - Edit these to configure the augmented dataset generation
# =============================================================================

# Which dataset to use: "pneumonia_mnist", "breast_mnist", "path_mnist", "derma_mnist"
DATASET_NAME = "pneumonia_mnist"

# Target output shape to match quantum dataset
# These should match the quantum dataset you want to compare against
# For example, if quantum output is (N, 16, 14, 14), set:
#   TARGET_CHANNELS = 16
#   TARGET_H = 14
#   TARGET_W = 14
TARGET_CHANNELS = 16  # num_kernels * kernel_size^2 (e.g., 4 topologies * 4 qubits = 16)
TARGET_H = 14  # (28 - kernel_size) // stride + 1
TARGET_W = 14

# Projection method: "rff", "random", or "cnn"
#   - "rff": Random Fourier Features (approximates RBF kernel)
#   - "random": Random matrix projection
#   - "cnn": Random CNN kernels (convolution with random weights)
PROJECTION_METHOD = "cnn"

# For RFF: gamma parameter for RBF kernel approximation (smaller = smoother)
# If None, will use 1/n_features as default
RFF_GAMMA = None

# For CNN method: kernel size and stride
# These determine the output spatial dimensions
CNN_KERNEL_SIZE = 2
CNN_STRIDE = 2

# Random seed for reproducibility
RANDOM_SEED = 42

# Which splits to process
SPLITS = ["train", "val", "test"]


# =============================================================================
# Helper functions
# =============================================================================


def load_medmnist_dataset(dataset_name, split, data_root):
    """Load a MedMNIST dataset by name and split."""

    name_to_class = {
        "pneumonia_mnist": "PneumoniaMNIST",
        "breast_mnist": "BreastMNIST",
        "path_mnist": "PathMNIST",
        "derma_mnist": "DermaMNIST",
    }

    if dataset_name not in name_to_class:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    class_name = name_to_class[dataset_name]

    try:
        import medmnist

        dataset_class = getattr(medmnist, class_name)
    except ImportError:
        raise ImportError("medmnist is required. Install with: pip install medmnist")

    transform = transforms.ToTensor()
    ds = dataset_class(
        split=split,
        download=True,
        root=str(data_root),
        transform=transform,
    )

    return ds


def extract_images_and_labels(dataset):
    """Extract all images and labels from a dataset as numpy arrays."""
    images = []
    labels = []

    for img, lbl in tqdm(dataset, desc="  Loading"):
        images.append(img.numpy())
        lbl_val = lbl.item() if hasattr(lbl, "item") else lbl[0]
        labels.append(lbl_val)

    X = np.array(images)  # (N, C, H, W)
    y = np.array(labels)  # (N,)

    return X, y


class RandomFourierFeatures:
    """
    Random Fourier Features for approximating RBF kernel.

    The RBF kernel k(x, y) = exp(-gamma * ||x - y||^2) can be approximated by:
        k(x, y) ≈ z(x)^T z(y)
    where z(x) = sqrt(2/D) * cos(Wx + b)

    W is drawn from N(0, 2*gamma) and b from Uniform(0, 2*pi).
    """

    def __init__(self, n_input, n_output, gamma=None, seed=None):
        self.n_input = n_input
        self.n_output = n_output

        # Default gamma = 1 / n_features (common heuristic)
        self.gamma = gamma if gamma is not None else 1.0 / n_input

        rng = np.random.RandomState(seed)

        # Sample random frequencies from N(0, sqrt(2*gamma))
        # Note: for RBF, we need W ~ N(0, 2*gamma*I), so std = sqrt(2*gamma)
        self.W = rng.randn(n_input, n_output) * np.sqrt(2 * self.gamma)

        # Sample random phases from Uniform(0, 2*pi)
        self.b = rng.uniform(0, 2 * np.pi, size=(1, n_output))

    def transform(self, X):
        """
        Transform input X using random Fourier features.

        Args:
            X: array of shape (N, n_input)
        Returns:
            Z: array of shape (N, n_output)
        """
        # z(x) = sqrt(2/D) * cos(Wx + b)
        projection = X @ self.W + self.b
        Z = np.sqrt(2.0 / self.n_output) * np.cos(projection)
        return Z


class RandomProjection:
    """
    Simple random matrix projection.

    Projects data using a random matrix with entries drawn from N(0, 1/sqrt(n_output)).
    This preserves distances approximately (Johnson-Lindenstrauss lemma).
    """

    def __init__(self, n_input, n_output, seed=None):
        self.n_input = n_input
        self.n_output = n_output

        rng = np.random.RandomState(seed)

        # Random projection matrix scaled to approximately preserve norms
        self.W = rng.randn(n_input, n_output) / np.sqrt(n_output)

    def transform(self, X):
        """
        Transform input X using random projection.

        Args:
            X: array of shape (N, n_input)
        Returns:
            Z: array of shape (N, n_output)
        """
        return X @ self.W


class RandomCNN:
    """
    Random CNN convolution layer (non-trained).

    Applies a convolution with random weights, mimicking the structure
    of a CNN feature extractor but without any learning.
    This is a fair baseline for quantum convolution which also uses
    fixed (non-trainable) transformations.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, seed=None, device="cpu"
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Create random conv layer (weights initialized randomly by default)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

        # Freeze weights (no training)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.conv.to(device)
        self.conv.eval()

        # Store weights as numpy for saving
        self.W = self.conv.weight.detach().cpu().numpy()
        self.b = (
            self.conv.bias.detach().cpu().numpy()
            if self.conv.bias is not None
            else None
        )

    def transform(self, X):
        """
        Apply random CNN convolution.

        Args:
            X: numpy array of shape (N, C, H, W)
        Returns:
            Z: numpy array of shape (N, out_channels, H_out, W_out)
        """
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            out = self.conv(X_tensor)
            return out.cpu().numpy()


def generate_output_filename(params):
    """
    Generate a descriptive filename for the augmented dataset.
    """
    method = params["projection_method"]

    if method == "cnn":
        filename = (
            f"{params['dataset_name']}__"
            f"augmented_{method}_"
            f"k{params['cnn_kernel_size']}_"
            f"s{params['cnn_stride']}_"
            f"c{params['target_channels']}_"
            f"seed{params['random_seed']}"
            ".npz"
        )
    else:
        filename = (
            f"{params['dataset_name']}__"
            f"augmented_{method}_"
            f"c{params['target_channels']}_"
            f"h{params['target_h']}_"
            f"w{params['target_w']}_"
            f"seed{params['random_seed']}"
            ".npz"
        )
    return filename


def main():
    print("=" * 60)
    print("AUGMENTED CLASSICAL DATASET GENERATOR")
    print("=" * 60)

    # Setup paths
    data_root = PROJECT_ROOT / "data"
    output_dir = (
        PROJECT_ROOT / "data" / "quantum_datasets"
    )  # Same dir for easy comparison
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, load training data to determine input dimensions
    print("\n--- Loading train split ---")
    train_ds = load_medmnist_dataset(DATASET_NAME, "train", data_root)
    X_train, y_train = extract_images_and_labels(train_ds)

    n_samples = X_train.shape[0]
    in_channels = X_train.shape[1]
    img_h, img_w = X_train.shape[2], X_train.shape[3]

    print(f"  Original shape: {X_train.shape}")

    # Handle CNN method separately (works on images directly)
    if PROJECTION_METHOD == "cnn":
        # Calculate output spatial dimensions for CNN
        # out_dim = (in_dim - kernel_size) // stride + 1
        out_h = (img_h - CNN_KERNEL_SIZE) // CNN_STRIDE + 1
        out_w = (img_w - CNN_KERNEL_SIZE) // CNN_STRIDE + 1

        print(f"\nParameters:")
        print(f"  Dataset:          {DATASET_NAME}")
        print(f"  Projection:       {PROJECTION_METHOD}")
        print(f"  CNN kernel size:  {CNN_KERNEL_SIZE}")
        print(f"  CNN stride:       {CNN_STRIDE}")
        print(f"  Output channels:  {TARGET_CHANNELS}")
        print(f"  Output spatial:   {out_h} x {out_w}")
        print(f"  Random seed:      {RANDOM_SEED}")
        print()

        # Create random CNN
        print(f"Initializing Random CNN...")
        projector = RandomCNN(
            in_channels=in_channels,
            out_channels=TARGET_CHANNELS,
            kernel_size=CNN_KERNEL_SIZE,
            stride=CNN_STRIDE,
            seed=RANDOM_SEED,
        )

        # Process all splits
        results = {}

        for split in SPLITS:
            print(f"\n--- Processing {split} split ---")

            if split == "train":
                X, y = X_train, y_train
            else:
                ds = load_medmnist_dataset(DATASET_NAME, split, data_root)
                X, y = extract_images_and_labels(ds)

            print(f"  Samples: {len(X)}")
            print(f"  Applying CNN convolution...")

            # CNN works on image tensors directly
            X_out = projector.transform(X)

            results[f"{split}_features"] = X_out.astype(np.float32)
            results[f"{split}_labels"] = y

            print(f"  Output shape: {X_out.shape}")

        # Build metadata for CNN
        metadata = {
            "dataset_name": DATASET_NAME,
            "projection_method": PROJECTION_METHOD,
            "cnn_kernel_size": CNN_KERNEL_SIZE,
            "cnn_stride": CNN_STRIDE,
            "target_channels": TARGET_CHANNELS,
            "target_h": int(out_h),
            "target_w": int(out_w),
            "target_features": int(TARGET_CHANNELS * out_h * out_w),
            "random_seed": RANDOM_SEED,
            "created_at": datetime.now().isoformat(),
            "train_samples": int(len(results["train_labels"])),
            "val_samples": int(len(results["val_labels"])),
            "test_samples": int(len(results["test_labels"])),
            "type": "augmented_classical",
        }

        # Save with CNN weights
        output_filename = generate_output_filename(metadata)
        output_path = output_dir / output_filename

        print(f"\nSaving augmented dataset to: {output_path}")

        save_dict = {
            "train_features": results["train_features"],
            "train_labels": results["train_labels"],
            "val_features": results["val_features"],
            "val_labels": results["val_labels"],
            "test_features": results["test_features"],
            "test_labels": results["test_labels"],
            "projection_W": projector.W,
            "metadata": json.dumps(metadata),
        }
        if projector.b is not None:
            save_dict["projection_b"] = projector.b

        np.savez_compressed(output_path, **save_dict)

    else:
        # RFF and Random projection methods (work on flattened data)
        target_features = TARGET_CHANNELS * TARGET_H * TARGET_W

        print(f"\nParameters:")
        print(f"  Dataset:          {DATASET_NAME}")
        print(f"  Projection:       {PROJECTION_METHOD}")
        print(f"  Target shape:     ({TARGET_CHANNELS}, {TARGET_H}, {TARGET_W})")
        print(f"  Target features:  {target_features}")
        print(f"  Random seed:      {RANDOM_SEED}")
        if PROJECTION_METHOD == "rff":
            print(
                f"  RFF gamma:        {RFF_GAMMA if RFF_GAMMA else 'auto (1/n_input)'}"
            )
        print()

        # Flatten images: (N, C, H, W) -> (N, C*H*W)
        n_input = np.prod(X_train.shape[1:])
        X_train_flat = X_train.reshape(n_samples, n_input)

        print(f"  Flattened input: {X_train_flat.shape}")
        print(f"  Target output: ({n_samples}, {target_features})")

        # Create the projection
        print(f"\nInitializing {PROJECTION_METHOD.upper()} projection...")
        if PROJECTION_METHOD == "rff":
            projector = RandomFourierFeatures(
                n_input=n_input,
                n_output=target_features,
                gamma=RFF_GAMMA,
                seed=RANDOM_SEED,
            )
            print(f"  RFF gamma used: {projector.gamma:.6f}")
        elif PROJECTION_METHOD == "random":
            projector = RandomProjection(
                n_input=n_input,
                n_output=target_features,
                seed=RANDOM_SEED,
            )
        else:
            raise ValueError(f"Unknown projection method: {PROJECTION_METHOD}")

        # Process all splits
        results = {}

        for split in SPLITS:
            print(f"\n--- Processing {split} split ---")

            if split == "train":
                X, y = X_train, y_train
                X_flat = X_train_flat
            else:
                ds = load_medmnist_dataset(DATASET_NAME, split, data_root)
                X, y = extract_images_and_labels(ds)
                X_flat = X.reshape(X.shape[0], n_input)

            print(f"  Samples: {len(X)}")

            # Apply projection
            print(f"  Applying {PROJECTION_METHOD} projection...")
            X_projected = projector.transform(X_flat)

            # Reshape to match quantum output shape: (N, C, H, W)
            X_reshaped = X_projected.reshape(-1, TARGET_CHANNELS, TARGET_H, TARGET_W)

            results[f"{split}_features"] = X_reshaped.astype(np.float32)
            results[f"{split}_labels"] = y

            print(f"  Output shape: {X_reshaped.shape}")

        # Build metadata
        metadata = {
            "dataset_name": DATASET_NAME,
            "projection_method": PROJECTION_METHOD,
            "target_channels": TARGET_CHANNELS,
            "target_h": TARGET_H,
            "target_w": TARGET_W,
            "target_features": int(target_features),
            "original_input_features": int(n_input),
            "random_seed": RANDOM_SEED,
            "rff_gamma": getattr(projector, "gamma", None),
            "created_at": datetime.now().isoformat(),
            "train_samples": int(len(results["train_labels"])),
            "val_samples": int(len(results["val_labels"])),
            "test_samples": int(len(results["test_labels"])),
            "type": "augmented_classical",
        }

        # Generate filename and save
        output_filename = generate_output_filename(metadata)
        output_path = output_dir / output_filename

        print(f"\nSaving augmented dataset to: {output_path}")

        np.savez_compressed(
            output_path,
            train_features=results["train_features"],
            train_labels=results["train_labels"],
            val_features=results["val_features"],
            val_labels=results["val_labels"],
            test_features=results["test_features"],
            test_labels=results["test_labels"],
            projection_W=projector.W,
            metadata=json.dumps(metadata),
        )

    # Save human-readable metadata
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_path}")
    print()
    print("Done!")
    print(
        f"Augmented dataset saved with {metadata['train_samples']} train, "
        f"{metadata['val_samples']} val, {metadata['test_samples']} test samples."
    )
    print()
    print("To compare with quantum datasets, add this to compare_datasets.py:")
    print(
        f'    {{"type": "quantum", "name": "{output_filename}", "label": "Classical + {PROJECTION_METHOD.upper()}"}}'
    )


if __name__ == "__main__":
    main()
