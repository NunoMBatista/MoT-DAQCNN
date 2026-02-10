"""
augment_classical_dimensionality.py

Create a "fair fight" baseline by projecting classical image data to the same
dimensionality as the quantum-transformed data using random feature expansion.

This helps verify whether any improvements from quantum features are due to
the quantum transformation itself, or just from having more features.

Four methods are available:
    1. Random Fourier Features (RFF): Approximates an RBF kernel
    2. Random Matrix Projection: Simple random linear projection
    3. Random CNN: Random convolutional kernels (non-trained)
    4. ResNet: Frozen early layers of pretrained ResNet-18 with PCA/MaxPool reduction

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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import models, transforms
from tqdm import tqdm

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# =============================================================================
# PARAMETERS - Edit these to configure the augmented dataset generation
# =============================================================================

# Which dataset to use: "pneumonia_mnist", "breast_mnist", "path_mnist", "derma_mnist"
DATASET_NAME = "derma_mnist"

# Target output shape to match quantum dataset
# These should match the quantum dataset you want to compare against
# For example, if quantum output is (N, 16, 14, 14), set:
#   TARGET_CHANNELS = 16
#   TARGET_H = 14
#   TARGET_W = 14

# 2x2 Grayscale 4 quantum kernel kernel dimensions
TARGET_CHANNELS = 16 # 18 FOR HSV # num_kernels * kernel_size^2 (e.g., 4 topologies * 4 qubits = 16)
TARGET_H = 14  # (28 - kernel_size) // stride + 1
TARGET_W = 14

# 3x3 Grayscale 4 quantum kernel dimensions
#TARGET_CHANNELS = 36 # 38 FOR HSV
#TARGET_H = 9
#TARGET_W = 9

# Projection method: "rff", "random", "cnn", or "resnet"
#   - "rff": Random Fourier Features (approximates RBF kernel)
#   - "random": Random matrix projection
#   - "cnn": Random CNN kernels (convolution with random weights)
#   - "resnet": Frozen early layers of pretrained ResNet-18
PROJECTION_METHOD = "rff"

# For RFF: gamma parameter for RBF kernel approximation (smaller = smoother)
# If None, will use 1/n_features as default
RFF_GAMMA = None

# For CNN method: kernel size and stride
# These determine the output spatial dimensions
CNN_KERNEL_SIZE = 2
CNN_STRIDE = 2

# For ResNet method: which layer to extract features from
# Options: "layer1" or "layer2"
# layer1[0].conv1 outputs 64 channels, layer2[0].conv1 outputs 128 channels
RESNET_LAYER = "layer1"

# For ResNet method: how to reduce channels to match target
# Options: "pca" or "maxpool"
#   - "pca": Use PCA to reduce channels (fit on train, apply to all)
#   - "maxpool": Use adaptive max pooling across channels
RESNET_REDUCTION = "pca"

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


class ResNetFeatureExtractor:
    """
    Frozen ResNet-18 early layer feature extractor.

    Uses pretrained ResNet-18 and extracts features from early layers.
    Since ResNet produces more channels than we need (64 or 128),
    we reduce dimensionality using PCA or MaxPooling.

    This is a strong classical baseline using learned (but frozen) features.
    """

    def __init__(
        self,
        target_channels,
        target_h,
        target_w,
        layer="layer1",
        reduction="pca",
        device="cpu",
    ):
        self.target_channels = target_channels
        self.target_h = target_h
        self.target_w = target_w
        self.layer = layer
        self.reduction = reduction
        self.device = device

        # Load pretrained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all weights
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.to(device)
        self.resnet.eval()

        # PCA will be fit on training data
        self.pca = None
        self.pca_fitted = False

        # Store which layer we're using for metadata
        if layer == "layer1":
            self.resnet_channels = 64
        elif layer == "layer2":
            self.resnet_channels = 128
        else:
            raise ValueError(f"Unknown layer: {layer}. Use 'layer1' or 'layer2'")

    def _extract_features(self, X):
        """
        Extract features from ResNet early layers.

        Args:
            X: numpy array of shape (N, C, H, W) with C=1 for grayscale
        Returns:
            features: numpy array of shape (N, resnet_channels, H', W')
        """
        with torch.no_grad():
            # Convert to tensor
            X_tensor = torch.from_numpy(X).float().to(self.device)

            # ResNet expects 3 channels, so repeat grayscale to RGB
            if X_tensor.shape[1] == 1:
                X_tensor = X_tensor.repeat(1, 3, 1, 1)

            # ResNet expects normalized ImageNet input
            # Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            X_tensor = (X_tensor - mean) / std

            # Forward through early layers
            x = self.resnet.conv1(X_tensor)  # 64 channels
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            if self.layer == "layer1":
                x = self.resnet.layer1(x)  # Still 64 channels
            elif self.layer == "layer2":
                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)  # 128 channels

            return x.cpu().numpy()

    def fit(self, X):
        """
        Fit the dimensionality reduction (PCA) on training data.

        Args:
            X: numpy array of shape (N, C, H, W)
        """
        if self.reduction != "pca":
            return  # No fitting needed for maxpool

        print("  Extracting ResNet features for PCA fitting...")
        features = self._extract_features(X)  # (N, resnet_channels, H', W')

        # Reshape to (N * H' * W', resnet_channels) for PCA
        N, C, H, W = features.shape
        features_flat = features.transpose(0, 2, 3, 1).reshape(-1, C)

        print(f"  Fitting PCA: {C} channels -> {self.target_channels} channels...")
        self.pca = PCA(n_components=self.target_channels)
        self.pca.fit(features_flat)
        self.pca_fitted = True

        explained_var = sum(self.pca.explained_variance_ratio_) * 100
        print(f"  PCA explained variance: {explained_var:.1f}%")

    def transform(self, X):
        """
        Extract and reduce ResNet features.

        Args:
            X: numpy array of shape (N, C, H, W)
        Returns:
            Z: numpy array of shape (N, target_channels, target_h, target_w)
        """
        # Extract ResNet features
        features = self._extract_features(X)  # (N, resnet_channels, H', W')
        N, C, H, W = features.shape

        if self.reduction == "pca":
            if not self.pca_fitted or self.pca is None:
                raise RuntimeError("PCA not fitted. Call fit() first on training data.")

            # Apply PCA to reduce channels
            # Reshape to (N * H' * W', resnet_channels)
            features_flat = features.transpose(0, 2, 3, 1).reshape(-1, C)

            # Transform (pca is guaranteed non-None here)
            pca = self.pca
            reduced = pca.transform(features_flat)  # (N * H' * W', target_channels)

            # Reshape back to (N, target_channels, H', W')
            reduced = reduced.reshape(N, H, W, self.target_channels)
            reduced = reduced.transpose(0, 3, 1, 2)

        elif self.reduction == "maxpool":
            # Use adaptive pooling across channels
            # Group channels and take max within each group
            features_tensor = torch.from_numpy(features).float()

            # Reshape: (N, resnet_channels, H, W) -> (N, target_channels, group_size, H, W)
            group_size = C // self.target_channels
            if C % self.target_channels != 0:
                # Pad channels to make divisible
                pad_size = self.target_channels - (C % self.target_channels)
                features_tensor = F.pad(features_tensor, (0, 0, 0, 0, 0, pad_size))
                C = features_tensor.shape[1]
                group_size = C // self.target_channels

            features_tensor = features_tensor.view(
                N, self.target_channels, group_size, H, W
            )
            # Max pool across the group dimension
            reduced = features_tensor.max(dim=2)[0]  # (N, target_channels, H, W)
            reduced = reduced.numpy()

        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Resize spatial dimensions to target
        reduced_tensor = torch.from_numpy(reduced).float()
        reduced_tensor = F.interpolate(
            reduced_tensor,
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )

        return reduced_tensor.numpy()


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
    elif method == "resnet":
        filename = (
            f"{params['dataset_name']}__"
            f"augmented_{method}_"
            f"{params['resnet_layer']}_"
            f"{params['resnet_reduction']}_"
            f"c{params['target_channels']}_"
            f"h{params['target_h']}_"
            f"w{params['target_w']}"
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

    elif PROJECTION_METHOD == "resnet":
        # ResNet frozen feature extractor with dimension reduction
        print(f"\nParameters:")
        print(f"  Dataset:          {DATASET_NAME}")
        print(f"  Projection:       {PROJECTION_METHOD}")
        print(f"  ResNet layer:     {RESNET_LAYER}")
        print(f"  Reduction method: {RESNET_REDUCTION}")
        print(f"  Target channels:  {TARGET_CHANNELS}")
        print(f"  Target spatial:   {TARGET_H} x {TARGET_W}")
        print()

        # Create ResNet feature extractor
        print("Initializing ResNet feature extractor...")
        projector = ResNetFeatureExtractor(
            target_channels=TARGET_CHANNELS,
            target_h=TARGET_H,
            target_w=TARGET_W,
            layer=RESNET_LAYER,
            reduction=RESNET_REDUCTION,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"  Using device: {projector.device}")
        print(f"  ResNet {RESNET_LAYER} outputs {projector.resnet_channels} channels")

        # Fit PCA on training data (if using PCA reduction)
        if RESNET_REDUCTION == "pca":
            projector.fit(X_train)

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
            print(f"  Extracting ResNet features and reducing...")

            # Process in batches to avoid memory issues
            batch_size = 64
            all_features = []
            for i in tqdm(range(0, len(X), batch_size), desc="  Batches"):
                batch = X[i : i + batch_size]
                batch_features = projector.transform(batch)
                all_features.append(batch_features)

            X_out = np.concatenate(all_features, axis=0)

            results[f"{split}_features"] = X_out.astype(np.float32)
            results[f"{split}_labels"] = y

            print(f"  Output shape: {X_out.shape}")

        # Build metadata for ResNet
        metadata = {
            "dataset_name": DATASET_NAME,
            "projection_method": PROJECTION_METHOD,
            "resnet_layer": RESNET_LAYER,
            "resnet_reduction": RESNET_REDUCTION,
            "resnet_channels": projector.resnet_channels,
            "target_channels": TARGET_CHANNELS,
            "target_h": TARGET_H,
            "target_w": TARGET_W,
            "target_features": int(TARGET_CHANNELS * TARGET_H * TARGET_W),
            "created_at": datetime.now().isoformat(),
            "train_samples": int(len(results["train_labels"])),
            "val_samples": int(len(results["val_labels"])),
            "test_samples": int(len(results["test_labels"])),
            "type": "augmented_classical",
        }

        if RESNET_REDUCTION == "pca" and projector.pca is not None:
            metadata["pca_explained_variance"] = float(
                sum(projector.pca.explained_variance_ratio_)
            )

        # Generate filename and save
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
            "metadata": json.dumps(metadata),
        }

        # Save PCA components if used
        if RESNET_REDUCTION == "pca" and projector.pca is not None:
            save_dict["pca_components"] = projector.pca.components_
            save_dict["pca_mean"] = projector.pca.mean_

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
