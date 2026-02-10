"""
compare_datasets.py

Load classical and/or quantum datasets and compute metrics to evaluate
how "separable" the classes are. This helps assess whether the quantum
transformation makes classification easier.

Metrics computed:
    - Fisher's Ratio: measures class separability (higher = more separable)
    - Silhouette Score: clustering quality metric (-1 to 1, higher = better)
    - 1-NN Accuracy: leave-one-out nearest neighbor accuracy
    - KTA (Kernel-Target Alignment): alignment between feature kernel and class structure (-1 to 1, higher = better)

Usage:
    python experiments/compare_datasets.py

Edit the DATASETS_TO_COMPARE list below to specify which datasets to analyze.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION - Edit this list to specify which datasets to compare
# =============================================================================

# Each entry is a dict with:
#   - "type": "classical" or "quantum"
#   - "name": dataset name (for classical) or path to .npz file (for quantum)
#   - "label": human-readable label for the output table (optional)


# PNEUMONIAMNIST DATASET
PNEUMONIA_MNIST_DATASETS = [
    # Classical baseline
    {
        "type": "classical",
        "name": "pneumonia_mnist",
        "label": "PneumoniaMNIST (classical)",
    },
    {
        "type": "classical",
        "name": "pneumonia_mnist__augmented_cnn_k2_s2_c16_seed42.npz",
        "label": "Classical + CNN (16, 14, 14)",
    },
    {
        "type": "classical",
        "name": "pneumonia_mnist__augmented_rff_c16_h14_w14_seed42.npz",
        "label": "Classical + RFF (16, 14, 14)",
    },
    {
        "type": "classical",
        "name": "pneumonia_mnist__augmented_cnn_k3_s3_c36_seed42.npz",
        "label": "Classical + CNN (36, 9, 9)",
    },
    {
        "type": "classical",
        "name": "pneumonia_mnist__augmented_rff_c36_h9_w9_seed42.npz",
        "label": "Classical + RFF (36, 9, 9)",
    },
    # 2x2 QUANTUM
    {
        "type": "quantum",
        "name": "pneumonia_mnist__k2_s2_tkin_ev6.28_sc1000.npz",
        "label": "QPM (ksize=2 knumber=1 (king))",
    },
    {
        "type": "quantum",
        "name": "pneumonia_mnist__k2_s2_thor_ev6.28_sc1000.npz",
        "label": "QPM (ksize=2 knumber=1 (horizontal))",
    },
    {
        "type": "quantum",
        "name": "pneumonia_mnist__k2_s2_tkin-hor-ver-u_s_ev6.28_sc1000.npz",
        "label": "QPM (ksize=2 knumber=4)",
    },
    # 3x3 QUANTUM
    {
        "type": "quantum",
        "name": "pneumonia_mnist__k3_s3_tkin_ev6.28_sc1000.npz",
        "label": "QPM (ksize=3 knumber=1 (king))",
    },
    {
        "type": "quantum",
        "name": "pneumonia_mnist__k3_s3_thor_ev6.28_sc1000.npz",
        "label": "QPM (ksize=3 knumber=1 (horizontal))",
    },
    {
        "type": "quantum",
        "name": "pneumonia_mnist__k3_s3_tkin-hor-cro-rin_ev6.28_sc1000.npz",
        "label": "QPM (ksize=3 knumber=4)",
    },
]

# BREASTMNIST DATASET
BREAST_MNIST_DATASETS = [
    # Classical baseline
    {
        "type": "classical",
        "name": "breast_mnist",
        "label": "BreastMNIST (classical)",
    },
    {
        "type": "quantum",
        "name": "breast_mnist__augmented_rff_c16_h14_w14_seed42.npz",
        "label": "Classical + RFF (16, 14, 14)",
    },
    {
        "type": "quantum",
        "name": "breast_mnist__augmented_cnn_k2_s2_c16_seed42.npz",
        "label": "Classical + CNN (16, 14, 14)",
    },
    {
        "type": "quantum",
        "name": "breast_mnist__augmented_rff_c36_h9_w9_seed42.npz",
        "label": "Classical + RFF (36, 9, 9)",
    },
    {
        "type": "quantum",
        "name": "breast_mnist__augmented_cnn_k3_s3_c36_seed42.npz",
        "label": "Classical + CNN (36, 9, 9)",
    },
    # 2x2 QUANTUM
    {
        "type": "quantum",
        "name": "breast_mnist__k2_s2_tkin_ev6.28_sc1000.npz",
        "label": "QBM (ksize=2 knumber=1 (king))",
    },
    {
        "type": "quantum",
        "name": "breast_mnist__k2_s2_thor_ev6.28_sc1000.npz",
        "label": "QBM (ksize=2 knumber=1 (horizontal))",
    },
    {
        "type": "quantum",
        "name": "breast_mnist__k2_s2_tkin-hor-ver-u_s_ev6.28_sc1000.npz",
        "label": "QBM (ksize=2 knumber=4)",
    },
    # 3x3 QUANTUM
    {
        "type": "quantum",
        "name": "breast_mnist__k3_s3_tkin_ev6.28_sc1000.npz",
        "label": "QBM (ksize=3 knumber=1 (king))",
    },
    {
        "type": "quantum",
        "name": "breast_mnist__k3_s3_thor_ev6.28_sc1000.npz",
        "label": "QBM (ksize=3 knumber=1 (horizontal))",
    },
    {
        "type": "quantum",
        "name": "breast_mnist__k3_s3_tkin-hor-cro-rin_ev6.28_sc1000.npz",
        "label": "QBM (ksize=3 knumber=4)",
    },
]

DERMA_MNIST_DATASETS = [
    # Classical baseline
    {
        "type": "classical",
        "name": "derma_mnist",
        "label": "DermaMNIST (classical)",
    },
    {
        "type": "quantum",
        "name": "derma_mnist__augmented_rff_c16_h14_w14_seed42.npz",
        "label": "Classical + RFF (16, 14, 14)",
    },
    {
        "type": "quantum",
        "name": "derma_mnist__augmented_cnn_k2_s2_c16_seed42.npz",
        "label": "Classical + CNN (16, 14, 14)",
    },
    {
        "type": "quantum",
        "name": "derma_mnist__augmented_rff_c36_h9_w9_seed42.npz",
        "label": "Classical + RFF (36, 9, 9)",
    },
    {
        "type": "quantum",
        "name": "derma_mnist__augmented_cnn_k3_s3_c36_seed42.npz",
        "label": "Classical + CNN (36, 9, 9)",
    },
    # 2x2 QUANTUM
    {
        "type": "quantum",
        "name": "derma_mnist__k2_s2_tkin_ev6.28_sc1000.npz",
        "label": "QDM (ksize=2 knumber=1 (king))",
    },
    {
        "type": "quantum",
        "name": "derma_mnist__k2_s2_thor_ev6.28_sc1000.npz",
        "label": "QDM (ksize=2 knumber=1 (horizontal))",
    },
    {
        "type": "quantum",
        "name": "derma_mnist__k2_s2_tkin-hor-ver-u_s_ev6.28_sc1000.npz",
        "label": "QDM (ksize=2 knumber=4)",
    },
    # 3x3 QUANTUM
    {
        "type": "quantum",
        "name": "derma_mnist__k3_s3_tkin_ev6.28_sc1000.npz",
        "label": "QDM (ksize=3 knumber=1 (king))",
    },
    {
        "type": "quantum",
        "name": "derma_mnist__k3_s3_thor_ev6.28_sc1000.npz",
        "label": "QDM (ksize=3 knumber=1 (horizontal))",
    },
    {
        "type": "quantum",
        "name": "derma_mnist__k3_s3_tkin-hor-cro-rin_ev6.28_sc1000.npz",
        "label": "QDM (ksize=3 knumber=4)",
    },
]

DATASETS_TO_COMPARE = PNEUMONIA_MNIST_DATASETS
DATASETS_TO_COMPARE = BREAST_MNIST_DATASETS
DATASETS_TO_COMPARE = DERMA_MNIST_DATASETS

# Which split to use for comparison (usually "test" for final evaluation)
SPLIT = "test"

# Max samples to use (set to None to use all samples)
# Using a subset can speed up computation significantly
MAX_SAMPLES = 5000


# =============================================================================
# Metric computation functions
# =============================================================================


def compute_fishers_ratio(X, y):
    """
    Compute Fisher's Discriminant Ratio for multi-class data.

    Fisher's ratio measures how well-separated the classes are:
        FR = (between-class variance) / (within-class variance)

    Higher values indicate better class separability.

    For multi-class, we compute the average pairwise Fisher's ratio.
    """
    classes = np.unique(y)
    n_classes = len(classes)

    if n_classes < 2:
        return 0.0

    # Flatten features if needed (e.g., images to vectors)
    X_flat = X.reshape(X.shape[0], -1)

    # Compute class means and overall mean
    class_means = {}
    class_vars = {}

    for c in classes:
        mask = y == c
        X_c = X_flat[mask]
        class_means[c] = np.mean(X_c, axis=0)
        # Within-class variance (average variance across features)
        class_vars[c] = np.var(X_c, axis=0) + 1e-10  # Small epsilon for stability

    # Compute pairwise Fisher's ratio and average
    ratios = []
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1 :]:
            # Between-class: squared difference of means
            between = (class_means[c1] - class_means[c2]) ** 2
            # Within-class: sum of variances
            within = class_vars[c1] + class_vars[c2]
            # Fisher's ratio per feature, then average across features
            fr = np.mean(between / within)
            ratios.append(fr)

    return np.mean(ratios)


def compute_silhouette(X, y, max_samples=5000):
    """
    Compute silhouette score.

    Silhouette score measures how similar samples are to their own cluster
    vs other clusters. Range: [-1, 1], higher is better.

    We subsample if dataset is large (silhouette is O(n^2)).
    """
    X_flat = X.reshape(X.shape[0], -1)

    # Subsample if too large
    if len(X_flat) > max_samples:
        idx = np.random.choice(len(X_flat), max_samples, replace=False)
        X_flat = X_flat[idx]
        y = y[idx]

    # Need at least 2 classes and 2 samples per class
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 2:
        return np.nan

    try:
        score = silhouette_score(X_flat, y)
    except ValueError:
        score = np.nan

    return score


def compute_1nn_accuracy(X, y, cv_folds=5, max_samples=5000):
    """
    Compute 1-Nearest Neighbor accuracy using cross-validation.

    1-NN accuracy gives a non-parametric estimate of how easily
    a simple classifier can separate the classes.
    """
    X_flat = X.reshape(X.shape[0], -1)

    # Subsample if too large
    if len(X_flat) > max_samples:
        idx = np.random.choice(len(X_flat), max_samples, replace=False)
        X_flat = X_flat[idx]
        y = y[idx]

    # Use 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1)

    # Cross-validation
    try:
        scores = cross_val_score(knn, X_flat, y, cv=cv_folds, scoring="accuracy")
        return np.mean(scores)
    except ValueError:
        return np.nan


def compute_kta(X, y, max_samples=10000):
    """
    Compute Kernel-Target Alignment (KTA).

    KTA measures how well the kernel (similarity matrix) computed from features
    aligns with the ideal kernel based on class labels. Higher values indicate
    that the feature representation aligns well with the classification task.

    KTA = <K, Y> / (||K|| * ||Y||)

    where:
        K is the kernel matrix (here, linear kernel: K = X @ X.T)
        Y is the target kernel matrix (Y_ij = 1 if labels match, -1 otherwise)
        <K, Y> is the Frobenius inner product
        ||K|| is the Frobenius norm

    Range: [-1, 1], higher is better.

    Due to O(n^2) memory and computation, we subsample for large datasets.
    """
    X_flat = X.reshape(X.shape[0], -1)

    # Subsample if too large (KTA is O(n^2) in memory)
    if len(X_flat) > max_samples:
        idx = np.random.choice(len(X_flat), max_samples, replace=False)
        X_flat = X_flat[idx]
        y = y[idx]

    # Normalize features for numerical stability
    X_normalized = X_flat / (np.linalg.norm(X_flat, axis=1, keepdims=True) + 1e-10)

    # Compute kernel matrix K (linear kernel)
    K = X_normalized @ X_normalized.T

    # Compute target kernel Y (Y_ij = 1 if y_i == y_j, else -1)
    n = len(y)
    Y = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Y[i, j] = 1.0 if y[i] == y[j] else -1.0

    # Compute KTA = <K, Y> / (||K|| * ||Y||)
    # Frobenius inner product: <K, Y> = trace(K.T @ Y) = sum(K * Y)
    numerator = np.sum(K * Y)

    # Frobenius norms
    norm_K = np.linalg.norm(K, "fro")
    norm_Y = np.linalg.norm(Y, "fro")

    if norm_K == 0 or norm_Y == 0:
        return np.nan

    kta = numerator / (norm_K * norm_Y)

    return kta


# =============================================================================
# Dataset loading functions
# =============================================================================


def load_classical_dataset(dataset_name, split, data_root):
    """Load a classical MedMNIST dataset and return (features, labels)."""

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

    # Extract all images and labels
    images = []
    labels = []
    for img, lbl in ds:
        images.append(img.numpy())
        lbl_val = lbl.item() if hasattr(lbl, "item") else lbl[0]
        labels.append(lbl_val)

    X = np.array(images)
    y = np.array(labels)

    return X, y


def load_quantum_dataset(npz_path, split):
    """Load a quantum dataset from .npz file and return (features, labels)."""

    data = np.load(npz_path, allow_pickle=True)

    features_key = f"{split}_features"
    labels_key = f"{split}_labels"

    if features_key not in data:
        raise ValueError(f"Split '{split}' not found in {npz_path}")

    X = data[features_key]
    y = data[labels_key]

    return X, y


def load_dataset(config, split, data_root, quantum_datasets_dir):
    """Load a dataset based on config dict."""

    dtype = config["type"]
    name = config["name"]

    if dtype == "classical":
        X, y = load_classical_dataset(name, split, data_root)
        metadata = {"type": "classical", "name": name}
    elif dtype == "quantum":
        # Check if name is absolute path or relative to quantum_datasets dir
        if os.path.isabs(name):
            npz_path = Path(name)
        else:
            npz_path = quantum_datasets_dir / name

        if not npz_path.exists():
            raise FileNotFoundError(f"Quantum dataset not found: {npz_path}")

        X, y = load_quantum_dataset(npz_path, split)

        # Try to load metadata
        json_path = npz_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)
            metadata["type"] = "quantum"
        else:
            metadata = {"type": "quantum", "name": name}
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")

    return X, y, metadata


# =============================================================================
# Main comparison logic
# =============================================================================


def main():
    print("=" * 70)
    print("DATASET COMPARISON: Classical vs Quantum Features")
    print("=" * 70)
    print()

    if len(DATASETS_TO_COMPARE) == 0:
        print("No datasets specified in DATASETS_TO_COMPARE. Exiting.")
        return

    data_root = PROJECT_ROOT / "data"
    quantum_datasets_dir = PROJECT_ROOT / "data" / "quantum_datasets"

    print(f"Split: {SPLIT}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'all'}")
    print()

    # Collect results
    results = []

    for config in DATASETS_TO_COMPARE:
        label = config.get("label", config["name"])
        print(f"Processing: {label}")
        print("-" * 50)

        try:
            X, y, metadata = load_dataset(
                config, SPLIT, data_root, quantum_datasets_dir
            )
        except (FileNotFoundError, ValueError, ImportError) as e:
            print(f"  ERROR: {e}")
            print()
            continue

        print(f"  Shape: {X.shape}")
        print(f"  Labels: {np.unique(y)}")
        print(f"  Samples per class: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Subsample if needed
        if MAX_SAMPLES and len(X) > MAX_SAMPLES:
            idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            X_sub = X[idx]
            y_sub = y[idx]
            print(f"  Subsampled to {MAX_SAMPLES} samples")
        else:
            X_sub = X
            y_sub = y

        # Compute metrics
        print("  Computing metrics...")

        fisher = compute_fishers_ratio(X_sub, y_sub)
        print(f"    Fisher's Ratio: {fisher:.4f}")

        silhouette = compute_silhouette(X_sub, y_sub)
        print(f"    Silhouette Score: {silhouette:.4f}")

        knn_acc = compute_1nn_accuracy(X_sub, y_sub)
        print(f"    1-NN Accuracy: {knn_acc:.4f}")

        kta = compute_kta(X_sub, y_sub)
        print(f"    KTA: {kta:.4f}")

        results.append(
            {
                "label": label,
                "type": config["type"],
                "shape": X.shape,
                "fisher": fisher,
                "silhouette": silhouette,
                "knn_acc": knn_acc,
                "kta": kta,
            }
        )

        print()

    # Print summary table
    if len(results) > 0:
        print("=" * 82)
        print("SUMMARY TABLE")
        print("=" * 82)
        print()
        print(
            f"{'Dataset':<40} {'Fisher':>10} {'Silhouette':>12} {'1-NN Acc':>10} {'KTA':>10}"
        )
        print("-" * 82)

        for r in results:
            fisher_str = f"{r['fisher']:.4f}" if not np.isnan(r["fisher"]) else "N/A"
            sil_str = (
                f"{r['silhouette']:.4f}" if not np.isnan(r["silhouette"]) else "N/A"
            )
            knn_str = f"{r['knn_acc']:.4f}" if not np.isnan(r["knn_acc"]) else "N/A"
            kta_str = f"{r['kta']:.4f}" if not np.isnan(r["kta"]) else "N/A"
            print(
                f"{r['label']:<40} {fisher_str:>10} {sil_str:>12} {knn_str:>10} {kta_str:>10}"
            )

        print("-" * 82)
        print()
        print("Interpretation:")
        print("  - Fisher's Ratio: Higher = better class separation")
        print("  - Silhouette Score: Higher (max 1.0) = better clustering")
        print("  - 1-NN Accuracy: Higher = easier to classify with simple model")
        print("  - KTA: Higher (max 1.0) = features align well with class structure")
        print()
        print("If quantum features show higher values, the quantum transformation")
        print("is making the classification problem 'easier' for downstream models.")

        print("-" * 82)
        print("BEST RESULTS FOR EACH METRIC:")
        best_fisher = max(
            results, key=lambda r: r["fisher"] if not np.isnan(r["fisher"]) else -np.inf
        )
        best_silhouette = max(
            results,
            key=lambda r: r["silhouette"] if not np.isnan(r["silhouette"]) else -np.inf,
        )
        best_knn = max(
            results,
            key=lambda r: r["knn_acc"] if not np.isnan(r["knn_acc"]) else -np.inf,
        )
        best_kta = max(
            results, key=lambda r: r["kta"] if not np.isnan(r["kta"]) else -np.inf
        )
        print(
            f"  - Best Fisher's Ratio: {best_fisher['label']} ({best_fisher['fisher']:.4f})"
        )
        print(
            f"  - Best Silhouette Score: {best_silhouette['label']} ({best_silhouette['silhouette']:.4f})"
        )
        print(
            f"  - Best 1-NN Accuracy: {best_knn['label']} ({best_knn['knn_acc']:.4f})"
        )
        print(f"  - Best KTA: {best_kta['label']} ({best_kta['kta']:.4f})")
        print("-" * 82)


if __name__ == "__main__":
    main()
