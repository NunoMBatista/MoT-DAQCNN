"""
kernel_parameter_optimization.py

Grid-search optimization for quantum kernel parameters (evolution_time and scaling_factor).
Evaluates different parameter combinations using metrics like 1-NN accuracy, KTA, Silhouette, or FDR.

Usage:
    python experiments/kernel_parameter_optimization.py

Edit the configuration variables below to customize the optimization.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.layers.quantum_convolution import QuantumConv2d
from src.utils.color_conversion import rgb_to_grayscale_tensor
from src.utils.data import load_medmnist_dataset
from src.utils.evaluate import (
    compute_1nn_accuracy,
    compute_fishers_ratio,
    compute_kta,
    compute_silhouette,
)

# =============================================================================
# CONFIGURATION - Edit these parameters
# =============================================================================

# Dataset selection
DATASET_NAME = "pneumonia_mnist"  # Options: "pneumonia_mnist", "breast_mnist", "path_mnist", "derma_mnist", "tissue_mnist"

# Metric to optimize: "1-nn", "kta", "silhouette", "fdr"
OPTIMIZATION_METRIC = "kta"

# Grid search ranges
# [0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
EVOLUTION_TIME_RANGE = [
    0.001,
    0.01,
    0.1,
    0.2,
    0.5,
    1.0,
    1.2,
    1.5,
    1.7,
    2.0,
    2.2,
    2.5,
    3.14,
    6.28,
    7,
    8,
    10,
]  # Time parameter for quantum evolution
SCALING_FACTOR_RANGE = [
    1,
    10,
    100,
    500,
    1000,
    2000,
    5000,
    10000,
]  # Scaling factor for Rydberg interaction
SCALING_FACTOR_RANGE = [1]

# Kernel configuration
KERNEL_SIZE = 3  # 2 or 3
STRIDE = 3  # Convolution stride
KERNEL_TOPOLOGY_NAMES = [
    "kings",
    "horizontal",
    "cross",
    "ring",
]  # Options for 2x2: ["kings", "horizontal", "vertical", "u_shape"]
# Options for 3x3: ["kings", "horizontal", "vertical", "cross", "ring"]

# Sampling configuration
NUM_SAMPLES = 200  # Total number of samples to use for evaluation
BALANCED_SAMPLING = True  # If True, sample equally from each class
SAMPLES_PER_CLASS = 100  # Only used if BALANCED_SAMPLING is True

# Other settings
BATCH_SIZE = 32  # Batch size for processing through quantum kernels
USE_GRAYSCALE = True  # Convert RGB images to grayscale
RANDOM_SEED = 1

# =============================================================================
# Helper Functions
# =============================================================================


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_dataset(dataset, num_samples, balanced=False, samples_per_class=None):
    """
    Sample a subset of the dataset.

    Args:
        dataset: The full dataset
        num_samples: Total number of samples to extract
        balanced: If True, sample equally from each class
        samples_per_class: Number of samples per class (only used if balanced=True)

    Returns:
        Subset of the dataset
    """
    if not balanced:
        # Random sampling
        indices = np.random.choice(
            len(dataset), size=min(num_samples, len(dataset)), replace=False
        )
        return Subset(dataset, indices)

    # Balanced sampling
    # First, collect indices by class
    class_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_val = (
            label.item()
            if hasattr(label, "item")
            else label[0]
            if isinstance(label, np.ndarray)
            else label
        )
        if label_val not in class_indices:
            class_indices[label_val] = []
        class_indices[label_val].append(idx)

    # Sample from each class
    selected_indices = []
    for class_label, indices in class_indices.items():
        n_samples = min(samples_per_class, len(indices))
        sampled = np.random.choice(indices, size=n_samples, replace=False)
        selected_indices.extend(sampled)

    return Subset(dataset, selected_indices)


def extract_features_and_labels(dataset, q_conv, batch_size, use_grayscale=False):
    """
    Extract features by passing dataset through quantum convolution.

    Returns:
        X: numpy array of features (N, out_channels * H * W)
        y: numpy array of labels (N,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []

    for images, labels in loader:
        with torch.no_grad():
            if use_grayscale and images.shape[1] == 3:
                images = rgb_to_grayscale_tensor(images)

            q_out = q_conv(images)  # (B, out_channels, H_out, W_out)

        all_features.append(q_out.cpu().numpy())

        lbl = labels.numpy()
        if lbl.ndim > 1:
            lbl = lbl.squeeze(-1)
        all_labels.append(lbl)

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Flatten features to (N, features)
    X = X.reshape(X.shape[0], -1)

    return X, y


def evaluate_metric(X, y, metric_name):
    """Evaluate the specified metric on the features."""
    if metric_name == "1-nn":
        return compute_1nn_accuracy(X, y)
    elif metric_name == "kta":
        return compute_kta(X, y)
    elif metric_name == "silhouette":
        return compute_silhouette(X, y)
    elif metric_name == "fdr":
        return compute_fishers_ratio(X, y)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


# =============================================================================
# Main Grid Search
# =============================================================================


def main():
    print("=" * 80)
    print("KERNEL PARAMETER OPTIMIZATION - GRID SEARCH")
    print("=" * 80)

    set_seed(RANDOM_SEED)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Dataset:               {DATASET_NAME}")
    print(f"  Optimization metric:   {OPTIMIZATION_METRIC}")
    print(f"  Kernel size:           {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"  Stride:                {STRIDE}")
    print(f"  Kernel topologies:     {KERNEL_TOPOLOGY_NAMES}")
    print(f"  Evolution time range:  {EVOLUTION_TIME_RANGE}")
    print(f"  Scaling factor range:  {SCALING_FACTOR_RANGE}")
    print(f"  Number of samples:     {NUM_SAMPLES}")
    print(f"  Balanced sampling:     {BALANCED_SAMPLING}")
    if BALANCED_SAMPLING:
        print(f"  Samples per class:     {SAMPLES_PER_CLASS}")
    print(f"  Use grayscale:         {USE_GRAYSCALE}")
    print(f"  Random seed:           {RANDOM_SEED}")
    print()

    # Load dataset
    data_root = PROJECT_ROOT / "data"
    print(f"Loading {DATASET_NAME} dataset...")
    full_dataset = load_medmnist_dataset(DATASET_NAME, "train", data_root)
    print(f"Full dataset size: {len(full_dataset)}")

    # Sample subset
    print(f"Sampling {NUM_SAMPLES} samples (balanced={BALANCED_SAMPLING})...")
    if BALANCED_SAMPLING:
        sampled_dataset = sample_dataset(
            full_dataset,
            NUM_SAMPLES,
            balanced=True,
            samples_per_class=SAMPLES_PER_CLASS,
        )
    else:
        sampled_dataset = sample_dataset(full_dataset, NUM_SAMPLES, balanced=False)
    print(f"Sampled dataset size: {len(sampled_dataset)}")

    # Verify class distribution
    labels_sample = []
    for _, label in sampled_dataset:
        label_val = (
            label.item()
            if hasattr(label, "item")
            else label[0]
            if isinstance(label, np.ndarray)
            else label
        )
        labels_sample.append(label_val)
    unique, counts = np.unique(labels_sample, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print()

    # Determine input channels
    sample_img, _ = full_dataset[0]
    in_channels = 1 if USE_GRAYSCALE else sample_img.shape[0]
    print(
        f"Input channels: {in_channels} ({'grayscale' if USE_GRAYSCALE else 'color'})"
    )
    print()

    # Grid search
    print(
        f"Starting grid search over {len(EVOLUTION_TIME_RANGE)} x {len(SCALING_FACTOR_RANGE)} = {len(EVOLUTION_TIME_RANGE) * len(SCALING_FACTOR_RANGE)} combinations..."
    )
    print()

    results = []
    result_matrix = np.zeros((len(EVOLUTION_TIME_RANGE), len(SCALING_FACTOR_RANGE)))

    total_combinations = len(EVOLUTION_TIME_RANGE) * len(SCALING_FACTOR_RANGE)
    pbar = tqdm(total=total_combinations, desc="Grid search progress")

    for i, evolution_time in enumerate(EVOLUTION_TIME_RANGE):
        for j, scaling_factor in enumerate(SCALING_FACTOR_RANGE):
            pbar.set_description(
                f"Testing ev={evolution_time:.2f}, sc={scaling_factor:.0f}"
            )

            # Build quantum convolution layer with current parameters
            q_conv = QuantumConv2d(
                in_channels=in_channels,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                kernel_topology_names=KERNEL_TOPOLOGY_NAMES,
                scaling_factor=scaling_factor,
                evolution_time=evolution_time,
                mode="trotter",
                quantum_device="default.qubit",
            )

            # Extract features
            X, y = extract_features_and_labels(
                sampled_dataset, q_conv, BATCH_SIZE, use_grayscale=USE_GRAYSCALE
            )

            # Evaluate metric
            score = evaluate_metric(X, y, OPTIMIZATION_METRIC)

            # Store results
            result_matrix[i, j] = score
            results.append(
                {
                    "evolution_time": evolution_time,
                    "scaling_factor": scaling_factor,
                    "score": score,
                }
            )

            pbar.update(1)

    pbar.close()

    # Find best parameters
    best_idx = np.nanargmax([r["score"] for r in results])
    best_result = results[best_idx]

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Best {OPTIMIZATION_METRIC.upper()} score: {best_result['score']:.4f}")
    print(f"Best evolution_time:  {best_result['evolution_time']}")
    print(f"Best scaling_factor:  {best_result['scaling_factor']}")
    print()

    # Print full results table
    print("Full results:")
    print(
        f"{'Evolution Time':<20} {'Scaling Factor':<20} {OPTIMIZATION_METRIC.upper():<15}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['evolution_time']:<20.2f} {r['scaling_factor']:<20.0f} {r['score']:<15.4f}"
        )
    print()

    # Create heatmap
    print("Generating heatmap...")
    plt.figure(figsize=(10, 8))

    # Create labels for axes
    x_labels = [f"{sf:.0f}" for sf in SCALING_FACTOR_RANGE]
    y_labels = [f"{et:.2f}" for et in EVOLUTION_TIME_RANGE]

    # Plot heatmap
    sns.heatmap(
        result_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={"label": OPTIMIZATION_METRIC.upper()},
    )

    plt.xlabel("Scaling Factor", fontsize=12)
    plt.ylabel("Evolution Time", fontsize=12)
    plt.title(
        f"Kernel Parameter Optimization - {OPTIMIZATION_METRIC.upper()}\n"
        f"Dataset: {DATASET_NAME}, Kernel: {KERNEL_SIZE}x{KERNEL_SIZE}, Topologies: {KERNEL_TOPOLOGY_NAMES}",
        fontsize=14,
    )
    plt.tight_layout()

    # Save figure
    output_dir = PROJECT_ROOT / "outputs" / "kernel_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_{DATASET_NAME}_k{KERNEL_SIZE}_{OPTIMIZATION_METRIC}_{timestamp}.png"
    output_path = output_dir / filename

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to: {output_path}")

    # Save results to JSON
    json_filename = f"optimization_{DATASET_NAME}_k{KERNEL_SIZE}_{OPTIMIZATION_METRIC}_{timestamp}.json"
    json_path = output_dir / json_filename

    output_data = {
        "configuration": {
            "dataset_name": DATASET_NAME,
            "optimization_metric": OPTIMIZATION_METRIC,
            "kernel_size": KERNEL_SIZE,
            "stride": STRIDE,
            "kernel_topology_names": KERNEL_TOPOLOGY_NAMES,
            "evolution_time_range": EVOLUTION_TIME_RANGE,
            "scaling_factor_range": SCALING_FACTOR_RANGE,
            "num_samples": NUM_SAMPLES,
            "balanced_sampling": BALANCED_SAMPLING,
            "samples_per_class": SAMPLES_PER_CLASS if BALANCED_SAMPLING else None,
            "use_grayscale": USE_GRAYSCALE,
            "random_seed": RANDOM_SEED,
        },
        "best_parameters": {
            "evolution_time": best_result["evolution_time"],
            "scaling_factor": best_result["scaling_factor"],
            "score": best_result["score"],
        },
        "all_results": results,
        "result_matrix": result_matrix.tolist(),
        "timestamp": timestamp,
    }

    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {json_path}")
    print()
    print("Optimization complete!")

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
