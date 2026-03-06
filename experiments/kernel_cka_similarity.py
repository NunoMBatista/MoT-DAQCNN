"""
kernel_cka_similarity.py

Measure pairwise similarity between quantum kernel feature maps using
Centered Kernel Alignment (CKA) on cached quantum `.npz` datasets.

CKA is a principled similarity index in [0, 1]:
  - CKA = 0  → feature maps are perfectly dissimilar (orthogonal).
  - CKA = 1  → feature maps are identical up to isotropic scaling.

For routing to be genuinely beneficial the kernels should produce
*complementary* representations, i.e. low CKA.  This script lets you
quickly check that assumption on BreastMNIST and PneumoniaMNIST without
running any training.

Why CKA over plain cosine / Gram-matrix correlation?
  CKA is invariant to orthogonal transformations and isotropic scaling of
  each kernel's feature matrix, which makes it a much fairer comparison
  than raw cosine similarity of flattened vectors.

Usage (from project root):
    # Compare all kernel pairs on both datasets (default)
    python experiments/kernel_cka_similarity.py

    # Compare only specific datasets
    python experiments/kernel_cka_similarity.py --datasets breast_mnist pneumonia_mnist

    # Use the test split instead of train
    python experiments/kernel_cka_similarity.py --split test

    # Limit to N samples (faster on large datasets)
    python experiments/kernel_cka_similarity.py --max_samples 1000

    # Save plots to a custom output directory
    python experiments/kernel_cka_similarity.py --output_dir outputs/cka

    # Skip saving figures (just print to stdout)
    python experiments/kernel_cka_similarity.py --no_plot
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import QUANTUM_DATASETS_DIR

# ---------------------------------------------------------------------------
# CKA implementation
# ---------------------------------------------------------------------------


def _centering_matrix(n: int) -> np.ndarray:
    """Return the n×n centering matrix H = I - (1/n) * 11^T."""
    return np.eye(n) - np.ones((n, n)) / n


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two feature matrices X and Y.

    Uses the HSIC-based formulation from Kornblith et al. (2019):
        CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    with K = X X^T and L = Y Y^T (linear kernels).

    A numerically efficient version avoids forming the full n×n Gram matrices
    by working directly with the feature matrices.  This gives O(n p + n q)
    time instead of O(n^2 (p + q)).

    Args:
        X: (N, P) feature matrix for kernel i.
        Y: (N, Q) feature matrix for kernel j.

    Returns:
        CKA value in [0, 1].
    """
    n = X.shape[0]
    # Center columns (equivalent to applying H from both sides in Gram space)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC(K, L) = (1/(n-1)^2) * ||Y^T X||_F^2   (unbiased estimator dropped
    # the 1/(n-1)^2 factor since it cancels in the ratio)
    dot = X.T @ Y  # (P, Q)
    hsic_xy = np.sum(dot**2)

    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        # Both feature maps are constant → ill-defined, return NaN
        return float("nan")

    return float(hsic_xy / denom)


# ---------------------------------------------------------------------------
# Feature extraction from cached .npz
# ---------------------------------------------------------------------------


def load_all_cached_datasets() -> list[Path]:
    """Return all .npz files found in QUANTUM_DATASETS_DIR."""
    if not QUANTUM_DATASETS_DIR.is_dir():
        return []
    return sorted(QUANTUM_DATASETS_DIR.glob("*.npz"))


def find_npz_for_dataset(dataset_name: str) -> Path | None:
    """
    Return the best matching .npz in QUANTUM_DATASETS_DIR for *dataset_name*.
    Prefers files whose name starts with the dataset name.
    """
    candidates = [
        p for p in load_all_cached_datasets() if p.stem.startswith(dataset_name)
    ]
    if not candidates:
        return None
    # Pick the one with the most kernels (largest file) as a tie-breaker
    return max(candidates, key=lambda p: p.stat().st_size)


def extract_per_kernel_features(
    npz_path: Path,
    split: str = "train",
    max_samples: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Load the cached quantum dataset and return a dict mapping each kernel
    topology name → flattened feature matrix of shape (N, P).

    P = num_qubits (channels per kernel) × H_out × W_out.

    Args:
        npz_path: Path to the .npz file.
        split: Which split to load ("train", "val", "test").
        max_samples: If set, randomly subsample this many images.

    Returns:
        kernel_features: dict  kernel_name → (N, P) float32 array.
        metadata: the parsed metadata dict from the .npz.
    """
    data = np.load(npz_path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))

    features: np.ndarray = data[f"{split}_features"]  # (N, C_total, H, W)

    if max_samples is not None and max_samples < features.shape[0]:
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(features.shape[0], size=max_samples, replace=False)
        indices.sort()
        features = features[indices]

    # Build kernel → channel indices mapping
    channel_map: list[dict] = metadata["channel_kernel_map"]
    kernel_to_channels: dict[str, list[int]] = {}
    for entry in channel_map:
        k = entry["kernel"]
        c = entry["channel"]
        kernel_to_channels.setdefault(k, []).append(c)

    # Extract and flatten each kernel's channels
    kernel_features: dict[str, np.ndarray] = {}
    for kernel_name, ch_indices in sorted(
        kernel_to_channels.items(), key=lambda kv: min(kv[1])
    ):
        ch_indices_sorted = sorted(ch_indices)
        # features[:, channels, H, W] → reshape to (N, channels * H * W)
        kernel_feat = features[:, ch_indices_sorted, :, :]  # (N, C_k, H, W)
        N = kernel_feat.shape[0]
        kernel_features[kernel_name] = kernel_feat.reshape(N, -1).astype(np.float32)

    return kernel_features, metadata


# ---------------------------------------------------------------------------
# CKA matrix computation
# ---------------------------------------------------------------------------


def compute_cka_matrix(
    kernel_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute the full pairwise CKA matrix for all kernels.

    Returns:
        cka_matrix: (K, K) symmetric matrix of CKA values.
        kernel_names: ordered list of kernel names.
    """
    kernel_names = list(kernel_features.keys())
    K = len(kernel_names)
    cka_matrix = np.full((K, K), fill_value=np.nan, dtype=np.float64)

    for i, ki in enumerate(kernel_names):
        for j, kj in enumerate(kernel_names):
            if i == j:
                cka_matrix[i, j] = 1.0
            elif j > i:
                val = linear_cka(kernel_features[ki], kernel_features[kj])
                cka_matrix[i, j] = val
                cka_matrix[j, i] = val  # symmetric

    return cka_matrix, kernel_names


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_cka_table(
    cka_matrix: np.ndarray,
    kernel_names: list[str],
    dataset_name: str,
    split: str,
    n_samples: int,
) -> None:
    """Pretty-print the CKA similarity matrix and per-pair interpretation."""
    K = len(kernel_names)
    col_w = 12

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"CKA Similarity Matrix  —  {dataset_name}  ({split} split, N={n_samples})")
    print(sep)

    # Header
    header = f"{'':>12}" + "".join(f"{n:>{col_w}}" for n in kernel_names)
    print(header)
    print("-" * len(header))

    for i, ki in enumerate(kernel_names):
        row = f"{ki:>12}"
        for j in range(K):
            val = cka_matrix[i, j]
            if np.isnan(val):
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{val:>{col_w}.4f}"
        print(row)

    # Pairwise summary
    print(f"\nPairwise CKA (off-diagonal):")
    print(f"  {'Pair':<28} {'CKA':>8}  Interpretation")
    print("  " + "-" * 58)

    for i in range(K):
        for j in range(i + 1, K):
            val = cka_matrix[i, j]
            if np.isnan(val):
                interp = "ill-defined (constant features)"
            elif val < 0.1:
                interp = "very low — highly complementary ✓✓"
            elif val < 0.3:
                interp = "low — good complementarity ✓"
            elif val < 0.5:
                interp = "moderate — some overlap"
            elif val < 0.75:
                interp = "high — significant redundancy"
            else:
                interp = "very high — near-duplicate features ✗"
            pair = f"{kernel_names[i]} vs {kernel_names[j]}"
            print(f"  {pair:<28} {val:>8.4f}  {interp}")

    # Summary stats
    off_diag_vals = [
        cka_matrix[i, j]
        for i in range(K)
        for j in range(i + 1, K)
        if not np.isnan(cka_matrix[i, j])
    ]
    if off_diag_vals:
        print(f"\nSummary:")
        print(f"  Mean CKA (off-diagonal) : {np.mean(off_diag_vals):.4f}")
        print(f"  Max  CKA (off-diagonal) : {np.max(off_diag_vals):.4f}")
        print(f"  Min  CKA (off-diagonal) : {np.min(off_diag_vals):.4f}")

    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    kernel_names: list[str],
    dataset_name: str,
    split: str,
    n_samples: int,
    output_path: Path,
) -> None:
    """
    Save a heatmap of the CKA matrix.

    Requires matplotlib. If not installed the plot is skipped gracefully.
    """
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not found — skipping heatmap.")
        return

    K = len(kernel_names)
    fig, ax = plt.subplots(figsize=(max(5, K * 1.4), max(4, K * 1.2)))

    # Use a diverging colormap anchored at 0 (orthogonal) → 1 (identical)
    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    im = ax.imshow(cka_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(K):
        for j in range(K):
            val = cka_matrix[i, j]
            text = f"{val:.3f}" if not np.isnan(val) else "N/A"
            # Pick white or black text for contrast
            bg = cmap(norm(val)) if not np.isnan(val) else (1, 1, 1, 1)
            luminance = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
            text_color = "white" if luminance < 0.45 else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(kernel_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(kernel_names, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA similarity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Title
    nice_name = dataset_name.replace("_", " ").title()
    ax.set_title(
        f"Pairwise Linear CKA — {nice_name}\n({split} split, N={n_samples} samples)",
        fontsize=11,
        pad=12,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved heatmap → {output_path}")


def plot_side_by_side_comparison(
    results: list[dict],
    output_path: Path,
) -> None:
    """
    Plot all CKA matrices side by side (one column per dataset) for easy
    visual comparison.
    """
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_datasets = len(results)
    if n_datasets == 0:
        return

    fig, axes = plt.subplots(
        1,
        n_datasets,
        figsize=(max(5, 5 * n_datasets), max(4, 4)),
        squeeze=False,
    )

    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for col, res in enumerate(results):
        ax = axes[0][col]
        matrix = res["cka_matrix"]
        names = res["kernel_names"]
        K = len(names)

        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        for i in range(K):
            for j in range(K):
                val = matrix[i, j]
                text = f"{val:.2f}" if not np.isnan(val) else "N/A"
                bg = cmap(norm(val)) if not np.isnan(val) else (1, 1, 1, 1)
                luminance = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
                text_color = "white" if luminance < 0.45 else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )

        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)

        nice_name = res["dataset_name"].replace("_", " ").title()
        ax.set_title(
            f"{nice_name}\n(N={res['n_samples']})",
            fontsize=10,
            pad=8,
        )

        # Add colorbar only on the last subplot
        if col == n_datasets - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("CKA", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Pairwise Linear CKA — Quantum Kernel Similarity Comparison\n"
        f"({results[0]['split']} split)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved comparison figure → {output_path}")


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------


def save_json_summary(results: list[dict], output_path: Path) -> None:
    """Serialize the CKA results to a human-readable JSON file."""
    summary = []
    for res in results:
        pairs = []
        names = res["kernel_names"]
        mat = res["cka_matrix"]
        K = len(names)
        for i in range(K):
            for j in range(i + 1, K):
                pairs.append(
                    {
                        "kernel_i": names[i],
                        "kernel_j": names[j],
                        "cka": round(float(mat[i, j]), 6)
                        if not np.isnan(mat[i, j])
                        else None,
                    }
                )
        off_diag = [p["cka"] for p in pairs if p["cka"] is not None]
        summary.append(
            {
                "dataset_name": res["dataset_name"],
                "split": res["split"],
                "n_samples": res["n_samples"],
                "kernel_names": names,
                "pairwise_cka": pairs,
                "mean_cka": round(float(np.mean(off_diag)), 6) if off_diag else None,
                "max_cka": round(float(np.max(off_diag)), 6) if off_diag else None,
                "min_cka": round(float(np.min(off_diag)), 6) if off_diag else None,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [json] Saved summary → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise linear CKA between quantum kernel feature maps "
            "loaded from cached .npz datasets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["breast_mnist", "pneumonia_mnist"],
        metavar="DATASET",
        help=(
            "Dataset name(s) to analyse. Must match the prefix of an .npz file "
            "in data/quantum_datasets/. "
            "(default: breast_mnist pneumonia_mnist)"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which data split to use for computing CKA. (default: train)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of samples to use. Useful to speed things up on "
            "large datasets. By default all samples are used."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "kernel_cka"),
        metavar="DIR",
        help="Directory where plots and JSON summary are saved.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip saving matplotlib figures (text output only).",
    )
    parser.add_argument(
        "--no_json",
        action="store_true",
        help="Skip saving the JSON summary file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Quantum Kernel CKA Similarity Analysis")
    print("=" * 70)
    print(f"  Datasets    : {args.datasets}")
    print(f"  Split       : {args.split}")
    print(f"  Max samples : {args.max_samples or 'all'}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 70)

    all_results: list[dict] = []

    for dataset_name in args.datasets:
        print(f"\n{'─' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─' * 70}")

        # Locate cached .npz
        npz_path = find_npz_for_dataset(dataset_name)
        if npz_path is None:
            print(
                f"  [SKIP] No cached .npz found for '{dataset_name}' in "
                f"{QUANTUM_DATASETS_DIR}.\n"
                f"         Run experiments/create_quantum_dataset.py first."
            )
            continue

        print(f"  Loaded : {npz_path.name}")

        # Extract per-kernel feature matrices
        kernel_features, metadata = extract_per_kernel_features(
            npz_path,
            split=args.split,
            max_samples=args.max_samples,
        )

        n_samples = next(iter(kernel_features.values())).shape[0]
        kernel_names = list(kernel_features.keys())
        feat_dim = {k: v.shape[1] for k, v in kernel_features.items()}

        print(f"  Kernels : {kernel_names}")
        print(f"  Samples : {n_samples}")
        print(f"  Feature dims per kernel: {feat_dim}")

        # Compute CKA matrix
        print(f"  Computing CKA matrix ({len(kernel_names)}×{len(kernel_names)}) …")
        cka_matrix, ordered_names = compute_cka_matrix(kernel_features)

        # Print table to stdout
        print_cka_table(
            cka_matrix,
            ordered_names,
            dataset_name,
            args.split,
            n_samples,
        )

        result = {
            "dataset_name": dataset_name,
            "split": args.split,
            "n_samples": n_samples,
            "kernel_names": ordered_names,
            "cka_matrix": cka_matrix,
            "metadata": metadata,
        }
        all_results.append(result)

        # Per-dataset heatmap
        if not args.no_plot:
            plot_path = output_dir / f"cka_{dataset_name}_{args.split}_{timestamp}.png"
            plot_cka_heatmap(
                cka_matrix,
                ordered_names,
                dataset_name,
                args.split,
                n_samples,
                plot_path,
            )

    # Side-by-side comparison (only if we have ≥2 datasets)
    if len(all_results) >= 2 and not args.no_plot:
        comparison_path = output_dir / f"cka_comparison_{args.split}_{timestamp}.png"
        plot_side_by_side_comparison(all_results, comparison_path)

    # JSON summary
    if all_results and not args.no_json:
        json_path = output_dir / f"cka_summary_{args.split}_{timestamp}.json"
        save_json_summary(all_results, json_path)

    # Final verdict: print a cross-dataset mean CKA comparison
    if len(all_results) >= 2:
        print(f"\n{'=' * 70}")
        print("Cross-dataset mean CKA comparison")
        print(
            "  (lower mean CKA = more complementary kernels = routing more likely to help)"
        )
        print(f"{'─' * 70}")
        for res in all_results:
            mat = res["cka_matrix"]
            K = len(res["kernel_names"])
            off_diag = [
                mat[i, j]
                for i in range(K)
                for j in range(i + 1, K)
                if not np.isnan(mat[i, j])
            ]
            mean_cka = np.mean(off_diag) if off_diag else float("nan")
            nice_name = res["dataset_name"].replace("_", " ").title()
            print(f"  {nice_name:<25} mean CKA = {mean_cka:.4f}")
        print("=" * 70)

    print("\nDone.")


if __name__ == "__main__":
    main()
