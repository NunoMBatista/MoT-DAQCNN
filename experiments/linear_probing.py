"""
Linear probing of frozen quantum feature maps.

For each config, loads the pre-computed quantum feature cache, flattens the
spatial feature map to a vector, fits a StandardScaler + LinearSVC on the
training split, and reports accuracy and AUC on the test split.

Why linear?
-----------
A linear SVM measures whether the quantum feature space is linearly separable.
This is independent of the trained classification head and tests the intrinsic
discriminability of the frozen quantum representations.  Using a nonlinear
classifier (e.g. RBF SVM) would confound kernel properties with the probing
classifier's own nonlinearity, making the comparison uninformative.

Primary comparison
------------------
Digital-ZZ vs Analog-ZZ: same feature dimensionality, different encoding mode.
This is the clean isolation of the encoding contribution.

Secondary comparison
--------------------
Digital-Z vs Digital-ZZ: different dimensionality (9 vs 45 per kernel), so any
AUC difference is confounded by feature count.  Treat as indicative only.

Classical random baseline (--random-filters N)
----------------------------------------------
Applies N Gaussian random 3×3 filters (stride matching the quantum config) to
the raw images and probes the resulting feature map with the same linear SVM.
This tests whether the quantum interaction structure contributes over unstructured
random projections of the same dimensionality.  For a fair comparison, set N to
match the quantum feature count per patch (e.g. 45 for ZZ, 9 for Z-only).

Usage
-----
    python experiments/linear_probing.py \\
        --configs configs/breast_mnist/original/1_kernel_3x3.yml \\
                  configs/breast_mnist/original/1_kernel_3x3_digital_zz.yml \\
                  configs/breast_mnist/original/1_kernel_3x3_analog_zz.yml \\
        --labels "Digital-Z" "Digital-ZZ" "Analog-ZZ" \\
        --raw-pixels --random-filters 45
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data import get_dataloaders
from src.utils.quantum_dataset_cache import find_cached_quantum_dataset, load_cached_quantum_dataset


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect(loader):
    """Flatten all batches from a DataLoader into (X, y) numpy arrays.

    Features are flattened across spatial dims: (N, C, H, W) -> (N, C*H*W).
    This preserves spatial information, which matters for classification.
    """
    xs, ys = [], []
    for x, y in loader:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        xs.append(x.reshape(x.shape[0], -1))
        ys.append(np.asarray(y).squeeze())
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Linear SVM probe
# ---------------------------------------------------------------------------

def probe(X_train, y_train, X_test, y_test, C=1.0):
    """Fit StandardScaler + LinearSVC on train, evaluate on test.

    AUC is computed from decision_function scores (no probability calibration
    needed).  For binary tasks this is the standard AUROC; for multiclass it
    is macro one-vs-rest AUC.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    svm = LinearSVC(C=C, max_iter=50_000, dual="auto")
    svm.fit(X_tr, y_train)

    y_pred = svm.predict(X_te)
    acc = accuracy_score(y_test, y_pred)

    scores = svm.decision_function(X_te)
    n_classes = len(np.unique(y_train))
    if n_classes == 2:
        auc = roc_auc_score(y_test, scores)
    else:
        auc = roc_auc_score(y_test, scores, multi_class="ovr", average="macro")

    return acc, auc


# ---------------------------------------------------------------------------
# Per-source probing
# ---------------------------------------------------------------------------

def probe_quantum(cfg_path, label, C=1.0):
    """Probe frozen quantum features for one config. Returns a result row."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cache_path = find_cached_quantum_dataset(cfg)
    if cache_path is None:
        print(f"  [SKIP] No cache found for '{label}' — run create_quantum_dataset.py first.")
        return None

    requested_kernels = cfg.get("model", {}).get("kernel_topology_names")
    batch_size = cfg.get("dataset", {}).get("batch_size", 64)

    train_loader, _, test_loader, n_classes, _ = load_cached_quantum_dataset(
        cache_path, batch_size, num_workers=0, requested_kernels=requested_kernels
    )

    X_train, y_train = collect(train_loader)
    X_test, y_test = collect(test_loader)
    n_features = X_train.shape[1]

    acc, auc = probe(X_train, y_train, X_test, y_test, C=C)
    print(f"  {label:<22s}  dim={n_features:>5d}  acc={acc:.4f}  AUC={auc:.4f}")
    return [label, n_features, f"{acc:.4f}", f"{auc:.4f}"]


def probe_raw_pixels(cfg_path, C=1.0):
    """Probe flattened raw pixel values. Uses the dataset from cfg_path."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_loader, _, test_loader, _ = get_dataloaders(cfg)

    X_train, y_train = collect(train_loader)
    X_test, y_test = collect(test_loader)
    n_features = X_train.shape[1]

    acc, auc = probe(X_train, y_train, X_test, y_test, C=C)
    print(f"  {'Raw pixels':<22s}  dim={n_features:>5d}  acc={acc:.4f}  AUC={auc:.4f}")
    return ["Raw pixels", n_features, f"{acc:.4f}", f"{auc:.4f}"]


def probe_random_filters(cfg_path, n_filters, C=1.0, seed=42):
    """Apply Gaussian random conv filters to raw images, probe with linear SVM.

    Filters are fixed (not trained).  Using the same stride and kernel_size as
    the quantum configs ensures the output spatial dimensions match, making the
    feature count comparable to the quantum ZZ features at the same n_filters.

    Filters are L2-normalised so no single random direction dominates.
    StandardScaler is still applied before SVM as with all other probes.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    kernel_size = cfg.get("model", {}).get("kernel_size", 3)
    stride = cfg.get("model", {}).get("stride", 3)

    # Random Gaussian filters, L2-normalised per filter
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n_filters, 1, kernel_size, kernel_size)).astype(np.float32)
    norms = np.linalg.norm(w.reshape(n_filters, -1), axis=1, keepdims=True)
    w /= (norms.reshape(n_filters, 1, 1, 1) + 1e-8)
    filters = torch.tensor(w)

    train_loader, _, test_loader, _ = get_dataloaders(cfg)

    def extract(loader):
        xs, ys = [], []
        for x, y in loader:
            out = F.conv2d(x.float(), filters, stride=stride)  # (B, n_filters, h, w)
            xs.append(out.detach().numpy().reshape(x.shape[0], -1))
            ys.append(np.asarray(y).squeeze())
        return np.concatenate(xs), np.concatenate(ys)

    X_train, y_train = extract(train_loader)
    X_test, y_test = extract(test_loader)
    n_features = X_train.shape[1]

    acc, auc = probe(X_train, y_train, X_test, y_test, C=C)
    label = f"Random ×{n_filters} filters"
    print(f"  {label:<22s}  dim={n_features:>5d}  acc={acc:.4f}  AUC={auc:.4f}")
    return [label, n_features, f"{acc:.4f}", f"{auc:.4f}"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Linear probing of frozen quantum features")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Config YAML paths, one per quantum variant")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display name for each config (same order as --configs)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="LinearSVC regularisation strength (default: 1.0)")
    parser.add_argument("--raw-pixels", action="store_true",
                        help="Also run a raw-pixel baseline (uses the dataset from the first config)")
    parser.add_argument("--random-filters", type=int, nargs="+", default=[], metavar="N",
                        help="Run random-filter baselines at each given filter count. "
                             "Use one value per quantum scale, e.g. --random-filters 45 180.")
    parser.add_argument("--output-csv", type=str, default=None, metavar="PATH",
                        help="Save results to a CSV file.")
    args = parser.parse_args()

    labels = args.labels or [
        os.path.basename(c).replace(".yml", "") for c in args.configs
    ]
    if len(labels) != len(args.configs):
        parser.error("--labels must have the same number of entries as --configs")

    print("\nLinear Probing — Frozen Quantum Features")
    print("=" * 60)
    print(f"  LinearSVC  C={args.C}")
    print(f"  Features standardised (fit on train split only)")
    print("=" * 60)

    rows = []

    if args.raw_pixels:
        row = probe_raw_pixels(args.configs[0], C=args.C)
        if row:
            rows.append(row)

    for n in args.random_filters:
        row = probe_random_filters(args.configs[0], n, C=args.C)
        if row:
            rows.append(row)

    for cfg_path, label in zip(args.configs, labels):
        row = probe_quantum(cfg_path, label, C=args.C)
        if row:
            rows.append(row)

    if not rows:
        print("\nNo results — check that caches exist.")
        return

    # Print table
    col_w = [max(len(r[0]) for r in rows), 10, 8, 8]
    header = f"{'Feature source':<{col_w[0]}}  {'Dimensions':>{col_w[1]}}  {'Acc':>{col_w[2]}}  {'AUC':>{col_w[3]}}"
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for r in rows:
        print(f"{r[0]:<{col_w[0]}}  {str(r[1]):>{col_w[1]}}  {r[2]:>{col_w[2]}}  {r[3]:>{col_w[3]}}")

    print()
    print("Note: Digital-ZZ vs Analog-ZZ is the dimensionality-matched comparison")
    print("      (same feature count). Digital-Z vs ZZ differs in dimension too.")

    if args.output_csv:
        import csv
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["feature_source", "dimensions", "acc", "auc"])
            writer.writerows(rows)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
