import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.daqcnn import DAQCNN  # noqa: E402
from src.models.daqcnn_training import train_one_epoch  # noqa: E402
from src.utils.data import get_dataloaders  # noqa: E402
from src.utils.evaluate import evaluate  # noqa: E402
from src.utils.quantum_dataset_cache import (  # noqa: E402
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_single_kernel_cfg(base_cfg: dict, kernel_name: str) -> dict:
    """Return a deep copy of base_cfg with kernel_topology_names set to a single kernel."""
    import copy

    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["kernel_topology_names"] = [kernel_name]
    return cfg


def train_classifier(cfg: dict, kernel_name: str, seed: int, device: str) -> dict:
    """
    Train a single DAQCNN classifier using only one kernel topology.

    Returns a dict with:
        - test_acc: float
        - test_auc: float
        - preds: np.ndarray of predicted class indices (N,)
        - labels: np.ndarray of true class indices (N,)
    """
    print(f"\n  [Kernel: {kernel_name}] Checking for cached quantum dataset...")
    cached_path = find_cached_quantum_dataset(cfg)

    using_cache = False
    cache_meta = None

    model_cfg = cfg.get("model", {})

    if cached_path is not None:
        print(f"  [Kernel: {kernel_name}] Cache found: {cached_path}")
        batch_size = cfg.get("dataset", {}).get("batch_size", 32)
        num_workers = cfg.get("dataset", {}).get("num_workers", 2)
        train_loader, val_loader, test_loader, n_classes, cache_meta = (
            load_cached_quantum_dataset(
                cached_path,
                batch_size,
                num_workers,
                requested_kernels=[kernel_name],
            )
        )
        using_cache = True
    else:
        print(f"  [Kernel: {kernel_name}] No cache found, computing from scratch.")
        train_loader, val_loader, test_loader, n_classes = get_dataloaders(cfg)

    num_classes = model_cfg.get("num_classes", n_classes)

    override_quantum_out_channels = None
    if using_cache and cache_meta:
        override_quantum_out_channels = cache_meta.get("out_channels")

    model_kwargs = {
        "num_classes": num_classes,
        "kernel_size": model_cfg.get("kernel_size", 3),
        "stride": model_cfg.get("stride", 3),
        "kernel_topology_names": [kernel_name],
        "scaling_factor": model_cfg.get("scaling_factor", 1.0),
        "evolution_time": model_cfg.get("evolution_time", 2.5),
        "mode": model_cfg.get("mode", "trotter"),
        "dropout": model_cfg.get("dropout", 0.6),
        "activation": model_cfg.get("activation", "relu"),
        "quantum_device": model_cfg.get("quantum_device", "default.qubit"),
        "quantum_device_kwargs": model_cfg.get("quantum_device_kwargs", None),
        "classical_device": device,
        "in_channels": model_cfg.get("in_channels", 1),
        "interface": model_cfg.get("interface", "torch"),
    }
    if override_quantum_out_channels is not None:
        model_kwargs["override_quantum_out_channels"] = override_quantum_out_channels

    model = DAQCNN(**model_kwargs)
    if using_cache:
        model.bypass_quantum = True
    model.to(device)

    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    epochs = optim_cfg.get("epochs", 10)
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = optim_cfg.get("patience", None)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, grad_clip, epoch_pbar=None
        )
        val_loss, val_acc = evaluate(model, val_loader, device, split_name="Val")
        print(
            f"  [Kernel: {kernel_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if patience is not None:
                epochs_without_improvement = 0
        elif patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"  [Kernel: {kernel_name}] Early stopping at epoch {epoch} "
                    f"(patience={patience})"
                )
                break

    print(f"  [Kernel: {kernel_name}] Evaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, device, split_name="Test", compute_full_metrics=True
    )

    # Derive hard predictions from stored probabilities
    probs = test_metrics["probs"]  # (N, C) numpy array
    labels = test_metrics["labels"]  # (N,)   numpy array
    preds = probs.argmax(axis=1)  # (N,)

    return {
        "test_acc": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "preds": preds,
        "labels": labels,
    }


def compute_orthogonality_table(
    preds_i: np.ndarray,
    preds_j: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Build the 2x2 correctness table between two classifiers Ci and Cj.

    Returns:
        N11: both correct
        N00: both wrong  (Double Faults)
        N10: Ci correct, Cj wrong
        N01: Ci wrong,   Cj correct
    """
    correct_i = preds_i == labels
    correct_j = preds_j == labels

    N11 = int(np.sum(correct_i & correct_j))
    N00 = int(np.sum(~correct_i & ~correct_j))
    N10 = int(np.sum(correct_i & ~correct_j))
    N01 = int(np.sum(~correct_i & correct_j))

    return {"N11": N11, "N00": N00, "N10": N10, "N01": N01}


def print_results(
    kernel_i: str,
    kernel_j: str,
    result_i: dict,
    result_j: dict,
    table: dict,
):
    total = table["N11"] + table["N00"] + table["N10"] + table["N01"]
    sep = "=" * 60

    print(f"\n{sep}")
    print("Orthogonality Check Results")
    print(sep)

    print(f"\n{'Kernel':<20} {'ROC-AUC':>10} {'ACC':>10}")
    print("-" * 42)
    print(
        f"  {kernel_i:<18} {result_i['test_auc']:>10.4f} {result_i['test_acc']:>10.4f}"
    )
    print(
        f"  {kernel_j:<18} {result_j['test_auc']:>10.4f} {result_j['test_acc']:>10.4f}"
    )

    print(f"\n{'Correctness Table':}")
    print(f"  (N={total} test samples)")
    print(f"\n  {'':>22} | Cj correct | Cj wrong")
    print(f"  {'-' * 44}")
    print(
        f"  {'Ci correct':>22} | "
        f"N11 = {table['N11']:>5} ({100 * table['N11'] / total:.1f}%) | "
        f"N10 = {table['N10']:>5} ({100 * table['N10'] / total:.1f}%)"
    )
    print(
        f"  {'Ci wrong':>22} | "
        f"N01 = {table['N01']:>5} ({100 * table['N01'] / total:.1f}%) | "
        f"N00 = {table['N00']:>5} ({100 * table['N00'] / total:.1f}%)"
    )
    print()
    print(
        f"  N11 (both correct)         : {table['N11']:>6}  ({100 * table['N11'] / total:.1f}%)"
    )
    print(
        f"  N00 (double faults)        : {table['N00']:>6}  ({100 * table['N00'] / total:.1f}%)"
    )
    print(
        f"  N10 (Ci correct, Cj wrong) : {table['N10']:>6}  ({100 * table['N10'] / total:.1f}%)"
    )
    print(
        f"  N01 (Ci wrong, Cj correct) : {table['N01']:>6}  ({100 * table['N01'] / total:.1f}%)"
    )
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Orthogonality check: train two single-kernel DAQCNN classifiers "
            "and compare their prediction agreement on the test set."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "pneumonia_mnist_multi_seed_3kern.yml"),
        help="Path to base YAML config file.",
    )
    parser.add_argument(
        "--kernel_i",
        type=str,
        default="kings",
        help="Name of the first kernel topology (Ci).",
    )
    parser.add_argument(
        "--kernel_j",
        type=str,
        default="horizontal",
        help="Name of the second kernel topology (Cj).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    set_seed(args.seed)

    requested_device = base_cfg.get("model", {}).get("classical_device", "auto")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device

    print("\n" + "=" * 60)
    print("Orthogonality Check")
    print("=" * 60)
    print(f"  Config   : {args.config}")
    print(f"  Dataset  : {base_cfg.get('dataset', {}).get('name', 'unknown')}")
    print(f"  Kernel i : {args.kernel_i}")
    print(f"  Kernel j : {args.kernel_j}")
    print(f"  Seed     : {args.seed}")
    print(f"  Device   : {device}")
    print("=" * 60)

    # --- Train Ci ---
    print(f"\n[1/2] Training classifier with kernel '{args.kernel_i}'...")
    cfg_i = build_single_kernel_cfg(base_cfg, args.kernel_i)
    set_seed(args.seed)
    result_i = train_classifier(cfg_i, args.kernel_i, args.seed, device)

    # --- Train Cj ---
    print(f"\n[2/2] Training classifier with kernel '{args.kernel_j}'...")
    cfg_j = build_single_kernel_cfg(base_cfg, args.kernel_j)
    set_seed(args.seed)
    result_j = train_classifier(cfg_j, args.kernel_j, args.seed, device)

    # Sanity check: both classifiers should have been evaluated on the same test set
    if not np.array_equal(result_i["labels"], result_j["labels"]):
        print(
            "WARNING: Test labels differ between the two runs. "
            "Make sure both use the same (deterministic) test split."
        )

    # --- Compute orthogonality table ---
    table = compute_orthogonality_table(
        result_i["preds"],
        result_j["preds"],
        result_i["labels"],
    )

    print_results(args.kernel_i, args.kernel_j, result_i, result_j, table)


if __name__ == "__main__":
    main()
