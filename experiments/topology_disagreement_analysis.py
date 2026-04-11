"""
Phase 1 Diagnostic: Per-Topology Disagreement Analysis

Trains a separate classifier on each topology's cached quantum features and
measures inter-topology disagreement to establish whether per-sample routing
can exploit complementary information across topologies.

Key metrics:
- Per-topology test accuracy (multi-seed)
- Pairwise correctness tables (who disagrees with whom)
- Oracle accuracy: best possible per-sample routing ceiling
- Complementarity score: fraction of samples with topology-dependent correctness
- Routing potential: fraction of samples where topologies disagree

Usage:
    python experiments/topology_disagreement_analysis.py \
        --config configs/breast_mnist/original/4_kernels_3x3.yml \
        --seeds 0 1 2 3 4
"""

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.daqcnn import DAQCNN  # noqa: E402
from src.models.daqcnn_training import train_one_epoch  # noqa: E402
from src.utils.evaluate import evaluate  # noqa: E402
from src.utils.quantum_dataset_cache import (  # noqa: E402
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)


# ── Utilities ────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_single_kernel_cfg(base_cfg: dict, kernel_name: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["kernel_topology_names"] = [kernel_name]
    return cfg


# ── Single-topology classifier training ──────────────────────────────────────


def train_single_topology_classifier(cfg, kernel_name, seed, device):
    """
    Train a DAQCNN classification head on a single topology's cached features.

    Returns dict with: test_acc, test_auc, test_f1, preds (N,), probs (N,C), labels (N,)
    """
    set_seed(seed)

    cached_path = find_cached_quantum_dataset(cfg)
    if cached_path is None:
        raise RuntimeError(
            f"No cached quantum dataset found for kernel '{kernel_name}'. "
            f"Run experiments/create_quantum_dataset.py first."
        )

    batch_size = cfg.get("dataset", {}).get("batch_size", 32)
    num_workers = cfg.get("dataset", {}).get("num_workers", 2)
    train_loader, val_loader, test_loader, n_classes, cache_meta = (
        load_cached_quantum_dataset(
            cached_path, batch_size, num_workers, requested_kernels=[kernel_name]
        )
    )

    model_cfg = cfg.get("model", {})
    model = DAQCNN(
        num_classes=model_cfg.get("num_classes", n_classes),
        kernel_size=model_cfg.get("kernel_size", 3),
        stride=model_cfg.get("stride", 3),
        kernel_topology_names=[kernel_name],
        scaling_factor=model_cfg.get("scaling_factor", 1.0),
        evolution_time=model_cfg.get("evolution_time", 2.5),
        mode=model_cfg.get("mode", "trotter"),
        dropout=model_cfg.get("dropout", 0.6),
        activation=model_cfg.get("activation", "relu"),
        quantum_device=model_cfg.get("quantum_device", "default.qubit"),
        classical_device=device,
        in_channels=model_cfg.get("in_channels", 1),
        override_quantum_out_channels=cache_meta.get("out_channels"),
        include_correlators=model_cfg.get("include_correlators", False),
    )
    model.bypass_quantum = True
    model.to(device)

    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    epochs = optim_cfg.get("epochs", 100)
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = optim_cfg.get("patience", None)
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, grad_clip, epoch_pbar=None
        )
        val_loss, val_acc = evaluate(model, val_loader, device, split_name="Val")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        elif patience is not None:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    [{kernel_name}] Early stop at epoch {epoch}")
                break

    # Restore best model for evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(
        model, test_loader, device, split_name="Test", compute_full_metrics=True
    )

    probs = test_metrics["probs"]  # (N, C)
    labels = test_metrics["labels"]  # (N,)
    preds = probs.argmax(axis=1)  # (N,)

    return {
        "test_acc": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "preds": preds,
        "probs": probs,
        "labels": labels,
    }


# ── Disagreement analysis ────────────────────────────────────────────────────


def compute_pairwise_tables(results, kernel_names):
    """Compute 2x2 correctness tables for all topology pairs."""
    labels = results[kernel_names[0]]["labels"]
    tables = {}

    for i, ki in enumerate(kernel_names):
        for j, kj in enumerate(kernel_names):
            if j <= i:
                continue

            correct_i = results[ki]["preds"] == labels
            correct_j = results[kj]["preds"] == labels

            tables[(ki, kj)] = {
                "N11": int(np.sum(correct_i & correct_j)),
                "N00": int(np.sum(~correct_i & ~correct_j)),
                "N10": int(np.sum(correct_i & ~correct_j)),
                "N01": int(np.sum(~correct_i & correct_j)),
            }

    return tables


def compute_disagreement_metrics(results, kernel_names):
    """Compute multi-topology disagreement and oracle routing metrics."""
    labels = results[kernel_names[0]]["labels"]
    N = len(labels)
    M = len(kernel_names)

    # Per-sample correctness matrix: (M, N)
    correct_matrix = np.array(
        [results[k]["preds"] == labels for k in kernel_names]
    )

    # How many topologies get each sample right
    num_correct = correct_matrix.sum(axis=0)  # (N,)

    # Oracle: correct if ANY topology gets it right
    oracle_correct = correct_matrix.any(axis=0)
    oracle_acc = float(oracle_correct.mean())

    # Best single topology
    best_single_acc = max(results[k]["test_acc"] for k in kernel_names)
    best_single_kernel = max(kernel_names, key=lambda k: results[k]["test_acc"])

    # Complementarity: at least one right but not all
    complementary = oracle_correct & (num_correct < M)
    complementarity_score = float(complementary.sum() / N)

    # Routing potential: samples where topologies disagree on correctness
    routing_potential = float(
        ((num_correct > 0) & (num_correct < M)).sum() / N
    )

    # Correctness distribution
    num_correct_dist = {
        str(i): int((num_correct == i).sum()) for i in range(M + 1)
    }

    # Per-sample best topology (by probability of correct class)
    best_topology_per_sample = np.zeros(N, dtype=int)
    for s in range(N):
        true_label = labels[s]
        best_prob = -1.0
        for k_idx, k_name in enumerate(kernel_names):
            prob = results[k_name]["probs"][s, true_label]
            if prob > best_prob:
                best_prob = prob
                best_topology_per_sample[s] = k_idx

    # Oracle routing distribution: how often each topology is "best"
    oracle_routing_dist = {
        k: int((best_topology_per_sample == i).sum())
        for i, k in enumerate(kernel_names)
    }

    return {
        "oracle_acc": oracle_acc,
        "best_single_acc": float(best_single_acc),
        "best_single_kernel": best_single_kernel,
        "complementarity_score": complementarity_score,
        "routing_potential": routing_potential,
        "per_topology_acc": {k: results[k]["test_acc"] for k in kernel_names},
        "per_topology_auc": {k: results[k]["test_auc"] for k in kernel_names},
        "num_correct_distribution": num_correct_dist,
        "oracle_routing_distribution": oracle_routing_dist,
        "hard_cases": int((num_correct == 0).sum()),
        "easy_cases": int((num_correct == M).sum()),
        "total_samples": N,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_seed_report(metrics, pairwise_tables, kernel_names, seed):
    """Print formatted per-seed diagnostic report."""
    sep = "=" * 70
    N = metrics["total_samples"]
    M = len(kernel_names)

    print(f"\n{sep}")
    print(f"  TOPOLOGY DISAGREEMENT — Seed {seed}")
    print(sep)

    # Per-topology accuracy
    print(f"\n  Per-Topology Test Accuracy:")
    print(f"  {'Topology':<20} {'Accuracy':>10} {'AUC':>10}")
    print(f"  {'-' * 42}")
    for k in kernel_names:
        marker = " <-- best" if k == metrics["best_single_kernel"] else ""
        print(
            f"  {k:<20} {metrics['per_topology_acc'][k]:>10.4f} "
            f"{metrics['per_topology_auc'][k]:>10.4f}{marker}"
        )

    # Key metrics
    print(f"\n  Key Routing Metrics:")
    print(f"  {'-' * 55}")
    print(f"  Oracle accuracy (best per-sample)  : {metrics['oracle_acc']:.4f}")
    print(
        f"  Best single topology               : "
        f"{metrics['best_single_acc']:.4f} ({metrics['best_single_kernel']})"
    )
    uplift = metrics["oracle_acc"] - metrics["best_single_acc"]
    print(f"  Oracle uplift                      : {uplift:+.4f} ({uplift:+.1%})")
    print(
        f"  Complementarity score              : {metrics['complementarity_score']:.4f}"
    )
    print(
        f"  Routing potential (disagreement)    : {metrics['routing_potential']:.4f}"
    )

    # Correctness distribution
    print(f"\n  Correctness Distribution (N={N}):")
    for i in range(M + 1):
        count = metrics["num_correct_distribution"][str(i)]
        pct = 100 * count / N
        bar = "#" * int(pct / 2)
        label = " (hard)" if i == 0 else (" (easy)" if i == M else "")
        print(f"    {i}/{M} correct: {count:>4} ({pct:>5.1f}%) {bar}{label}")

    # Oracle routing distribution
    print(f"\n  Oracle Routing Distribution (which topology is best per sample):")
    for k in kernel_names:
        count = metrics["oracle_routing_distribution"][k]
        pct = 100 * count / N
        print(f"    {k:<20} {count:>4} ({pct:>5.1f}%)")

    # Pairwise tables
    print(f"\n  Pairwise Disagreement:")
    header = (
        f"  {'Pair':<30} {'Both OK':>8} {'Only i':>8} "
        f"{'Only j':>8} {'Both Bad':>8} {'Disagree':>10}"
    )
    print(header)
    print(f"  {'-' * 78}")
    for (ki, kj), table in sorted(pairwise_tables.items()):
        disagree = table["N10"] + table["N01"]
        disagree_pct = 100 * disagree / N
        print(
            f"  {ki} vs {kj:<18} "
            f"{table['N11']:>8} {table['N10']:>8} {table['N01']:>8} "
            f"{table['N00']:>8} {disagree_pct:>9.1f}%"
        )

    print(sep)


def print_aggregate_report(
    kernel_names, per_topology_accs, per_topology_aucs, oracle_accs,
    all_seed_metrics, all_seed_pairwise, seeds,
):
    """Print aggregated multi-seed report with interpretation."""
    sep = "=" * 70
    n_seeds = len(seeds)

    print(f"\n{sep}")
    print(f"  AGGREGATE RESULTS ({n_seeds} seeds)")
    print(sep)

    # Per-topology stats
    print(f"\n  Per-Topology Test Accuracy (mean +/- std):")
    print(f"  {'Topology':<20} {'Acc Mean':>10} {'Acc Std':>10} {'AUC Mean':>10}")
    print(f"  {'-' * 54}")
    for k in kernel_names:
        print(
            f"  {k:<20} {np.mean(per_topology_accs[k]):>10.4f} "
            f"{np.std(per_topology_accs[k]):>10.4f} "
            f"{np.mean(per_topology_aucs[k]):>10.4f}"
        )

    # Oracle stats
    best_single_mean = max(np.mean(per_topology_accs[k]) for k in kernel_names)
    oracle_mean = np.mean(oracle_accs)
    uplift = oracle_mean - best_single_mean

    print(f"\n  Oracle Accuracy     : {oracle_mean:.4f} +/- {np.std(oracle_accs):.4f}")
    print(f"  Best Single Topology: {best_single_mean:.4f}")
    print(f"  Oracle Uplift       : {uplift:+.4f} ({uplift:+.1%})")

    # Routing potential and complementarity
    routing_potentials = [m["routing_potential"] for m in all_seed_metrics]
    complementarities = [m["complementarity_score"] for m in all_seed_metrics]
    print(
        f"  Routing Potential   : {np.mean(routing_potentials):.4f} "
        f"+/- {np.std(routing_potentials):.4f}"
    )
    print(
        f"  Complementarity     : {np.mean(complementarities):.4f} "
        f"+/- {np.std(complementarities):.4f}"
    )

    # Average pairwise disagreement
    print(f"\n  Average Pairwise Disagreement:")
    print(f"  {'Pair':<30} {'Disagree':>12}")
    print(f"  {'-' * 44}")
    for ki_kj in sorted(all_seed_pairwise[0].keys()):
        disagrees = []
        for pw in all_seed_pairwise:
            t = pw[ki_kj]
            total = t["N11"] + t["N00"] + t["N10"] + t["N01"]
            disagrees.append(100 * (t["N10"] + t["N01"]) / total)
        ki, kj = ki_kj
        print(
            f"  {ki} vs {kj:<18} "
            f"{np.mean(disagrees):>5.1f}% +/- {np.std(disagrees):.1f}%"
        )

    # Average oracle routing distribution
    print(f"\n  Average Oracle Routing Distribution:")
    for k in kernel_names:
        counts = [m["oracle_routing_distribution"][k] for m in all_seed_metrics]
        totals = [m["total_samples"] for m in all_seed_metrics]
        pcts = [100 * c / t for c, t in zip(counts, totals)]
        print(f"    {k:<20} {np.mean(pcts):>5.1f}% +/- {np.std(pcts):.1f}%")

    # Interpretation
    print(f"\n{sep}")
    print("  INTERPRETATION")
    print(sep)

    if uplift > 0.05:
        print(
            f"  Oracle uplift of {uplift:.1%} is STRONG — "
            "routing has significant potential."
        )
    elif uplift > 0.02:
        print(
            f"  Oracle uplift of {uplift:.1%} is MODERATE — "
            "routing could help but ceiling is limited."
        )
    elif uplift > 0.005:
        print(
            f"  Oracle uplift of {uplift:.1%} is WEAK — "
            "topologies are mostly redundant."
        )
    else:
        print(
            f"  Oracle uplift of {uplift:.1%} is NEGLIGIBLE — "
            "routing adds no value here."
        )

    avg_routing = np.mean(routing_potentials)
    if avg_routing > 0.3:
        print(
            f"  Routing potential {avg_routing:.1%} — "
            "substantial disagreement across topologies."
        )
    elif avg_routing > 0.15:
        print(
            f"  Routing potential {avg_routing:.1%} — "
            "moderate disagreement."
        )
    else:
        print(
            f"  Routing potential {avg_routing:.1%} — "
            "topologies mostly agree on samples."
        )

    # Is routing even theoretically useful?
    if uplift < 0.01 and avg_routing < 0.15:
        print(
            "\n  CONCLUSION: Topologies provide largely redundant information. "
            "Per-sample routing is unlikely to help. Consider whether "
            "multi-topology features are better used via simple concatenation."
        )
    elif uplift > 0.02 and avg_routing > 0.2:
        print(
            "\n  CONCLUSION: There IS complementary signal across topologies. "
            "Proceed to Phase 2 (oracle routing experiment) to validate "
            "that a classifier can exploit this signal."
        )
    else:
        print(
            "\n  CONCLUSION: Marginal complementarity exists. "
            "Proceed to Phase 2 with tempered expectations."
        )

    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Diagnostic: Per-topology disagreement analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "breast_mnist", "original", "4_kernels_3x3.yml"),
        help="Path to base YAML config file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds for multi-seed evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results. Auto-generated if not provided.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    kernel_names = base_cfg["model"]["kernel_topology_names"]

    # Device resolution
    requested_device = base_cfg.get("model", {}).get("classical_device", "auto")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device

    # Output directory
    if args.output_dir is None:
        dataset_name = base_cfg.get("dataset", {}).get("name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            "outputs", f"topology_disagreement_{dataset_name}_{timestamp}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Phase 1: Topology Disagreement Analysis")
    print("=" * 70)
    print(f"  Config     : {args.config}")
    print(f"  Dataset    : {base_cfg.get('dataset', {}).get('name', 'unknown')}")
    print(f"  Topologies : {kernel_names}")
    print(f"  Seeds      : {args.seeds}")
    print(f"  Device     : {device}")
    print(f"  Output     : {args.output_dir}")
    print("=" * 70)

    # ── Per-seed loop ─────────────────────────────────────────────────────

    all_seed_metrics = []
    all_seed_pairwise = []
    per_topology_accs = {k: [] for k in kernel_names}
    per_topology_aucs = {k: [] for k in kernel_names}
    oracle_accs = []

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'─' * 70}")
        print(f"  Seed {seed} ({seed_idx + 1}/{len(args.seeds)})")
        print(f"{'─' * 70}")

        seed_results = {}
        for kernel_name in kernel_names:
            print(f"\n  Training classifier for '{kernel_name}'...")
            cfg = build_single_kernel_cfg(base_cfg, kernel_name)
            result = train_single_topology_classifier(cfg, kernel_name, seed, device)
            seed_results[kernel_name] = result
            print(
                f"    -> acc={result['test_acc']:.4f}, "
                f"auc={result['test_auc']:.4f}, "
                f"f1={result['test_f1']:.4f}"
            )

        # Compute metrics
        pairwise = compute_pairwise_tables(seed_results, kernel_names)
        metrics = compute_disagreement_metrics(seed_results, kernel_names)

        print_seed_report(metrics, pairwise, kernel_names, seed)

        all_seed_metrics.append(metrics)
        all_seed_pairwise.append(pairwise)

        for k in kernel_names:
            per_topology_accs[k].append(seed_results[k]["test_acc"])
            per_topology_aucs[k].append(seed_results[k]["test_auc"])
        oracle_accs.append(metrics["oracle_acc"])

    # ── Aggregate ─────────────────────────────────────────────────────────

    print_aggregate_report(
        kernel_names, per_topology_accs, per_topology_aucs, oracle_accs,
        all_seed_metrics, all_seed_pairwise, args.seeds,
    )

    # ── Save results ──────────────────────────────────────────────────────

    best_single_mean = max(np.mean(per_topology_accs[k]) for k in kernel_names)

    aggregate = {
        "config": args.config,
        "dataset": base_cfg.get("dataset", {}).get("name", "unknown"),
        "seeds": args.seeds,
        "kernel_names": kernel_names,
        "per_topology_acc": {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "values": v,
            }
            for k, v in per_topology_accs.items()
        },
        "per_topology_auc": {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "values": v,
            }
            for k, v in per_topology_aucs.items()
        },
        "oracle_acc": {
            "mean": float(np.mean(oracle_accs)),
            "std": float(np.std(oracle_accs)),
            "values": [float(x) for x in oracle_accs],
        },
        "oracle_uplift": float(np.mean(oracle_accs) - best_single_mean),
        "avg_routing_potential": float(
            np.mean([m["routing_potential"] for m in all_seed_metrics])
        ),
        "avg_complementarity": float(
            np.mean([m["complementarity_score"] for m in all_seed_metrics])
        ),
        "per_seed_metrics": all_seed_metrics,
        "per_seed_pairwise": [
            {f"{ki}_vs_{kj}": table for (ki, kj), table in pw.items()}
            for pw in all_seed_pairwise
        ],
    }

    out_path = os.path.join(args.output_dir, "disagreement_analysis.json")
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
