"""
Phase 2: Oracle Routing Experiment

Tests whether a classifier can exploit oracle-routed sparse features.
Compares three conditions:
  A) All-topologies  — full feature tensor, no routing (DAQCNN baseline)
  B) Best-single     — only the best topology's channels (cheapest option)
  C) Oracle-sparse   — per-sample perfect routing (routing ceiling)

The oracle picks, for each sample, the topology whose per-topology classifier
assigns the highest P(true_class). This is a per-SAMPLE oracle (same topology
for all spatial positions within one image).

Usage:
    python experiments/oracle_routing_experiment.py \
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
from torch.utils.data import DataLoader, TensorDataset
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.daqcnn_training import train_one_epoch  # noqa: E402
from src.utils.evaluate import evaluate  # noqa: E402
from src.utils.kernel_mapping import build_kernel_to_channels_map  # noqa: E402
from src.utils.quantum_dataset_cache import find_cached_quantum_dataset  # noqa: E402
from src.utils.sparse_reconstruction import build_sparse_tensor_fast  # noqa: E402
from src.utils.training_utils import build_classification_head  # noqa: E402


# ── Utilities ────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(features, labels, batch_size, shuffle=True, num_workers=2):
    ds = TensorDataset(features, labels)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def load_all_features(cfg):
    """Load full cached quantum features (all topologies) as raw tensors."""
    cached_path = find_cached_quantum_dataset(cfg)
    if cached_path is None:
        raise RuntimeError(
            "No cached quantum dataset found. "
            "Run experiments/create_quantum_dataset.py first."
        )
    print(f"  Loading cached features from: {cached_path}")

    data = np.load(cached_path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))

    splits = {}
    for split in ["train", "val", "test"]:
        splits[f"{split}_features"] = torch.from_numpy(
            data[f"{split}_features"]
        ).float()
        splits[f"{split}_labels"] = torch.from_numpy(
            data[f"{split}_labels"]
        ).long()

    return splits, metadata


def get_channel_groups(metadata):
    """Get ordered channel indices for each topology."""
    kernel_to_channels = build_kernel_to_channels_map(
        metadata["channel_kernel_map"]
    )
    kernel_names = metadata["kernel_topology_names"]
    return [sorted(kernel_to_channels[k]) for k in kernel_names]


# ── Training ─────────────────────────────────────────────────────────────────


def train_and_evaluate(
    model, train_loader, val_loader, test_loader, cfg, device, label=""
):
    """Train with early stopping, return test metrics dict."""
    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0)),
    )

    epochs = optim_cfg.get("epochs", 100)
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = optim_cfg.get("patience", None)
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model, train_loader, optimizer, device, grad_clip, epoch_pbar=None
        )
        val_loss, _ = evaluate(model, val_loader, device, split_name="Val")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        elif patience:
            no_improve += 1
            if no_improve >= patience:
                if label:
                    print(f"    [{label}] Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    metrics = evaluate(
        model, test_loader, device, split_name="Test", compute_full_metrics=True
    )
    return metrics


def make_head(in_channels, cfg, device):
    """Build a classification head from config and move to device."""
    model_cfg = cfg.get("model", {})
    head = build_classification_head(
        in_channels=in_channels,
        num_classes=model_cfg.get("num_classes", 2),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
    )
    return head.to(device)


# ── Oracle routing ───────────────────────────────────────────────────────────


def compute_oracle_routing(splits, channel_groups, kernel_names, cfg, device, seed):
    """
    Train per-topology classifiers and compute oracle routing for all splits.

    Returns:
        oracle_routing: dict[split_name -> np.ndarray of shape (N,)]
            Per-sample topology index that gives highest P(true_class).
        per_topo_test_acc: dict[kernel_name -> float]
            Test accuracy of each per-topology classifier.
    """
    model_cfg = cfg.get("model", {})
    batch_size = cfg.get("dataset", {}).get("batch_size", 32)
    n_qubits = len(channel_groups[0])

    per_topo_test_acc = {}
    # Store probs per topology per split: {kernel: {split: (N, C) ndarray}}
    all_probs = {k: {} for k in kernel_names}

    for k_idx, k_name in enumerate(kernel_names):
        print(f"    Training per-topology classifier: {k_name}...")
        set_seed(seed)

        channels = channel_groups[k_idx]

        # Extract this topology's channels
        loaders = {}
        for split, shuffle in [("train", True), ("val", False), ("test", False)]:
            feat = splits[f"{split}_features"][:, channels]
            labs = splits[f"{split}_labels"]
            loaders[split] = make_loader(feat, labs, batch_size, shuffle=shuffle)

        head = make_head(n_qubits, cfg, device)
        test_metrics = train_and_evaluate(
            head, loaders["train"], loaders["val"], loaders["test"],
            cfg, device, label=k_name,
        )
        per_topo_test_acc[k_name] = test_metrics["accuracy"]
        print(f"      {k_name}: test_acc={test_metrics['accuracy']:.4f}")

        # Collect probs on all splits
        head.eval()
        for split in ["train", "val", "test"]:
            metrics = evaluate(
                head, loaders[split], device,
                split_name=split, compute_full_metrics=True,
            )
            all_probs[k_name][split] = metrics["probs"]  # (N, C)

    # Compute oracle routing: per sample, pick topology with max P(true_class)
    oracle_routing = {}
    for split in ["train", "val", "test"]:
        labels = splits[f"{split}_labels"].numpy()
        N = len(labels)
        M = len(kernel_names)

        p_correct = np.zeros((M, N))
        for k_idx, k_name in enumerate(kernel_names):
            probs = all_probs[k_name][split]  # (N, C)
            p_correct[k_idx] = probs[np.arange(N), labels]

        oracle_routing[split] = p_correct.argmax(axis=0)  # (N,)

    return oracle_routing, per_topo_test_acc


def build_oracle_sparse(features, oracle_indices, channel_groups, chunk_size=2048):
    """Build sparse features using per-sample oracle routing, processed in chunks."""
    N = features.shape[0]
    H, W = features.shape[2], features.shape[3]
    chunks = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        batch_feat = features[start:end]
        batch_idx = oracle_indices[start:end]
        B = batch_feat.shape[0]

        # Per-sample routing → broadcast to all spatial positions
        routing_map = torch.from_numpy(batch_idx).long()
        routing_map = routing_map.unsqueeze(1).unsqueeze(2).expand(B, H, W)

        sparse_batch = build_sparse_tensor_fast(
            batch_feat, routing_map, channel_groups
        )
        chunks.append(sparse_batch)

    return torch.cat(chunks, dim=0)


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_seed_report(seed, results, per_topo_acc, kernel_names):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  ORACLE ROUTING EXPERIMENT — Seed {seed}")
    print(sep)

    print(f"\n  Per-Topology Classifier Accuracy:")
    for k in kernel_names:
        print(f"    {k:<20} {per_topo_acc[k]:.4f}")

    print(f"\n  {'Condition':<25} {'Acc':>10} {'AUC':>10} {'F1':>10}")
    print(f"  {'-' * 57}")
    for cond_name, m in results.items():
        print(
            f"  {cond_name:<25} {m['accuracy']:>10.4f} "
            f"{m['auc']:>10.4f} {m['f1']:>10.4f}"
        )

    # Highlight key comparisons
    all_acc = results["all_topologies"]["accuracy"]
    sparse_acc = results["oracle_sparse"]["accuracy"]
    single_acc = results["best_single"]["accuracy"]
    print(f"\n  Oracle-sparse vs all-topologies : {sparse_acc - all_acc:+.4f}")
    print(f"  Oracle-sparse vs best-single    : {sparse_acc - single_acc:+.4f}")
    print(sep)


def print_aggregate_report(all_results, seeds, kernel_names):
    sep = "=" * 70
    n = len(seeds)

    print(f"\n{sep}")
    print(f"  AGGREGATE RESULTS ({n} seeds)")
    print(sep)

    conditions = ["all_topologies", "best_single", "oracle_sparse"]
    print(f"\n  {'Condition':<25} {'Acc Mean':>10} {'Acc Std':>10} {'AUC Mean':>10}")
    print(f"  {'-' * 57}")
    for cond in conditions:
        accs = [r[cond]["accuracy"] for r in all_results]
        aucs = [r[cond]["auc"] for r in all_results]
        print(
            f"  {cond:<25} {np.mean(accs):>10.4f} "
            f"{np.std(accs):>10.4f} {np.mean(aucs):>10.4f}"
        )

    all_accs = [r["all_topologies"]["accuracy"] for r in all_results]
    sparse_accs = [r["oracle_sparse"]["accuracy"] for r in all_results]
    single_accs = [r["best_single"]["accuracy"] for r in all_results]

    diff_vs_all = np.mean(sparse_accs) - np.mean(all_accs)
    diff_vs_single = np.mean(sparse_accs) - np.mean(single_accs)

    print(f"\n  Oracle-sparse vs all-topologies : {diff_vs_all:+.4f} ({diff_vs_all:+.1%})")
    print(f"  Oracle-sparse vs best-single    : {diff_vs_single:+.4f} ({diff_vs_single:+.1%})")

    # Interpretation
    print(f"\n{sep}")
    print("  INTERPRETATION")
    print(sep)

    if diff_vs_all > 0.01:
        print(
            f"  Oracle-sparse BEATS all-topologies by {diff_vs_all:.1%}."
        )
        print(
            "  Routing is genuinely better than concatenation — strong signal "
            "for fixing the router."
        )
    elif diff_vs_all > -0.01:
        print(
            f"  Oracle-sparse MATCHES all-topologies (delta {diff_vs_all:+.1%})."
        )
        print(
            "  Routing can match full accuracy at 1/M quantum compute — "
            "efficiency win if the router is good enough."
        )
    else:
        print(
            f"  Oracle-sparse LOSES to all-topologies by {abs(diff_vs_all):.1%}."
        )
        print(
            "  The classifier struggles with sparse inputs. This suggests "
            "an architectural issue with the sparse reconstruction approach."
        )

    if diff_vs_single > 0.02:
        print(
            f"\n  Oracle-sparse beats best-single by {diff_vs_single:.1%} — "
            "multi-topology routing is better than always using one topology."
        )
    elif diff_vs_single > 0:
        print(
            f"\n  Oracle-sparse marginally beats best-single ({diff_vs_single:+.1%}) — "
            "the routing benefit is slim."
        )
    else:
        print(
            f"\n  Oracle-sparse doesn't beat best-single ({diff_vs_single:+.1%}) — "
            "routing adds no value over the cheapest option."
        )

    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Oracle routing experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (should match the one used for disagreement analysis).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    kernel_names = base_cfg["model"]["kernel_topology_names"]

    # Device
    requested = base_cfg.get("model", {}).get("classical_device", "auto")
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    else:
        device = requested

    # Output
    if args.output_dir is None:
        dataset_name = base_cfg.get("dataset", {}).get("name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            "outputs", f"oracle_routing_{dataset_name}_{timestamp}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all features once
    splits, metadata = load_all_features(base_cfg)
    channel_groups = get_channel_groups(metadata)
    total_channels = len(metadata["channel_kernel_map"])
    n_qubits = len(channel_groups[0])
    batch_size = base_cfg.get("dataset", {}).get("batch_size", 32)

    print("=" * 70)
    print("  Phase 2: Oracle Routing Experiment")
    print("=" * 70)
    print(f"  Config       : {args.config}")
    print(f"  Dataset      : {metadata.get('dataset_name', 'unknown')}")
    print(f"  Topologies   : {kernel_names} (M={len(kernel_names)})")
    print(f"  Channels     : {total_channels} total, {n_qubits} per topology")
    print(f"  Seeds        : {args.seeds}")
    print(f"  Device       : {device}")
    print(f"  Output       : {args.output_dir}")
    print("=" * 70)

    all_results = []

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'─' * 70}")
        print(f"  Seed {seed} ({seed_idx + 1}/{len(args.seeds)})")
        print(f"{'─' * 70}")

        # ── Step 1: Oracle routing from per-topology classifiers ──────

        print("\n  Step 1: Computing oracle routing...")
        oracle_routing, per_topo_acc = compute_oracle_routing(
            splits, channel_groups, kernel_names, base_cfg, device, seed
        )

        # Oracle routing stats
        for split in ["train", "val", "test"]:
            routing = oracle_routing[split]
            print(f"    Oracle routing ({split}):")
            for k_idx, k_name in enumerate(kernel_names):
                count = (routing == k_idx).sum()
                pct = 100 * count / len(routing)
                print(f"      {k_name:<20} {count:>6} ({pct:>5.1f}%)")

        # Determine best single topology for this seed
        best_k_name = max(kernel_names, key=lambda k: per_topo_acc[k])
        best_k_idx = kernel_names.index(best_k_name)
        best_channels = channel_groups[best_k_idx]
        print(f"\n    Best single topology: {best_k_name} ({per_topo_acc[best_k_name]:.4f})")

        # ── Step 2: Build sparse features ─────────────────────────────

        print("\n  Step 2: Building oracle-sparse features...")
        sparse_features = {}
        for split in ["train", "val", "test"]:
            sparse_features[split] = build_oracle_sparse(
                splits[f"{split}_features"],
                oracle_routing[split],
                channel_groups,
            )
            print(f"    {split}: {sparse_features[split].shape}")

        # ── Step 3: Train and evaluate all three conditions ───────────

        seed_results = {}

        # Condition A: All-topologies (full features)
        print("\n  Step 3a: Training all-topologies classifier...")
        set_seed(seed)
        head_all = make_head(total_channels, base_cfg, device)
        loaders_all = {
            s: make_loader(
                splits[f"{s}_features"], splits[f"{s}_labels"],
                batch_size, shuffle=(s == "train"),
            )
            for s in ["train", "val", "test"]
        }
        metrics_all = train_and_evaluate(
            head_all, loaders_all["train"], loaders_all["val"],
            loaders_all["test"], base_cfg, device, label="all-topologies",
        )
        seed_results["all_topologies"] = {
            "accuracy": metrics_all["accuracy"],
            "auc": metrics_all["auc"],
            "f1": metrics_all["f1"],
        }
        print(f"    all-topologies: acc={metrics_all['accuracy']:.4f}")

        # Condition B: Best single topology
        print(f"\n  Step 3b: Training best-single classifier ({best_k_name})...")
        set_seed(seed)
        head_single = make_head(n_qubits, base_cfg, device)
        loaders_single = {
            s: make_loader(
                splits[f"{s}_features"][:, best_channels],
                splits[f"{s}_labels"],
                batch_size, shuffle=(s == "train"),
            )
            for s in ["train", "val", "test"]
        }
        metrics_single = train_and_evaluate(
            head_single, loaders_single["train"], loaders_single["val"],
            loaders_single["test"], base_cfg, device, label="best-single",
        )
        seed_results["best_single"] = {
            "accuracy": metrics_single["accuracy"],
            "auc": metrics_single["auc"],
            "f1": metrics_single["f1"],
            "kernel": best_k_name,
        }
        print(f"    best-single ({best_k_name}): acc={metrics_single['accuracy']:.4f}")

        # Condition C: Oracle-sparse
        print("\n  Step 3c: Training oracle-sparse classifier...")
        set_seed(seed)
        head_sparse = make_head(total_channels, base_cfg, device)
        loaders_sparse = {
            s: make_loader(
                sparse_features[s], splits[f"{s}_labels"],
                batch_size, shuffle=(s == "train"),
            )
            for s in ["train", "val", "test"]
        }
        metrics_sparse = train_and_evaluate(
            head_sparse, loaders_sparse["train"], loaders_sparse["val"],
            loaders_sparse["test"], base_cfg, device, label="oracle-sparse",
        )
        seed_results["oracle_sparse"] = {
            "accuracy": metrics_sparse["accuracy"],
            "auc": metrics_sparse["auc"],
            "f1": metrics_sparse["f1"],
        }
        print(f"    oracle-sparse: acc={metrics_sparse['accuracy']:.4f}")

        print_seed_report(seed, seed_results, per_topo_acc, kernel_names)
        all_results.append(seed_results)

    # ── Aggregate ─────────────────────────────────────────────────────

    print_aggregate_report(all_results, args.seeds, kernel_names)

    # ── Save ──────────────────────────────────────────────────────────

    aggregate = {
        "config": args.config,
        "dataset": metadata.get("dataset_name", "unknown"),
        "seeds": args.seeds,
        "kernel_names": kernel_names,
        "conditions": {},
    }
    for cond in ["all_topologies", "best_single", "oracle_sparse"]:
        accs = [r[cond]["accuracy"] for r in all_results]
        aucs = [r[cond]["auc"] for r in all_results]
        f1s = [r[cond]["f1"] for r in all_results]
        aggregate["conditions"][cond] = {
            "acc": {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "values": accs},
            "auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "values": aucs},
            "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "values": f1s},
        }
    # Add best-single kernel info per seed
    aggregate["best_single_kernels"] = [
        r["best_single"]["kernel"] for r in all_results
    ]
    aggregate["per_seed_results"] = all_results

    out_path = os.path.join(args.output_dir, "oracle_routing_results.json")
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
