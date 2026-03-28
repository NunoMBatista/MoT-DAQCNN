"""
Training pipeline for the Oracle-Routed Mixture-of-Topologies.

Three-phase pipeline:
    Phase 1 — Train per-topology classifiers on cached quantum features and
              generate oracle routing labels (which topology is best per sample).
    Phase 2 — Train a lightweight image-level router (OracleRouter) to predict
              the oracle topology from raw pixels.
    Phase 3 — Build sparse quantum feature tensors using the router's predictions
              and train a final classifier on them.

The result dict returned by ``run_oracle_routed`` matches the interface expected
by ``robust_test_original_daqcnn.py`` and ``hyperparameter_search.py``.
"""

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.daqcnn_training import get_model_metadata
from src.models.oracle_router import OracleRouter
from src.utils.data import get_dataloaders
from src.utils.evaluate import evaluate
from src.utils.kernel_mapping import (
    build_kernel_to_channels_map,
    get_channels_per_kernel,
    get_kernel_names,
    get_num_kernels,
)
from src.utils.plotting import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_roc_curve,
)
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)
from src.utils.sparse_reconstruction import build_sparse_tensor_fast
from src.utils.training_utils import build_classification_head, resolve_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader(features, labels, batch_size, shuffle=True, num_workers=2):
    ds = TensorDataset(features, labels)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
    )


def _train_head(
    head, train_loader, val_loader, optim_cfg, device, grad_clip, label="",
    verbose=True,
):
    """Train a classification head with early stopping.  Returns best state dict."""
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0)),
    )

    epochs = optim_cfg.get("epochs", 100)
    patience = optim_cfg.get("patience", None)
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    use_scheduler = optim_cfg.get("use_scheduler", False)
    scheduler = None
    if use_scheduler:
        T_max = optim_cfg.get("scheduler_T_max", epochs)
        eta_min = float(optim_cfg.get("scheduler_eta_min", 0.0))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        # Train
        head.train()
        criterion = nn.CrossEntropyLoss()
        loss_sum, acc_sum, count = 0.0, 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            logits = head(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()
            bs = labels.size(0)
            loss_sum += loss.item() * bs
            acc_sum += (logits.argmax(1) == labels).float().sum().item()
            count += bs

        train_loss = loss_sum / count
        train_acc = acc_sum / count
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = evaluate(head, val_loader, device, split_name="Val")
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())
            no_improve = 0
        elif patience:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    [{label}] Early stop at epoch {epoch}")
                break

    if best_state:
        head.load_state_dict(best_state)

    return best_state, train_losses, val_losses, train_accs, val_accs


# ---------------------------------------------------------------------------
# Phase 1: Per-topology classifiers → oracle routing labels
# ---------------------------------------------------------------------------


def _run_phase1(
    splits, channel_groups, kernel_names, cfg, device, seed, set_seed_fn,
    batch_size, verbose,
):
    """Train per-topology classifiers and compute oracle labels for all splits.

    Returns:
        oracle_labels: dict[split -> np.ndarray of shape (N,)]
        per_topo_acc: dict[kernel_name -> float]  (test accuracy)
    """
    model_cfg = cfg.get("model", {})
    optim_cfg = cfg.get("optim", {})
    # Allow oracle_router config to override per-topo training params
    or_cfg = cfg.get("oracle_router", {})
    phase1_optim = copy.deepcopy(optim_cfg)
    if "phase1_lr" in or_cfg:
        phase1_optim["lr"] = or_cfg["phase1_lr"]
    if "phase1_epochs" in or_cfg:
        phase1_optim["epochs"] = or_cfg["phase1_epochs"]

    num_classes = model_cfg.get("num_classes", 2)
    n_qubits = len(channel_groups[0])
    grad_clip = optim_cfg.get("grad_clip", 0.0)

    per_topo_acc = {}
    # Collect probs: {kernel_name: {split: (N, C) ndarray}}
    all_probs = {k: {} for k in kernel_names}

    for k_idx, k_name in enumerate(kernel_names):
        if verbose:
            print(f"    Phase 1: training per-topology classifier [{k_name}]...")
        if set_seed_fn:
            set_seed_fn(seed)

        channels = channel_groups[k_idx]
        loaders = {}
        for split, shuffle in [("train", True), ("val", False), ("test", False)]:
            feat = splits[f"{split}_features"][:, channels]
            labs = splits[f"{split}_labels"]
            loaders[split] = _make_loader(feat, labs, batch_size, shuffle=shuffle)

        head = build_classification_head(
            in_channels=n_qubits,
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
        ).to(device)

        _train_head(
            head, loaders["train"], loaders["val"], phase1_optim,
            device, grad_clip, label=k_name, verbose=verbose,
        )

        # Test accuracy
        test_m = evaluate(
            head, loaders["test"], device,
            split_name="Test", compute_full_metrics=True,
        )
        per_topo_acc[k_name] = test_m["accuracy"]
        if verbose:
            print(f"      {k_name}: test_acc={test_m['accuracy']:.4f}")

        # Collect probs for all splits (for oracle label generation)
        head.eval()
        for split in ["train", "val", "test"]:
            m = evaluate(
                head, loaders[split], device,
                split_name=split, compute_full_metrics=True,
            )
            all_probs[k_name][split] = m["probs"]

    # Oracle labels: for each sample, pick topology with max P(true_class)
    oracle_labels = {}
    for split in ["train", "val", "test"]:
        labels_np = splits[f"{split}_labels"].numpy()
        N = len(labels_np)
        M = len(kernel_names)
        p_correct = np.zeros((M, N))
        for k_idx, k_name in enumerate(kernel_names):
            probs = all_probs[k_name][split]
            p_correct[k_idx] = probs[np.arange(N), labels_np]
        oracle_labels[split] = p_correct.argmax(axis=0)

    return oracle_labels, per_topo_acc


# ---------------------------------------------------------------------------
# Phase 2: Train image-level router on oracle labels
# ---------------------------------------------------------------------------


def _run_phase2(
    oracle_labels, cfg, device, seed, set_seed_fn, verbose,
):
    """Train OracleRouter on raw images → oracle topology labels.

    Returns:
        router: trained OracleRouter model
        router_agreement: float (accuracy on test set oracle labels)
        train_losses, val_losses: lists
    """
    if set_seed_fn:
        set_seed_fn(seed)

    or_cfg = cfg.get("oracle_router", {})
    model_cfg = cfg.get("model", {})

    # Load raw images
    raw_train_loader, raw_val_loader, raw_test_loader, _ = get_dataloaders(cfg)

    # Determine image properties from first batch
    sample_batch = next(iter(raw_train_loader))[0]
    in_channels = sample_batch.shape[1]

    kernel_names = model_cfg.get("kernel_topology_names", [])
    num_topologies = len(kernel_names)

    # Build router
    router = OracleRouter(
        in_channels=in_channels,
        num_topologies=num_topologies,
        hidden_dims=or_cfg.get("router_hidden_dims", [16, 32]),
        fc_dim=or_cfg.get("router_fc_dim", 64),
        dropout=or_cfg.get("router_dropout", 0.3),
    ).to(device)

    # Training config
    lr = float(or_cfg.get("router_lr", 1e-3))
    weight_decay = float(or_cfg.get("router_weight_decay", 1e-5))
    epochs = or_cfg.get("router_epochs", 30)
    patience = or_cfg.get("router_patience", 10)
    grad_clip = cfg.get("optim", {}).get("grad_clip", 1.0)

    optimizer = torch.optim.Adam(
        router.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Build dataloaders with oracle labels replacing the original class labels
    def _replace_labels(loader, oracle_labs):
        """Iterate original loader, yield (images, oracle_topology_label)."""
        all_imgs, all_labs = [], []
        for imgs, _ in loader:
            all_imgs.append(imgs)
        all_imgs = torch.cat(all_imgs, dim=0)
        oracle_tensor = torch.from_numpy(oracle_labs).long()
        ds = TensorDataset(all_imgs, oracle_tensor)
        return DataLoader(
            ds, batch_size=loader.batch_size,
            shuffle=isinstance(loader.sampler, torch.utils.data.sampler.RandomSampler),
            num_workers=0,  # data already in memory
        )

    router_train_loader = _replace_labels(raw_train_loader, oracle_labels["train"])
    router_val_loader = _replace_labels(raw_val_loader, oracle_labels["val"])
    router_test_loader = _replace_labels(raw_test_loader, oracle_labels["test"])

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        router.train()
        loss_sum, acc_sum, count = 0.0, 0.0, 0
        for imgs, labs in router_train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            logits = router(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(router.parameters(), grad_clip)
            optimizer.step()
            bs = labs.size(0)
            loss_sum += loss.item() * bs
            acc_sum += (logits.argmax(1) == labs).float().sum().item()
            count += bs

        train_loss = loss_sum / count
        train_acc = acc_sum / count
        train_losses.append(train_loss)

        # Validate
        router.eval()
        val_loss_sum, val_acc_sum, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, labs in router_val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                logits = router(imgs)
                loss = criterion(logits, labs)
                bs = labs.size(0)
                val_loss_sum += loss.item() * bs
                val_acc_sum += (logits.argmax(1) == labs).float().sum().item()
                val_count += bs
        val_loss = val_loss_sum / val_count
        val_acc = val_acc_sum / val_count
        val_losses.append(val_loss)

        if verbose and epoch % 5 == 0:
            print(
                f"    Phase 2 epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(router.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                if verbose:
                    print(f"    Phase 2: early stop at epoch {epoch}")
                break

    if best_state:
        router.load_state_dict(best_state)

    # Test agreement with oracle labels
    router.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in router_test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            preds = router.predict_topology(imgs)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    router_agreement = correct / total

    if verbose:
        print(f"    Phase 2: router test agreement = {router_agreement:.4f}")

    return router, router_agreement, train_losses, val_losses


# ---------------------------------------------------------------------------
# Phase 3: Sparse features → final classifier
# ---------------------------------------------------------------------------


def _build_sparse_split(features, routing_indices, channel_groups, chunk_size=2048):
    """Build sparse tensor from per-sample routing, processed in chunks."""
    N = features.shape[0]
    H, W = features.shape[2], features.shape[3]
    chunks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        batch_feat = features[start:end]
        B = batch_feat.shape[0]
        routing_map = torch.from_numpy(
            routing_indices[start:end]
        ).long().unsqueeze(1).unsqueeze(2).expand(B, H, W)
        chunks.append(build_sparse_tensor_fast(batch_feat, routing_map, channel_groups))
    return torch.cat(chunks, dim=0)


def _get_router_predictions(router, cfg, device):
    """Run the router on all raw images to get per-sample topology predictions."""
    raw_train, raw_val, raw_test, _ = get_dataloaders(cfg)

    preds = {}
    router.eval()
    with torch.no_grad():
        for split, loader in [("train", raw_train), ("val", raw_val), ("test", raw_test)]:
            all_preds = []
            for imgs, _ in loader:
                imgs = imgs.to(device)
                all_preds.append(router.predict_topology(imgs).cpu().numpy())
            preds[split] = np.concatenate(all_preds, axis=0)
    return preds


def _run_phase3(
    splits, channel_groups, router_preds, cfg, device, seed, set_seed_fn,
    batch_size, verbose,
):
    """Build sparse features from router predictions and train final classifier.

    Returns:
        test_metrics: dict from evaluate() with compute_full_metrics=True
        train_losses, val_losses, train_accs, val_accs: lists
    """
    if set_seed_fn:
        set_seed_fn(seed)

    model_cfg = cfg.get("model", {})
    optim_cfg = cfg.get("optim", {})
    or_cfg = cfg.get("oracle_router", {})

    # Allow overrides for Phase 3 training
    phase3_optim = copy.deepcopy(optim_cfg)
    if "final_lr" in or_cfg:
        phase3_optim["lr"] = or_cfg["final_lr"]
    if "final_epochs" in or_cfg:
        phase3_optim["epochs"] = or_cfg["final_epochs"]

    total_channels = splits["train_features"].shape[1]
    num_classes = model_cfg.get("num_classes", 2)
    grad_clip = optim_cfg.get("grad_clip", 0.0)

    # Build sparse features
    if verbose:
        print("    Phase 3: building sparse features...")
    sparse = {}
    for split in ["train", "val", "test"]:
        sparse[split] = _build_sparse_split(
            splits[f"{split}_features"], router_preds[split], channel_groups,
        )
        if verbose:
            print(f"      {split}: {sparse[split].shape}")

    # DataLoaders
    loaders = {}
    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        loaders[split] = _make_loader(
            sparse[split], splits[f"{split}_labels"], batch_size, shuffle=shuffle,
        )

    # Final classifier
    head = build_classification_head(
        in_channels=total_channels,
        num_classes=num_classes,
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
    ).to(device)

    _, train_losses, val_losses, train_accs, val_accs = _train_head(
        head, loaders["train"], loaders["val"], phase3_optim,
        device, grad_clip, label="final", verbose=verbose,
    )

    # Test
    test_metrics = evaluate(
        head, loaders["test"], device,
        split_name="Test", compute_full_metrics=True,
    )
    if verbose:
        print(f"    Phase 3: test_acc={test_metrics['accuracy']:.4f}")

    return test_metrics, train_losses, val_losses, train_accs, val_accs


# ---------------------------------------------------------------------------
# Top-level pipeline entry point
# ---------------------------------------------------------------------------


def run_oracle_routed(cfg, seed, output_dir, verbose=True, set_seed_fn=None):
    """Run the full oracle-routed MoT pipeline for a single seed.

    This is the main entry point called by ``robust_test_original_daqcnn.py``
    and ``hyperparameter_search.py``.

    Args:
        cfg: Full YAML config dict (must have ``model.architecture: "oracle-routed"``).
        seed: Random seed for this run.
        output_dir: Directory for checkpoints, plots, and results.
        verbose: If True, print progress.
        set_seed_fn: Callable ``set_seed_fn(seed)`` for reproducibility.

    Returns:
        dict with keys matching ``run_single_seed`` output for compatibility.
    """
    t_start = time.time()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Oracle-Routed MoT — seed {seed}")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    if set_seed_fn:
        set_seed_fn(seed)

    device = resolve_device(cfg, verbose=verbose)
    model_cfg = cfg.get("model", {})
    batch_size = cfg.get("dataset", {}).get("batch_size", 32)
    kernel_names = model_cfg.get("kernel_topology_names", [])

    # ------------------------------------------------------------------
    # 2. Load cached quantum features (mandatory)
    # ------------------------------------------------------------------
    cached_path = find_cached_quantum_dataset(cfg)
    if cached_path is None:
        raise RuntimeError(
            "Oracle-routed pipeline requires a pre-computed quantum dataset "
            "cache. Generate one with:\n"
            "    python experiments/create_quantum_dataset.py --config <config.yml>"
        )

    if verbose:
        print(f"  Cached dataset: {cached_path}")

    data = np.load(cached_path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))

    # Handle kernel subset extraction
    cached_kernels = metadata.get("kernel_topology_names", [])
    requested_kernels = model_cfg.get("kernel_topology_names", cached_kernels)
    extract_subset = set(requested_kernels) != set(cached_kernels)

    if extract_subset:
        # Use load_cached_quantum_dataset for subset extraction, then grab tensors
        tl, vl, tel, n_classes, metadata = load_cached_quantum_dataset(
            cached_path, batch_size, 0, requested_kernels=requested_kernels,
        )
        splits = {
            "train_features": torch.cat([b[0] for b in tl], dim=0),
            "train_labels": torch.cat([b[1] for b in tl], dim=0),
            "val_features": torch.cat([b[0] for b in vl], dim=0),
            "val_labels": torch.cat([b[1] for b in vl], dim=0),
            "test_features": torch.cat([b[0] for b in tel], dim=0),
            "test_labels": torch.cat([b[1] for b in tel], dim=0),
        }
    else:
        splits = {}
        for split in ["train", "val", "test"]:
            splits[f"{split}_features"] = torch.from_numpy(
                data[f"{split}_features"]
            ).float()
            splits[f"{split}_labels"] = torch.from_numpy(
                data[f"{split}_labels"]
            ).long()

    # Channel groups
    kernel_to_channels = build_kernel_to_channels_map(metadata["channel_kernel_map"])
    kernel_names_ordered = get_kernel_names(kernel_to_channels)
    channel_groups = [sorted(kernel_to_channels[k]) for k in kernel_names_ordered]
    total_channels = metadata["out_channels"]
    num_kernels = len(kernel_names_ordered)

    if verbose:
        print(f"  Topologies ({num_kernels}): {kernel_names_ordered}")
        print(f"  Channels: {total_channels} total, {len(channel_groups[0])} per topology")
        n_train = len(splits["train_labels"])
        n_test = len(splits["test_labels"])
        print(f"  Samples: {n_train} train, {n_test} test\n")

    # ------------------------------------------------------------------
    # Phase 1: Per-topology classifiers → oracle labels
    # ------------------------------------------------------------------
    if verbose:
        print("  Phase 1: Training per-topology classifiers...\n")

    oracle_labels, per_topo_acc = _run_phase1(
        splits, channel_groups, kernel_names_ordered, cfg, device, seed,
        set_seed_fn, batch_size, verbose,
    )

    if verbose:
        print(f"\n    Oracle label distribution (test):")
        for k_idx, k_name in enumerate(kernel_names_ordered):
            count = (oracle_labels["test"] == k_idx).sum()
            pct = 100 * count / len(oracle_labels["test"])
            print(f"      {k_name:<20} {count:>6} ({pct:>5.1f}%)")

    # ------------------------------------------------------------------
    # Phase 2: Train image-level router
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  Phase 2: Training image-level router...\n")

    router, router_agreement, router_train_losses, router_val_losses = _run_phase2(
        oracle_labels, cfg, device, seed, set_seed_fn, verbose,
    )

    # Save router checkpoint
    router_ckpt_path = os.path.join(output_dir, f"router_seed_{seed}.pt")
    torch.save({
        "model_state_dict": router.state_dict(),
        "config": cfg.get("oracle_router", {}),
        "seed": seed,
        "router_agreement": router_agreement,
        "kernel_names": kernel_names_ordered,
    }, router_ckpt_path)

    # ------------------------------------------------------------------
    # Phase 3: Sparse features → final classifier
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  Phase 3: Training final classifier on routed features...\n")

    # Get router predictions on raw images
    router_preds = _get_router_predictions(router, cfg, device)

    # Compute router agreement with oracle for each split
    if verbose:
        for split in ["train", "val", "test"]:
            agreement = (router_preds[split] == oracle_labels[split]).mean()
            print(f"    Router agreement ({split}): {agreement:.4f}")

    test_metrics, final_train_losses, final_val_losses, final_train_accs, final_val_accs = (
        _run_phase3(
            splits, channel_groups, router_preds, cfg, device, seed,
            set_seed_fn, batch_size, verbose,
        )
    )

    pipeline_time = time.time() - t_start

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # Final classifier loss curves
    loss_plot = os.path.join(output_dir, f"loss_curve_seed_{seed}.png")
    plot_loss_curves(final_train_losses, final_val_losses, loss_plot)

    # ROC
    num_classes = model_cfg.get("num_classes", 2)
    roc_plot = os.path.join(output_dir, f"roc_curve_seed_{seed}.png")
    plot_roc_curve(
        test_metrics["labels"], test_metrics["probs"],
        roc_plot, num_classes=num_classes,
    )

    # Confusion matrix
    cm_plot = os.path.join(output_dir, f"confusion_matrix_seed_{seed}.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], cm_plot, num_classes=num_classes,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Oracle-Routed MoT Results (seed={seed})")
        print(f"{'=' * 60}")
        print(f"  Router agreement (test): {router_agreement:.4f}")
        print(f"  Final test accuracy:     {test_metrics['accuracy']:.4f}")
        print(f"  Final test AUC:          {test_metrics['auc']:.4f}")
        print(f"  Final test F1:           {test_metrics['f1']:.4f}")
        print(f"  Pipeline time:           {pipeline_time:.1f}s")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Return standardized result dict
    # ------------------------------------------------------------------
    return {
        "seed": seed,
        "architecture": "oracle-routed",
        # Final classifier histories (used by multi-seed plots)
        "train_losses": final_train_losses,
        "val_losses": final_val_losses,
        "train_accs": final_train_accs,
        "val_accs": final_val_accs,
        # Test metrics
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_recall": test_metrics["recall"],
        "test_probs": test_metrics["probs"].tolist(),
        "test_labels": test_metrics["labels"].tolist(),
        "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "num_classes": num_classes,
        "final_train_loss": final_train_losses[-1] if final_train_losses else 0.0,
        "final_val_loss": final_val_losses[-1] if final_val_losses else 0.0,
        "final_train_acc": final_train_accs[-1] if final_train_accs else 0.0,
        "final_val_acc": final_val_accs[-1] if final_val_accs else 0.0,
        # Oracle-routed specific
        "per_topo_acc": per_topo_acc,
        "router_agreement": router_agreement,
        "router_train_losses": router_train_losses,
        "router_val_losses": router_val_losses,
        "pipeline_time_s": pipeline_time,
        "kernel_names": kernel_names_ordered,
    }
