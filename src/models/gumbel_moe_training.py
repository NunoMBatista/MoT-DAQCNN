"""
Training loop for the Gumbel-Softmax Mixture-of-Experts (GS-MoE) model.

Mirrors the structure of ``daqcnn_training.py`` but adds:
- Gumbel temperature annealing (exponential decay)
- Hard/soft routing switching
- Budget (sparsity) loss alongside the task loss
- Routing distribution logging per epoch
"""

import math
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.daqcnn_training import get_model_metadata
from src.models.gumbel_moe import GumbelMoE
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
    plot_routing_prob_histogram,
    plot_routing_ratio_over_epochs,
    plot_tau_entropy_curve,
)
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)
from src.utils.training_utils import resolve_device

# ---------------------------------------------------------------------------
# Eval wrapper — evaluate() expects model(x) -> single tensor
# ---------------------------------------------------------------------------


class _GumbelEvalWrapper(nn.Module):
    """Wraps GumbelMoE so ``evaluate()`` only sees class logits."""

    def __init__(self, model: GumbelMoE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_gumbel_moe_from_cfg(
    cfg: dict, cache_meta: dict, num_classes: int, device: str
) -> GumbelMoE:
    """Build a ``GumbelMoE`` instance from a YAML config and cache metadata.

    Args:
        cfg: Full YAML config dict.
        cache_meta: Metadata dict returned by ``load_cached_quantum_dataset``.
        num_classes: Number of output classes.
        device: Torch device string.

    Returns:
        GumbelMoE model moved to *device*.
    """
    model_cfg = cfg.get("model", {})
    gumbel_cfg = cfg.get("gumbel", {})

    # Derive kernel structure from cache metadata
    channel_kernel_map = cache_meta["channel_kernel_map"]
    kernel_to_channels = build_kernel_to_channels_map(channel_kernel_map)

    kernel_names = get_kernel_names(kernel_to_channels)
    num_kernels = get_num_kernels(kernel_to_channels)
    channels_per_kernel = get_channels_per_kernel(kernel_to_channels)
    total_channels = cache_meta["out_channels"]

    # Gumbel-specific config
    temperature = float(gumbel_cfg.get("tau_start", 1.0))
    hard = False  # always start soft; training loop switches later
    sparsity_weight = float(gumbel_cfg.get("sparsity_weight", 0.01))
    router_hidden_dim = gumbel_cfg.get("router_hidden_dim", None)
    # router_hidden_dim may come through as YAML null -> Python None; keep as-is

    model = GumbelMoE(
        num_classes=num_classes,
        total_channels=total_channels,
        num_kernels=num_kernels,
        channels_per_kernel=channels_per_kernel,
        kernel_names=kernel_names,
        temperature=temperature,
        hard=hard,
        sparsity_weight=sparsity_weight,
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
        router_hidden_dim=router_hidden_dim,
    )

    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Single-epoch training
# ---------------------------------------------------------------------------


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch_gumbel(
    model: GumbelMoE,
    loader,
    optimizer,
    device: str,
    grad_clip: float,
    epoch: int,
):
    """Train for one epoch with combined task + budget loss.

    Temperature and hard-flag updates are the caller's responsibility —
    this function does NOT modify ``model.router_block.temperature`` or
    ``model.router_block.hard``.

    Args:
        model: GumbelMoE instance (already on *device*).
        loader: Training DataLoader yielding (features, labels).
        optimizer: Torch optimizer.
        device: Torch device string.
        grad_clip: Max gradient norm (0 = no clipping).
        epoch: Current epoch (1-indexed), used only for progress display.

    Returns:
        Tuple of (avg_total_loss, avg_task_loss, avg_budget_loss, avg_acc).
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss_sum = 0.0
    task_loss_sum = 0.0
    budget_loss_sum = 0.0
    acc_sum = 0.0
    count = 0

    batch_pbar = tqdm(loader, desc=f"Epoch {epoch} training", leave=False)
    for batch_idx, (imgs, labels) in enumerate(batch_pbar):
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()

        logits, routing_probs = model(imgs)
        total_loss, task_loss_val, budget_loss_val = model.compute_loss(
            logits, routing_probs, labels, criterion
        )

        total_loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = labels.size(0)
        total_loss_sum += total_loss.item() * bs
        task_loss_sum += task_loss_val * bs
        budget_loss_sum += budget_loss_val * bs
        acc_sum += _accuracy(logits, labels) * bs
        count += bs

        batch_pbar.set_description(
            f"Epoch {epoch} | batch {batch_idx + 1}/{len(loader)} "
            f"| loss={total_loss.item():.4f}"
        )

    batch_pbar.close()

    avg_total = total_loss_sum / count
    avg_task = task_loss_sum / count
    avg_budget = budget_loss_sum / count
    avg_acc = acc_sum / count
    return avg_total, avg_task, avg_budget, avg_acc


# ---------------------------------------------------------------------------
# Top-level run function (called from robust_test_original_daqcnn.py)
# ---------------------------------------------------------------------------


def run_gumbel_moe(cfg, seed, output_dir, verbose=True, set_seed_fn=None):
    """Train and evaluate a GumbelMoE model for a single seed.

    This function is a drop-in replacement for ``run_single_seed`` — it returns
    a dict with the **exact same top-level keys** so the downstream aggregation,
    plotting, and JSON saving in ``robust_test_original_daqcnn.py`` works
    without modification.

    Args:
        cfg: Full YAML config dict.
        seed: Random seed for this run.
        output_dir: Directory to write checkpoints, plots, and results.
        verbose: If True, print progress and diagnostics.
        set_seed_fn: Callable ``set_seed_fn(seed)`` for reproducibility.

    Returns:
        dict with keys matching ``run_single_seed`` output.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running GumbelMoE experiment with seed={seed}")
        print(f"{'=' * 60}\n")

    # -----------------------------------------------------------------------
    # 1. Seed
    # -----------------------------------------------------------------------
    if set_seed_fn:
        set_seed_fn(seed)

    # -----------------------------------------------------------------------
    # 2. Device
    # -----------------------------------------------------------------------
    device = resolve_device(cfg, verbose=verbose)

    # -----------------------------------------------------------------------
    # 3. Cache lookup (mandatory for GS-MoE)
    # -----------------------------------------------------------------------
    cached_path = find_cached_quantum_dataset(cfg)
    if cached_path is None:
        raise RuntimeError(
            "GumbelMoE requires a pre-computed quantum dataset cache, but none "
            "was found matching the current config.\n"
            "Generate one first with:\n"
            "    python experiments/create_quantum_dataset.py --config <your_config.yml>\n"
            "Then re-run this experiment."
        )

    if verbose:
        print(f"Found cached quantum dataset: {cached_path}")
        print("Loading pre-computed features...\n")

    # -----------------------------------------------------------------------
    # 4. Load cached data
    # -----------------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    batch_size = cfg.get("dataset", {}).get("batch_size", 32)
    num_workers = cfg.get("dataset", {}).get("num_workers", 2)
    requested_kernels = model_cfg.get("kernel_topology_names", None)

    train_loader, val_loader, test_loader, n_classes, cache_meta = (
        load_cached_quantum_dataset(
            cached_path,
            batch_size,
            num_workers,
            requested_kernels=requested_kernels,
        )
    )

    if verbose:
        print(
            f"Dataset loaded: {len(train_loader.dataset)} train, "
            f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples"
        )
        # Cache info diagnostics
        kernel_to_channels = build_kernel_to_channels_map(
            cache_meta["channel_kernel_map"]
        )
        knames = get_kernel_names(kernel_to_channels)
        print(f"  Kernels ({len(knames)}): {knames}")
        print(f"  Channels per kernel: {get_channels_per_kernel(kernel_to_channels)}")
        print(f"  Total feature channels: {cache_meta['out_channels']}\n")

    # -----------------------------------------------------------------------
    # 5. Build model
    # -----------------------------------------------------------------------
    num_classes = model_cfg.get("num_classes", n_classes)
    model = build_gumbel_moe_from_cfg(cfg, cache_meta, num_classes, device)

    if verbose:
        print(f"GumbelMoE built and moved to {device}")
        print(f"  Router hidden dim: {model.router.net[2].normalized_shape[0]}")
        print(f"  Sparsity weight: {model.sparsity_weight}")
        print()

    # Eval wrapper (used for all evaluate() calls)
    eval_wrapper = _GumbelEvalWrapper(model)

    # -----------------------------------------------------------------------
    # 6. Optimizer
    # -----------------------------------------------------------------------
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    epochs = optim_cfg.get("epochs", 1)
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = optim_cfg.get("patience", None)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -----------------------------------------------------------------------
    # 7. Scheduler (optional)
    # -----------------------------------------------------------------------
    use_scheduler = optim_cfg.get("use_scheduler", False)
    scheduler = None
    if use_scheduler:
        T_max = optim_cfg.get("scheduler_T_max", epochs)
        eta_min = float(optim_cfg.get("scheduler_eta_min", 0.0))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # -----------------------------------------------------------------------
    # 8. Temperature annealing & hard-switch config
    # -----------------------------------------------------------------------
    gumbel_cfg = cfg.get("gumbel", {})
    tau_start = float(gumbel_cfg.get("tau_start", 1.0))
    tau_end = float(gumbel_cfg.get("tau_end", 0.1))
    tau_anneal_epochs = int(gumbel_cfg.get("tau_anneal_epochs", 80))
    hard_after_epoch = int(gumbel_cfg.get("hard_after_epoch", 80))

    # -----------------------------------------------------------------------
    # 9. Training loop
    # -----------------------------------------------------------------------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Also track component losses for loss curve plots
    task_losses_history = []
    budget_losses_history = []

    # Gumbel-specific diagnostics
    routing_history = []  # per-epoch routing ratio dicts (kernel_name -> fraction)
    tau_history = []  # per-epoch Gumbel temperature
    entropy_history = []  # per-epoch normalized router entropy

    # Output subdirectory for per-epoch routing probability histograms (seed-specific)
    routing_hist_dir = os.path.join(output_dir, f"routing_prob_histograms_seed_{seed}")
    os.makedirs(routing_hist_dir, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    hard_switch_logged = False

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Seed {seed}", leave=True)
    for epoch in epoch_pbar:
        t0 = time.time()

        # --- Temperature annealing (exponential decay) ---
        progress = min(epoch / max(tau_anneal_epochs, 1), 1.0)
        tau = tau_start * (tau_end / tau_start) ** progress
        tau = max(min(tau, tau_start), tau_end)  # clamp for float precision
        model.router_block.temperature = tau

        # --- Hard switching ---
        if epoch >= hard_after_epoch and not model.router_block.hard:
            model.router_block.hard = True
            if not hard_switch_logged:
                if verbose:
                    tqdm.write(
                        f"\n>>> Switching to hard routing (straight-through) "
                        f"at epoch {epoch} <<<"
                    )
                hard_switch_logged = True

        # --- Train one epoch ---
        total_loss, task_loss, budget_loss, train_acc = train_one_epoch_gumbel(
            model, train_loader, optimizer, device, grad_clip, epoch
        )

        # --- Validate ---
        val_loss, val_acc = evaluate(eval_wrapper, val_loader, device, split_name="Val")

        train_losses.append(total_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        task_losses_history.append(task_loss)
        budget_losses_history.append(budget_loss)

        dt = time.time() - t0

        # --- Record tau for this epoch ---
        tau_history.append(tau)

        # --- Routing distribution log ---
        routing_stats = model.get_routing_stats()
        routing_str = ""
        if routing_stats is not None:
            routing_history.append(routing_stats["routing_ratio"])
            parts = [
                f"{name}: {ratio * 100:.1f}%"
                for name, ratio in routing_stats["routing_ratio"].items()
            ]
            routing_str = " | routing=[" + ", ".join(parts) + "]"

        # --- Router entropy ---
        norm_entropy = 0.0
        entropy_str = ""
        if model.last_routing_probs is not None:
            probs = model.last_routing_probs  # (B*H*W, K)
            eps = 1e-8
            h = -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()
            M = model.num_kernels
            max_entropy = math.log(M) if M > 1 else 1.0
            norm_entropy = h.item() / max_entropy
            entropy_str = f" | H={norm_entropy:.3f}"

            # --- Per-epoch routing probability histogram ---
            prob_dict = {}
            for k, name in enumerate(model.kernel_names):
                prob_dict[name] = probs[:, k]
            hist_path = os.path.join(
                routing_hist_dir, f"epoch_{epoch:03d}_all_kernels.png"
            )
            plot_routing_prob_histogram(prob_dict, epoch, hist_path)

        entropy_history.append(norm_entropy)

        hard_str = " [HARD]" if model.router_block.hard else ""

        epoch_pbar.set_description(
            f"Seed {seed} | Ep {epoch}/{epochs} | "
            f"train={total_loss:.4f} val={val_loss:.4f} | "
            f"tau={tau:.3f}{hard_str}"
        )

        if verbose:
            tqdm.write(
                f"Epoch {epoch}/{epochs} | tau={tau:.3f}{hard_str} | "
                f"task={task_loss:.4f} budget={budget_loss:.4f} total={total_loss:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"{dt:.1f}s{routing_str}{entropy_str}"
            )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            if patience is not None:
                epochs_without_improvement = 0
        elif patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(patience={patience})"
                    )
                break

        # --- Scheduler step ---
        if scheduler is not None:
            scheduler.step()

    epoch_pbar.close()

    # -----------------------------------------------------------------------
    # 10. Model metadata
    # -----------------------------------------------------------------------
    model_metadata = get_model_metadata(model)

    if verbose:
        print(
            f"\nModel parameters: {model_metadata['total_params']:,} total, "
            f"{model_metadata['trainable_params']:,} trainable"
        )

    # -----------------------------------------------------------------------
    # 11. Load best model for test evaluation
    # -----------------------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -----------------------------------------------------------------------
    # 12. Save checkpoints
    # -----------------------------------------------------------------------
    final_model_path = os.path.join(output_dir, f"final_model_seed_{seed}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "seed": seed,
            "num_classes": num_classes,
        },
        final_model_path,
    )
    if verbose:
        print(f"Final model saved to: {final_model_path}")

    if best_model_state is not None:
        best_model_path = os.path.join(output_dir, f"best_model_seed_{seed}.pt")
        torch.save(
            {
                "model_state_dict": best_model_state,
                "config": cfg,
                "seed": seed,
                "num_classes": num_classes,
                "best_val_loss": best_val_loss,
            },
            best_model_path,
        )
        if verbose:
            print(f"Best model saved to: {best_model_path}")

    # -----------------------------------------------------------------------
    # 13. Test evaluation with full metrics
    # -----------------------------------------------------------------------
    if verbose:
        print("\nEvaluating on test set...")

    test_metrics = evaluate(
        eval_wrapper, test_loader, device, split_name="Test", compute_full_metrics=True
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Test Results (seed={seed})")
        print(f"{'=' * 60}")
        print(f"Test Loss:     {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUC:      {test_metrics['auc']:.4f}")
        print(f"Test F1:       {test_metrics['f1']:.4f}")
        print(f"Test Recall:   {test_metrics['recall']:.4f}")
        print(f"{'=' * 60}\n")

    # -----------------------------------------------------------------------
    # 14. Save plots
    # -----------------------------------------------------------------------
    # Loss curves (with task and budget component lines)
    seed_plot_path = os.path.join(output_dir, f"loss_curve_seed_{seed}.png")
    plot_loss_curves(
        train_losses,
        val_losses,
        seed_plot_path,
        ce_losses=task_losses_history,
        ent_losses=budget_losses_history,
        ent_label="Budget Loss",
    )
    if verbose:
        print(f"Loss curve saved to: {seed_plot_path}")

    # Routing ratio over epochs
    if routing_history:
        routing_path = os.path.join(output_dir, f"gumbel_routing_ratio_seed_{seed}.png")
        plot_routing_ratio_over_epochs(
            routing_history, model.kernel_names, routing_path
        )
        if verbose:
            print(f"Routing ratio plot saved to: {routing_path}")

    # Tau vs entropy curve
    if tau_history and entropy_history:
        tau_ent_path = os.path.join(output_dir, f"tau_entropy_curve_seed_{seed}.png")
        plot_tau_entropy_curve(tau_history, entropy_history, tau_ent_path)
        if verbose:
            print(f"Tau-entropy curve saved to: {tau_ent_path}")

    # ROC curve
    roc_plot_path = os.path.join(output_dir, f"roc_curve_seed_{seed}.png")
    plot_roc_curve(
        test_metrics["labels"],
        test_metrics["probs"],
        roc_plot_path,
        num_classes=num_classes,
    )
    if verbose:
        print(f"ROC curve saved to: {roc_plot_path}")

    # Confusion matrix
    cm_plot_path = os.path.join(output_dir, f"confusion_matrix_seed_{seed}.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        cm_plot_path,
        num_classes=num_classes,
    )
    if verbose:
        print(f"Confusion matrix saved to: {cm_plot_path}")

    # -----------------------------------------------------------------------
    # 15. Return result dict (same keys as run_single_seed)
    # -----------------------------------------------------------------------
    return {
        "seed": seed,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_recall": test_metrics["recall"],
        "test_probs": test_metrics["probs"].tolist(),
        "test_labels": test_metrics["labels"].tolist(),
        "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "num_classes": num_classes,
        "final_train_loss": train_losses[-1] if train_losses else 0.0,
        "final_val_loss": val_losses[-1] if val_losses else 0.0,
        "final_train_acc": train_accs[-1] if train_accs else 0.0,
        "final_val_acc": val_accs[-1] if val_accs else 0.0,
        "model_metadata": model_metadata,
        # Gumbel-specific diagnostics
        "routing_history": routing_history,
        "tau_history": tau_history,
        "entropy_history": entropy_history,
        "task_losses_history": task_losses_history,
        "budget_losses_history": budget_losses_history,
        "kernel_names": model.kernel_names,
    }
