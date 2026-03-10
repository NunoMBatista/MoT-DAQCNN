import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

# ---------------------------------------------------------------------------
# TS-MoE specific plotting
# ---------------------------------------------------------------------------


def plot_routing_confusion_matrix(cm, kernel_names, save_path):
    """Plot the M x M routing confusion matrix (Teacher labels vs Student predictions).

    Used during student distillation to verify that the Student faithfully
    reproduces the Teacher's routing decisions. Off-diagonal mass indicates
    disagreement; a single dominant column indicates lazy bias (Student
    always predicts one kernel).

    Args:
        cm: Numpy array of shape (M, M). Rows are Teacher labels, columns
            are Student predictions.
        kernel_names: List of kernel topology names (length M), used as
            tick labels on both axes.
        save_path: Where to save the PNG.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=kernel_names,
        yticklabels=kernel_names,
    )
    plt.xlabel("Student Prediction", fontsize=12)
    plt.ylabel("Teacher Label", fontsize=12)
    plt.title("Routing Confusion Matrix (Teacher vs Student)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_alpha_histogram(alpha_values, kernel_name, epoch, save_path, num_bins=20):
    """Plot distribution of alpha routing weights for a single kernel.

    Used during teacher training to verify that the SE block learns decisive
    routing. A successful teacher will show a bimodal distribution with peaks
    at 0.0 and 1.0 by the final epoch.

    Args:
        alpha_values: 1-D numpy array or tensor of alpha values for one kernel
            across all patches in the dataset. Values in [0, 1].
        kernel_name: Name of the kernel topology (e.g., "kings").
        epoch: Current training epoch (for the title).
        save_path: Where to save the PNG.
        num_bins: Number of histogram bins.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if hasattr(alpha_values, "numpy"):
        alpha_values = alpha_values.numpy()
    alpha_values = np.asarray(alpha_values).ravel()

    plt.figure(figsize=(8, 5))
    plt.hist(
        alpha_values,
        bins=num_bins,
        range=(0.0, 1.0),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )

    # Reference lines
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.axvline(1.0, color="green", linestyle="--", linewidth=1, alpha=0.6)

    plt.xlabel("Alpha value", fontsize=12)
    plt.ylabel("Number of patches", fontsize=12)
    plt.title(f"Alpha distribution — {kernel_name} (epoch {epoch})", fontsize=13)
    plt.xlim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_alpha_histogram_combined(alpha_values_dict, epoch, save_path, num_bins=20):
    """Plot alpha routing weights for all kernels in a single histogram.

    Each kernel is drawn as a separate histogram with a different color,
    overlaid on the same plot. This makes it easy to compare routing
    distributions across kernels at a glance.

    Args:
        alpha_values_dict: Dict mapping kernel_name -> 1-D array of alpha values.
        epoch: Current training epoch (for the title).
        save_path: Where to save the PNG.
        num_bins: Number of histogram bins.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Define a color palette (expand if needed for more kernels)
    colors = ["steelblue", "coral", "seagreen", "purple", "orange", "pink"]

    plt.figure(figsize=(10, 6))

    for idx, (kernel_name, alpha_values) in enumerate(alpha_values_dict.items()):
        if hasattr(alpha_values, "numpy"):
            alpha_values = alpha_values.numpy()
        alpha_values = np.asarray(alpha_values).ravel()

        color = colors[idx % len(colors)]
        plt.hist(
            alpha_values,
            bins=num_bins,
            range=(0.0, 1.0),
            alpha=0.5,
            label=kernel_name,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    # Reference lines
    plt.axvline(
        0.0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="0.0 (reject)"
    )
    plt.axvline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="0.5 (uncertain)",
    )
    plt.axvline(
        1.0, color="green", linestyle="--", linewidth=1, alpha=0.6, label="1.0 (select)"
    )

    plt.xlabel("Alpha value", fontsize=12)
    plt.ylabel("Number of patches", fontsize=12)
    plt.title(f"Alpha distribution — All kernels (epoch {epoch})", fontsize=13)
    plt.xlim(-0.05, 1.05)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_routing_ratio_over_epochs(routing_history, kernel_names, save_path):
    """Plot how the global routing ratio for each kernel evolves over epochs.

    Args:
        routing_history: List of dicts (one per epoch), each mapping
            kernel_name -> fraction of patches routed to that kernel.
        kernel_names: Ordered list of kernel names.
        save_path: Where to save the PNG.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(routing_history) + 1)
    plt.figure(figsize=(10, 6))

    for name in kernel_names:
        values = [r[name] for r in routing_history]
        plt.plot(epochs, values, marker="o", markersize=3, linewidth=2, label=name)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Fraction of patches", fontsize=12)
    plt.title("Global routing ratio per kernel", fontsize=14)
    plt.legend(fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_routing_prob_histogram(routing_probs_dict, epoch, save_path, num_bins=20):
    """Plot soft routing probability distribution for all kernels (Gumbel-Softmax).

    This is the Gumbel analog of ``plot_alpha_histogram_combined``. Each kernel's
    soft routing probabilities are drawn as a separate histogram with a different
    color, overlaid on the same plot.

    The shape of the distribution is diagnostic:
        - **Bimodal** (peaks at 0 and 1): decisive routing — the router is
          confidently assigning patches to specific kernels.
        - **Uniform / bell-shaped**: indecisive routing — the router is confused
          or temperature is too high.
        - **Single peak at 1/K**: collapsed / uniform routing — no specialisation.

    Args:
        routing_probs_dict: Dict mapping kernel_name -> 1-D array/tensor of soft
            routing probabilities for that kernel across all patches. Values in
            [0, 1].
        epoch: Current training epoch (for the title).
        save_path: Where to save the PNG.
        num_bins: Number of histogram bins.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    colors = ["steelblue", "coral", "seagreen", "purple", "orange", "pink"]

    plt.figure(figsize=(10, 6))

    for idx, (kernel_name, prob_values) in enumerate(routing_probs_dict.items()):
        if hasattr(prob_values, "numpy"):
            prob_values = prob_values.cpu().numpy()
        prob_values = np.asarray(prob_values).ravel()

        color = colors[idx % len(colors)]
        plt.hist(
            prob_values,
            bins=num_bins,
            range=(0.0, 1.0),
            alpha=0.5,
            label=kernel_name,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    num_kernels = len(routing_probs_dict)
    if num_kernels > 1:
        uniform_val = 1.0 / num_kernels
        plt.axvline(
            uniform_val,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"1/K = {uniform_val:.2f} (uniform)",
        )

    plt.xlabel("Soft routing probability", fontsize=12)
    plt.ylabel("Number of patches", fontsize=12)
    plt.title(
        f"Gumbel soft routing probability distribution (epoch {epoch})", fontsize=13
    )
    plt.xlim(-0.05, 1.05)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tau_entropy_curve(tau_history, entropy_history, save_path):
    """Plot Gumbel temperature (tau) and normalized router entropy over epochs.

    Uses a dual y-axis: left for tau (log scale), right for normalized entropy
    (linear, 0-1). This is the single most important diagnostic for tuning
    Gumbel-Softmax annealing — you want to see entropy decrease as tau
    decreases, confirming that lower temperature drives more decisive routing.

    Args:
        tau_history: List of floats — Gumbel temperature at each epoch.
        entropy_history: List of floats — normalized router entropy at each
            epoch (range [0, 1]).
        save_path: Where to save the PNG.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(tau_history) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: tau (log scale)
    color_tau = "tab:blue"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Temperature (τ)", color=color_tau, fontsize=12)
    ax1.plot(epochs, tau_history, color=color_tau, linewidth=2, label="τ (temperature)")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=color_tau)
    ax1.grid(True, alpha=0.3)

    # Right axis: entropy (linear 0-1)
    ax2 = ax1.twinx()
    color_ent = "tab:red"
    ax2.set_ylabel("Normalized entropy", color=color_ent, fontsize=12)
    ax2.plot(
        epochs,
        entropy_history,
        color=color_ent,
        linewidth=2,
        linestyle="--",
        label="Entropy (normalized)",
    )
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor=color_ent)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    plt.title("Gumbel temperature & router entropy over training", fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Standard plotting functions
# ---------------------------------------------------------------------------


def plot_loss_curves(
    train_losses,
    val_losses,
    save_path,
    ce_losses=None,
    ent_losses=None,
    ce_label="CE Loss",
    ent_label="λ·Entropy Loss",
):
    """Plot training and validation loss curves.

    The blue (train) and red (val) lines show the **joint** loss that the
    optimiser actually minimises (CE + regulariser).  When ``ce_losses``
    and/or ``ent_losses`` are provided, additional lines are drawn so you
    can see how each component evolves independently:

        - **Yellow** — pure cross-entropy / task-loss component
        - **Green**  — regularisation component (entropy for Teacher,
          budget/sparsity for Gumbel-Softmax, etc.)

    Args:
        train_losses: List of training losses per epoch (joint loss).
        val_losses: List of validation losses per epoch (joint loss).
        save_path: Path to save the plot.
        ce_losses: Optional list of per-epoch CE-only losses (yellow line).
        ent_losses: Optional list of per-epoch regulariser losses (green line).
        ce_label: Legend label for the CE component line (default ``"CE Loss"``).
        ent_label: Legend label for the regulariser component line
            (default ``"λ·Entropy Loss"``).  Pass e.g. ``"Budget Loss"`` for
            the Gumbel-Softmax pipeline.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Train Loss (joint)", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Val Loss (joint)", linewidth=2)

    if ce_losses is not None:
        plt.plot(
            epochs,
            ce_losses,
            color="#DAA520",
            linestyle="-",
            label=ce_label,
            linewidth=1.5,
            alpha=0.85,
        )
    if ent_losses is not None:
        plt.plot(
            epochs,
            ent_losses,
            color="green",
            linestyle="-",
            label=ent_label,
            linewidth=1.5,
            alpha=0.85,
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_multi_seed_loss_curves(
    all_train_losses,
    all_val_losses,
    save_path,
    all_ce_losses=None,
    all_ent_losses=None,
    ce_label="CE Loss",
    ent_label="Entropy Loss",
):
    """Plot loss curves with mean and std across multiple seeds.

    When ``all_ce_losses`` and/or ``all_ent_losses`` are supplied (one list
    per seed, same as the joint losses), additional mean ± std bands are
    drawn so you can see how the individual loss components behave across
    seeds:

        - **Yellow band** — CE / task-loss component
        - **Green band**  — regularisation component (entropy for Teacher,
          budget/sparsity for Gumbel-Softmax, etc.)

    Args:
        all_train_losses: List of lists, each inner list is train losses for one seed.
        all_val_losses: List of lists, each inner list is val losses for one seed.
        save_path: Path to save the plot.
        all_ce_losses: Optional list of lists — per-seed CE-only losses.
        all_ent_losses: Optional list of lists — per-seed regulariser losses.
        ce_label: Legend label for the CE component (default ``"CE Loss"``).
        ent_label: Legend label for the regulariser component
            (default ``"Entropy Loss"``).  Pass e.g. ``"Budget Loss"`` for
            the Gumbel-Softmax pipeline.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Early stopping can produce ragged lists — pad shorter runs with their
    # last value so we get a rectangular array for mean/std.
    max_len = max(len(losses) for losses in all_train_losses)

    def _pad(lst, length):
        return lst + [lst[-1]] * (length - len(lst))

    train_losses_arr = np.array([_pad(losses, max_len) for losses in all_train_losses])
    val_losses_arr = np.array([_pad(losses, max_len) for losses in all_val_losses])

    train_mean = train_losses_arr.mean(axis=0)
    train_std = train_losses_arr.std(axis=0)
    val_mean = val_losses_arr.mean(axis=0)
    val_std = val_losses_arr.std(axis=0)

    epochs = range(1, len(train_mean) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_mean, "b-", label="Train Loss (mean)", linewidth=2)
    plt.fill_between(
        epochs, train_mean - train_std, train_mean + train_std, color="b", alpha=0.2
    )

    plt.plot(epochs, val_mean, "r-", label="Val Loss (mean)", linewidth=2)
    plt.fill_between(
        epochs, val_mean - val_std, val_mean + val_std, color="r", alpha=0.2
    )

    # --- Optional CE component ---
    if all_ce_losses:
        ce_arr = np.array([_pad(losses, max_len) for losses in all_ce_losses])
        ce_mean = ce_arr.mean(axis=0)
        ce_std = ce_arr.std(axis=0)
        plt.plot(
            epochs,
            ce_mean,
            color="#DAA520",
            linestyle="-",
            label=f"{ce_label} (mean)",
            linewidth=1.5,
            alpha=0.85,
        )
        plt.fill_between(
            epochs,
            ce_mean - ce_std,
            ce_mean + ce_std,
            color="#DAA520",
            alpha=0.15,
        )

    # --- Optional entropy component ---
    if all_ent_losses:
        ent_arr = np.array([_pad(losses, max_len) for losses in all_ent_losses])
        ent_mean = ent_arr.mean(axis=0)
        ent_std = ent_arr.std(axis=0)
        plt.plot(
            epochs,
            ent_mean,
            color="green",
            linestyle="-",
            label=f"{ent_label} (mean)",
            linewidth=1.5,
            alpha=0.85,
        )
        plt.fill_between(
            epochs,
            ent_mean - ent_std,
            ent_mean + ent_std,
            color="green",
            alpha=0.15,
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Loss Curves (mean ± std, n={len(all_train_losses)} seeds)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(labels, probs, save_path, num_classes=2):
    """Plot ROC curve for binary or multi-class classification.

    Args:
        labels: Ground truth labels (numpy array)
        probs: Predicted probabilities (numpy array, shape [n_samples, n_classes])
        save_path: Path to save the plot
        num_classes: Number of classes
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))

    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    else:
        # Multi-class: plot ROC for each class
        for i in range(num_classes):
            binary_labels = (labels == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f"Class {i} (AUC = {roc_auc:.3f})")

    # Plot diagonal
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_multi_seed_roc_curves(all_labels, all_probs, save_path, num_classes=2):
    """Plot ROC curves with mean across multiple seeds.

    Args:
        all_labels: List of label arrays, one per seed
        all_probs: List of probability arrays, one per seed
        save_path: Path to save the plot
        num_classes: Number of classes
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))

    if num_classes == 2:
        # Binary classification - compute mean ROC
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        for labels, probs in zip(all_labels, all_probs):
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            # Interpolate to mean_fpr
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)
        mean_tpr[-1] = 1.0

        # Compute AUC
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std([auc(mean_fpr, tpr) for tpr in tprs])

        plt.plot(
            mean_fpr,
            mean_tpr,
            linewidth=2,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        )
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
    else:
        # Multi-class: plot mean ROC for each class
        mean_fpr = np.linspace(0, 1, 100)

        for class_idx in range(num_classes):
            tprs = []
            for labels, probs in zip(all_labels, all_probs):
                binary_labels = (labels == class_idx).astype(int)
                fpr, tpr, _ = roc_curve(binary_labels, probs[:, class_idx])
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)

            tprs = np.array(tprs)
            mean_tpr = tprs.mean(axis=0)
            std_tpr = tprs.std(axis=0)
            mean_tpr[-1] = 1.0

            mean_auc_val = auc(mean_fpr, mean_tpr)
            std_auc = np.std([auc(mean_fpr, tpr) for tpr in tprs])

            plt.plot(
                mean_fpr,
                mean_tpr,
                linewidth=2,
                label=f"Class {class_idx} (AUC = {mean_auc_val:.3f} ± {std_auc:.3f})",
            )
            plt.fill_between(
                mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2
            )

    # Plot diagonal
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves (mean ± std, n={len(all_labels)} seeds)", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, save_path, num_classes=2):
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix (numpy array)
        save_path: Path to save the plot
        num_classes: Number of classes
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
