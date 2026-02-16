import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

# ---------------------------------------------------------------------------
# TS-MoE specific plotting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Standard plotting functions
# ---------------------------------------------------------------------------


def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_multi_seed_loss_curves(all_train_losses, all_val_losses, save_path):
    """Plot loss curves with mean and std across multiple seeds.

    Args:
        all_train_losses: List of lists, each inner list is train losses for one seed
        all_val_losses: List of lists, each inner list is val losses for one seed
        save_path: Path to save the plot
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
