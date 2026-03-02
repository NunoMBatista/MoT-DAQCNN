import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute accuracy, AUC, F1, and Recall metrics."""
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    acc = (preds == labels_np).mean()

    # For binary classification
    if probs.shape[1] == 2:
        auc = roc_auc_score(labels_np, probs[:, 1])
    else:
        # Multi-class: use one-vs-rest
        auc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")

    f1 = f1_score(labels_np, preds, average="macro", zero_division="warn")
    recall = recall_score(labels_np, preds, average="macro", zero_division="warn")
    cm = confusion_matrix(labels_np, preds)

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "recall": float(recall),
        "confusion_matrix": cm,
    }


def evaluate(model, loader, device, split_name="Val", compute_full_metrics=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    all_logits = []
    all_labels = []

    batch_pbar = tqdm(loader, desc=f"{split_name} evaluation", leave=False)
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(batch_pbar):
            imgs = imgs.to(device)
            labels = labels.squeeze().long().to(device)

            batch_pbar.set_description(f"{split_name} forward pass")
            logits = model(imgs)
            loss = loss_fn(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, labels) * bs
            total_count += bs

            if compute_full_metrics:
                all_logits.append(logits)
                all_labels.append(labels)

            batch_pbar.set_description(
                f"{split_name} batch {batch_idx + 1}/{len(loader)} | loss={loss.item():.4f}"
            )

    batch_pbar.close()

    if compute_full_metrics:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_logits, all_labels)
        metrics["loss"] = total_loss / total_count
        # Add probabilities and labels for ROC plotting
        probs = torch.softmax(all_logits, dim=1).cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        metrics["probs"] = probs
        metrics["labels"] = labels_np
        return metrics
    else:
        return total_loss / total_count, total_acc / total_count


# =============================================================================
# Class Separability Metrics
# =============================================================================


def compute_1nn_accuracy(X, y, cv_folds=5, max_samples=5000):
    """
    Compute 1-Nearest Neighbor accuracy using cross-validation.

    1-NN accuracy gives a non-parametric estimate of how easily
    a simple classifier can separate the classes.

    Args:
        X: Feature array of shape (N, features) or (N, C, H, W)
        y: Label array of shape (N,)
        cv_folds: Number of cross-validation folds
        max_samples: Maximum samples to use (for computational efficiency)

    Returns:
        float: Mean cross-validated accuracy
    """
    X_flat = X.reshape(X.shape[0], -1)

    # Subsample if too large
    if len(X_flat) > max_samples:
        idx = np.random.choice(len(X_flat), max_samples, replace=False)
        X_flat = X_flat[idx]
        y = y[idx]

    # Use 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1)

    # Cross-validation
    try:
        scores = cross_val_score(knn, X_flat, y, cv=cv_folds, scoring="accuracy")
        return np.mean(scores)
    except ValueError:
        return np.nan


def compute_kta(X, y, max_samples=10000):
    """
    Compute Kernel-Target Alignment (KTA).

    KTA measures how well the kernel (similarity matrix) computed from features
    aligns with the ideal kernel based on class labels. Higher values indicate
    that the feature representation aligns well with the classification task.

    KTA = <K, Y> / (||K|| * ||Y||)

    where:
        K is the kernel matrix (here, linear kernel: K = X @ X.T)
        Y is the target kernel matrix (Y_ij = 1 if labels match, 0 otherwise)
        <K, Y> is the Frobenius inner product
        ||K|| is the Frobenius norm

    Args:
        X: Feature array of shape (N, features) or (N, C, H, W)
        y: Label array of shape (N,)
        max_samples: Maximum samples to use (KTA is O(n^2) in memory)

    Returns:
        float: KTA score in range [-1, 1], higher is better
    """
    X_flat = X.reshape(X.shape[0], -1)

    # Subsample if too large (KTA is O(n^2) in memory)
    if len(X_flat) > max_samples:
        idx = np.random.choice(len(X_flat), max_samples, replace=False)
        X_flat = X_flat[idx]
        y = y[idx]

    # Normalize features for numerical stability
    X_normalized = X_flat / (np.linalg.norm(X_flat, axis=1, keepdims=True) + 1e-10)

    # Compute kernel matrix K (linear kernel)
    K = X_normalized @ X_normalized.T

    # Compute target kernel Y (Y_ij = 1 if y_i == y_j, else 0)
    # Vectorized outer equality: broadcasts y as a column and a row, no Python loop.
    # Example: y = [0, 0, 7, 7] produces Y = [[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]]
    # Note: Uses EQUALITY comparison, so class 7 is NOT treated as "bigger" than class 0.
    Y = (y[:, None] == y[None, :]).astype(np.float64)

    # Compute KTA = <K, Y> / (||K|| * ||Y||)
    # Frobenius inner product: <K, Y> = trace(K.T @ Y) = sum(K * Y)
    numerator = np.sum(K * Y)

    # Frobenius norms
    norm_K = np.linalg.norm(K, "fro")
    norm_Y = np.linalg.norm(Y, "fro")

    if norm_K == 0 or norm_Y == 0:
        return np.nan

    kta = numerator / (norm_K * norm_Y)

    return kta


def compute_silhouette(X, y, max_samples=5000):
    """
    Compute silhouette score.

    Silhouette score measures how similar samples are to their own cluster
    vs other clusters. Range: [-1, 1], higher is better.

    We subsample if dataset is large (silhouette is O(n^2)).

    Args:
        X: Feature array of shape (N, features) or (N, C, H, W)
        y: Label array of shape (N,)
        max_samples: Maximum samples to use (for computational efficiency)

    Returns:
        float: Silhouette score in range [-1, 1], higher is better
    """
    X_flat = X.reshape(X.shape[0], -1)

    # Subsample if too large
    if len(X_flat) > max_samples:
        idx = np.random.choice(len(X_flat), max_samples, replace=False)
        X_flat = X_flat[idx]
        y = y[idx]

    # Need at least 2 classes and 2 samples per class
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 2:
        return np.nan

    try:
        score = silhouette_score(X_flat, y)
    except ValueError:
        score = np.nan

    return score


def compute_fishers_ratio(X, y):
    """
    Compute Fisher's Discriminant Ratio for multi-class data.

    Fisher's ratio measures how well-separated the classes are:
        FR = (between-class variance) / (within-class variance)

    Higher values indicate better class separability.

    For multi-class, we compute the average pairwise Fisher's ratio.

    Args:
        X: Feature array of shape (N, features) or (N, C, H, W)
        y: Label array of shape (N,)

    Returns:
        float: Fisher's ratio, higher is better
    """
    classes = np.unique(y)
    n_classes = len(classes)

    if n_classes < 2:
        return 0.0

    # Flatten features if needed (e.g., images to vectors)
    X_flat = X.reshape(X.shape[0], -1)

    # Compute class means and overall mean
    class_means = {}
    class_vars = {}

    for c in classes:
        mask = y == c
        X_c = X_flat[mask]
        class_means[c] = np.mean(X_c, axis=0)
        # Within-class variance (average variance across features)
        class_vars[c] = np.var(X_c, axis=0) + 1e-10  # Small epsilon for stability

    # Compute pairwise Fisher's ratio and average
    ratios = []
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1 :]:
            # Between-class: squared difference of means
            between = (class_means[c1] - class_means[c2]) ** 2
            # Within-class: sum of variances
            within = class_vars[c1] + class_vars[c2]
            # Fisher's ratio per feature, then average across features
            fr = np.mean(between / within)
            ratios.append(fr)

    return np.mean(ratios)
