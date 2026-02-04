import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
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
