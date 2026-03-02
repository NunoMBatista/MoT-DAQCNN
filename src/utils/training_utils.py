"""
Shared utilities for the TS-MoE (Teacher-Student Mixture of Experts) pipeline.

Includes loss functions, common training helpers, and the shared classification
head factory used by DAQCNN, TeacherMoE, and FinalClassifier.
"""

import torch
import torch.nn as nn


def resolve_device(cfg: dict, verbose: bool = True) -> str:
    """Resolve the torch device from config, with CPU fallback.

    Reads ``model.classical_device`` from the config dict. Accepts "auto",
    "cuda", or "cpu". If "cuda" is requested but not available, falls back
    to "cpu" with an optional warning.

    Args:
        cfg: Loaded YAML config dict.
        verbose: If True, prints the resolved device and any fallback warning.

    Returns:
        Device string: "cuda" or "cpu".
    """
    requested = cfg.get("model", {}).get("classical_device", "auto")
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = requested
    if verbose:
        print(f"Device: {device}")
    return device


def entropy_loss(alpha):
    """Compute the mean entropy of routing weights across all patches.

    Entropy is maximized when alpha is uniform (bad — indecisive routing)
    and minimized when alpha is one-hot (good — decisive routing).

    By adding ``lambda * entropy_loss(alpha)`` to the classification loss,
    we penalize indecisive routing and encourage the SE block to commit
    to a single kernel per patch.

    Args:
        alpha: Tensor of shape (B, M, H, W) with softmax-normalized
            routing weights. M is the number of kernels.

    Returns:
        Scalar tensor: mean entropy across all patches in the batch.
            Range is [0, log(M)] where M = alpha.shape[1].
    """
    # Small epsilon to avoid log(0)
    eps = 1e-8
    # Entropy per patch: -sum_k alpha_k * log(alpha_k), summed over M kernels
    # alpha shape: (B, M, H, W)
    h = -torch.sum(alpha * torch.log(alpha + eps), dim=1)  # (B, H, W)
    return h.mean()


def build_classification_head(in_channels, num_classes, dropout=0.1, activation="relu"):
    """Build the shared CNN classification head used by DAQCNN, TeacherMoE, and FinalClassifier.

    All three models use an identical head architecture so their results are
    directly comparable.  Centralising it here means a single change propagates
    everywhere and there is no risk of the copies drifting apart.

    Architecture (mirrors the original paper):
        Conv2d(in_channels -> 64, 2×2, stride=1, no padding)
        BatchNorm2d(64)
        Activation (ReLU or GELU)
        MaxPool2d(2×2, stride=2)
        Conv2d(64 -> 64, 2×2, stride=1, no padding)
        Activation
        Dropout(dropout)
        Flatten
        Dropout(dropout)
        LazyLinear(num_classes)   ← infers spatial dim at first forward pass

    Args:
        in_channels: Number of input feature channels (M*N for quantum features).
        num_classes: Number of output logits.
        dropout: Dropout probability applied before and after Flatten.
        activation: "relu" (default) or "gelu".

    Returns:
        nn.Sequential — the classification head, ready to be assigned as
        ``self.head`` on any model.
    """
    act_fn = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=2, stride=1, padding=0),
        nn.BatchNorm2d(64),
        act_fn,
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
        act_fn,
        nn.Dropout(dropout),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.LazyLinear(num_classes),
    )


def compute_lambda(epoch, warmup_epochs, lambda_max, lambda_start=0.0):
    """Compute the entropy regularization weight with linear annealing.

    Lambda starts at ``lambda_start`` (typically 0 to let the network explore)
    and ramps linearly to ``lambda_max`` over ``warmup_epochs`` epochs.
    After that it stays at ``lambda_max``.

    Args:
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of epochs over which to anneal lambda.
            If 0, lambda is always lambda_max.
        lambda_max: Maximum value of lambda.
        lambda_start: Starting value of lambda (default 0.0).

    Returns:
        float: Current lambda value.
    """
    if warmup_epochs <= 0:
        return lambda_max
    progress = min(epoch / warmup_epochs, 1.0)
    return lambda_start + (lambda_max - lambda_start) * progress
