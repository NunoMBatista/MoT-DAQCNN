"""
Shared utilities for the TS-MoE (Teacher-Student Mixture of Experts) pipeline.

Includes loss functions, common training helpers, and the shared classification
head factory used by DAQCNN, TeacherMoE, and FinalClassifier.
"""

import math

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
    """Compute the **normalised** mean entropy of routing weights across all patches.

    Entropy is maximized when alpha is uniform (bad — indecisive routing)
    and minimized when alpha is one-hot (good — decisive routing).

    By adding ``lambda * entropy_loss(alpha)`` to the classification loss,
    we penalize indecisive routing and encourage the SE block to commit
    to a single kernel per patch.

    **Normalisation:** Raw entropy ranges from 0 to log(M), where
    M is the number of kernels. Without normalisation, the entropy term's
    magnitude scales with M, meaning the same ``lambda_max`` value produces
    very different gradient magnitudes for 2-kernel vs 4-kernel configs.
    With M=4, raw entropy can reach ~1.39, and ``lambda_max=0.5`` produces
    a penalty of ~0.7 — comparable to the CE loss itself, causing the gate
    and classification head to fight each other.

    We now divide by log(M) so the returned value is always in [0, 1]:
        0.0 = perfectly decisive (one-hot alpha)
        1.0 = maximally indecisive (uniform alpha)

    This makes ``lambda_max`` values interpretable consistently across any
    number of kernels, and prevents the entropy penalty from overwhelming
    the classification signal.

    Args:
        alpha: Tensor of shape (B, M, H, W) with softmax-normalized
            routing weights. M is the number of kernels.

    Returns:
        Scalar tensor: mean normalised entropy across all patches in the
            batch.  Range is [0, 1] regardless of M.
    """
    M = alpha.shape[1]
    # Small epsilon to avoid log(0)
    eps = 1e-8
    # Entropy per patch: -sum_k alpha_k * log(alpha_k), summed over M kernels
    # alpha shape: (B, M, H, W)
    h = -torch.sum(alpha * torch.log(alpha + eps), dim=1)  # (B, H, W)
    # Normalise by maximum possible entropy so the value is in [0, 1].
    # For M=1 (degenerate case), max_entropy=0; guard against division by zero.
    max_entropy = math.log(M) if M > 1 else 1.0
    return h.mean() / max_entropy


def build_classification_head(in_channels, num_classes, dropout=0.1, activation="relu", hidden_channels=64):
    """Build the shared CNN classification head.

    Args:
        in_channels: Number of input feature channels (M*N).
        num_classes: Number of output logits.
        dropout: Dropout probability.
        activation: "relu" or "gelu".
        hidden_channels: Number of filters in the first two Conv2d layers.
    """
    act_fn = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=2, stride=1, padding=0),
        nn.BatchNorm2d(hidden_channels),
        act_fn,
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=2, stride=1, padding=0),
        act_fn,
        nn.Dropout(dropout),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.LazyLinear(num_classes),
    )


def compute_lambda(epoch, warmup_epochs, lambda_max, lambda_start=0.0, start_epoch=0):
    """Compute the entropy regularization weight with linear annealing.

    Lambda stays at ``lambda_start`` until ``start_epoch``, then ramps
    linearly to ``lambda_max`` over ``warmup_epochs`` epochs.

    Args:
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of epochs over which to anneal lambda.
            If 0, lambda is always lambda_max.
        lambda_max: Maximum value of lambda.
        lambda_start: Starting value of lambda (default 0.0).
        start_epoch: Epoch at which lambda starts growing (default 0).

    Returns:
        float: Current lambda value.
    """
    if epoch < start_epoch:
        return lambda_start
    if warmup_epochs <= 0:
        return lambda_max
    progress = min((epoch - start_epoch) / warmup_epochs, 1.0)
    return lambda_start + (lambda_max - lambda_start) * progress
