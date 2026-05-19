"""
Shared training utilities for DAQCNN.

Includes common training helpers and the shared classification head factory.
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


# ---------------------------------------------------------------------------
# Capacity-sweep head variants
# All input tensors are 4D (B, C, H, W); the initial Flatten handles that.
# LazyLinear infers the input dim on the first forward pass, so these heads
# work for any feature-source dimensionality without changes.
# ---------------------------------------------------------------------------

def _activation(name: str) -> nn.Module:
    return nn.GELU() if name.lower() == "gelu" else nn.ReLU()


def build_linear_head(num_classes: int, **_ignored) -> nn.Sequential:
    """Pure linear classifier on flattened features.

    The Adam-trained analogue of the LinearSVC probe. Accepts and ignores
    dropout/activation kwargs so it shares a uniform call signature with
    the other head builders.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(num_classes),
    )


def build_mlp1_head(num_classes: int, hidden: int = 32, dropout: float = 0.5,
                    activation: str = "gelu") -> nn.Sequential:
    """Single hidden layer MLP head."""
    return nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(hidden),
        _activation(activation),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
    )


def build_mlp2_head(num_classes: int, h1: int = 64, h2: int = 32,
                    dropout: float = 0.5, activation: str = "gelu") -> nn.Sequential:
    """Two hidden layer MLP head."""
    return nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(h1),
        _activation(activation),
        nn.Dropout(dropout),
        nn.Linear(h1, h2),
        _activation(activation),
        nn.Dropout(dropout),
        nn.Linear(h2, num_classes),
    )


