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


