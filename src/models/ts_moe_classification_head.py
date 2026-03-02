"""
Final Classifier for the Mixture-of-Topologies (MoT) pipeline (Phase 3).

After the Student Gatekeeper routes each patch to a single kernel topology,
we get sparse tensors where only the selected kernel's channels contain data
and the rest are zeroed. The Teacher's classification head can't be reused
here because it was trained on soft-mixed (all kernels weighted) features.

This module provides a new classification head trained from scratch on the
sparse routed tensors, learning to interpret zeros as "not selected" rather
than "low signal".

Architecture:
    Sparse routed features (B, M*N [+1], H, W)
        -> CNN classification head (same structure as Teacher/DAQCNN head)
        -> Logits (B, num_classes)
"""

import torch.nn as nn

from src.utils.kernel_mapping import (
    build_kernel_to_channels_map,
    get_channels_per_kernel,
    get_kernel_names,
    get_num_kernels,
)
from src.utils.training_utils import build_classification_head


class FinalClassifier(nn.Module):
    """Classification head for sparse routed quantum features.

    Same CNN architecture as the Teacher/DAQCNN head so results are
    directly comparable — the only difference is the input distribution
    (sparse vs soft-mixed).

    Args:
        in_channels: Number of input channels. Normally M*N, or M*N+1
            if a mask channel is appended.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        activation: Activation function name ("relu" or "gelu").
    """

    def __init__(self, in_channels, num_classes, dropout=0.1, activation="relu"):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.head = build_classification_head(
            in_channels, num_classes, dropout, activation
        )

    def forward(self, x):
        """Forward pass on sparse routed features.

        Args:
            x: Sparse tensor of shape (B, in_channels, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.head(x)


def build_final_classifier_from_metadata(
    metadata, num_classes, dropout=0.1, activation="relu"
):
    """Build a FinalClassifier from cached dataset metadata.

    Args:
        metadata: Dict from ``load_cached_quantum_dataset()``. Must contain
            "out_channels".
        num_classes: Number of output classes.
        dropout: Dropout for the classification head.
        activation: Activation function ("relu" or "gelu").

    Returns:
        FinalClassifier instance.
    """
    in_channels = metadata["out_channels"]

    return FinalClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
    )
