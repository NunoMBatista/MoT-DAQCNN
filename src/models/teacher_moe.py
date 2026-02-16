"""
Teacher model for the Mixture-of-Topologies (MoT) pipeline.

The Teacher runs ALL kernel topologies on every patch (brute force via cached
quantum features), then uses a Grouped SE Block to learn which kernel is best
for each patch. A classification head trains on the soft-weighted features,
providing gradients that teach the SE block to route decisively.

This model works with pre-computed quantum features loaded from cached datasets.
It never runs quantum circuits itself — it just learns how to weight and
classify the quantum outputs.

Architecture:
    Cached quantum features (B, M*N, H, W)
        -> Grouped SE Block (soft routing weights per patch)
        -> Weighted features (B, M*N, H, W)
        -> Classification head (same as original DAQCNN head)
        -> Logits (B, num_classes)
"""

import torch
import torch.nn as nn

from src.layers.grouped_se_block import GroupedSEBlock
from src.utils.kernel_mapping import (
    build_kernel_to_channels_map,
    get_channels_per_kernel,
    get_kernel_names,
    get_num_kernels,
    validate_kernel_map,
)


class TeacherMoE(nn.Module):
    """Teacher model with Grouped SE routing over kernel topologies.

    Args:
        num_classes: Number of output classes for classification.
        total_channels: Total number of quantum feature channels (M * N).
        kernel_to_channels: Dict mapping kernel name -> list of channel indices.
            Built from cached dataset metadata via ``build_kernel_to_channels_map()``.
        dropout: Dropout probability for the classification head.
        activation: Activation function name ("relu" or "gelu").
        se_hidden_dim: Hidden dimension for the SE gating MLP. If None, uses
            a sensible default based on the number of kernels.

    Attributes:
        se_block: The Grouped SE Block that produces routing weights.
        head: Classification CNN head (mirrors the original DAQCNN head).
        kernel_names: Ordered list of kernel topology names.
        num_kernels: Number of kernel topologies (M).
        channels_per_kernel: Channels per kernel (N = kernel_size^2).
    """

    def __init__(
        self,
        num_classes,
        total_channels,
        kernel_to_channels,
        dropout=0.1,
        activation="relu",
        se_hidden_dim=None,
    ):
        super().__init__()

        # Store kernel mapping info
        self.kernel_names = get_kernel_names(kernel_to_channels)
        self.num_kernels = get_num_kernels(kernel_to_channels)
        self.channels_per_kernel = get_channels_per_kernel(kernel_to_channels)
        self.total_channels = total_channels

        # Validate that the mapping is consistent
        validate_kernel_map(kernel_to_channels, total_channels)

        # Build ordered channel groups (list of lists, matching kernel_names order)
        channel_groups = [kernel_to_channels[name] for name in self.kernel_names]

        # Grouped SE Block — the "Judge"
        self.se_block = GroupedSEBlock(
            num_kernels=self.num_kernels,
            channels_per_kernel=self.channels_per_kernel,
            channel_groups=channel_groups,
            hidden_dim=se_hidden_dim,
        )

        # Classification Head — the "Driver"
        # Same architecture as the original DAQCNN head so results are comparable.
        act_fn = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.head = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=2, stride=1, padding=0),
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

    @property
    def last_alpha(self):
        """Access the most recent alpha weights from the SE block.

        Shape: (B, M, H, W) where M = num_kernels.
        Available after a forward pass. Used for:
            - Logging alpha histograms during training
            - Generating distillation labels for the student
        """
        return self.se_block.last_alpha

    def forward(self, x):
        """Forward pass through SE block and classification head.

        Args:
            x: Quantum features of shape (B, M*N, H, W) from cached dataset.

        Returns:
            Logits of shape (B, num_classes).

        Note:
            After calling forward(), ``self.last_alpha`` contains the
            routing weights (B, M, H, W) for this batch.
        """
        # SE block: learn per-patch kernel routing weights and reweight channels
        x = self.se_block(x)

        # Classification head
        logits = self.head(x)
        return logits

    def get_routing_stats(self):
        """Compute routing statistics from the last forward pass.

        Returns:
            dict with:
                - "routing_ratio": dict mapping kernel name -> fraction of patches
                  where this kernel had the highest alpha.
                - "mean_alpha": dict mapping kernel name -> mean alpha across all patches.
                - "alpha_flat": dict mapping kernel name -> 1D tensor of all alpha values
                  (for histogram plotting).
        """
        alpha = self.last_alpha  # (B, M, H, W)
        if alpha is None:
            return None

        # Which kernel won at each patch
        winners = alpha.argmax(dim=1)  # (B, H, W)
        total_patches = winners.numel()

        routing_ratio = {}
        mean_alpha = {}
        alpha_flat = {}

        for k, name in enumerate(self.kernel_names):
            # Fraction of patches where this kernel won
            routing_ratio[name] = (winners == k).float().sum().item() / total_patches
            # Mean alpha value across all patches
            mean_alpha[name] = alpha[:, k, :, :].mean().item()
            # Flat alpha values for histogram plotting
            alpha_flat[name] = alpha[:, k, :, :].reshape(-1).cpu()

        return {
            "routing_ratio": routing_ratio,
            "mean_alpha": mean_alpha,
            "alpha_flat": alpha_flat,
        }


def build_teacher_from_metadata(
    metadata, num_classes, dropout=0.1, activation="relu", se_hidden_dim=None
):
    """Convenience factory: build a TeacherMoE directly from cached dataset metadata.

    Args:
        metadata: Dict from ``load_cached_quantum_dataset()``. Must contain
            "channel_kernel_map" and "out_channels".
        num_classes: Number of output classes.
        dropout: Dropout for the classification head.
        activation: Activation function ("relu" or "gelu").
        se_hidden_dim: Hidden dimension for SE gating MLP.

    Returns:
        TeacherMoE instance.
    """
    channel_kernel_map = metadata["channel_kernel_map"]
    total_channels = metadata["out_channels"]

    kernel_to_channels = build_kernel_to_channels_map(channel_kernel_map)

    return TeacherMoE(
        num_classes=num_classes,
        total_channels=total_channels,
        kernel_to_channels=kernel_to_channels,
        dropout=dropout,
        activation=activation,
        se_hidden_dim=se_hidden_dim,
    )
