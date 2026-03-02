"""
Teacher model for the Mixture-of-Topologies (MoT) pipeline.

The Teacher runs ALL kernel topologies on every patch (brute force via cached
quantum features), then uses a Kernel Channel Attention Block to learn which
kernel is best for each patch. A classification head trains on the soft-weighted
features, providing gradients that teach the attention block to route decisively.

This model works with pre-computed quantum features loaded from cached datasets.
It never runs quantum circuits itself — it just learns how to weight and
classify the quantum outputs.

Architecture:
    Cached quantum features (B, M*N, H, W)
        -> Kernel Channel Attention Block (soft routing weights per patch)
        -> Weighted features (B, M*N, H, W)
        -> Classification head (same as original DAQCNN head)
        -> Logits (B, num_classes)
"""

import torch
import torch.nn as nn

from src.layers.kernel_channel_attention_block import KernelChannelAttentionBlock
from src.utils.kernel_mapping import (
    build_kernel_to_channels_map,
    get_channels_per_kernel,
    get_kernel_names,
    get_num_kernels,
    validate_kernel_map,
)
from src.utils.training_utils import build_classification_head


class TeacherMoE(nn.Module):
    """Teacher model with Kernel Channel Attention routing over kernel topologies.

    Args:
        num_classes: Number of output classes for classification.
        total_channels: Total number of quantum feature channels (M * N).
        kernel_to_channels: Dict mapping kernel name -> list of channel indices.
            Built from cached dataset metadata via ``build_kernel_to_channels_map()``.
        dropout: Dropout probability for the classification head.
        activation: Activation function name ("relu" or "gelu").
        attention_hidden_dim: Hidden dimension for the attention gating MLP.
            If None, uses a sensible default based on total channels.

    Attributes:
        attention_block: The Kernel Channel Attention Block that produces routing weights.
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
        attention_hidden_dim=None,
        gate_zero_init=False,
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

        # Kernel Channel Attention Block — the "Judge"
        self.attention_block = KernelChannelAttentionBlock(
            num_kernels=self.num_kernels,
            channels_per_kernel=self.channels_per_kernel,
            channel_groups=channel_groups,
            hidden_dim=attention_hidden_dim,
            gate_zero_init=gate_zero_init,
        )

        # Classification Head — the "Driver"
        # Same architecture as the original DAQCNN head so results are comparable.
        # Built via the shared factory in training_utils so all three models stay in sync.
        self.head = build_classification_head(
            total_channels, num_classes, dropout, activation
        )

    @property
    def last_alpha(self):
        """Access the most recent alpha weights from the attention block.

        Shape: (B, M, H, W) where M = num_kernels.
        Available after a forward pass. Used for:
            - Logging alpha histograms during training
            - Generating distillation labels for the student
        """
        return self.attention_block.last_alpha

    @property
    def last_logits(self):
        """Access the most recent pre-softmax logits from the attention block.

        Shape: (B, M, H, W) where M = num_kernels.
        Available after a forward pass. Used for knowledge distillation.
        """
        return self.attention_block.last_logits

    def forward(self, x):
        """Forward pass through attention block and classification head.

        Args:
            x: Quantum features of shape (B, M*N, H, W) from cached dataset.

        Returns:
            Logits of shape (B, num_classes).

        Note:
            After calling forward(), ``self.last_alpha`` contains the
            routing weights (B, M, H, W) for this batch.
        """
        # Attention block: learn per-patch kernel routing weights and reweight channels
        x = self.attention_block(x)

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
    metadata,
    num_classes,
    dropout=0.1,
    activation="relu",
    attention_hidden_dim=None,
    gate_zero_init=False,
):
    """Convenience factory: build a TeacherMoE directly from cached dataset metadata.

    Args:
        metadata: Dict from ``load_cached_quantum_dataset()``. Must contain
            "channel_kernel_map" and "out_channels".
        num_classes: Number of output classes.
        dropout: Dropout for the classification head.
        activation: Activation function ("relu" or "gelu").
        attention_hidden_dim: Hidden dimension for attention gating MLP.
        gate_zero_init: If True, zero-init the last gate layer for uniform
            initial alphas.

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
        attention_hidden_dim=attention_hidden_dim,
        gate_zero_init=gate_zero_init,
    )
