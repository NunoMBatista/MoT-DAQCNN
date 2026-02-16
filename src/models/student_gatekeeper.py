"""
Student Gatekeeper for the Mixture-of-Topologies (MoT) pipeline.

The Student is a lightweight classical network that predicts which kernel
topology the Teacher would assign to each image patch, WITHOUT running
any quantum circuits. It learns to mimic the Teacher's routing decisions
via knowledge distillation (hard labels from argmax of Teacher alpha weights).

At inference time the Student runs first on raw image patches to decide
routing, so only the selected kernel's quantum circuit needs to run per patch.

For the patch sizes used in this project (2x2 or 3x3), convolutional layers
are degenerate (kernel would cover the entire input). The architecture is
therefore an MLP operating on flattened patch pixels.

Architecture:
    Flattened patch pixels (B, C * ks * ks)
        -> MLP with configurable hidden layers
        -> Logits (B, M) — one score per kernel topology
"""

import torch.nn as nn

from src.utils.kernel_mapping import (
    build_kernel_to_channels_map,
    get_kernel_names,
    get_num_kernels,
)


class StudentGatekeeper(nn.Module):
    """Lightweight MLP that predicts per-patch kernel routing.

    Trained via distillation: the Teacher's argmax(alpha) provides hard
    labels, and the Student learns to predict them from raw pixel patches.

    Args:
        patch_dim: Number of input features per patch (C * kernel_size^2).
        num_kernels: Number of kernel topologies to choose from (M).
        hidden_dims: Tuple of hidden layer sizes. Default (32, 16) keeps
            the model well under 5k parameters for typical patch sizes.

    Attributes:
        net: The sequential MLP.
        patch_dim: Stored for reshaping convenience.
        num_kernels: Number of output classes (M).
    """

    def __init__(self, patch_dim, num_kernels, hidden_dims=(32, 16)):
        super().__init__()

        self.patch_dim = patch_dim
        self.num_kernels = num_kernels

        layers = []
        in_dim = patch_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_kernels))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Predict kernel routing logits for a batch of patches.

        Args:
            x: Flattened patch pixels of shape (B, patch_dim).
                Also accepts (B, C, H, W) which will be auto-flattened.

        Returns:
            Logits of shape (B, num_kernels).
        """
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)

    def predict(self, x):
        """Return hard routing decisions (argmax) for a batch of patches.

        Args:
            x: Same as ``forward()``.

        Returns:
            Tensor of shape (B,) with values in {0, ..., num_kernels - 1}.
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_student_from_metadata(metadata, hidden_dims=(32, 16), in_channels=1):
    """Build a StudentGatekeeper from cached dataset metadata.

    Reads kernel_size, num_kernels, and kernel names from the metadata
    dict produced by ``load_cached_quantum_dataset()``.

    Args:
        metadata: Dict from cached quantum dataset. Must contain
            "kernel_size", "channel_kernel_map".
        hidden_dims: Hidden layer sizes for the MLP.
        in_channels: Number of image channels the student will receive.
            Typically 1 for grayscale or grayscale-converted datasets.

    Returns:
        Tuple of (StudentGatekeeper, kernel_names) where kernel_names
        is the ordered list of kernel topology names.
    """
    kernel_size = metadata["kernel_size"]
    channel_kernel_map = metadata["channel_kernel_map"]
    kernel_to_channels = build_kernel_to_channels_map(channel_kernel_map)

    num_kernels = get_num_kernels(kernel_to_channels)
    kernel_names = get_kernel_names(kernel_to_channels)
    patch_dim = in_channels * kernel_size * kernel_size

    student = StudentGatekeeper(
        patch_dim=patch_dim,
        num_kernels=num_kernels,
        hidden_dims=hidden_dims,
    )

    return student, kernel_names
