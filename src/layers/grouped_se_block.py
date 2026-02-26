"""
Grouped Squeeze-and-Excitation Block for patch-wise kernel routing.

This block is the core "Judge" of the Teacher model. Given quantum features
from M kernel topologies (each producing N channels), it decides how much
to weight each kernel's output at every spatial position (patch).

Key design choices:
    - Operates PATCH-WISE: each spatial position (i, j) gets its own M weights.
    - Outputs M weights (one per kernel group), NOT M*N (not per channel).
    - Uses 1x1 convolutions for spatial efficiency (equivalent to per-pixel MLP).
    - Alpha weights are softmax-normalized across kernel groups.
"""

import torch
import torch.nn as nn


class GroupedSEBlock(nn.Module):
    """Grouped Squeeze-and-Excitation block for kernel routing.

    Takes a tensor with M*N channels (M kernels, N channels each), groups them
    by kernel, squeezes each group to a single value via average pooling, and
    produces M soft routing weights per spatial position.

    Args:
        num_kernels: Number of kernel topologies (M).
        channels_per_kernel: Number of channels per kernel (N = kernel_size^2).
        channel_groups: List of lists, where channel_groups[k] contains the
            channel indices belonging to kernel k. If None, assumes sequential
            grouping: kernel 0 = channels [0..N-1], kernel 1 = [N..2N-1], etc.
        hidden_dim: Hidden dimension for the gating MLP. Defaults to
            max(num_kernels, 4) to keep it small but expressive.
        use_std: If True, use both mean and std for squeeze (input becomes 2*M).
            If False, use only mean (input is M). Default: False.

    Input shape:  (B, M*N, H, W)
    Output shape: (B, M*N, H, W) — same shape, but channels are reweighted.

    The alpha weights (B, M, H, W) are stored after each forward pass:
        - ``self.last_alpha_live``: live tensor (gradient-attached), for entropy loss.
        - ``self.last_alpha``: detached copy, safe for logging and distillation.
    """

    def __init__(
        self,
        num_kernels,
        channels_per_kernel,
        channel_groups=None,
        hidden_dim=None,
        use_std=False,
    ):
        super().__init__()

        self.num_kernels = num_kernels  # M
        self.channels_per_kernel = channels_per_kernel  # N
        self.total_channels = num_kernels * channels_per_kernel  # M * N
        self.use_std = use_std  # Whether to include std in squeeze

        # Channel groups: which channel indices belong to each kernel.
        # Stored as a list of lists, ordered by kernel index.
        if channel_groups is not None:
            assert len(channel_groups) == num_kernels, (
                f"Expected {num_kernels} channel groups, got {len(channel_groups)}"
            )
            self.channel_groups = [sorted(g) for g in channel_groups]
        else:
            # Default: sequential grouping
            self.channel_groups = [
                list(range(k * channels_per_kernel, (k + 1) * channels_per_kernel))
                for k in range(num_kernels)
            ]

        # Gating network: small bottleneck MLP implemented as 1x1 convolutions
        # so it processes all spatial positions in parallel.
        if hidden_dim is None:
            hidden_dim = max(num_kernels, 4)

        # Input to gate is M (mean only) or 2*M (mean + std)
        gate_input_channels = 2 * num_kernels if use_std else num_kernels

        self.gate = nn.Sequential(
            nn.Conv2d(gate_input_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_kernels, kernel_size=1, bias=True),
        )

        # Stored after each forward pass:
        # - live version keeps gradients (for entropy loss backprop)
        # - detached version is safe for logging / numpy conversion
        # - logits are pre-softmax values for knowledge distillation
        self.last_alpha_live = None
        self.last_alpha = None
        self.last_logits = None

    def forward(self, x):
        """Forward pass: squeeze, gate, and reweight.

        Args:
            x: Tensor of shape (B, M*N, H, W) with quantum features from
               all M kernels concatenated along the channel dimension.

        Returns:
            Tensor of shape (B, M*N, H, W) with channels reweighted by
            the learned per-patch alpha values.
        """
        B, C, H, W = x.shape
        assert C == self.total_channels, (
            f"Expected {self.total_channels} channels, got {C}"
        )

        # --- Squeeze: average each kernel group (optionally include std) ---
        # Result: (B, M, H, W) if use_std=False, or (B, 2*M, H, W) if use_std=True
        if self.use_std:
            # Use both mean and std: more information about each kernel group
            pooled = torch.zeros(
                B, 2 * self.num_kernels, H, W, device=x.device, dtype=x.dtype
            )
            for k, group in enumerate(self.channel_groups):
                pooled[:, 2 * k, :, :] = x[:, group, :, :].mean(dim=1)  # mean
                pooled[:, 2 * k + 1, :, :] = x[:, group, :, :].std(dim=1)  # std
        else:
            # Use only mean (original behavior)
            pooled = torch.zeros(
                B, self.num_kernels, H, W, device=x.device, dtype=x.dtype
            )
            for k, group in enumerate(self.channel_groups):
                pooled[:, k, :, :] = x[:, group, :, :].mean(dim=1)

        # --- Gate: produce M weights per patch via small MLP ---
        # (B, M or 2*M, H, W) -> (B, M, H, W)
        alpha_logits = self.gate(pooled)  # pre-softmax logits
        alpha = torch.softmax(alpha_logits, dim=1)  # normalize across kernels

        # Store logits (for KD), live alpha (for entropy loss), and detached alpha (for logging)
        self.last_logits = alpha_logits.detach()
        self.last_alpha_live = alpha
        self.last_alpha = alpha.detach()

        # --- Excite: multiply alpha_k into all N channels of kernel group k ---
        out = torch.zeros_like(x)
        for k, group in enumerate(self.channel_groups):
            # alpha[:, k:k+1, :, :] has shape (B, 1, H, W) — broadcasts over N channels
            out[:, group, :, :] = x[:, group, :, :] * alpha[:, k : k + 1, :, :]

        return out
