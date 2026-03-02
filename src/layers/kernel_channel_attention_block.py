"""
Kernel Channel Attention Block for patch-wise kernel routing.

This block is the core "Judge" of the Teacher model. Given quantum features
from M kernel topologies (each producing N channels), it decides how much
to weight each kernel's output at every spatial position (patch).

Unlike the original Squeeze-and-Excitation approach, this block does NOT
squeeze (average) each kernel group down to a single scalar. Instead, the
gate sees ALL raw channels — the full qubit activation pattern — so it can
make routing decisions based on the complete quantum texture fingerprint
rather than a lossy summary.

A per-group BatchNorm2d normalizes each kernel group's channels before the
gate, preventing topologies with inherently larger raw output magnitudes
(e.g., Kings graph with its stronger 1/r^6 interactions) from dominating
the routing decision purely through scale.

Key design choices:
    - Operates PATCH-WISE: each spatial position (i, j) gets its own M weights.
    - Outputs M weights (one per kernel group), NOT M*N (not per channel).
    - Uses 1x1 convolutions for spatial efficiency (equivalent to per-pixel MLP).
    - Alpha weights are softmax-normalized across kernel groups.
    - BatchNorm2d applied per kernel group before gating for scale fairness.
"""

import torch
import torch.nn as nn


class KernelChannelAttentionBlock(nn.Module):
    """Kernel Channel Attention block for kernel routing.

    Takes a tensor with M*N channels (M kernels, N channels each), normalizes
    each kernel group via BatchNorm2d, feeds ALL raw channels into a gating
    MLP, and produces M soft routing weights per spatial position.

    Args:
        num_kernels: Number of kernel topologies (M).
        channels_per_kernel: Number of channels per kernel (N = kernel_size^2).
        channel_groups: List of lists, where channel_groups[k] contains the
            channel indices belonging to kernel k. If None, assumes sequential
            grouping: kernel 0 = channels [0..N-1], kernel 1 = [N..2N-1], etc.
        hidden_dim: Hidden dimension for the gating MLP. Defaults to
            max(total_channels // 2, num_kernels * 2, 8).
        gate_zero_init: If True, zero-initialize the weights and bias of the
            last Conv2d in the gate so that initial alpha values are uniform
            (1/M) across all kernels for every patch. This removes random-init
            asymmetry and ensures symmetry is broken by data gradients rather
            than by the initial weight lottery. Default False for backward
            compatibility.

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
        gate_zero_init=False,
    ):
        super().__init__()

        self.num_kernels = num_kernels  # M
        self.channels_per_kernel = channels_per_kernel  # N
        self.total_channels = num_kernels * channels_per_kernel  # M * N

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

        # Per-group BatchNorm2d: normalizes each kernel group's channels
        # independently, so topologies with different raw output scales
        # are put on equal footing before the gate makes its decision.
        self.group_norms = nn.ModuleList(
            [nn.BatchNorm2d(channels_per_kernel) for _ in range(num_kernels)]
        )

        # Gating network: MLP implemented as 1x1 convolutions so it
        # processes all spatial positions in parallel.
        # Input is ALL M*N channels (no squeeze).
        if hidden_dim is None:
            hidden_dim = max(self.total_channels // 2, num_kernels * 2, 8)

        self.gate = nn.Sequential(
            nn.Conv2d(self.total_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_kernels, kernel_size=1, bias=True),
        )

        # Optionally zero-init the last gate layer so that initial logits are
        # all zero -> softmax produces uniform 1/M alphas on the first forward
        # pass, regardless of seed. Symmetry is then broken by gradients.
        if gate_zero_init:
            last_conv = self.gate[-1]
            nn.init.zeros_(last_conv.weight)
            nn.init.zeros_(last_conv.bias)

        # Stored after each forward pass:
        # - live version keeps gradients (for entropy loss backprop)
        # - detached version is safe for logging / numpy conversion
        # - logits are pre-softmax values for knowledge distillation
        self.last_alpha_live = None
        self.last_alpha = None
        self.last_logits = None

    def forward(self, x):
        """Forward pass: normalize, gate, and reweight.

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

        # --- Normalize: BatchNorm each kernel group independently ---
        # Each group is a contiguous slice (channels_per_kernel channels wide).
        # We split, norm each slice, then reassemble — no Python-level loop,
        # no temporary zero-filled buffer.
        # split() returns a tuple of (B, N, H, W) tensors, one per kernel group.
        groups = x.split(self.channels_per_kernel, dim=1)  # tuple of M × (B, N, H, W)
        normed_groups = [self.group_norms[k](g) for k, g in enumerate(groups)]
        x_normed = torch.cat(normed_groups, dim=1)  # (B, M*N, H, W)

        # --- Gate: produce M weights per patch from ALL normalized channels ---
        # (B, M*N, H, W) -> (B, M, H, W)
        alpha_logits = self.gate(x_normed)  # pre-softmax logits
        alpha = torch.softmax(alpha_logits, dim=1)  # normalize across kernels

        # Store logits (for KD), live alpha (for entropy loss), and detached alpha (for logging)
        self.last_logits = alpha_logits.detach()
        self.last_alpha_live = alpha
        self.last_alpha = alpha.detach()

        # --- Excite: multiply alpha_k into all N channels of kernel group k ---
        # Expand alpha from (B, M, H, W) to (B, M*N, H, W) by repeating each
        # scalar weight N times so it broadcasts directly over the channel group.
        #
        # BUG FIX: Apply alpha to the NORMALISED features (x_normed) instead of
        # the raw features (x).  The gate's softmax decision is based on x_normed,
        # so the classification head must see the same scale.  Previously, using
        # raw x here meant that a kernel topology with inherently larger output
        # magnitudes (e.g. Kings graph with stronger 1/r^6 interactions) would
        # dominate the classification head's input even when the gate assigned it
        # a LOW alpha weight — the raw magnitude leaked through and overwhelmed
        # the gate's attenuation.  This caused the CE loss to send inconsistent
        # gradient signals back to the gate:
        #   - Gate says "down-weight kernel 0" (low alpha)
        #   - But kernel 0's raw magnitude still dominates the head's input
        #   - CE loss blames the gate for kernel 0's influence and oscillates
        #
        # Using x_normed ensures that the head sees features whose relative
        # contributions are purely determined by the gate's alpha weights,
        # giving the gate clean, consistent gradient feedback.
        N = self.channels_per_kernel
        alpha_expanded = alpha.repeat_interleave(N, dim=1)  # (B, M*N, H, W)
        out = x_normed * alpha_expanded

        return out
