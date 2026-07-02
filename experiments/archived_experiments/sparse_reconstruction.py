"""
Sparse tensor reconstruction from student routing predictions.

After the Student Gatekeeper predicts which kernel topology to use for each
image patch, we need to build a sparse tensor where only the selected kernel's
channels are filled with quantum feature data and all other channels are zeroed.

This sparse tensor is then fed to a new Final Classifier that learns to
interpret the routed data (where zeros mean "not selected", not "low signal").

Example for M=2 kernels, N=9 channels per kernel:
    - Patch (i,j) assigned to kernel 0: channels 0-8 filled, channels 9-17 = 0
    - Patch (i,j) assigned to kernel 1: channels 0-8 = 0, channels 9-17 filled
"""

import torch


def build_sparse_tensor_fast(
    quantum_features, routing_map, channel_groups, per_group_stats=None
):
    """Build a sparse tensor by routing each patch to its selected kernel.

    For every spatial position (i, j), the routing map says which kernel k
    was selected. We keep that kernel's channels and zero out the rest.

    When ``per_group_stats`` is provided, each kernel group's channels are
    normalised BEFORE masking using the supplied per-group mean and std.
    These statistics should be computed from the Phase 3 training split so
    that the Final Classifier trains and is evaluated on a consistently
    normalised feature space — with no dependence on the Teacher's internal
    BatchNorm running statistics or training dynamics.

    Args:
        quantum_features: (B, M*N, H, W) tensor with all kernels' quantum outputs.
        routing_map: (B, H, W) LongTensor with values in {0, ..., M-1},
            indicating the selected kernel at each spatial position.
        channel_groups: List of lists, where channel_groups[k] is the list
            of channel indices for kernel k. Must be ordered consistently
            with the routing map indices.
        per_group_stats: Optional list of (mean, std) pairs, one per kernel
            group. Each mean and std has shape (1, N, 1, 1) so it broadcasts
            over (B, N, H, W). When provided, each group's channels are
            normalised as (x - mean) / std before masking. std is clamped to
            a minimum of 1e-6 to avoid division by zero.

    Returns:
        Sparse tensor of shape (B, M*N, H, W) where only the selected
        kernel's channels are non-zero at each spatial position.
    """
    B, C, H, W = quantum_features.shape
    device = quantum_features.device
    routing_map = routing_map.to(device)

    # --- Per-group normalisation ---
    # Normalise ALL groups first, THEN apply the sparse mask.  This way
    # statistics are computed over the full (non-sparse) features, which is
    # the only consistent thing to do — computing stats after masking would
    # include the structural zeros introduced by routing.
    if per_group_stats is not None:
        channels_per_kernel = len(channel_groups[0])
        groups = quantum_features.split(channels_per_kernel, dim=1)  # M × (B, N, H, W)
        normed_groups = []
        for k, g in enumerate(groups):
            mean, std = per_group_stats[k]
            mean = mean.to(device)
            std = std.to(device)
            normed_groups.append((g - mean) / std.clamp(min=1e-6))
        quantum_features = torch.cat(normed_groups, dim=1)

    sparse = torch.zeros_like(quantum_features)

    for k, channels in enumerate(channel_groups):
        # (B, 1, H, W) boolean mask — True where kernel k was selected
        mask = (routing_map == k).unsqueeze(1)  # (B, 1, H, W)
        if not mask.any():
            continue

        ch_idx = torch.tensor(channels, device=device)
        # quantum_features[:, ch_idx] has shape (B, N, H, W)
        # mask broadcasts over the N channels in the group
        sparse[:, ch_idx, :, :] = quantum_features[:, ch_idx, :, :] * mask

    return sparse
