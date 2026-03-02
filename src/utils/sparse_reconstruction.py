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
    quantum_features, routing_map, channel_groups, use_mask_channel=False
):
    """Build a sparse tensor by routing each patch to its selected kernel.

    For every spatial position (i, j), the routing map says which kernel k
    was selected. We keep that kernel's channels and zero out the rest.

    Args:
        quantum_features: (B, M*N, H, W) tensor with all kernels' quantum outputs.
        routing_map: (B, H, W) LongTensor with values in {0, ..., M-1},
            indicating the selected kernel at each spatial position.
        channel_groups: List of lists, where channel_groups[k] is the list
            of channel indices for kernel k. Must be ordered consistently
            with the routing map indices.
        use_mask_channel: Whether to append a mask channel.

    Returns:
        Sparse tensor of shape (B, M*N, H, W), or (B, M*N+1, H, W) if
        use_mask_channel is True.
    """
    B, C, H, W = quantum_features.shape
    device = quantum_features.device

    sparse = torch.zeros_like(quantum_features)

    for k, channels in enumerate(channel_groups):
        mask = routing_map == k  # (B, H, W)
        if not mask.any():
            continue

        mask_expanded = mask.unsqueeze(1)  # (B, 1, H, W)
        for ch in channels:
            sparse[:, ch, :, :] = torch.where(
                mask_expanded[:, 0, :, :],
                quantum_features[:, ch, :, :],
                sparse[:, ch, :, :],
            )

    if use_mask_channel:
        # Every patch is assigned exactly one kernel, so the mask is always all-ones.
        mask_ch = torch.ones(B, 1, H, W, device=device, dtype=quantum_features.dtype)
        sparse = torch.cat([sparse, mask_ch], dim=1)

    return sparse
