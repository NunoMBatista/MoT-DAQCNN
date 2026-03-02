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


def build_sparse_tensor_fast(quantum_features, routing_map, channel_groups):
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

    Returns:
        Sparse tensor of shape (B, M*N, H, W) where only the selected
        kernel's channels are non-zero at each spatial position.
    """
    B, C, H, W = quantum_features.shape
    device = quantum_features.device

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
