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


def build_sparse_tensor_fast(quantum_features, routing_map, channel_groups, group_norms=None):
    """Build a sparse tensor by routing each patch to its selected kernel.

    For every spatial position (i, j), the routing map says which kernel k
    was selected. We keep that kernel's channels and zero out the rest.

    When ``group_norms`` is provided, each kernel group's channels are
    normalised BEFORE masking.  This matches the Teacher's behavior, where
    per-group BatchNorm is applied before routing weights, and ensures the
    Final Classifier sees the same feature scale the Teacher learned.

    Args:
        quantum_features: (B, M*N, H, W) tensor with all kernels' quantum outputs.
        routing_map: (B, H, W) LongTensor with values in {0, ..., M-1},
            indicating the selected kernel at each spatial position.
        channel_groups: List of lists, where channel_groups[k] is the list
            of channel indices for kernel k. Must be ordered consistently
            with the routing map indices.
        group_norms: Optional nn.ModuleList of BatchNorm2d layers (one per
            kernel group).  When provided, normalises each group's channels
            before applying the sparse mask.  Should be the Teacher's trained
            ``attention_block.group_norms`` for consistency.

    Returns:
        Sparse tensor of shape (B, M*N, H, W) where only the selected
        kernel's channels are non-zero at each spatial position.
    """
    B, C, H, W = quantum_features.shape

    # --- Per-group normalisation (matches Teacher behavior) ---
    # Normalise ALL groups first, THEN apply sparse mask.  This way the
    # BatchNorm statistics are computed over the full (non-sparse) features,
    # exactly as the Teacher does.
    if group_norms is not None:
        # Move features to same device as group_norms (they may be on CUDA)
        norm_device = next(group_norms.parameters()).device
        quantum_features = quantum_features.to(norm_device)
        routing_map = routing_map.to(norm_device)
        channels_per_kernel = len(channel_groups[0])
        groups = quantum_features.split(channels_per_kernel, dim=1)
        normed_groups = [group_norms[k](g) for k, g in enumerate(groups)]
        quantum_features = torch.cat(normed_groups, dim=1)

    device = quantum_features.device
    routing_map = routing_map.to(device)  # Ensure routing_map is on same device
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
