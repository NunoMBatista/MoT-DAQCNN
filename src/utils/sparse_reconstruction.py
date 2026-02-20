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

from src.utils.kernel_mapping import (
    build_kernel_to_channels_map,
    get_channels_per_kernel,
    get_kernel_names,
    get_num_kernels,
)


def build_sparse_tensor(
    quantum_features, routing_map, kernel_to_channels, use_mask_channel=False
):
    """Build a sparse tensor by routing each patch to its selected kernel.

    For every spatial position (i, j), the routing map says which kernel k
    was selected. We keep that kernel's channels and zero out the rest.

    Args:
        quantum_features: Tensor of shape (B, M*N, H, W) with all kernels'
            quantum outputs. Same format as cached quantum dataset features.
        routing_map: LongTensor of shape (B, H, W) with values in {0, ..., M-1},
            indicating the selected kernel at each spatial position.
        kernel_to_channels: Dict mapping kernel name -> list of channel indices.
            Built from ``build_kernel_to_channels_map()``.
        use_mask_channel: If True, append an extra channel that is 1.0 where
            data is present and 0.0 where masked. This helps the classifier
            distinguish "zero because masked" from "zero because signal is zero".

    Returns:
        Sparse tensor of shape (B, C_out, H, W) where C_out = M*N if
        use_mask_channel is False, or M*N + 1 if True.
    """
    B, C, H, W = quantum_features.shape
    device = quantum_features.device

    kernel_names = get_kernel_names(kernel_to_channels)
    num_kernels = get_num_kernels(kernel_to_channels)

    # Build ordered list of channel index groups (one per kernel)
    channel_groups = [kernel_to_channels[name] for name in kernel_names]

    # Start with all zeros
    sparse = torch.zeros_like(quantum_features)

    # For each kernel, find patches assigned to it and copy those channels
    for k in range(num_kernels):
        # Boolean mask: which patches chose kernel k — shape (B, H, W)
        mask = routing_map == k
        if not mask.any():
            continue

        channels = channel_groups[k]
        # Expand mask to cover the N channels of this kernel group
        # mask: (B, H, W) -> (B, 1, H, W) -> broadcasts over channels
        mask_expanded = mask.unsqueeze(1)  # (B, 1, H, W)

        for ch in channels:
            sparse[:, ch, :, :] = torch.where(
                mask_expanded[:, 0, :, :],
                quantum_features[:, ch, :, :],
                sparse[:, ch, :, :],
            )

    if use_mask_channel:
        # Build a single mask channel: 1.0 wherever any kernel was active
        # (which is everywhere, since every patch gets exactly one kernel)
        mask_ch = torch.ones(B, 1, H, W, device=device, dtype=quantum_features.dtype)
        sparse = torch.cat([sparse, mask_ch], dim=1)

    return sparse


def build_sparse_tensor_fast(
    quantum_features, routing_map, channel_groups, use_mask_channel=False
):
    """Same as ``build_sparse_tensor`` but takes pre-computed channel groups.

    This avoids repeated dict lookups when called in a training loop.

    Args:
        quantum_features: (B, M*N, H, W) tensor.
        routing_map: (B, H, W) LongTensor with kernel indices.
        channel_groups: List of lists, where channel_groups[k] is the list
            of channel indices for kernel k. Must be ordered consistently
            with the routing map indices.
        use_mask_channel: Whether to append a mask channel.

    Returns:
        Sparse tensor of shape (B, C_out, H, W).
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
        mask_ch = torch.ones(B, 1, H, W, device=device, dtype=quantum_features.dtype)
        sparse = torch.cat([sparse, mask_ch], dim=1)

    return sparse


def route_dataset(
    quantum_loader,
    student,
    raw_image_datasets,
    kernel_size,
    stride,
    color_space,
    kernel_to_channels,
    device,
    use_mask_channel=False,
    batch_size=32,
):
    """Route an entire dataset through the Student and build sparse tensors.

    This is the full Phase 3 pipeline for one split:
        1. Extract raw patches from original images.
        2. Run the Student on each patch to get routing decisions.
        3. Build sparse tensors from quantum features + routing map.

    Args:
        quantum_loader: DataLoader yielding (quantum_features, class_labels).
            Must be unshuffled so indices align with raw_image_datasets.
        student: Trained StudentGatekeeper model (set to eval mode).
        raw_image_datasets: Raw image dataset (indexable, returns (image, label)).
        kernel_size: Patch size for extraction.
        stride: Stride for patch extraction.
        color_space: Color space used ("RGB", "HSV", "GRAYSCALE").
        kernel_to_channels: Dict mapping kernel name -> channel indices.
        device: Device string.
        use_mask_channel: Whether to add mask channel.
        batch_size: Batch size for student inference.

    Returns:
        sparse_features: Tensor (N, C_out, H, W) with routed quantum features.
        routing_maps: LongTensor (N, H, W) with routing decisions.
        class_labels: LongTensor (N,) with original class labels.
    """
    from src.models.student_training import extract_patches
    from src.utils.color_conversion import apply_color_conversion

    student.eval()
    kernel_names = get_kernel_names(kernel_to_channels)
    channel_groups = [kernel_to_channels[name] for name in kernel_names]

    all_sparse = []
    all_routing = []
    all_labels = []

    q_idx = 0  # track position in quantum dataset
    for q_features, q_labels in quantum_loader:
        B = q_features.shape[0]
        _, _, H, W = q_features.shape

        # Get corresponding raw images for patch extraction
        raw_images = []
        for i in range(B):
            img, _ = raw_image_datasets[q_idx + i]
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            raw_images.append(img)
        raw_images = torch.stack(raw_images)
        q_idx += B

        # Apply same color conversion as quantum pipeline
        raw_images = apply_color_conversion(raw_images, color_space)
        if color_space == "HSV" and raw_images.shape[1] == 3:
            raw_images = raw_images[:, 2:3, :, :]

        # Extract patches: (B, n_patches, patch_dim)
        patches = extract_patches(raw_images, kernel_size, stride)
        n_patches = patches.shape[1]

        # Flatten to (B * n_patches, patch_dim) for student
        flat_patches = patches.reshape(-1, patches.shape[-1]).to(device)

        # Student routing predictions
        with torch.no_grad():
            routing_flat = student.predict(flat_patches).cpu()  # (B * n_patches,)

        # Reshape to (B, H, W)
        routing_map = routing_flat.reshape(B, H, W)

        # Build sparse tensor
        sparse = build_sparse_tensor_fast(
            q_features, routing_map, channel_groups, use_mask_channel
        )

        all_sparse.append(sparse)
        all_routing.append(routing_map)

        labels = q_labels.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        all_labels.append(labels)

    sparse_features = torch.cat(all_sparse, dim=0)
    routing_maps = torch.cat(all_routing, dim=0)
    class_labels = torch.cat(all_labels, dim=0).long()

    return sparse_features, routing_maps, class_labels
