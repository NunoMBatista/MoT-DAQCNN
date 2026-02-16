"""
Kernel-to-channel mapping utilities.

Parses the `channel_kernel_map` from cached quantum dataset metadata
and builds convenient lookup structures for grouping channels by kernel topology.
"""


def build_kernel_to_channels_map(channel_kernel_map):
    """Build a mapping from kernel topology name to its channel indices.

    The cached quantum datasets store a `channel_kernel_map` list in their
    metadata. Each entry looks like:
        {"channel": 0, "kernel": "kings", "qubit": 0}

    This function groups those entries by kernel name, returning a dict
    that maps each kernel name to a sorted list of channel indices.

    Args:
        channel_kernel_map: List of dicts from cached dataset metadata.
            Each dict has keys "channel", "kernel", and "qubit".

    Returns:
        dict: ``{"kings": [0, 1, ..., 8], "horizontal": [9, 10, ..., 17], ...}``

    Example:
        >>> meta = metadata['channel_kernel_map']
        >>> k2c = build_kernel_to_channels_map(meta)
        >>> k2c["kings"]
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    kernel_to_channels = {}
    for entry in channel_kernel_map:
        kernel_name = entry["kernel"]
        channel_idx = entry["channel"]
        if kernel_name not in kernel_to_channels:
            kernel_to_channels[kernel_name] = []
        kernel_to_channels[kernel_name].append(channel_idx)

    # Sort channel indices within each group for consistency
    for name in kernel_to_channels:
        kernel_to_channels[name].sort()

    return kernel_to_channels


def get_kernel_names(kernel_to_channels):
    """Return ordered list of kernel names, sorted by first channel index.

    This gives a stable ordering: the kernel whose channels appear first
    in the channel dimension comes first in the list.

    Args:
        kernel_to_channels: Dict from ``build_kernel_to_channels_map()``.

    Returns:
        list[str]: Kernel names ordered by first channel index.
    """
    return sorted(kernel_to_channels.keys(), key=lambda k: kernel_to_channels[k][0])


def get_num_kernels(kernel_to_channels):
    """Return the number of distinct kernel topologies (M)."""
    return len(kernel_to_channels)


def get_channels_per_kernel(kernel_to_channels):
    """Return the number of channels per kernel (N = kernel_size^2).

    Assumes all kernels have the same number of channels.
    Raises ValueError if they don't.
    """
    sizes = [len(chs) for chs in kernel_to_channels.values()]
    if len(set(sizes)) != 1:
        raise ValueError(
            f"Kernels have different channel counts: {dict(zip(kernel_to_channels.keys(), sizes))}. "
            "All kernels must produce the same number of output channels."
        )
    return sizes[0]


def validate_kernel_map(kernel_to_channels, total_channels):
    """Sanity-check that the kernel map covers exactly the expected channels.

    Args:
        kernel_to_channels: Dict from ``build_kernel_to_channels_map()``.
        total_channels: Expected total number of channels (M * N).

    Raises:
        ValueError: If the mapping is inconsistent.
    """
    all_channels = []
    for chs in kernel_to_channels.values():
        all_channels.extend(chs)

    all_channels_sorted = sorted(all_channels)
    expected = list(range(total_channels))

    if all_channels_sorted != expected:
        raise ValueError(
            f"Kernel map channels {all_channels_sorted} don't match "
            f"expected range [0, {total_channels}). "
            f"Got {len(all_channels)} channels, expected {total_channels}."
        )
