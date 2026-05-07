"""
Quantum dataset cache utilities.

Look up pre-computed quantum datasets in data/quantum_datasets/ and, when one
matches the current config, load it as regular PyTorch DataLoaders so the
training loop can skip the expensive quantum convolution entirely.
"""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import QUANTUM_DATASETS_DIR


def _match_metadata(meta, cfg):
    """
    Check whether the metadata dict from a cached quantum dataset matches
    the quantum-relevant parameters in the training config.

    We compare: dataset_name, image_size, kernel_size, stride,
    kernel_topology_names (order-insensitive or subset), num_kernels,
    scaling_factor, evolution_time, color_space.

    Returns:
        "exact" if perfect match, "subset" if requested kernels are a subset of cached,
        False otherwise.
    """
    from src.config import DATASET_IMAGE_SIZE

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    dataset_name = dataset_cfg.get("name")

    # Dataset name
    if meta.get("dataset_name") != dataset_name:
        return False

    # Image resolution — must match so 28px and 64px caches are never confused.
    # Configs may supply dataset.image_size explicitly; otherwise we derive it
    # from the dataset name (e.g. "oct_mnist_64" → 64).
    config_image_size = dataset_cfg.get(
        "image_size", DATASET_IMAGE_SIZE.get(dataset_name, 28)
    )
    cached_image_size = meta.get("image_size", 28)  # legacy caches omit this → 28
    if cached_image_size != config_image_size:
        return False

    # Color space (default to RGB for backward compatibility)
    cached_color_space = meta.get("color_space", "RGB")
    config_color_space = dataset_cfg.get("color_space", "RGB")
    if cached_color_space != config_color_space:
        return False

    # Kernel size
    if meta.get("kernel_size") != model_cfg.get("kernel_size"):
        return False

    # Stride
    if meta.get("stride") != model_cfg.get("stride"):
        return False

    # Topology names - check for exact match or subset
    cached_topos = sorted(meta.get("kernel_topology_names", []))
    config_topos = sorted(model_cfg.get("kernel_topology_names", []))

    exact_match = cached_topos == config_topos
    subset_match = set(config_topos).issubset(set(cached_topos))

    if not exact_match and not subset_match:
        return False

    # Scaling factor — compare with a small tolerance for float rounding
    if (
        abs(
            float(meta.get("scaling_factor", 0))
            - float(model_cfg.get("scaling_factor", 0))
        )
        > 1e-6
    ):
        return False

    # Evolution time
    if (
        abs(
            float(meta.get("evolution_time", 0))
            - float(model_cfg.get("evolution_time", 0))
        )
        > 1e-6
    ):
        return False

    # ZZ correlators — backward compat: missing field means False
    cached_correlators = meta.get("include_correlators", False)
    config_correlators = model_cfg.get("include_correlators", False)
    if cached_correlators != config_correlators:
        return False

    # Encoding mode — backward compat: missing field means "digital"
    cached_enc = meta.get("encoding_mode", "digital")
    config_enc = model_cfg.get("encoding_mode", "digital")
    if cached_enc != config_enc:
        return False

    # Noise — backward compat: missing field means no noise (False)
    # A clean cache must not be used for a noisy run, and vice-versa.
    cached_noise = meta.get("noise_enabled", False)
    config_noise  = cfg.get("noise", {}).get("enabled", False)
    if cached_noise != config_noise:
        return False
    if config_noise:
        # Both are noisy — also match the physical parameters
        noise_cfg = cfg["noise"]
        for meta_key, cfg_key in (
            ("noise_T1_us",     "T1_us"),
            ("noise_T2_us",     "T2_us"),
            ("noise_p_gate_1q", "p_gate_1q"),
            ("noise_omega_mhz", "omega_mhz"),
        ):
            if abs(float(meta.get(meta_key) or 0) - float(noise_cfg.get(cfg_key, 0))) > 1e-6:
                return False

    return "exact" if exact_match else "subset"


def find_cached_quantum_dataset(cfg, datasets_dir=None):
    """
    Scan ``data/quantum_datasets/`` for a ``.json`` metadata sidecar whose
    parameters match *cfg*.  Returns the path to the corresponding ``.npz``
    file, or ``None`` if nothing matches.

    Prefers exact matches, but will return a superset match if available
    (the loader will extract only the needed kernel channels).
    """
    if datasets_dir is None:
        datasets_dir = QUANTUM_DATASETS_DIR

    if not datasets_dir.is_dir():
        return None

    exact_match = None
    subset_match = None

    for json_path in sorted(datasets_dir.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        match_type = _match_metadata(meta, cfg)
        if match_type:
            npz_path = json_path.with_suffix(".npz")
            if npz_path.is_file():
                if match_type == "exact":
                    exact_match = npz_path
                    break  # Exact match found, stop searching
                elif match_type == "subset":
                    subset_match = npz_path

    # Prefer exact match, fall back to subset
    return exact_match if exact_match else subset_match


def load_cached_quantum_dataset(
    npz_path, batch_size=32, num_workers=2, requested_kernels=None
):
    """
    Load a cached quantum ``.npz`` and return DataLoaders that yield
    ``(features, labels)`` tensors, plus the number of classes and the
    metadata dict.

    Args:
        npz_path: Path to the cached .npz file.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of workers for DataLoaders.
        requested_kernels: Optional list of kernel names to extract. If provided
            and different from cached kernels, will extract only the channels
            corresponding to the requested kernels. If None, loads all kernels.

    Returns:
        (train_loader, val_loader, test_loader, n_classes, metadata)
    """
    data = np.load(npz_path, allow_pickle=True)

    # Parse metadata
    metadata = json.loads(str(data["metadata"]))

    # Check if we need to extract a subset of kernels
    cached_kernels = metadata.get("kernel_topology_names", [])
    extract_subset = (
        requested_kernels is not None
        and set(requested_kernels) != set(cached_kernels)
        and set(requested_kernels).issubset(set(cached_kernels))
    )

    if extract_subset:
        # Build channel mapping to extract only requested kernel channels
        from src.utils.kernel_mapping import build_kernel_to_channels_map

        cached_channel_map = metadata["channel_kernel_map"]
        kernel_to_channels = build_kernel_to_channels_map(cached_channel_map)

        # Collect channel indices for requested kernels.
        # IMPORTANT: iterate in channel-index order (the order kernels appear in
        # the feature tensor), NOT alphabetical order.  get_kernel_names() uses
        # the same ordering (sorted by first channel index), so the rebuilt
        # channel_kernel_map stays consistent with the multi-kernel channel layout
        # derived from it.  Alphabetical order would silently misalign channels
        # whenever kernel names don't sort the same way as their channel positions.
        channel_indices = []
        new_channel_map = []
        new_out_idx = 0

        # Sort requested kernels by their first channel index in the cached dataset
        requested_sorted_by_channel = sorted(
            requested_kernels, key=lambda name: kernel_to_channels[name][0]
        )

        for kernel_name in requested_sorted_by_channel:
            if kernel_name in kernel_to_channels:
                old_indices = sorted(kernel_to_channels[kernel_name])
                channel_indices.extend(old_indices)

                # Update channel mapping for the subset (preserve dict format)
                for old_idx in old_indices:
                    # Find the original entry to get qubit info
                    original_entry = next(
                        e for e in cached_channel_map if e["channel"] == old_idx
                    )
                    new_entry = dict(original_entry)
                    new_entry["channel"] = new_out_idx
                    new_channel_map.append(new_entry)
                    new_out_idx += 1

        # Update metadata to reflect the subset
        metadata = metadata.copy()
        metadata["kernel_topology_names"] = requested_sorted_by_channel
        metadata["num_kernels"] = len(requested_kernels)
        metadata["out_channels"] = len(channel_indices)
        metadata["channel_kernel_map"] = new_channel_map

        # Log subset extraction info
        print(
            f"- Extracting subset from cached dataset:"
            f"{len(requested_kernels)}/{len(cached_kernels)} kernels "
            f"({len(channel_indices)}/{len(cached_channel_map)} channels)"
        )
        print(f"   Requested: {requested_sorted_by_channel}")
        print(f"   Available: {sorted(cached_kernels)}")
    else:
        channel_indices = None  # Use all channels

    def _make_loader(split, shuffle):
        features = torch.from_numpy(data[f"{split}_features"]).float()

        # Extract subset of channels if needed
        if channel_indices is not None:
            features = features[:, channel_indices, :, :]

        labels = torch.from_numpy(data[f"{split}_labels"]).long()
        ds = TensorDataset(features, labels)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    train_loader = _make_loader("train", shuffle=True)
    val_loader = _make_loader("val", shuffle=False)
    test_loader = _make_loader("test", shuffle=False)

    n_classes = len(set(data["train_labels"].tolist()))

    return train_loader, val_loader, test_loader, n_classes, metadata
