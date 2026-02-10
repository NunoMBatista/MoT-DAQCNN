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

    We compare: dataset_name, kernel_size, stride, kernel_topology_names
    (order-insensitive), num_kernels, scaling_factor, evolution_time, color_space.
    """
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    dataset_name = dataset_cfg.get("name")

    # Dataset name
    if meta.get("dataset_name") != dataset_name:
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

    # Topology names (sorted so order doesn't matter)
    cached_topos = sorted(meta.get("kernel_topology_names", []))
    config_topos = sorted(model_cfg.get("kernel_topology_names", []))
    if cached_topos != config_topos:
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

    return True


def find_cached_quantum_dataset(cfg, datasets_dir=None):
    """
    Scan ``data/quantum_datasets/`` for a ``.json`` metadata sidecar whose
    parameters match *cfg*.  Returns the path to the corresponding ``.npz``
    file, or ``None`` if nothing matches.
    """
    if datasets_dir is None:
        datasets_dir = QUANTUM_DATASETS_DIR

    if not datasets_dir.is_dir():
        return None

    for json_path in sorted(datasets_dir.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if _match_metadata(meta, cfg):
            npz_path = json_path.with_suffix(".npz")
            if npz_path.is_file():
                return npz_path

    return None


def load_cached_quantum_dataset(npz_path, batch_size=32, num_workers=2):
    """
    Load a cached quantum ``.npz`` and return DataLoaders that yield
    ``(features, labels)`` tensors, plus the number of classes and the
    metadata dict.

    Returns:
        (train_loader, val_loader, test_loader, n_classes, metadata)
    """
    data = np.load(npz_path, allow_pickle=True)

    # Parse metadata
    metadata = json.loads(str(data["metadata"]))

    def _make_loader(split, shuffle):
        features = torch.from_numpy(data[f"{split}_features"]).float()
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
