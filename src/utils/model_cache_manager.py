import os
from typing import Any, Dict, Optional

import torch

from src.models.daqcnn import DAQCNN


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> tuple:
    """
    Load a DAQCNN model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on ('cpu', 'cuda', etc.)

    Returns:
        Tuple of (model, checkpoint_dict) where checkpoint_dict contains metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config and metadata
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    num_classes = checkpoint.get("num_classes", model_cfg.get("num_classes", 2))

    # Recreate model
    model = DAQCNN(
        num_classes=num_classes,
        kernel_size=model_cfg.get("kernel_size", 2),
        stride=model_cfg.get("stride", 1),
        kernel_topology_names=model_cfg.get("kernel_topology_names", None),
        scaling_factor=model_cfg.get("scaling_factor", 1.0),
        evolution_time=model_cfg.get("evolution_time", 0.2),
        mode=model_cfg.get("mode", "trotter"),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
        quantum_device=model_cfg.get("quantum_device", "default.qubit"),
        quantum_device_kwargs=model_cfg.get("quantum_device_kwargs", None),
        classical_device=device,
        in_channels=model_cfg.get("in_channels", 1),
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.

    Args:
        checkpoint_path: Path to the .pt checkpoint file

    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    info = {
        "seed": checkpoint.get("seed", "unknown"),
        "num_classes": checkpoint.get("num_classes", "unknown"),
        "config": checkpoint.get("config", {}),
    }

    # Add best_val_loss if it's a best model checkpoint
    if "best_val_loss" in checkpoint:
        info["best_val_loss"] = checkpoint["best_val_loss"]

    return info


def find_model_checkpoints(output_dir: str) -> Dict[str, list]:
    """
    Find all model checkpoints in an output directory.

    Args:
        output_dir: Path to the output directory

    Returns:
        Dictionary with 'best' and 'final' keys, each containing lists of checkpoint paths
    """
    checkpoints = {"best": [], "final": []}

    if not os.path.exists(output_dir):
        return checkpoints

    for filename in os.listdir(output_dir):
        if filename.startswith("best_model_") and filename.endswith(".pt"):
            checkpoints["best"].append(os.path.join(output_dir, filename))
        elif filename.startswith("final_model_") and filename.endswith(".pt"):
            checkpoints["final"].append(os.path.join(output_dir, filename))

    # Sort by filename for consistency
    checkpoints["best"].sort()
    checkpoints["final"].sort()

    return checkpoints


def scan_all_outputs(outputs_root: str = "outputs") -> list:
    """
    Scan all output directories and find available model checkpoints.

    Args:
        outputs_root: Root directory containing all output runs

    Returns:
        List of dictionaries, each containing run info and checkpoint paths
    """
    runs = []

    if not os.path.exists(outputs_root):
        return runs

    for run_dir in sorted(os.listdir(outputs_root)):
        run_path = os.path.join(outputs_root, run_dir)
        if not os.path.isdir(run_path):
            continue

        checkpoints = find_model_checkpoints(run_path)

        # Skip runs with no checkpoints
        if not checkpoints["best"] and not checkpoints["final"]:
            continue

        # Try to load config
        config_path = os.path.join(run_path, "config.yml")
        config = None
        if os.path.exists(config_path):
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        # Try to load aggregate metrics
        metrics_path = os.path.join(run_path, "aggregate_metrics.json")
        metrics = None
        if os.path.exists(metrics_path):
            import json

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        runs.append(
            {
                "run_name": run_dir,
                "run_path": run_path,
                "checkpoints": checkpoints,
                "config": config,
                "metrics": metrics,
            }
        )

    return runs
