"""
Model checkpoint loading and output directory scanning for DAQCNN models.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from src.config import OUTPUTS_DIR
from src.models.daqcnn import DAQCNN

_DAQCNN_PREFIXES = ("best_model_", "final_model_")


def detect_checkpoint_type(checkpoint_path: str) -> str:
    basename = os.path.basename(checkpoint_path)
    if any(basename.startswith(p) for p in _DAQCNN_PREFIXES):
        return "daqcnn"
    return "unknown"


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[torch.nn.Module, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _load_daqcnn(checkpoint, device)
    return model, checkpoint


def _load_daqcnn(checkpoint: dict, device: str) -> DAQCNN:
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    num_classes = checkpoint.get("num_classes", model_cfg.get("num_classes", 2))

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
        include_correlators=model_cfg.get("include_correlators", False),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    info: Dict[str, Any] = {
        "seed": checkpoint.get("seed", "unknown"),
        "num_classes": checkpoint.get("num_classes", "unknown"),
        "config": checkpoint.get("config", {}),
        "type": detect_checkpoint_type(checkpoint_path),
    }

    for key in ("best_val_loss", "agreement", "test_accuracy", "kernel_names"):
        if key in checkpoint:
            info[key] = checkpoint[key]

    return info


def find_model_checkpoints(output_dir: str) -> Dict[str, list]:
    checkpoints: Dict[str, List[str]] = {"best": [], "final": []}

    if not os.path.exists(output_dir):
        return checkpoints

    for fname in os.listdir(output_dir):
        if not fname.endswith(".pt"):
            continue
        filepath = os.path.join(output_dir, fname)
        if "best" in fname:
            checkpoints["best"].append(filepath)
        elif "final" in fname:
            checkpoints["final"].append(filepath)

    checkpoints["best"].sort()
    checkpoints["final"].sort()
    return checkpoints


def scan_all_outputs(outputs_root: Optional[str] = None) -> list:
    if outputs_root is None:
        outputs_root = str(OUTPUTS_DIR)

    runs: List[Dict[str, Any]] = []

    if not os.path.exists(outputs_root):
        return runs

    for run_dir in sorted(os.listdir(outputs_root)):
        run_path = os.path.join(outputs_root, run_dir)
        if not os.path.isdir(run_path):
            continue

        checkpoints = find_model_checkpoints(run_path)
        if not checkpoints["best"] and not checkpoints["final"]:
            continue

        config = _load_yaml_safe(os.path.join(run_path, "config.yml"))
        metrics = _load_json_safe(os.path.join(run_path, "aggregate_metrics.json"))

        model_metadata = None
        individual_results = _load_json_safe(
            os.path.join(run_path, "individual_results.json")
        )
        if individual_results and isinstance(individual_results, list):
            if len(individual_results) > 0:
                model_metadata = individual_results[0].get("model_metadata")

        runs.append({
            "run_name": run_dir,
            "run_path": run_path,
            "checkpoints": checkpoints,
            "config": config,
            "metrics": metrics,
            "model_metadata": model_metadata,
            "architecture": "original",
        })

    return runs


def _load_yaml_safe(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _load_json_safe(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None
