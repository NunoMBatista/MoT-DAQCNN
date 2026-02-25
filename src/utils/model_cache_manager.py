"""
Model checkpoint loading and output directory scanning.

Supports both the original DAQCNN models and TS-MoE pipeline models
(Teacher, Student Gatekeeper, Final Classifier). Automatically detects
the model type from checkpoint contents and reconstructs the correct
architecture.

Output directory scanning handles both flat layouts (original DAQCNN)
and nested TS-MoE layouts (seed_N/teacher|student|final_classifier/).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from src.config import OUTPUTS_DIR
from src.models.daqcnn import DAQCNN

# -------------------------------------------------------------------------
# Checkpoint type detection
# -------------------------------------------------------------------------

# Checkpoint filename prefixes for each model type
_DAQCNN_PREFIXES = ("best_model_", "final_model_")
_TEACHER_PREFIXES = ("teacher_best_seed_", "teacher_final_seed_")
_STUDENT_PREFIXES = ("student_best_seed_", "student_final_seed_")
_FINAL_CLF_PREFIXES = (
    "final_classifier_seed_",
    "final_classifier_best_seed_",
)


def detect_checkpoint_type(checkpoint_path: str) -> str:
    """Detect the model type from a checkpoint filename.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint file.

    Returns:
        One of ``"daqcnn"``, ``"teacher"``, ``"student"``,
        ``"final_classifier"``, or ``"unknown"``.
    """
    basename = os.path.basename(checkpoint_path)
    if any(basename.startswith(p) for p in _TEACHER_PREFIXES):
        return "teacher"
    if any(basename.startswith(p) for p in _STUDENT_PREFIXES):
        return "student"
    if any(basename.startswith(p) for p in _FINAL_CLF_PREFIXES):
        return "final_classifier"
    if any(basename.startswith(p) for p in _DAQCNN_PREFIXES):
        return "daqcnn"
    return "unknown"


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[torch.nn.Module, dict]:
    """Load a model from a checkpoint file, auto-detecting the architecture.

    Supports original DAQCNN, TS-MoE Teacher, Student Gatekeeper, and
    Final Classifier checkpoints.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Device to load the model on (``"cpu"``, ``"cuda"``, etc.).

    Returns:
        Tuple of ``(model, checkpoint_dict)`` where ``checkpoint_dict``
        contains all saved metadata (config, seed, metrics, etc.).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the checkpoint type cannot be determined.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_type = detect_checkpoint_type(checkpoint_path)

    if ckpt_type == "teacher":
        model = _load_teacher(checkpoint, device)
    elif ckpt_type == "student":
        model = _load_student(checkpoint, device)
    elif ckpt_type == "final_classifier":
        model = _load_final_classifier(checkpoint, device)
    elif ckpt_type == "daqcnn":
        model = _load_daqcnn(checkpoint, device)
    else:
        # Fallback: try DAQCNN (legacy behaviour)
        model = _load_daqcnn(checkpoint, device)

    return model, checkpoint


def _load_daqcnn(checkpoint: dict, device: str) -> DAQCNN:
    """Reconstruct a DAQCNN model from a checkpoint dict."""
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
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_teacher(checkpoint: dict, device: str) -> torch.nn.Module:
    """Reconstruct a TeacherMoE model from a checkpoint dict."""
    from src.models.teacher_moe import build_teacher_from_metadata

    metadata = checkpoint["metadata"]
    num_classes = checkpoint.get("num_classes", 2)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})

    model = build_teacher_from_metadata(
        metadata,
        num_classes=num_classes,
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
    )

    # Initialize lazy layers before loading state dict
    total_ch = metadata["out_channels"]
    ks = metadata["kernel_size"]
    stride = metadata["stride"]
    spatial = (28 - ks) // stride + 1
    dummy = torch.randn(1, total_ch, spatial, spatial)
    with torch.no_grad():
        model(dummy)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_student(checkpoint: dict, device: str) -> torch.nn.Module:
    """Reconstruct a StudentGatekeeper model from a checkpoint dict."""
    from src.models.student_gatekeeper import StudentGatekeeper

    model = StudentGatekeeper(
        patch_dim=checkpoint["patch_dim"],
        num_kernels=checkpoint["num_kernels"],
        hidden_dims=checkpoint.get("hidden_dims", (32, 16)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_final_classifier(checkpoint: dict, device: str) -> torch.nn.Module:
    """Reconstruct a FinalClassifier model from a checkpoint dict."""
    from src.models.ts_moe_classification_head import FinalClassifier

    metadata = checkpoint["metadata"]
    num_classes = checkpoint.get("num_classes", 2)
    use_mask = checkpoint.get("use_mask_channel", False)
    total_ch = metadata["out_channels"]
    in_channels = total_ch + (1 if use_mask else 0)

    model = FinalClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
    )

    # Initialize lazy layers before loading state dict
    ks = metadata["kernel_size"]
    stride = metadata["stride"]
    spatial = (28 - ks) // stride + 1
    dummy = torch.randn(1, in_channels, spatial, spatial)
    with torch.no_grad():
        model(dummy)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------------------------------
# Checkpoint info (lightweight, no full model load)
# -------------------------------------------------------------------------


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Get metadata about a checkpoint without loading the full model.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.

    Returns:
        Dictionary with ``seed``, ``num_classes``, ``config``, ``type``,
        and optional keys like ``best_val_loss`` or ``agreement``.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    info: Dict[str, Any] = {
        "seed": checkpoint.get("seed", "unknown"),
        "num_classes": checkpoint.get("num_classes", "unknown"),
        "config": checkpoint.get("config", {}),
        "type": detect_checkpoint_type(checkpoint_path),
    }

    if "best_val_loss" in checkpoint:
        info["best_val_loss"] = checkpoint["best_val_loss"]

    if "agreement" in checkpoint:
        info["agreement"] = checkpoint["agreement"]

    if "test_accuracy" in checkpoint:
        info["test_accuracy"] = checkpoint["test_accuracy"]

    if "kernel_names" in checkpoint:
        info["kernel_names"] = checkpoint["kernel_names"]

    return info


# -------------------------------------------------------------------------
# Checkpoint discovery
# -------------------------------------------------------------------------


def find_model_checkpoints(output_dir: str) -> Dict[str, list]:
    """Find all model checkpoints in an output directory.

    Handles both flat directory layouts (original DAQCNN with
    ``best_model_*`` / ``final_model_*``) and TS-MoE layouts where
    checkpoints live in ``seed_N/teacher/``, ``seed_N/student/``, and
    ``seed_N/final_classifier/`` subdirectories.

    Args:
        output_dir: Path to the output directory.

    Returns:
        Dictionary with ``"best"`` and ``"final"`` keys, each containing
        a sorted list of checkpoint file paths.
    """
    checkpoints: Dict[str, List[str]] = {"best": [], "final": []}

    if not os.path.exists(output_dir):
        return checkpoints

    # Collect .pt files from the top level and from TS-MoE nested dirs
    pt_files = _collect_pt_files(output_dir)

    for filepath in pt_files:
        basename = os.path.basename(filepath)
        if "best" in basename:
            checkpoints["best"].append(filepath)
        elif "final" in basename or basename.startswith("final_classifier_seed_"):
            checkpoints["final"].append(filepath)

    checkpoints["best"].sort()
    checkpoints["final"].sort()

    return checkpoints


def _collect_pt_files(output_dir: str) -> List[str]:
    """Recursively collect ``.pt`` checkpoint files from a run directory.

    Searches the top-level directory and up to two levels deep to catch
    TS-MoE's ``seed_N/phase/`` structure without doing a full recursive
    walk of unrelated subdirectories.
    """
    pt_files: List[str] = []

    if not os.path.isdir(output_dir):
        return pt_files

    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)

        if entry.endswith(".pt") and os.path.isfile(entry_path):
            pt_files.append(entry_path)

        elif os.path.isdir(entry_path) and entry.startswith("seed_"):
            # TS-MoE layout: seed_N/{teacher,student,final_classifier}/
            for phase_dir in os.listdir(entry_path):
                phase_path = os.path.join(entry_path, phase_dir)
                if os.path.isdir(phase_path):
                    for fname in os.listdir(phase_path):
                        if fname.endswith(".pt"):
                            pt_files.append(os.path.join(phase_path, fname))

    return pt_files


# -------------------------------------------------------------------------
# Full output directory scanning
# -------------------------------------------------------------------------


def scan_all_outputs(outputs_root: Optional[str] = None) -> list:
    """Scan all output directories and find available model checkpoints.

    Discovers both original DAQCNN runs and TS-MoE pipeline runs.  For
    TS-MoE runs the architecture is tagged as ``"TS-MoE"`` and extra
    pipeline summary data (teacher accuracy, student agreement, etc.) is
    attached.

    Args:
        outputs_root: Root directory containing all output runs. Defaults
            to the project's ``outputs/`` directory.

    Returns:
        List of dicts, each with keys: ``run_name``, ``run_path``,
        ``checkpoints``, ``config``, ``metrics``, ``model_metadata``,
        and ``architecture``.
    """
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

        # Skip runs with no checkpoints
        if not checkpoints["best"] and not checkpoints["final"]:
            continue

        # Detect architecture from config or directory structure
        config = _load_yaml_safe(os.path.join(run_path, "config.yml"))
        architecture = _detect_run_architecture(run_path, config)

        # Load aggregate metrics
        metrics = _load_json_safe(os.path.join(run_path, "aggregate_metrics.json"))

        # Load model metadata from individual results
        model_metadata = None
        individual_results = _load_json_safe(
            os.path.join(run_path, "individual_results.json")
        )
        if individual_results and isinstance(individual_results, list):
            if len(individual_results) > 0:
                model_metadata = individual_results[0].get("model_metadata")

        # For TS-MoE runs, if metadata not found, try loading from teacher checkpoint
        if architecture == "TS-MoE" and model_metadata is None:
            model_metadata = _load_metadata_from_teacher_checkpoint(run_path)

        # For TS-MoE runs, load pipeline summary for extra info
        pipeline_summary = None
        if architecture == "TS-MoE":
            pipeline_summary = _load_ts_moe_summary(run_path)

        run_info: Dict[str, Any] = {
            "run_name": run_dir,
            "run_path": run_path,
            "checkpoints": checkpoints,
            "config": config,
            "metrics": metrics,
            "model_metadata": model_metadata,
            "architecture": architecture,
        }
        if pipeline_summary is not None:
            run_info["pipeline_summary"] = pipeline_summary

        runs.append(run_info)

    return runs


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------


def _detect_run_architecture(run_path: str, config: Optional[dict]) -> str:
    """Detect whether a run is original DAQCNN or TS-MoE.

    Checks the config first, then falls back to directory structure
    heuristics (presence of ``seed_N/teacher/`` dirs).
    """
    if config:
        arch = config.get("model", {}).get("architecture", "original")
        if arch == "TS-MoE":
            return "TS-MoE"

    # Heuristic: look for seed_*/teacher/ dirs
    for entry in os.listdir(run_path):
        entry_path = os.path.join(run_path, entry)
        if os.path.isdir(entry_path) and entry.startswith("seed_"):
            if os.path.isdir(os.path.join(entry_path, "teacher")):
                return "TS-MoE"

    return "original"


def _load_ts_moe_summary(run_path: str) -> Optional[dict]:
    """Load the first pipeline_summary.json found in a TS-MoE run."""
    for entry in sorted(os.listdir(run_path)):
        seed_path = os.path.join(run_path, entry)
        if os.path.isdir(seed_path) and entry.startswith("seed_"):
            summary_path = os.path.join(seed_path, "pipeline_summary.json")
            summary = _load_json_safe(summary_path)
            if summary is not None:
                return summary

    # Also check aggregate_results.json at run root
    return _load_json_safe(os.path.join(run_path, "aggregate_results.json"))


def _load_yaml_safe(path: str) -> Optional[dict]:
    """Load a YAML file, returning None if it doesn't exist or fails."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _load_json_safe(path: str) -> Optional[Any]:
    """Load a JSON file, returning None if it doesn't exist or fails."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _load_metadata_from_teacher_checkpoint(run_path: str) -> Optional[dict]:
    """Load metadata from the first teacher checkpoint in a TS-MoE run.

    Args:
        run_path: Path to the TS-MoE run directory.

    Returns:
        Dictionary containing model metadata, or None if not found.
    """
    for entry in sorted(os.listdir(run_path)):
        seed_path = os.path.join(run_path, entry)
        if os.path.isdir(seed_path) and entry.startswith("seed_"):
            teacher_dir = os.path.join(seed_path, "teacher")
            if not os.path.isdir(teacher_dir):
                continue

            # Find teacher checkpoint
            for fname in sorted(os.listdir(teacher_dir)):
                if fname.endswith(".pt") and fname.startswith("teacher_"):
                    teacher_ckpt_path = os.path.join(teacher_dir, fname)
                    try:
                        ckpt = torch.load(
                            teacher_ckpt_path, map_location="cpu", weights_only=False
                        )
                        metadata = ckpt.get("metadata", {})
                        if metadata:
                            return metadata
                    except Exception:
                        continue

    return None
