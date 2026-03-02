"""
Phase 6 TUI integration tests: TS-MoE model loading and output scanning.

Verifies that the model_cache_manager correctly:
    - Detects checkpoint types from filenames
    - Loads Teacher, Student, and Final Classifier models from checkpoints
    - Scans TS-MoE output directory structures (seed_N/phase/)
    - Detects architecture from config and directory heuristics
    - Returns pipeline summary data for TS-MoE runs

Run with:
    python -m pytest tests/test_phase6_tui_integration.py -v
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.train_ts_moe import run_ts_moe_pipeline  # noqa: E402
from src.utils.model_cache_manager import (  # noqa: E402
    _collect_pt_files,
    _detect_run_architecture,
    detect_checkpoint_type,
    find_model_checkpoints,
    get_checkpoint_info,
    load_model_from_checkpoint,
    scan_all_outputs,
)

# =========================================================================
# Constants
# =========================================================================

KERNEL_NAMES = ["kings", "horizontal"]
KERNEL_SIZE = 2
NUM_CLASSES = 2


# =========================================================================
# Helpers
# =========================================================================


def _create_fake_cached_dataset(tmpdir):
    """Write a minimal synthetic quantum dataset (npz + json) into tmpdir."""
    n_qubits = KERNEL_SIZE * KERNEL_SIZE
    M = len(KERNEL_NAMES)
    total_channels = M * n_qubits
    stride = KERNEL_SIZE
    spatial = (28 - KERNEL_SIZE) // stride + 1

    n_train, n_val, n_test = 24, 12, 12

    channel_kernel_map = []
    ch = 0
    for name in KERNEL_NAMES:
        for q in range(n_qubits):
            channel_kernel_map.append({"channel": ch, "kernel": name, "qubit": q})
            ch += 1

    metadata = {
        "dataset_name": "pneumonia_mnist",
        "kernel_size": KERNEL_SIZE,
        "stride": stride,
        "kernel_topology_names": KERNEL_NAMES,
        "num_kernels": M,
        "scaling_factor": 1.0,
        "evolution_time": 0.2,
        "out_channels": total_channels,
        "channel_kernel_map": channel_kernel_map,
        "color_space": "GRAYSCALE",
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
    }

    rng = np.random.RandomState(7)
    datasets_dir = Path(tmpdir) / "data" / "quantum_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    npz_path = str(datasets_dir / "fake_dataset.npz")
    np.savez_compressed(
        npz_path,
        train_features=rng.randn(n_train, total_channels, spatial, spatial).astype(
            np.float32
        ),
        train_labels=rng.randint(0, NUM_CLASSES, n_train).astype(np.int64),
        val_features=rng.randn(n_val, total_channels, spatial, spatial).astype(
            np.float32
        ),
        val_labels=rng.randint(0, NUM_CLASSES, n_val).astype(np.int64),
        test_features=rng.randn(n_test, total_channels, spatial, spatial).astype(
            np.float32
        ),
        test_labels=rng.randint(0, NUM_CLASSES, n_test).astype(np.int64),
        metadata=json.dumps(metadata),
    )

    json_path = str(datasets_dir / "fake_dataset.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    cfg = {
        "dataset": {
            "name": "pneumonia_mnist",
            "data_root": str(Path(tmpdir) / "data"),
            "batch_size": 8,
            "num_workers": 0,
            "color_space": "GRAYSCALE",
        },
        "model": {
            "architecture": "TS-MoE",
            "num_classes": NUM_CLASSES,
            "kernel_size": KERNEL_SIZE,
            "stride": KERNEL_SIZE,
            "kernel_topology_names": KERNEL_NAMES,
            "scaling_factor": 1.0,
            "evolution_time": 0.2,
            "dropout": 0.1,
            "activation": "relu",
            "classical_device": "cpu",
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 2,
            "grad_clip": 1.0,
            "patience": None,
            "use_scheduler": False,
        },
        "ts_moe": {
            "lambda_max": 0.1,
            "lambda_warmup_epochs": 1,
            "student_epochs": 2,
            "student_lr": 1e-3,
            "student_hidden_dims": [16, 8],
            "final_epochs": 2,
            "final_lr": 1e-3,
            "confidence_threshold": 0.0,
        },
        "misc": {"seed": 7},
    }

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    return datasets_dir, cfg, split_counts


def _make_synthetic_images(split_counts):
    """Create synthetic raw image TensorDatasets matching cached sizes."""
    rng = torch.Generator().manual_seed(7)
    datasets = {}
    for split in ("train", "val", "test"):
        n = split_counts[split]
        images = torch.rand(n, 1, 28, 28, generator=rng)
        labels = torch.randint(0, NUM_CLASSES, (n, 1), generator=rng)
        datasets[split] = TensorDataset(images, labels)
    return datasets


# =========================================================================
# Fixture: run pipeline once and keep outputs for scanning tests
# =========================================================================


@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory):
    """Run the full TS-MoE pipeline once, return paths and results."""
    tmpdir = str(tmp_path_factory.mktemp("phase6"))
    datasets_dir, cfg, split_counts = _create_fake_cached_dataset(tmpdir)
    raw_datasets = _make_synthetic_images(split_counts)

    # Use a realistic output directory structure
    outputs_root = os.path.join(tmpdir, "outputs")
    run_dir = os.path.join(outputs_root, "moe_run_20250101_000000")
    os.makedirs(run_dir, exist_ok=True)

    # Save config (as the experiment runner would)
    import yaml

    config_path = os.path.join(run_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    result = run_ts_moe_pipeline(
        cfg,
        seed=7,
        output_dir=run_dir,
        verbose=False,
        datasets_dir=datasets_dir,
        raw_image_datasets=raw_datasets,
    )

    seed_dir = os.path.join(run_dir, "seed_7")

    return {
        "result": result,
        "outputs_root": outputs_root,
        "run_dir": run_dir,
        "seed_dir": seed_dir,
        "cfg": cfg,
    }


# =========================================================================
# Tests: Checkpoint type detection
# =========================================================================


class TestCheckpointTypeDetection:
    """Verify detect_checkpoint_type identifies models from filenames."""

    def test_daqcnn_best(self):
        assert detect_checkpoint_type("outputs/run_1/best_model_seed_0.pt") == "daqcnn"

    def test_daqcnn_final(self):
        assert detect_checkpoint_type("outputs/run_1/final_model_seed_0.pt") == "daqcnn"

    def test_teacher_best(self):
        assert (
            detect_checkpoint_type("seed_0/teacher/teacher_best_seed_0.pt") == "teacher"
        )

    def test_teacher_final(self):
        assert (
            detect_checkpoint_type("seed_0/teacher/teacher_final_seed_0.pt")
            == "teacher"
        )

    def test_student_best(self):
        assert (
            detect_checkpoint_type("seed_0/student/student_best_seed_0.pt") == "student"
        )

    def test_student_final(self):
        assert (
            detect_checkpoint_type("seed_0/student/student_final_seed_0.pt")
            == "student"
        )

    def test_final_classifier(self):
        assert (
            detect_checkpoint_type("final_classifier/final_classifier_seed_0.pt")
            == "final_classifier"
        )

    def test_final_classifier_best(self):
        assert (
            detect_checkpoint_type("final_classifier/final_classifier_best_seed_0.pt")
            == "final_classifier"
        )

    def test_unknown(self):
        assert detect_checkpoint_type("some_random_model.pt") == "unknown"

    def test_basename_only(self):
        """Should work with just a filename, no directory prefix."""
        assert detect_checkpoint_type("teacher_final_seed_42.pt") == "teacher"


# =========================================================================
# Tests: Model loading from checkpoints
# =========================================================================


class TestTeacherModelLoading:
    """Verify Teacher model loads correctly from checkpoint."""

    def test_load_teacher_checkpoint(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(seed_dir, "teacher", "teacher_final_seed_7.pt")

        model, ckpt = load_model_from_checkpoint(ckpt_path, device="cpu")

        # Should be a TeacherMoE instance
        from src.models.teacher_moe import TeacherMoE

        assert isinstance(model, TeacherMoE)

    def test_teacher_inference_works(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(seed_dir, "teacher", "teacher_final_seed_7.pt")

        model, ckpt = load_model_from_checkpoint(ckpt_path, device="cpu")

        metadata = ckpt["metadata"]
        total_ch = metadata["out_channels"]
        ks = metadata["kernel_size"]
        stride = metadata["stride"]
        spatial = (28 - ks) // stride + 1

        dummy = torch.randn(2, total_ch, spatial, spatial)
        with torch.no_grad():
            out = model(dummy)

        assert out.shape == (2, NUM_CLASSES)

    def test_teacher_best_checkpoint_loadable(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        best_path = os.path.join(seed_dir, "teacher", "teacher_best_seed_7.pt")
        if os.path.isfile(best_path):
            model, ckpt = load_model_from_checkpoint(best_path, device="cpu")
            from src.models.teacher_moe import TeacherMoE

            assert isinstance(model, TeacherMoE)


class TestStudentModelLoading:
    """Verify Student model loads correctly from checkpoint."""

    def test_load_student_checkpoint(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(seed_dir, "student", "student_final_seed_7.pt")

        model, ckpt = load_model_from_checkpoint(ckpt_path, device="cpu")

        from src.models.student_gatekeeper import StudentGatekeeper

        assert isinstance(model, StudentGatekeeper)

    def test_student_inference_works(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(seed_dir, "student", "student_final_seed_7.pt")

        model, ckpt = load_model_from_checkpoint(ckpt_path, device="cpu")

        patch_dim = ckpt["patch_dim"]
        num_kernels = ckpt["num_kernels"]

        dummy = torch.randn(4, patch_dim)
        with torch.no_grad():
            out = model(dummy)

        assert out.shape == (4, num_kernels)

    def test_student_checkpoint_has_agreement(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(seed_dir, "student", "student_final_seed_7.pt")

        info = get_checkpoint_info(ckpt_path)
        assert "agreement" in info
        assert 0.0 <= info["agreement"] <= 1.0


class TestFinalClassifierModelLoading:
    """Verify Final Classifier model loads correctly from checkpoint."""

    def test_load_final_classifier_checkpoint(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(
            seed_dir, "final_classifier", "final_classifier_seed_7.pt"
        )

        model, ckpt = load_model_from_checkpoint(ckpt_path, device="cpu")

        from src.models.ts_moe_classification_head import FinalClassifier

        assert isinstance(model, FinalClassifier)

    def test_final_classifier_inference_works(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(
            seed_dir, "final_classifier", "final_classifier_seed_7.pt"
        )

        model, ckpt = load_model_from_checkpoint(ckpt_path, device="cpu")

        metadata = ckpt["metadata"]
        in_channels = metadata["out_channels"]
        ks = metadata["kernel_size"]
        stride = metadata["stride"]
        spatial = metadata.get("feature_spatial_size", (28 - ks) // stride + 1)

        dummy = torch.randn(2, in_channels, spatial, spatial)
        with torch.no_grad():
            out = model(dummy)

        assert out.shape == (2, NUM_CLASSES)

    def test_final_classifier_checkpoint_has_accuracy(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(
            seed_dir, "final_classifier", "final_classifier_seed_7.pt"
        )

        info = get_checkpoint_info(ckpt_path)
        assert "test_accuracy" in info


# =========================================================================
# Tests: Checkpoint info (lightweight metadata)
# =========================================================================


class TestCheckpointInfo:
    """Verify get_checkpoint_info returns correct metadata."""

    def test_teacher_info(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(seed_dir, "teacher", "teacher_final_seed_7.pt")

        info = get_checkpoint_info(ckpt_path)
        assert info["seed"] == 7
        assert info["num_classes"] == NUM_CLASSES
        assert info["type"] == "teacher"
        assert "kernel_names" in info
        assert info["kernel_names"] == KERNEL_NAMES

    def test_student_info(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(seed_dir, "student", "student_final_seed_7.pt")

        info = get_checkpoint_info(ckpt_path)
        assert info["seed"] == 7
        assert info["type"] == "student"
        assert "agreement" in info

    def test_final_classifier_info(self, pipeline_outputs):
        seed_dir = pipeline_outputs["seed_dir"]
        ckpt_path = os.path.join(
            seed_dir, "final_classifier", "final_classifier_seed_7.pt"
        )

        info = get_checkpoint_info(ckpt_path)
        assert info["seed"] == 7
        assert info["type"] == "final_classifier"
        assert "test_accuracy" in info

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_checkpoint_info(str(tmp_path / "nonexistent.pt"))


# =========================================================================
# Tests: Checkpoint file discovery
# =========================================================================


class TestCheckpointDiscovery:
    """Verify find_model_checkpoints handles TS-MoE directory layouts."""

    def test_finds_ts_moe_checkpoints(self, pipeline_outputs):
        run_dir = pipeline_outputs["run_dir"]
        checkpoints = find_model_checkpoints(run_dir)

        # Should find teacher, student, and final classifier checkpoints
        all_ckpts = checkpoints["best"] + checkpoints["final"]
        assert len(all_ckpts) > 0, "No checkpoints found"

        # Check that we found at least one from each phase
        types_found = {detect_checkpoint_type(p) for p in all_ckpts}
        assert "teacher" in types_found, "No teacher checkpoint found"
        assert "student" in types_found, "No student checkpoint found"
        assert "final_classifier" in types_found, "No final_classifier checkpoint found"

    def test_best_checkpoints_contain_best(self, pipeline_outputs):
        run_dir = pipeline_outputs["run_dir"]
        checkpoints = find_model_checkpoints(run_dir)

        for path in checkpoints["best"]:
            assert "best" in os.path.basename(path)

    def test_final_checkpoints_contain_final(self, pipeline_outputs):
        run_dir = pipeline_outputs["run_dir"]
        checkpoints = find_model_checkpoints(run_dir)

        for path in checkpoints["final"]:
            basename = os.path.basename(path)
            assert "final" in basename or basename.startswith("final_classifier_seed_")

    def test_empty_dir_returns_empty(self, tmp_path):
        checkpoints = find_model_checkpoints(str(tmp_path))
        assert checkpoints == {"best": [], "final": []}

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        checkpoints = find_model_checkpoints(str(tmp_path / "nonexistent"))
        assert checkpoints == {"best": [], "final": []}


class TestCollectPtFiles:
    """Verify _collect_pt_files finds files in nested TS-MoE structures."""

    def test_collects_from_nested_structure(self, pipeline_outputs):
        run_dir = pipeline_outputs["run_dir"]
        pt_files = _collect_pt_files(run_dir)

        assert len(pt_files) > 0
        for f in pt_files:
            assert f.endswith(".pt")

    def test_collects_from_flat_structure(self, tmp_path):
        """Should also work with flat DAQCNN layout."""
        (tmp_path / "best_model_seed_0.pt").touch()
        (tmp_path / "final_model_seed_0.pt").touch()

        pt_files = _collect_pt_files(str(tmp_path))
        assert len(pt_files) == 2


# =========================================================================
# Tests: Architecture detection
# =========================================================================


class TestArchitectureDetection:
    """Verify _detect_run_architecture correctly identifies TS-MoE runs."""

    def test_ts_moe_from_config(self, pipeline_outputs):
        run_dir = pipeline_outputs["run_dir"]
        cfg = pipeline_outputs["cfg"]
        arch = _detect_run_architecture(run_dir, cfg)
        assert arch == "TS-MoE"

    def test_original_from_config(self, tmp_path):
        cfg = {"model": {"architecture": "original"}}
        arch = _detect_run_architecture(str(tmp_path), cfg)
        assert arch == "original"

    def test_original_from_missing_config(self, tmp_path):
        arch = _detect_run_architecture(str(tmp_path), None)
        assert arch == "original"

    def test_ts_moe_from_directory_heuristic(self, tmp_path):
        """Should detect TS-MoE from seed_N/teacher/ directory structure."""
        teacher_dir = tmp_path / "seed_0" / "teacher"
        teacher_dir.mkdir(parents=True)

        arch = _detect_run_architecture(str(tmp_path), None)
        assert arch == "TS-MoE"

    def test_original_when_no_seed_dirs(self, tmp_path):
        (tmp_path / "best_model_seed_0.pt").touch()
        arch = _detect_run_architecture(str(tmp_path), None)
        assert arch == "original"


# =========================================================================
# Tests: Full output scanning
# =========================================================================


class TestScanAllOutputs:
    """Verify scan_all_outputs discovers TS-MoE runs with metadata."""

    def test_scan_finds_ts_moe_run(self, pipeline_outputs):
        outputs_root = pipeline_outputs["outputs_root"]
        runs = scan_all_outputs(outputs_root)

        assert len(runs) >= 1
        ts_moe_runs = [r for r in runs if r.get("architecture") == "TS-MoE"]
        assert len(ts_moe_runs) >= 1, "No TS-MoE runs found by scanner"

    def test_scan_includes_architecture_field(self, pipeline_outputs):
        outputs_root = pipeline_outputs["outputs_root"]
        runs = scan_all_outputs(outputs_root)

        for run in runs:
            assert "architecture" in run

    def test_scan_loads_config(self, pipeline_outputs):
        outputs_root = pipeline_outputs["outputs_root"]
        runs = scan_all_outputs(outputs_root)

        ts_moe_runs = [r for r in runs if r.get("architecture") == "TS-MoE"]
        assert ts_moe_runs[0]["config"] is not None
        assert ts_moe_runs[0]["config"]["model"]["architecture"] == "TS-MoE"

    def test_scan_loads_pipeline_summary(self, pipeline_outputs):
        outputs_root = pipeline_outputs["outputs_root"]
        runs = scan_all_outputs(outputs_root)

        ts_moe_runs = [r for r in runs if r.get("architecture") == "TS-MoE"]
        run = ts_moe_runs[0]

        assert "pipeline_summary" in run
        ps = run["pipeline_summary"]
        assert ps is not None
        assert "teacher_test_acc" in ps
        assert "student_agreement" in ps
        assert "final_test_acc" in ps

    def test_scan_has_checkpoints(self, pipeline_outputs):
        outputs_root = pipeline_outputs["outputs_root"]
        runs = scan_all_outputs(outputs_root)

        ts_moe_runs = [r for r in runs if r.get("architecture") == "TS-MoE"]
        run = ts_moe_runs[0]

        all_ckpts = run["checkpoints"]["best"] + run["checkpoints"]["final"]
        assert len(all_ckpts) > 0

    def test_scan_empty_root(self, tmp_path):
        runs = scan_all_outputs(str(tmp_path))
        assert runs == []

    def test_scan_nonexistent_root(self, tmp_path):
        runs = scan_all_outputs(str(tmp_path / "nonexistent"))
        assert runs == []

    def test_scan_mixed_runs(self, pipeline_outputs, tmp_path):
        """Scanner should handle a mix of DAQCNN and TS-MoE runs."""
        outputs_root = str(tmp_path / "mixed_outputs")
        os.makedirs(outputs_root, exist_ok=True)

        # Create a fake DAQCNN run
        daqcnn_dir = os.path.join(outputs_root, "default_daqcnn_run_20250101")
        os.makedirs(daqcnn_dir, exist_ok=True)
        # Create a dummy checkpoint file
        torch.save(
            {"model_state_dict": {}, "config": {"model": {"num_classes": 2}}},
            os.path.join(daqcnn_dir, "best_model_seed_0.pt"),
        )
        import yaml

        with open(os.path.join(daqcnn_dir, "config.yml"), "w") as f:
            yaml.dump({"model": {"architecture": "original"}}, f)

        # Create a fake TS-MoE run structure
        moe_dir = os.path.join(outputs_root, "moe_run_20250102")
        os.makedirs(moe_dir, exist_ok=True)
        seed_teacher = os.path.join(moe_dir, "seed_0", "teacher")
        os.makedirs(seed_teacher, exist_ok=True)
        torch.save(
            {"model_state_dict": {}, "config": {}},
            os.path.join(seed_teacher, "teacher_final_seed_0.pt"),
        )

        runs = scan_all_outputs(outputs_root)
        architectures = {r["architecture"] for r in runs}
        assert "original" in architectures
        assert "TS-MoE" in architectures


# =========================================================================
# Tests: Error handling
# =========================================================================


class TestErrorHandling:
    """Verify graceful error handling for edge cases."""

    def test_load_nonexistent_checkpoint(self):
        with pytest.raises(FileNotFoundError):
            load_model_from_checkpoint("/nonexistent/path.pt")

    def test_get_info_nonexistent_checkpoint(self):
        with pytest.raises(FileNotFoundError):
            get_checkpoint_info("/nonexistent/path.pt")
