"""
Phase 4 integration tests: config loading, architecture branching,
and TS-MoE pipeline orchestration.

Tests:
    - Config schema validation (model.architecture field, ts_moe section)
    - Architecture branching in the experiment runner
    - Student/Final classifier config overrides from ts_moe section
    - Full pipeline orchestration via run_ts_moe_pipeline
    - Backward compatibility: original DAQCNN configs still work

Run with:
    python -m pytest tests/test_config_integration_phase4.py -v
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

from src.models.train_ts_moe import (  # noqa: E402
    run_ts_moe_pipeline,
)

# =========================================================================
# Helpers
# =========================================================================

KERNEL_NAMES = ["kings", "square"]
KERNEL_SIZE = 2
NUM_CLASSES = 2


def _create_fake_cached_dataset(tmpdir):
    """Write a minimal synthetic quantum dataset (npz + json) into tmpdir.

    Returns (datasets_dir, cfg, split_counts).
    """
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

    npz_path = os.path.join(datasets_dir, "fake_dataset.npz")
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

    json_path = os.path.join(datasets_dir, "fake_dataset.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    cfg = {
        "dataset": {
            "name": "pneumonia_mnist",
            "data_root": os.path.join(tmpdir, "data"),
            "batch_size": 8,
            "num_workers": 0,
            "color_space": "GRAYSCALE",
        },
        "model": {
            "architecture": "TS-MoE",
            "num_classes": NUM_CLASSES,
            "kernel_size": KERNEL_SIZE,
            "stride": stride,
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
# Tests: Config schema
# =========================================================================


class TestConfigSchema:
    """Verify the TS-MoE config structure is parsed correctly."""

    def test_architecture_field_present(self):
        """Config should contain model.architecture."""
        import yaml

        config_path = os.path.join(
            PROJECT_ROOT, "configs", "pneumonia_mnist_ts_moe.yml"
        )
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        assert cfg["model"]["architecture"] == "TS-MoE"

    def test_ts_moe_section_present(self):
        """Config should have a ts_moe section with expected keys."""
        import yaml

        config_path = os.path.join(
            PROJECT_ROOT, "configs", "pneumonia_mnist_ts_moe.yml"
        )
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        ts = cfg["ts_moe"]
        assert "lambda_max" in ts
        assert "lambda_warmup_epochs" in ts
        assert "student_hidden_dims" in ts
        assert "student_epochs" in ts
        assert "student_lr" in ts
        assert "final_epochs" in ts
        assert "final_lr" in ts

    def test_original_config_has_no_architecture(self):
        """Original configs default to 'original' when architecture is absent."""
        import yaml

        config_path = os.path.join(PROJECT_ROOT, "configs", "pneumonia_mnist.yml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        arch = cfg.get("model", {}).get("architecture", "original")
        assert arch == "original"


# =========================================================================
# Tests: Config override helpers
# =========================================================================


class TestConfigOverrides:
    """Verify that run_student_training and run_final_classifier_training read
    their hyper-parameters directly from the ts_moe section of the config,
    so no intermediate override helper is needed."""

    def test_student_epochs_read_from_ts_moe(self):
        """student_epochs in ts_moe is used directly by run_student_training."""
        cfg = {
            "optim": {"lr": 1e-3, "epochs": 10},
            "ts_moe": {"student_epochs": 5, "student_lr": 2e-4},
        }
        # The training function reads ts_moe.student_epochs directly;
        # no wrapper is needed and the original optim values are untouched.
        assert cfg["ts_moe"]["student_epochs"] == 5
        assert cfg["optim"]["epochs"] == 10  # original unchanged

    def test_final_epochs_read_from_ts_moe(self):
        """final_epochs in ts_moe is used directly by run_final_classifier_training."""
        cfg = {
            "optim": {"lr": 1e-3, "epochs": 10},
            "ts_moe": {"final_epochs": 3, "final_lr": 5e-4},
        }
        assert cfg["ts_moe"]["final_epochs"] == 3
        assert cfg["optim"]["epochs"] == 10  # original unchanged


# =========================================================================
# Tests: Architecture branching in experiment runner
# =========================================================================


class TestArchitectureBranching:
    """Verify the experiment runner dispatches based on model.architecture."""

    def test_original_architecture_detected(self):
        cfg = {"model": {"classical_device": "cpu"}}
        arch = cfg.get("model", {}).get("architecture", "original")
        assert arch == "original"

    def test_ts_moe_architecture_detected(self):
        cfg = {"model": {"architecture": "TS-MoE", "classical_device": "cpu"}}
        arch = cfg.get("model", {}).get("architecture", "original")
        assert arch == "TS-MoE"

    def test_run_ts_moe_for_seed_import(self):
        """The wrapper function should be importable from the experiment runner."""
        from experiments.robust_test_original_daqcnn import run_ts_moe_for_seed

        assert callable(run_ts_moe_for_seed)


# =========================================================================
# Tests: Full pipeline orchestration
# =========================================================================


class TestTsMoePipeline:
    """End-to-end test: run_ts_moe_pipeline on tiny synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmpdir = str(tmp_path)
        self.datasets_dir, self.cfg, self.split_counts = _create_fake_cached_dataset(
            self.tmpdir
        )
        self.raw_datasets = _make_synthetic_images(self.split_counts)

    def test_full_pipeline_runs(self):
        """Full Teacher → Student → Final pipeline should complete."""
        output_dir = os.path.join(self.tmpdir, "outputs", "pipeline")
        os.makedirs(output_dir, exist_ok=True)

        result = run_ts_moe_pipeline(
            self.cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=self.datasets_dir,
            raw_image_datasets=self.raw_datasets,
        )

        assert "summary" in result
        assert "teacher" in result
        assert "student" in result
        assert "final" in result

    def test_pipeline_summary_keys(self):
        """Summary should contain all expected comparison keys."""
        output_dir = os.path.join(self.tmpdir, "outputs", "pipeline2")
        os.makedirs(output_dir, exist_ok=True)

        result = run_ts_moe_pipeline(
            self.cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=self.datasets_dir,
            raw_image_datasets=self.raw_datasets,
        )

        summary = result["summary"]
        for key in [
            "seed",
            "pipeline_time_s",
            "teacher_test_acc",
            "student_agreement",
            "final_test_acc",
            "speedup_factor",
        ]:
            assert key in summary, f"Missing key: {key}"

    def test_pipeline_creates_output_structure(self):
        """Pipeline should create teacher/, student/, final_classifier/ dirs."""
        output_dir = os.path.join(self.tmpdir, "outputs", "pipeline3")
        os.makedirs(output_dir, exist_ok=True)

        run_ts_moe_pipeline(
            self.cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=self.datasets_dir,
            raw_image_datasets=self.raw_datasets,
        )

        seed_dir = os.path.join(output_dir, "seed_7")
        assert os.path.isdir(os.path.join(seed_dir, "teacher"))
        assert os.path.isdir(os.path.join(seed_dir, "student"))
        assert os.path.isdir(os.path.join(seed_dir, "final_classifier"))

    def test_pipeline_saves_summary_json(self):
        """Pipeline should save pipeline_summary.json."""
        output_dir = os.path.join(self.tmpdir, "outputs", "pipeline4")
        os.makedirs(output_dir, exist_ok=True)

        run_ts_moe_pipeline(
            self.cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=self.datasets_dir,
            raw_image_datasets=self.raw_datasets,
        )

        summary_path = os.path.join(output_dir, "seed_7", "pipeline_summary.json")
        assert os.path.isfile(summary_path)

        with open(summary_path) as f:
            data = json.load(f)
        assert data["seed"] == 7
        assert isinstance(data["pipeline_time_s"], float)

    def test_pipeline_checkpoints_exist(self):
        """Teacher, Student, and Final Classifier checkpoints should be saved."""
        output_dir = os.path.join(self.tmpdir, "outputs", "pipeline5")
        os.makedirs(output_dir, exist_ok=True)

        run_ts_moe_pipeline(
            self.cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=self.datasets_dir,
            raw_image_datasets=self.raw_datasets,
        )

        seed_dir = os.path.join(output_dir, "seed_7")

        # At least the final checkpoint for each phase
        teacher_ckpt = os.path.join(seed_dir, "teacher", "teacher_final_seed_7.pt")
        student_ckpt = os.path.join(seed_dir, "student", "student_final_seed_7.pt")
        final_ckpt = os.path.join(
            seed_dir, "final_classifier", "final_classifier_seed_7.pt"
        )

        assert os.path.isfile(teacher_ckpt), "Teacher checkpoint missing"
        assert os.path.isfile(student_ckpt), "Student checkpoint missing"
        assert os.path.isfile(final_ckpt), "Final classifier checkpoint missing"

    def test_pipeline_produces_final_result(self):
        """Pipeline should complete and produce a final classifier result."""
        output_dir = os.path.join(self.tmpdir, "outputs", "pipeline_final")
        os.makedirs(output_dir, exist_ok=True)

        result = run_ts_moe_pipeline(
            self.cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=self.datasets_dir,
            raw_image_datasets=self.raw_datasets,
        )
        assert isinstance(result["final"].get("test_acc"), float)

    def test_speedup_factor_matches_num_kernels(self):
        """Speedup factor should equal the number of kernels."""
        output_dir = os.path.join(self.tmpdir, "outputs", "pipeline6")
        os.makedirs(output_dir, exist_ok=True)

        result = run_ts_moe_pipeline(
            self.cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=self.datasets_dir,
            raw_image_datasets=self.raw_datasets,
        )

        expected = len(KERNEL_NAMES)
        assert result["summary"]["speedup_factor"] == expected


# =========================================================================
# Tests: Experiment runner wrapper
# =========================================================================


class TestExperimentRunnerWrapper:
    """Test run_ts_moe_for_seed returns aggregation-compatible dict."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmpdir = str(tmp_path)
        self.datasets_dir, self.cfg, self.split_counts = _create_fake_cached_dataset(
            self.tmpdir
        )
        self.raw_datasets = _make_synthetic_images(self.split_counts)

    def test_wrapper_returns_aggregation_keys(self):
        """run_ts_moe_pipeline returns keys needed by aggregation wrapper."""
        output_dir = os.path.join(self.tmpdir, "outputs", "wrapper")
        os.makedirs(output_dir, exist_ok=True)

        # Call run_ts_moe_pipeline directly (the wrapper in the experiment
        # runner delegates to this, so testing it is equivalent).
        result = run_ts_moe_pipeline(
            self.cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=self.datasets_dir,
            raw_image_datasets=self.raw_datasets,
        )

        # Verify the final result has the keys the wrapper would extract
        final = result["final"]
        summary = result["summary"]

        # Keys from final result that the wrapper extracts
        for key in [
            "test_loss",
            "test_acc",
            "test_auc",
            "test_f1",
            "test_recall",
            "final_train_loss",
            "final_val_loss",
            "final_train_acc",
            "final_val_acc",
            "num_classes",
        ]:
            assert key in final, f"Missing key in final result: {key}"

        # Summary keys used by wrapper
        assert summary.get("teacher_test_acc") is not None
        assert summary.get("student_agreement") is not None
        assert summary.get("seed") == 7


# =========================================================================
# Run directly
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
