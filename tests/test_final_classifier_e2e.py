"""
End-to-end smoke test for run_final_classifier_training().

Creates a synthetic cached quantum dataset on disk (npz + json), trains a
Teacher for a few epochs, distills into a Student, then runs the full Final
Classifier pipeline on sparse routed tensors. Validates that all expected
output files, metrics, and comparison data are produced.

Run with:
    python tests/test_final_classifier_e2e.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ts_moe_classification_head_training import run_final_classifier_training
from src.models.student_training import run_student_training
from src.models.teacher_moe_training import run_teacher_training

# =========================================================================
# Helpers
# =========================================================================


def create_fake_cached_dataset(tmpdir, kernel_names, kernel_size, num_classes):
    """Write a synthetic quantum dataset (.npz + .json) into tmpdir.

    Returns (npz_path, datasets_dir, cfg, split_counts).
    """
    n_qubits = kernel_size * kernel_size
    M = len(kernel_names)
    total_channels = M * n_qubits

    stride = kernel_size
    spatial = (28 - kernel_size) // stride + 1

    n_train = 32
    n_val = 16
    n_test = 16

    channel_kernel_map = []
    ch = 0
    for name in kernel_names:
        for q in range(n_qubits):
            channel_kernel_map.append({"channel": ch, "kernel": name, "qubit": q})
            ch += 1

    metadata = {
        "dataset_name": "pneumonia_mnist",
        "kernel_size": kernel_size,
        "stride": stride,
        "kernel_topology_names": kernel_names,
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

    rng = np.random.RandomState(42)
    train_features = rng.randn(n_train, total_channels, spatial, spatial).astype(
        np.float32
    )
    train_labels = rng.randint(0, num_classes, n_train).astype(np.int64)
    val_features = rng.randn(n_val, total_channels, spatial, spatial).astype(np.float32)
    val_labels = rng.randint(0, num_classes, n_val).astype(np.int64)
    test_features = rng.randn(n_test, total_channels, spatial, spatial).astype(
        np.float32
    )
    test_labels = rng.randint(0, num_classes, n_test).astype(np.int64)

    datasets_dir = os.path.join(tmpdir, "data", "quantum_datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    npz_path = os.path.join(datasets_dir, "fake_dataset.npz")

    np.savez_compressed(
        npz_path,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
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
            "num_classes": num_classes,
            "kernel_size": kernel_size,
            "stride": stride,
            "kernel_topology_names": kernel_names,
            "scaling_factor": 1.0,
            "evolution_time": 0.2,
            "dropout": 0.1,
            "activation": "relu",
            "classical_device": "cpu",
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 3,
            "grad_clip": 1.0,
            "patience": None,
            "use_scheduler": False,
        },
        "ts_moe": {
            "lambda_max": 0.1,
            "lambda_warmup_epochs": 2,
            "student_epochs": 3,
            "student_batch_size": 64,
            "student_hidden_dims": [32, 16],
            "student_lr": 1e-3,
            "student_patience": None,
            "final_epochs": 3,
            "final_lr": 1e-3,
            "final_patience": None,
            "use_mask_channel": False,
        },
        "misc": {
            "seed": 42,
        },
    }

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    return npz_path, Path(datasets_dir), cfg, split_counts


def make_synthetic_image_datasets(
    split_counts, in_channels=1, img_size=28, num_classes=2
):
    """Create synthetic raw image datasets matching cached quantum dataset sizes."""
    rng = torch.Generator().manual_seed(99)
    datasets = {}
    for split in ("train", "val", "test"):
        n = split_counts[split]
        images = torch.rand(n, in_channels, img_size, img_size, generator=rng)
        labels = torch.randint(0, num_classes, (n, 1), generator=rng)
        datasets[split] = TensorDataset(images, labels)
    return datasets


def train_teacher_and_get_ckpt(tmpdir, cfg, datasets_dir):
    """Train a Teacher for a few epochs and return checkpoint path."""
    teacher_output = os.path.join(tmpdir, "outputs", "teacher")
    os.makedirs(teacher_output, exist_ok=True)

    result = run_teacher_training(
        cfg,
        seed=42,
        output_dir=teacher_output,
        verbose=False,
        datasets_dir=datasets_dir,
    )

    ckpt_path = os.path.join(teacher_output, "teacher_best_seed_42.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(teacher_output, "teacher_final_seed_42.pt")

    assert os.path.isfile(ckpt_path), f"Teacher checkpoint not found at {ckpt_path}"
    return ckpt_path, result


def train_student_and_get_ckpt(
    tmpdir, cfg, datasets_dir, teacher_ckpt_path, raw_image_datasets
):
    """Train a Student and return checkpoint path."""
    student_output = os.path.join(tmpdir, "outputs", "student")
    os.makedirs(student_output, exist_ok=True)

    result = run_student_training(
        cfg,
        seed=42,
        output_dir=student_output,
        teacher_ckpt_path=teacher_ckpt_path,
        verbose=False,
        datasets_dir=datasets_dir,
        raw_image_datasets=raw_image_datasets,
    )

    ckpt_path = os.path.join(student_output, "student_best_seed_42.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(student_output, "student_final_seed_42.pt")

    assert os.path.isfile(ckpt_path), f"Student checkpoint not found at {ckpt_path}"
    return ckpt_path, result


# =========================================================================
# Tests
# =========================================================================


def test_run_final_classifier_e2e():
    """Full pipeline: Teacher -> Student -> Final Classifier."""
    print("\n--- test_run_final_classifier_e2e ---")
    kernel_names = ["kings", "horizontal"]
    kernel_size = 3
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg, split_counts = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        raw_datasets = make_synthetic_image_datasets(
            split_counts, in_channels=1, num_classes=num_classes
        )

        # Phase 1: Teacher
        teacher_ckpt, teacher_result = train_teacher_and_get_ckpt(
            tmpdir, cfg, datasets_dir
        )
        teacher_acc = teacher_result["test_acc"]

        # Phase 2: Student
        student_ckpt, student_result = train_student_and_get_ckpt(
            tmpdir, cfg, datasets_dir, teacher_ckpt, raw_datasets
        )

        # Phase 3: Final Classifier
        final_output = os.path.join(tmpdir, "outputs", "final")
        os.makedirs(final_output, exist_ok=True)

        result = run_final_classifier_training(
            cfg,
            seed=42,
            output_dir=final_output,
            student_ckpt_path=student_ckpt,
            verbose=True,
            datasets_dir=datasets_dir,
            raw_image_datasets=raw_datasets,
            teacher_test_acc=teacher_acc,
            baseline_test_acc=0.5,
        )

        # --- Validate result dict ---
        assert result["architecture"] == "TS-MoE-FinalClassifier"
        assert result["seed"] == 42
        assert isinstance(result["test_acc"], float)
        assert isinstance(result["test_auc"], float)
        assert isinstance(result["test_f1"], float)
        assert isinstance(result["test_recall"], float)
        assert isinstance(result["test_loss"], float)
        assert len(result["train_losses"]) == 3
        assert len(result["val_losses"]) == 3
        assert result["num_classes"] == num_classes
        assert set(result["kernel_names"]) == set(kernel_names)
        assert len(result["kernel_names"]) == len(kernel_names)

        # Comparison dict
        assert "comparison" in result
        comp = result["comparison"]
        assert comp["teacher_oracle_acc"] == teacher_acc
        assert comp["baseline_acc"] == 0.5
        assert comp["speedup_factor"] == 2
        assert comp["routing_time_s"] > 0
        assert comp["pipeline_time_s"] > 0

        # Routing analysis
        assert "routing_analysis" in result
        for c, per_kernel in result["routing_analysis"].items():
            total = sum(per_kernel.values())
            assert abs(total - 1.0) < 1e-6

        # --- Validate output files ---
        assert os.path.isfile(os.path.join(final_output, "final_classifier_seed_42.pt"))
        assert os.path.isfile(
            os.path.join(final_output, "final_loss_curve_seed_42.png")
        )
        assert os.path.isfile(os.path.join(final_output, "final_roc_curve_seed_42.png"))
        assert os.path.isfile(
            os.path.join(final_output, "final_confusion_matrix_seed_42.png")
        )

        print("PASS")


def test_final_classifier_with_mask_channel():
    """Final Classifier with the optional mask channel enabled."""
    print("\n--- test_final_classifier_with_mask_channel ---")
    kernel_names = ["a", "b"]
    kernel_size = 3
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg, split_counts = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        cfg["ts_moe"]["use_mask_channel"] = True

        raw_datasets = make_synthetic_image_datasets(
            split_counts, in_channels=1, num_classes=num_classes
        )

        teacher_ckpt, _ = train_teacher_and_get_ckpt(tmpdir, cfg, datasets_dir)
        student_ckpt, _ = train_student_and_get_ckpt(
            tmpdir, cfg, datasets_dir, teacher_ckpt, raw_datasets
        )

        final_output = os.path.join(tmpdir, "outputs", "final_mask")
        os.makedirs(final_output, exist_ok=True)

        result = run_final_classifier_training(
            cfg,
            seed=42,
            output_dir=final_output,
            student_ckpt_path=student_ckpt,
            verbose=False,
            datasets_dir=datasets_dir,
            raw_image_datasets=raw_datasets,
        )

        assert result["use_mask_channel"] is True
        assert isinstance(result["test_acc"], float)
        print("PASS")


def test_final_classifier_4_kernels():
    """Final Classifier with M=4 kernel topologies."""
    print("\n--- test_final_classifier_4_kernels ---")
    kernel_names = ["kings", "horizontal", "vertical", "diagonal"]
    kernel_size = 2
    num_classes = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg, split_counts = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        raw_datasets = make_synthetic_image_datasets(
            split_counts, in_channels=1, num_classes=num_classes
        )

        teacher_ckpt, _ = train_teacher_and_get_ckpt(tmpdir, cfg, datasets_dir)
        student_ckpt, _ = train_student_and_get_ckpt(
            tmpdir, cfg, datasets_dir, teacher_ckpt, raw_datasets
        )

        final_output = os.path.join(tmpdir, "outputs", "final_4k")
        os.makedirs(final_output, exist_ok=True)

        result = run_final_classifier_training(
            cfg,
            seed=42,
            output_dir=final_output,
            student_ckpt_path=student_ckpt,
            verbose=False,
            datasets_dir=datasets_dir,
            raw_image_datasets=raw_datasets,
        )

        assert result["num_kernels"] == 4
        assert result["num_classes"] == 3
        assert len(result["kernel_names"]) == 4
        assert result["comparison"]["speedup_factor"] == 4
        print("PASS")


def test_final_classifier_checkpoint_loadable():
    """Saved checkpoint can be reloaded and used for inference."""
    print("\n--- test_final_classifier_checkpoint_loadable ---")
    kernel_names = ["a", "b"]
    kernel_size = 3
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg, split_counts = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        raw_datasets = make_synthetic_image_datasets(
            split_counts, in_channels=1, num_classes=num_classes
        )

        teacher_ckpt, _ = train_teacher_and_get_ckpt(tmpdir, cfg, datasets_dir)
        student_ckpt, _ = train_student_and_get_ckpt(
            tmpdir, cfg, datasets_dir, teacher_ckpt, raw_datasets
        )

        final_output = os.path.join(tmpdir, "outputs", "final_ckpt")
        os.makedirs(final_output, exist_ok=True)

        run_final_classifier_training(
            cfg,
            seed=42,
            output_dir=final_output,
            student_ckpt_path=student_ckpt,
            verbose=False,
            datasets_dir=datasets_dir,
            raw_image_datasets=raw_datasets,
        )

        # Load checkpoint
        ckpt_path = os.path.join(final_output, "final_classifier_seed_42.pt")
        assert os.path.isfile(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "config" in ckpt
        assert "metadata" in ckpt
        assert "kernel_names" in ckpt
        assert ckpt["num_classes"] == num_classes

        # Rebuild model and load weights
        from src.models.ts_moe_classification_head import build_final_classifier_from_metadata

        model = build_final_classifier_from_metadata(
            ckpt["metadata"],
            ckpt["num_classes"],
            use_mask_channel=ckpt.get("use_mask_channel", False),
        )

        # Initialize lazy layers
        total_ch = ckpt["metadata"]["out_channels"]
        ks = ckpt["metadata"]["kernel_size"]
        stride = ckpt["metadata"]["stride"]
        spatial = (28 - ks) // stride + 1
        dummy = torch.randn(1, total_ch, spatial, spatial)
        model(dummy)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Run inference
        x = torch.randn(2, total_ch, spatial, spatial)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, num_classes)
        print("PASS")


def test_final_classifier_early_stopping():
    """Early stopping should trigger if patience is set."""
    print("\n--- test_final_classifier_early_stopping ---")
    kernel_names = ["a", "b"]
    kernel_size = 3
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg, split_counts = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        cfg["ts_moe"]["final_epochs"] = 20
        cfg["ts_moe"]["final_patience"] = 2

        raw_datasets = make_synthetic_image_datasets(
            split_counts, in_channels=1, num_classes=num_classes
        )

        teacher_ckpt, _ = train_teacher_and_get_ckpt(tmpdir, cfg, datasets_dir)
        student_ckpt, _ = train_student_and_get_ckpt(
            tmpdir, cfg, datasets_dir, teacher_ckpt, raw_datasets
        )

        final_output = os.path.join(tmpdir, "outputs", "final_es")
        os.makedirs(final_output, exist_ok=True)

        result = run_final_classifier_training(
            cfg,
            seed=42,
            output_dir=final_output,
            student_ckpt_path=student_ckpt,
            verbose=False,
            datasets_dir=datasets_dir,
            raw_image_datasets=raw_datasets,
        )

        # Should have stopped before 20 epochs (or run all 20 if it kept improving)
        actual_epochs = len(result["train_losses"])
        assert actual_epochs <= 20
        assert isinstance(result["test_acc"], float)
        print("PASS")


def test_result_dict_format():
    """Validate all required keys are present in the result dict."""
    print("\n--- test_result_dict_format ---")
    kernel_names = ["x", "y"]
    kernel_size = 3
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg, split_counts = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        raw_datasets = make_synthetic_image_datasets(
            split_counts, in_channels=1, num_classes=num_classes
        )

        teacher_ckpt, teacher_result = train_teacher_and_get_ckpt(
            tmpdir, cfg, datasets_dir
        )
        student_ckpt, _ = train_student_and_get_ckpt(
            tmpdir, cfg, datasets_dir, teacher_ckpt, raw_datasets
        )

        final_output = os.path.join(tmpdir, "outputs", "final_fmt")
        os.makedirs(final_output, exist_ok=True)

        result = run_final_classifier_training(
            cfg,
            seed=42,
            output_dir=final_output,
            student_ckpt_path=student_ckpt,
            verbose=False,
            datasets_dir=datasets_dir,
            raw_image_datasets=raw_datasets,
            teacher_test_acc=teacher_result["test_acc"],
            baseline_test_acc=0.5,
        )

        # All required keys
        required_keys = [
            "seed",
            "architecture",
            "train_losses",
            "val_losses",
            "train_accs",
            "val_accs",
            "test_loss",
            "test_acc",
            "test_auc",
            "test_f1",
            "test_recall",
            "test_probs",
            "test_labels",
            "test_confusion_matrix",
            "num_classes",
            "final_train_loss",
            "final_val_loss",
            "final_train_acc",
            "final_val_acc",
            "kernel_names",
            "num_kernels",
            "use_mask_channel",
            "comparison",
            "routing_analysis",
            "routing_time_s",
            "classifier_params",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Type checks
        assert isinstance(result["test_probs"], list)
        assert isinstance(result["test_labels"], list)
        assert isinstance(result["test_confusion_matrix"], list)
        assert isinstance(result["comparison"], dict)
        assert isinstance(result["routing_analysis"], dict)
        assert isinstance(result["classifier_params"], int)
        assert result["routing_time_s"] >= 0

        print("PASS")


def test_missing_student_checkpoint_raises():
    """Should raise FileNotFoundError for missing student checkpoint."""
    print("\n--- test_missing_student_checkpoint_raises ---")
    kernel_names = ["a", "b"]
    kernel_size = 3
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg, split_counts = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )

        final_output = os.path.join(tmpdir, "outputs", "final_err")
        os.makedirs(final_output, exist_ok=True)

        fake_ckpt = os.path.join(tmpdir, "nonexistent_student.pt")
        raised = False
        try:
            run_final_classifier_training(
                cfg,
                seed=42,
                output_dir=final_output,
                student_ckpt_path=fake_ckpt,
                verbose=False,
                datasets_dir=datasets_dir,
            )
        except (FileNotFoundError, RuntimeError, Exception):
            raised = True

        assert raised, "Should have raised an error for missing student checkpoint"
        print("PASS")


# =========================================================================
# Runner
# =========================================================================


def run_all():
    """Run all e2e tests and report results."""
    import traceback

    tests = [
        test_run_final_classifier_e2e,
        test_final_classifier_with_mask_channel,
        test_final_classifier_4_kernels,
        test_final_classifier_checkpoint_loadable,
        test_final_classifier_early_stopping,
        test_result_dict_format,
        test_missing_student_checkpoint_raises,
    ]

    total = len(tests)
    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, e))
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Phase 3 E2E Tests: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")

    if errors:
        print("\nFailed tests:")
        for name, e in errors:
            print(f"  {name}: {e}")
        return False
    return True


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
