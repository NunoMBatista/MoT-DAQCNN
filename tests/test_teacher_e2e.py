"""
End-to-end smoke test for run_teacher_training().

Creates a synthetic cached quantum dataset on disk (npz + json), points a
config at it, and runs the full teacher training pipeline for a few epochs.
Validates that all expected output files and metrics are produced.

Run with:
    python tests/test_teacher_e2e.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.teacher_moe_training import run_teacher_training


def create_fake_cached_dataset(tmpdir, kernel_names, kernel_size, num_classes):
    """Write a synthetic quantum dataset (.npz + .json) into tmpdir.

    Returns the path to the .npz file, the datasets_dir Path, and the config
    dict that will match it.
    """
    n_qubits = kernel_size * kernel_size
    M = len(kernel_names)
    total_channels = M * n_qubits

    # For a 28x28 image with stride=kernel_size (non-overlapping):
    stride = kernel_size
    spatial = (28 - kernel_size) // stride + 1

    # Build channel-kernel map
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
        "train_samples": 32,
        "val_samples": 16,
        "test_samples": 16,
    }

    # Synthetic features: random, shape (N, total_channels, spatial, spatial)
    rng = np.random.RandomState(42)
    train_features = rng.randn(32, total_channels, spatial, spatial).astype(np.float32)
    train_labels = rng.randint(0, num_classes, 32).astype(np.int64)
    val_features = rng.randn(16, total_channels, spatial, spatial).astype(np.float32)
    val_labels = rng.randint(0, num_classes, 16).astype(np.int64)
    test_features = rng.randn(16, total_channels, spatial, spatial).astype(np.float32)
    test_labels = rng.randint(0, num_classes, 16).astype(np.int64)

    # Save npz
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

    # Save sidecar json (for find_cached_quantum_dataset to match)
    json_path = os.path.join(datasets_dir, "fake_dataset.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Build a config that matches this dataset
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
        },
        "misc": {
            "seed": 42,
        },
    }

    return npz_path, Path(datasets_dir), cfg


def test_run_teacher_training_e2e():
    """Full end-to-end: create fake data, run training, check outputs."""
    kernel_names = ["kings", "horizontal"]
    kernel_size = 3
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path, datasets_dir, cfg = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )

        output_dir = os.path.join(tmpdir, "outputs", "teacher_test")
        os.makedirs(output_dir, exist_ok=True)

        result = run_teacher_training(
            cfg,
            seed=42,
            output_dir=output_dir,
            verbose=True,
            datasets_dir=datasets_dir,
        )

        # ----- Validate result dict -----
        assert result is not None, "run_teacher_training returned None"
        assert result["seed"] == 42
        assert result["architecture"] == "TS-MoE-Teacher"
        assert result["num_classes"] == num_classes
        assert len(result["kernel_names"]) == 2

        # Check metric lists have one entry per epoch
        n_epochs = cfg["optim"]["epochs"]
        assert len(result["train_losses"]) == n_epochs
        assert len(result["val_losses"]) == n_epochs
        assert len(result["train_ce_losses"]) == n_epochs
        assert len(result["train_ent_losses"]) == n_epochs
        assert len(result["lambda_history"]) == n_epochs
        assert len(result["train_accs"]) == n_epochs
        assert len(result["val_accs"]) == n_epochs
        assert len(result["routing_history"]) == n_epochs

        # Check test metrics are present and finite
        for key in ["test_loss", "test_acc", "test_auc", "test_f1", "test_recall"]:
            assert key in result, f"Missing key: {key}"
            v = result[key]
            assert v == v, f"{key} is NaN"  # NaN check

        assert 0.0 <= result["test_acc"] <= 1.0
        assert 0.0 <= result["test_auc"] <= 1.0

        # Lambda should anneal correctly
        assert result["lambda_history"][0] == 0.0, (
            f"Lambda at epoch 0 should be 0, got {result['lambda_history'][0]}"
        )
        # Last lambda should be > 0
        assert result["lambda_history"][-1] > 0.0

        # Routing history: each entry should have all kernel names summing to ~1
        for epoch_routing in result["routing_history"]:
            assert set(epoch_routing.keys()) == set(kernel_names)
            total = sum(epoch_routing.values())
            assert abs(total - 1.0) < 1e-5, f"Routing sum = {total}"

        # ----- Validate output files -----
        # Model checkpoints
        assert os.path.isfile(os.path.join(output_dir, "teacher_final_seed_42.pt")), (
            "Final model checkpoint not saved"
        )
        assert os.path.isfile(os.path.join(output_dir, "teacher_best_seed_42.pt")), (
            "Best model checkpoint not saved"
        )

        # Plots
        assert os.path.isfile(
            os.path.join(output_dir, "teacher_loss_curve_seed_42.png")
        ), "Loss curve plot not saved"
        assert os.path.isfile(
            os.path.join(output_dir, "teacher_roc_curve_seed_42.png")
        ), "ROC curve plot not saved"
        assert os.path.isfile(
            os.path.join(output_dir, "teacher_confusion_matrix_seed_42.png")
        ), "Confusion matrix plot not saved"
        assert os.path.isfile(
            os.path.join(output_dir, "teacher_routing_ratio_seed_42.png")
        ), "Routing ratio plot not saved"

        # Alpha histograms: should have one combined histogram per epoch
        hist_dir = os.path.join(output_dir, "alpha_histograms")
        assert os.path.isdir(hist_dir), "Alpha histogram directory not created"
        for epoch in range(1, n_epochs + 1):
            hist_file = os.path.join(hist_dir, f"epoch_{epoch:03d}_all_kernels.png")
            assert os.path.isfile(hist_file), (
                f"Missing alpha histogram: {hist_file}"
                )

        hist_count = len([f for f in os.listdir(hist_dir) if f.endswith(".png")])
        expected_count = n_epochs
        assert hist_count == expected_count, (
            f"Expected {expected_count} histogram files (one per epoch), got {hist_count}"
        )

        print("\n  All E2E checks passed!")


def test_run_teacher_training_4_kernels():
    """Same E2E test but with 4 kernels to verify generalization."""
    kernel_names = ["kings", "horizontal", "ring", "cross"]
    kernel_size = 3
    num_classes = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        cfg["model"]["num_classes"] = num_classes
        cfg["optim"]["epochs"] = 2  # quick

        output_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        result = run_teacher_training(
            cfg,
            seed=0,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=datasets_dir,
        )

        assert len(result["kernel_names"]) == 4
        assert len(result["routing_history"]) == 2

        for epoch_routing in result["routing_history"]:
            assert len(epoch_routing) == 4
            total = sum(epoch_routing.values())
            assert abs(total - 1.0) < 1e-5

        print("  4-kernel E2E test passed!")


def test_missing_cached_dataset_raises():
    """Should raise FileNotFoundError when no cached dataset exists."""
    cfg = {
        "dataset": {"name": "nonexistent_dataset", "color_space": "GRAYSCALE"},
        "model": {
            "kernel_size": 2,
            "stride": 1,
            "kernel_topology_names": ["kings"],
            "scaling_factor": 1.0,
            "evolution_time": 0.2,
            "classical_device": "cpu",
        },
        "optim": {"epochs": 1},
        "ts_moe": {},
        "misc": {"seed": 0},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Point at an empty directory so no dataset can be found
        empty_datasets_dir = Path(os.path.join(tmpdir, "empty_datasets"))
        empty_datasets_dir.mkdir(parents=True, exist_ok=True)

        try:
            run_teacher_training(
                cfg,
                seed=0,
                output_dir=output_dir,
                verbose=False,
                datasets_dir=empty_datasets_dir,
            )
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "cached quantum dataset" in str(e).lower()
            print("  Missing dataset error raised correctly!")


def test_model_checkpoint_loadable():
    """Saved checkpoint should be loadable and contain expected keys."""
    kernel_names = ["kings", "horizontal"]
    kernel_size = 2
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        cfg["optim"]["epochs"] = 1

        output_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        run_teacher_training(
            cfg,
            seed=7,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=datasets_dir,
        )

        # Load and inspect checkpoint
        ckpt_path = os.path.join(output_dir, "teacher_final_seed_7.pt")
        assert os.path.isfile(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "config" in ckpt
        assert "seed" in ckpt
        assert "num_classes" in ckpt
        assert "kernel_names" in ckpt
        assert "metadata" in ckpt
        assert ckpt["seed"] == 7
        assert ckpt["num_classes"] == num_classes
        assert ckpt["kernel_names"] == kernel_names

        # Verify we can load the state dict into a new model
        from src.models.teacher_moe import build_teacher_from_metadata

        model = build_teacher_from_metadata(ckpt["metadata"], num_classes=num_classes)
        # Need to initialize lazy layers first
        total_ch = len(kernel_names) * kernel_size * kernel_size
        spatial = (28 - kernel_size) // kernel_size + 1
        dummy = torch.randn(1, total_ch, spatial, spatial)
        with torch.no_grad():
            model(dummy)
        model.load_state_dict(ckpt["model_state_dict"])

        print("  Checkpoint load test passed!")


def test_early_stopping():
    """Early stopping should stop training before all epochs complete."""
    kernel_names = ["kings", "horizontal"]
    kernel_size = 3
    num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        _, datasets_dir, cfg = create_fake_cached_dataset(
            tmpdir, kernel_names, kernel_size, num_classes
        )
        cfg["optim"]["epochs"] = 50  # many epochs
        cfg["optim"]["patience"] = 2  # stop after 2 epochs without improvement

        output_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        result = run_teacher_training(
            cfg,
            seed=42,
            output_dir=output_dir,
            verbose=False,
            datasets_dir=datasets_dir,
        )

        # Should have stopped well before 50 epochs
        actual_epochs = len(result["train_losses"])
        assert actual_epochs < 50, (
            f"Early stopping didn't trigger: ran all {actual_epochs} epochs"
        )
        assert actual_epochs >= 3, (
            f"Stopped too early at epoch {actual_epochs} (need at least 3: "
            "1 to set best, 2 patience epochs)"
        )

        print(f"  Early stopping triggered at epoch {actual_epochs} — correct!")


# =========================================================================
# Runner
# =========================================================================


def run_all():
    tests = [
        ("E2E teacher training (2 kernels)", test_run_teacher_training_e2e),
        ("E2E teacher training (4 kernels)", test_run_teacher_training_4_kernels),
        ("Missing cached dataset raises error", test_missing_cached_dataset_raises),
        ("Checkpoint is loadable", test_model_checkpoint_loadable),
        ("Early stopping", test_early_stopping),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, fn in tests:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        try:
            fn()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            import traceback

            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1
            failures.append((name, str(e)))

    print(f"\n{'=' * 50}")
    print(f"  E2E Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
