"""
Unit and smoke tests for Phase 2: Student Gatekeeper.

Tests cover:
    - StudentGatekeeper model (forward, predict, parameter count)
    - build_student_from_metadata factory
    - Routing label generation from Teacher
    - Patch extraction from images
    - Student DataLoader creation
    - Training loop (one epoch)
    - Evaluation and agreement computation
    - Routing confusion matrix computation
    - Prediction distribution computation
    - Plotting (routing confusion matrix)

Run with:
    python tests/test_student_phase2.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader, TensorDataset

from src.layers.kernel_channel_attention_block import KernelChannelAttentionBlock
from src.models.student_gatekeeper import (
    StudentGatekeeper,
    build_student_from_metadata,
    count_parameters,
)
from src.models.student_training import (
    compute_prediction_distribution,
    compute_routing_confusion,
    create_student_dataloaders,
    evaluate_student,
    extract_all_patches,
    extract_patches,
    generate_routing_labels,
    train_student_one_epoch,
)
from src.models.teacher_moe import TeacherMoE
from src.utils.kernel_mapping import build_kernel_to_channels_map
from src.utils.plotting import plot_routing_confusion_matrix

# =========================================================================
# Helpers
# =========================================================================


def _make_kernel_to_channels(kernel_names, kernel_size):
    """Build a simple sequential kernel-to-channels map for testing."""
    n_qubits = kernel_size * kernel_size
    k2c = {}
    ch = 0
    for name in kernel_names:
        k2c[name] = list(range(ch, ch + n_qubits))
        ch += n_qubits
    return k2c


def _make_metadata(kernel_names, kernel_size, stride):
    """Build a fake metadata dict matching cached quantum dataset format."""
    n_qubits = kernel_size * kernel_size
    total_channels = len(kernel_names) * n_qubits

    channel_kernel_map = []
    ch = 0
    for name in kernel_names:
        for q in range(n_qubits):
            channel_kernel_map.append({"channel": ch, "kernel": name, "qubit": q})
            ch += 1

    return {
        "dataset_name": "pneumonia_mnist",
        "kernel_size": kernel_size,
        "stride": stride,
        "kernel_topology_names": kernel_names,
        "num_kernels": len(kernel_names),
        "out_channels": total_channels,
        "channel_kernel_map": channel_kernel_map,
        "color_space": "GRAYSCALE",
    }


def _make_teacher(kernel_names, kernel_size, stride, num_classes=2):
    """Build a TeacherMoE for testing and initialize its lazy layers."""
    n_qubits = kernel_size * kernel_size
    total_channels = len(kernel_names) * n_qubits
    k2c = _make_kernel_to_channels(kernel_names, kernel_size)

    teacher = TeacherMoE(
        num_classes=num_classes,
        total_channels=total_channels,
        kernel_to_channels=k2c,
    )

    spatial = (28 - kernel_size) // stride + 1
    dummy = torch.randn(1, total_channels, spatial, spatial)
    with torch.no_grad():
        teacher(dummy)

    return teacher, total_channels, spatial


def _make_quantum_loader(
    n_samples, total_channels, spatial, num_classes=2, batch_size=8
):
    """Build a DataLoader mimicking cached quantum features."""
    features = torch.randn(n_samples, total_channels, spatial, spatial)
    labels = torch.randint(0, num_classes, (n_samples,))
    ds = TensorDataset(features, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_image_loader(n_samples, in_channels=1, img_size=28, batch_size=8):
    """Build a DataLoader mimicking raw MedMNIST images."""
    images = torch.rand(n_samples, in_channels, img_size, img_size)
    labels = torch.randint(0, 2, (n_samples,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# =========================================================================
# Test classes
# =========================================================================


class TestStudentGatekeeper:
    """Tests for the StudentGatekeeper model."""

    def test_forward_shape(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=(32, 16))
        x = torch.randn(16, 9)
        out = model(x)
        assert out.shape == (16, 2), f"Expected (16, 2), got {out.shape}"

    def test_forward_4d_input(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=3, hidden_dims=(16,))
        x = torch.randn(8, 1, 3, 3)
        out = model(x)
        assert out.shape == (8, 3), f"Expected (8, 3), got {out.shape}"

    def test_forward_2x2_patch(self):
        model = StudentGatekeeper(patch_dim=4, num_kernels=2, hidden_dims=(8,))
        x = torch.randn(4, 4)
        out = model(x)
        assert out.shape == (4, 2)

    def test_predict(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=(16,))
        x = torch.randn(10, 9)
        preds = model.predict(x)
        assert preds.shape == (10,), f"Expected (10,), got {preds.shape}"
        assert preds.min() >= 0
        assert preds.max() <= 1

    def test_predict_values_in_range(self):
        model = StudentGatekeeper(patch_dim=4, num_kernels=5, hidden_dims=(16,))
        x = torch.randn(100, 4)
        preds = model.predict(x)
        assert preds.min() >= 0
        assert preds.max() < 5

    def test_parameter_count_under_5k(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=(32, 16))
        total, trainable = count_parameters(model)
        assert total == trainable, "All params should be trainable"
        assert total < 5000, f"Student has {total} params, target is <5000"

    def test_parameter_count_2x2(self):
        model = StudentGatekeeper(patch_dim=4, num_kernels=2, hidden_dims=(16, 8))
        total, _ = count_parameters(model)
        assert total < 5000, f"Student has {total} params, target is <5000"

    def test_parameter_count_4_kernels(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=4, hidden_dims=(32, 16))
        total, _ = count_parameters(model)
        assert total < 5000, f"Student has {total} params, target is <5000"

    def test_gradients_flow(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=(16,))
        x = torch.randn(4, 9, requires_grad=True)
        labels = torch.randint(0, 2, (4,))
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None, "Gradient should exist"
            assert p.grad.abs().sum() > 0, "Gradient should be nonzero"

    def test_different_hidden_dims(self):
        for dims in [(8,), (16, 8), (64, 32, 16), (32,)]:
            model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=dims)
            x = torch.randn(4, 9)
            out = model(x)
            assert out.shape == (4, 2)

    def test_single_kernel(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=1, hidden_dims=(8,))
        x = torch.randn(4, 9)
        out = model(x)
        assert out.shape == (4, 1)

    def test_attributes_stored(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=3, hidden_dims=(32, 16))
        assert model.patch_dim == 9
        assert model.num_kernels == 3


class TestBuildStudentFromMetadata:
    """Tests for the factory function."""

    def test_basic_build(self):
        metadata = _make_metadata(["kings", "horizontal"], kernel_size=3, stride=3)
        student, kernel_names = build_student_from_metadata(metadata)
        assert isinstance(student, StudentGatekeeper)
        assert student.patch_dim == 9
        assert student.num_kernels == 2
        assert kernel_names == ["kings", "horizontal"]  # sorted by first channel index

    def test_build_2x2(self):
        metadata = _make_metadata(["kings", "horizontal"], kernel_size=2, stride=2)
        student, kernel_names = build_student_from_metadata(metadata)
        assert student.patch_dim == 4
        assert student.num_kernels == 2

    def test_build_4_kernels(self):
        metadata = _make_metadata(
            ["kings", "horizontal", "cross", "ring"], kernel_size=3, stride=3
        )
        student, kernel_names = build_student_from_metadata(metadata)
        assert student.num_kernels == 4
        assert len(kernel_names) == 4

    def test_custom_hidden_dims(self):
        metadata = _make_metadata(["kings", "horizontal"], kernel_size=3, stride=3)
        student, _ = build_student_from_metadata(metadata, hidden_dims=(64, 32))
        total, _ = count_parameters(student)
        # With larger dims, still should work
        assert total > 0

    def test_custom_in_channels(self):
        metadata = _make_metadata(["kings"], kernel_size=3, stride=3)
        student, _ = build_student_from_metadata(metadata, in_channels=3)
        assert student.patch_dim == 3 * 9  # 3 channels * 3*3

    def test_forward_after_build(self):
        metadata = _make_metadata(["kings", "horizontal"], kernel_size=3, stride=3)
        student, _ = build_student_from_metadata(metadata)
        x = torch.randn(8, 9)
        out = student(x)
        assert out.shape == (8, 2)


class TestGenerateRoutingLabels:
    """Tests for label generation from the Teacher."""

    def test_basic_label_generation(self):
        kernel_names = ["kings", "horizontal"]
        teacher, total_ch, spatial = _make_teacher(kernel_names, 3, 3)
        loader = _make_quantum_loader(16, total_ch, spatial)

        labels = generate_routing_labels(teacher, loader, "cpu")

        assert labels.shape == (16, spatial, spatial)
        assert labels.dtype == torch.long
        assert labels.min() >= 0
        assert labels.max() < 2

    def test_label_values_match_argmax(self):
        kernel_names = ["kings", "horizontal"]
        teacher, total_ch, spatial = _make_teacher(kernel_names, 3, 3)

        features = torch.randn(4, total_ch, spatial, spatial)
        loader = DataLoader(
            TensorDataset(features, torch.zeros(4)), batch_size=4, shuffle=False
        )

        labels = generate_routing_labels(teacher, loader, "cpu")

        # Verify against manual argmax
        teacher.eval()
        with torch.no_grad():
            teacher(features)
            alpha = teacher.last_alpha
            expected = alpha.argmax(dim=1)

        assert torch.equal(labels, expected)

    def test_4_kernels(self):
        kernel_names = ["kings", "horizontal", "cross", "ring"]
        teacher, total_ch, spatial = _make_teacher(kernel_names, 3, 3)
        loader = _make_quantum_loader(8, total_ch, spatial)

        labels = generate_routing_labels(teacher, loader, "cpu")
        assert labels.max() < 4

    def test_multiple_batches(self):
        kernel_names = ["kings", "horizontal"]
        teacher, total_ch, spatial = _make_teacher(kernel_names, 3, 3)
        loader = _make_quantum_loader(20, total_ch, spatial, batch_size=4)

        labels = generate_routing_labels(teacher, loader, "cpu")
        assert labels.shape[0] == 20

    def test_teacher_stays_in_eval(self):
        kernel_names = ["kings", "horizontal"]
        teacher, total_ch, spatial = _make_teacher(kernel_names, 3, 3)
        teacher.train()
        loader = _make_quantum_loader(4, total_ch, spatial)

        generate_routing_labels(teacher, loader, "cpu")
        assert not teacher.training, "Teacher should be in eval mode after label gen"


class TestExtractPatches:
    """Tests for patch extraction from images."""

    def test_basic_extract(self):
        images = torch.randn(4, 1, 28, 28)
        patches = extract_patches(images, kernel_size=3, stride=3)

        h_out = (28 - 3) // 3 + 1  # 9
        w_out = h_out
        n_patches = h_out * w_out

        assert patches.shape == (4, n_patches, 9), f"Got {patches.shape}"

    def test_stride_1(self):
        images = torch.randn(2, 1, 28, 28)
        patches = extract_patches(images, kernel_size=3, stride=1)

        h_out = (28 - 3) // 1 + 1  # 26
        n_patches = h_out * h_out  # 676

        assert patches.shape == (2, n_patches, 9)

    def test_2x2_kernel(self):
        images = torch.randn(3, 1, 28, 28)
        patches = extract_patches(images, kernel_size=2, stride=1)

        h_out = 27
        n_patches = h_out * h_out

        assert patches.shape == (3, n_patches, 4)

    def test_multichannel(self):
        images = torch.randn(2, 3, 28, 28)
        patches = extract_patches(images, kernel_size=3, stride=3)

        h_out = 9
        n_patches = h_out * h_out
        patch_dim = 3 * 9  # 3 channels * 3*3

        assert patches.shape == (2, n_patches, patch_dim)

    def test_patch_content(self):
        """Verify that the extracted patch contains the correct pixel values."""
        images = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        patches = extract_patches(images, kernel_size=2, stride=2)

        # Top-left patch should be [0, 1, 4, 5]
        expected_tl = torch.tensor([[0, 1, 4, 5]], dtype=torch.float32)
        assert torch.allclose(patches[0, 0:1, :], expected_tl), (
            f"Expected {expected_tl}, got {patches[0, 0:1, :]}"
        )

    def test_spatial_alignment_with_quantum(self):
        """Patches should match the quantum conv's spatial grid."""
        images = torch.randn(2, 1, 28, 28)
        kernel_size = 3
        stride = 3

        patches = extract_patches(images, kernel_size, stride)
        h_out = (28 - kernel_size) // stride + 1

        # Number of patches should equal h_out * w_out
        assert patches.shape[1] == h_out * h_out


class TestCreateStudentDataloaders:
    """Tests for student DataLoader creation."""

    def test_basic_creation(self):
        N, n_patches, patch_dim = 16, 81, 9
        patches = torch.randn(N, n_patches, patch_dim)
        labels = torch.randint(0, 2, (N, 9, 9))  # H=W=9

        train_loader, val_loader = create_student_dataloaders(
            patches, labels, batch_size=32
        )

        assert val_loader is None
        total_samples = N * n_patches
        assert len(train_loader.dataset) == total_samples

    def test_with_validation(self):
        N, n_patches, patch_dim = 16, 81, 9
        patches = torch.randn(N, n_patches, patch_dim)
        labels = torch.randint(0, 2, (N, 9, 9))

        Nv = 8
        val_patches = torch.randn(Nv, n_patches, patch_dim)
        val_labels = torch.randint(0, 2, (Nv, 9, 9))

        train_loader, val_loader = create_student_dataloaders(
            patches,
            labels,
            batch_size=32,
            val_patches=val_patches,
            val_routing_labels=val_labels,
        )

        assert val_loader is not None
        assert len(train_loader.dataset) == N * n_patches
        assert len(val_loader.dataset) == Nv * n_patches

    def test_batch_shapes(self):
        N, n_patches, patch_dim = 8, 16, 4
        patches = torch.randn(N, n_patches, patch_dim)
        labels = torch.randint(0, 2, (N, 4, 4))

        train_loader, _ = create_student_dataloaders(patches, labels, batch_size=10)

        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[1] == patch_dim
        assert batch_y.dim() == 1
        assert batch_y.dtype == torch.long

    def test_size_mismatch_raises(self):
        patches = torch.randn(10, 81, 9)
        labels = torch.randint(0, 2, (10, 8, 8))  # wrong spatial dim

        try:
            create_student_dataloaders(patches, labels, batch_size=32)
            assert False, "Should have raised an AssertionError"
        except AssertionError:
            pass


class TestTrainStudentOneEpoch:
    """Tests for one epoch of student training."""

    def test_basic_training(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=(16,))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        patches = torch.randn(64, 9)
        labels = torch.randint(0, 2, (64,))
        ds = TensorDataset(patches, labels)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        epoch_pbar = None

        # Use a simple range as epoch_pbar substitute
        from unittest.mock import MagicMock

        epoch_pbar = MagicMock()
        epoch_pbar.set_description = MagicMock()

        loss, acc = train_student_one_epoch(
            model, loader, optimizer, "cpu", grad_clip=0, epoch_pbar=epoch_pbar
        )

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 1

    def test_loss_decreases(self):
        torch.manual_seed(42)
        model = StudentGatekeeper(patch_dim=4, num_kernels=2, hidden_dims=(16,))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Create easy-to-learn data: class 0 = zeros, class 1 = ones
        patches = torch.cat([torch.zeros(50, 4), torch.ones(50, 4)])
        labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        ds = TensorDataset(patches, labels)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        from unittest.mock import MagicMock

        epoch_pbar = MagicMock()
        epoch_pbar.set_description = MagicMock()

        losses = []
        for _ in range(5):
            loss, acc = train_student_one_epoch(
                model, loader, optimizer, "cpu", grad_clip=0, epoch_pbar=epoch_pbar
            )
            losses.append(loss)

        assert losses[-1] < losses[0], (
            f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_with_grad_clip(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=(16,))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        patches = torch.randn(32, 9)
        labels = torch.randint(0, 2, (32,))
        loader = DataLoader(TensorDataset(patches, labels), batch_size=16)

        from unittest.mock import MagicMock

        epoch_pbar = MagicMock()
        epoch_pbar.set_description = MagicMock()

        loss, acc = train_student_one_epoch(
            model, loader, optimizer, "cpu", grad_clip=1.0, epoch_pbar=epoch_pbar
        )
        assert isinstance(loss, float)


class TestEvaluateStudent:
    """Tests for student evaluation."""

    def test_basic_eval(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=(16,))
        patches = torch.randn(32, 9)
        labels = torch.randint(0, 2, (32,))
        loader = DataLoader(TensorDataset(patches, labels), batch_size=16)

        loss, acc = evaluate_student(model, loader, "cpu")

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 1

    def test_perfect_prediction(self):
        """A model that always predicts 0 should have ~50% acc on balanced data."""
        torch.manual_seed(0)
        model = StudentGatekeeper(patch_dim=4, num_kernels=2, hidden_dims=(8,))

        # Override model to always predict class 0 with high confidence
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            # Set bias of last layer to strongly favor class 0
            last_layer = list(model.net.children())[-1]
            last_layer.bias[0] = 10.0
            last_layer.bias[1] = -10.0

        # All-zero labels = all class 0
        patches = torch.randn(20, 4)
        labels = torch.zeros(20).long()
        loader = DataLoader(TensorDataset(patches, labels), batch_size=20)

        loss, acc = evaluate_student(model, loader, "cpu")
        assert acc > 0.99, f"Expected near-perfect accuracy, got {acc}"

    def test_model_in_eval_mode(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=2, hidden_dims=(16,))
        model.train()
        patches = torch.randn(8, 9)
        labels = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(patches, labels), batch_size=8)

        evaluate_student(model, loader, "cpu")
        assert not model.training, "Model should be in eval mode after evaluation"


class TestRoutingConfusion:
    """Tests for routing confusion matrix computation."""

    def test_basic_confusion(self):
        model = StudentGatekeeper(patch_dim=4, num_kernels=2, hidden_dims=(8,))
        patches = torch.randn(32, 4)
        labels = torch.randint(0, 2, (32,))
        loader = DataLoader(TensorDataset(patches, labels), batch_size=16)

        cm, agreement, per_kernel = compute_routing_confusion(
            model, loader, "cpu", num_kernels=2
        )

        assert cm.shape == (2, 2)
        assert cm.sum() == 32
        assert 0 <= agreement <= 1
        assert len(per_kernel) == 2

    def test_perfect_agreement(self):
        """Force model to predict the same as labels."""
        torch.manual_seed(99)
        model = StudentGatekeeper(patch_dim=4, num_kernels=2, hidden_dims=(8,))

        # Rig model to output class 0 always
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            last_layer = list(model.net.children())[-1]
            last_layer.bias[0] = 100.0
            last_layer.bias[1] = -100.0

        # Labels all 0
        patches = torch.randn(20, 4)
        labels = torch.zeros(20).long()
        loader = DataLoader(TensorDataset(patches, labels), batch_size=20)

        cm, agreement, per_kernel = compute_routing_confusion(
            model, loader, "cpu", num_kernels=2
        )

        assert agreement > 0.99
        assert cm[0, 0] == 20
        assert per_kernel[0] > 0.99

    def test_4_kernels_shape(self):
        model = StudentGatekeeper(patch_dim=9, num_kernels=4, hidden_dims=(16,))
        patches = torch.randn(40, 9)
        labels = torch.randint(0, 4, (40,))
        loader = DataLoader(TensorDataset(patches, labels), batch_size=20)

        cm, agreement, per_kernel = compute_routing_confusion(
            model, loader, "cpu", num_kernels=4
        )

        assert cm.shape == (4, 4)
        assert cm.sum() == 40
        assert len(per_kernel) == 4

    def test_agreement_matches_diagonal(self):
        model = StudentGatekeeper(patch_dim=4, num_kernels=3, hidden_dims=(8,))
        patches = torch.randn(60, 4)
        labels = torch.randint(0, 3, (60,))
        loader = DataLoader(TensorDataset(patches, labels), batch_size=30)

        cm, agreement, _ = compute_routing_confusion(
            model, loader, "cpu", num_kernels=3
        )

        expected_agreement = cm.trace() / cm.sum()
        assert abs(agreement - expected_agreement) < 1e-6


class TestPredictionDistribution:
    """Tests for prediction distribution computation."""

    def test_basic_distribution(self):
        model = StudentGatekeeper(patch_dim=4, num_kernels=2, hidden_dims=(8,))
        patches = torch.randn(40, 4)
        labels = torch.zeros(40).long()
        loader = DataLoader(TensorDataset(patches, labels), batch_size=20)

        dist = compute_prediction_distribution(model, loader, "cpu", num_kernels=2)

        assert len(dist) == 2
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-5, f"Distribution sums to {total}"

    def test_all_one_class(self):
        model = StudentGatekeeper(patch_dim=4, num_kernels=2, hidden_dims=(8,))

        # Force all predictions to class 1
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            last_layer = list(model.net.children())[-1]
            last_layer.bias[0] = -100.0
            last_layer.bias[1] = 100.0

        patches = torch.randn(20, 4)
        labels = torch.zeros(20).long()
        loader = DataLoader(TensorDataset(patches, labels), batch_size=20)

        dist = compute_prediction_distribution(model, loader, "cpu", num_kernels=2)

        assert dist[1] > 0.99, f"Expected all class 1, got dist={dist}"
        assert dist[0] < 0.01


class TestPlotRoutingConfusion:
    """Tests for the routing confusion matrix plot."""

    def test_creates_file(self):
        cm = np.array([[40, 10], [5, 45]])
        kernel_names = ["kings", "horizontal"]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "routing_cm.png")
            plot_routing_confusion_matrix(cm, kernel_names, save_path)
            assert os.path.isfile(save_path), "Plot file not created"
            assert os.path.getsize(save_path) > 100, "Plot file is too small"

    def test_4_kernels(self):
        cm = np.random.randint(0, 50, (4, 4))
        kernel_names = ["kings", "horizontal", "cross", "ring"]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "routing_cm_4k.png")
            plot_routing_confusion_matrix(cm, kernel_names, save_path)
            assert os.path.isfile(save_path)

    def test_nested_directory(self):
        cm = np.array([[10, 2], [3, 15]])
        kernel_names = ["a", "b"]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "sub", "dir", "cm.png")
            plot_routing_confusion_matrix(cm, kernel_names, save_path)
            assert os.path.isfile(save_path)


class TestExtractAllPatches:
    """Tests for full-dataset patch extraction (using a fake dataset)."""

    def test_basic_extract_all(self):
        """Use a TensorDataset as a stand-in for a MedMNIST dataset."""
        images = torch.rand(20, 1, 28, 28)
        labels = torch.randint(0, 2, (20, 1))
        ds = TensorDataset(images, labels)

        patches, class_labels = extract_all_patches(
            ds, kernel_size=3, stride=3, color_space="RGB", batch_size=8
        )

        h_out = (28 - 3) // 3 + 1  # 9
        n_patches = h_out * h_out  # 81

        assert patches.shape == (20, n_patches, 9), f"Got {patches.shape}"
        assert class_labels.shape == (20,)

    def test_grayscale_rgb_input(self):
        """GRAYSCALE color_space on 3-channel images should reduce to 1 channel."""
        images = torch.rand(10, 3, 28, 28)
        labels = torch.randint(0, 2, (10, 1))
        ds = TensorDataset(images, labels)

        patches, _ = extract_all_patches(
            ds, kernel_size=3, stride=3, color_space="GRAYSCALE", batch_size=10
        )

        h_out = 9
        # After grayscale conversion: 1 channel * 3*3 = 9 features per patch
        assert patches.shape == (10, h_out * h_out, 9)

    def test_rgb_passthrough(self):
        """RGB color_space should keep all 3 channels."""
        images = torch.rand(5, 3, 28, 28)
        labels = torch.randint(0, 2, (5, 1))
        ds = TensorDataset(images, labels)

        patches, _ = extract_all_patches(
            ds, kernel_size=3, stride=3, color_space="RGB", batch_size=5
        )

        h_out = 9
        # 3 channels * 3*3 = 27 features per patch
        assert patches.shape == (5, h_out * h_out, 27)


class TestEndToEndSmoke:
    """Smoke test combining label generation + patch extraction + training."""

    def test_full_pipeline_synthetic(self):
        torch.manual_seed(42)
        kernel_names = ["kings", "horizontal"]
        kernel_size = 3
        stride = 3
        num_classes = 2

        # Build teacher
        teacher, total_ch, spatial = _make_teacher(
            kernel_names, kernel_size, stride, num_classes
        )

        # Synthetic quantum features
        N_train, N_val = 16, 8
        q_train_loader = _make_quantum_loader(
            N_train, total_ch, spatial, num_classes, batch_size=8
        )
        q_val_loader = _make_quantum_loader(
            N_val, total_ch, spatial, num_classes, batch_size=8
        )

        # Generate routing labels
        train_labels = generate_routing_labels(teacher, q_train_loader, "cpu")
        val_labels = generate_routing_labels(teacher, q_val_loader, "cpu")

        assert train_labels.shape == (N_train, spatial, spatial)
        assert val_labels.shape == (N_val, spatial, spatial)

        # Synthetic raw images -> patches
        train_images = torch.rand(N_train, 1, 28, 28)
        val_images = torch.rand(N_val, 1, 28, 28)

        train_patches = extract_patches(train_images, kernel_size, stride)
        val_patches = extract_patches(val_images, kernel_size, stride)

        # Create student DataLoaders
        train_loader, val_loader = create_student_dataloaders(
            train_patches,
            train_labels,
            batch_size=64,
            val_patches=val_patches,
            val_routing_labels=val_labels,
        )

        # Build student
        metadata = _make_metadata(kernel_names, kernel_size, stride)
        student, s_kernel_names = build_student_from_metadata(metadata)

        # Train for a few epochs
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        from unittest.mock import MagicMock

        epoch_pbar = MagicMock()
        epoch_pbar.set_description = MagicMock()

        for _ in range(3):
            train_student_one_epoch(
                student,
                train_loader,
                optimizer,
                "cpu",
                grad_clip=0,
                epoch_pbar=epoch_pbar,
            )

        # Evaluate
        val_loss, val_acc = evaluate_student(student, val_loader, "cpu")
        assert isinstance(val_loss, float)
        assert 0 <= val_acc <= 1

        # Routing confusion
        cm, agreement, per_kernel = compute_routing_confusion(
            student, val_loader, "cpu", num_kernels=2
        )
        assert cm.shape == (2, 2)
        assert 0 <= agreement <= 1

        # Prediction distribution
        dist = compute_prediction_distribution(
            student, val_loader, "cpu", num_kernels=2
        )
        assert abs(sum(dist.values()) - 1.0) < 1e-5

        print("  Full synthetic pipeline completed successfully!")


# =========================================================================
# Runner
# =========================================================================


def run_all():
    test_classes = [
        TestStudentGatekeeper,
        TestBuildStudentFromMetadata,
        TestGenerateRoutingLabels,
        TestExtractPatches,
        TestCreateStudentDataloaders,
        TestTrainStudentOneEpoch,
        TestEvaluateStudent,
        TestRoutingConfusion,
        TestPredictionDistribution,
        TestPlotRoutingConfusion,
        TestExtractAllPatches,
        TestEndToEndSmoke,
    ]

    passed = 0
    failed = 0
    failures = []

    for cls in test_classes:
        print(f"\n{'─' * 50}")
        print(f"  {cls.__name__}")
        print(f"{'─' * 50}")

        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in sorted(methods):
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                import traceback

                print(f"  ✗ {method_name}: {e}")
                traceback.print_exc()
                failed += 1
                failures.append((f"{cls.__name__}.{method_name}", str(e)))

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
