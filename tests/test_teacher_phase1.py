"""
Smoke and sanity tests for Phase 1: Teacher MoE components.

Tests cover:
    1. kernel_mapping utilities
    2. KernelChannelAttentionBlock shape transformations and behavior
    3. Entropy loss and lambda annealing
    4. TeacherMoE model forward pass and routing stats
    5. Alpha histogram plotting (file creation)
    6. Mini training loop (end-to-end smoke test)

Run with:
    python -m pytest tests/test_teacher_phase1.py -v
    or
    python tests/test_teacher_phase1.py
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.layers.kernel_channel_attention_block import KernelChannelAttentionBlock
from src.models.teacher_moe import TeacherMoE, build_teacher_from_metadata
from src.utils.kernel_mapping import (
    build_kernel_to_channels_map,
    get_channels_per_kernel,
    get_kernel_names,
    get_num_kernels,
    validate_kernel_map,
)
from src.utils.plotting import plot_alpha_histogram, plot_routing_ratio_over_epochs
from src.utils.training_utils import compute_lambda, entropy_loss

# =========================================================================
# Helpers to build fake data / metadata
# =========================================================================


def make_channel_kernel_map(kernel_names, channels_per_kernel):
    """Build a fake channel_kernel_map like the cached dataset metadata."""
    mapping = []
    ch = 0
    for name in kernel_names:
        for q in range(channels_per_kernel):
            mapping.append({"channel": ch, "kernel": name, "qubit": q})
            ch += 1
    return mapping


def make_fake_metadata(kernel_names, kernel_size):
    """Build a minimal metadata dict mimicking a cached quantum dataset."""
    n_qubits = kernel_size * kernel_size
    total_channels = len(kernel_names) * n_qubits
    return {
        "channel_kernel_map": make_channel_kernel_map(kernel_names, n_qubits),
        "out_channels": total_channels,
        "kernel_size": kernel_size,
        "num_kernels": len(kernel_names),
    }


def make_fake_loaders(
    num_samples, total_channels, spatial_size, num_classes, batch_size=8
):
    """Create fake DataLoaders with random quantum features and labels."""
    features = torch.randn(num_samples, total_channels, spatial_size, spatial_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


# =========================================================================
# 1. Kernel mapping tests
# =========================================================================


class TestKernelMapping:
    def test_build_basic(self):
        """Two kernels, 4 channels each (2x2 grid)."""
        ckm = make_channel_kernel_map(["kings", "horizontal"], 4)
        k2c = build_kernel_to_channels_map(ckm)

        assert set(k2c.keys()) == {"kings", "horizontal"}
        assert k2c["kings"] == [0, 1, 2, 3]
        assert k2c["horizontal"] == [4, 5, 6, 7]

    def test_build_three_kernels(self):
        """Three kernels, 9 channels each (3x3 grid)."""
        ckm = make_channel_kernel_map(["kings", "ring", "cross"], 9)
        k2c = build_kernel_to_channels_map(ckm)

        assert get_num_kernels(k2c) == 3
        assert get_channels_per_kernel(k2c) == 9
        assert k2c["cross"] == list(range(18, 27))

    def test_kernel_names_ordered(self):
        """Kernel names should be sorted by first channel index."""
        ckm = make_channel_kernel_map(["ring", "kings", "cross"], 4)
        k2c = build_kernel_to_channels_map(ckm)
        names = get_kernel_names(k2c)

        # ring has channels 0-3, kings 4-7, cross 8-11
        assert names == ["ring", "kings", "cross"]

    def test_validate_ok(self):
        """Validation passes when map covers all channels."""
        ckm = make_channel_kernel_map(["a", "b"], 4)
        k2c = build_kernel_to_channels_map(ckm)
        validate_kernel_map(k2c, total_channels=8)  # should not raise

    def test_validate_bad(self):
        """Validation fails when total channels don't match."""
        ckm = make_channel_kernel_map(["a", "b"], 4)
        k2c = build_kernel_to_channels_map(ckm)
        try:
            validate_kernel_map(k2c, total_channels=10)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_unequal_channels_raises(self):
        """get_channels_per_kernel should raise if kernels have different counts."""
        k2c = {"a": [0, 1, 2], "b": [3, 4]}
        try:
            get_channels_per_kernel(k2c)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# =========================================================================
# 2. KernelChannelAttentionBlock tests
# =========================================================================


class TestKernelChannelAttentionBlock:
    def test_output_shape_matches_input(self):
        """Output shape must equal input shape."""
        M, N = 2, 4
        block = KernelChannelAttentionBlock(num_kernels=M, channels_per_kernel=N)
        x = torch.randn(3, M * N, 5, 5)
        out = block(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_output_shape_3_kernels_9ch(self):
        """3 kernels x 9 channels = 27 total channels."""
        M, N = 3, 9
        block = KernelChannelAttentionBlock(num_kernels=M, channels_per_kernel=N)
        x = torch.randn(4, 27, 7, 7)
        out = block(x)
        assert out.shape == (4, 27, 7, 7)

    def test_alpha_shape(self):
        """Alpha should have shape (B, M, H, W)."""
        M, N = 2, 4
        block = KernelChannelAttentionBlock(num_kernels=M, channels_per_kernel=N)
        x = torch.randn(3, 8, 5, 5)
        block(x)
        assert block.last_alpha.shape == (3, M, 5, 5)

    def test_alpha_sums_to_one(self):
        """Alpha values across kernels should sum to 1 at every spatial position."""
        M, N = 4, 9
        block = KernelChannelAttentionBlock(num_kernels=M, channels_per_kernel=N)
        x = torch.randn(2, M * N, 3, 3)
        block(x)
        alpha_sum = block.last_alpha.sum(dim=1)  # (B, H, W)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5), (
            f"Alpha sum should be 1.0, got min={alpha_sum.min():.4f} max={alpha_sum.max():.4f}"
        )

    def test_alpha_in_valid_range(self):
        """All alpha values should be in [0, 1]."""
        block = KernelChannelAttentionBlock(num_kernels=3, channels_per_kernel=4)
        x = torch.randn(5, 12, 4, 4)
        block(x)
        assert block.last_alpha.min() >= 0.0
        assert block.last_alpha.max() <= 1.0

    def test_custom_channel_groups(self):
        """Non-sequential channel groups should work."""
        # Kernel 0 uses channels [0, 2, 4], kernel 1 uses [1, 3, 5]
        groups = [[0, 2, 4], [1, 3, 5]]
        block = KernelChannelAttentionBlock(
            num_kernels=2, channels_per_kernel=3, channel_groups=groups
        )
        x = torch.randn(2, 6, 3, 3)
        out = block(x)
        assert out.shape == x.shape

    def test_gradients_flow(self):
        """Gradients should flow through the attention block."""
        block = KernelChannelAttentionBlock(num_kernels=2, channels_per_kernel=4)
        x = torch.randn(2, 8, 3, 3, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient on input"
        # Check gate parameters got gradients
        for p in block.gate.parameters():
            assert p.grad is not None, "No gradient on gate parameter"

    def test_live_alpha_has_grad(self):
        """last_alpha_live should be part of the computation graph."""
        block = KernelChannelAttentionBlock(num_kernels=2, channels_per_kernel=4)
        x = torch.randn(2, 8, 3, 3)
        block(x)
        assert block.last_alpha_live.requires_grad, (
            "last_alpha_live should require grad"
        )
        assert not block.last_alpha.requires_grad, (
            "last_alpha (detached) should NOT require grad"
        )

    def test_single_kernel(self):
        """Edge case: M=1 kernel should work (alpha always 1.0)."""
        block = KernelChannelAttentionBlock(num_kernels=1, channels_per_kernel=4)
        x = torch.randn(2, 4, 3, 3)
        out = block(x)
        assert out.shape == x.shape
        # With softmax over M=1, alpha is always 1.0
        assert torch.allclose(
            block.last_alpha, torch.ones_like(block.last_alpha), atol=1e-5
        )

    def test_wrong_channels_raises(self):
        """Passing wrong number of channels should raise."""
        block = KernelChannelAttentionBlock(num_kernels=2, channels_per_kernel=4)
        x = torch.randn(2, 10, 3, 3)  # 10 != 2*4 = 8
        try:
            block(x)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass


# =========================================================================
# 3. Loss function tests
# =========================================================================


class TestLosses:
    def test_entropy_uniform_is_high(self):
        """Uniform alpha (all equal) should give maximum normalised entropy (1.0)."""
        M = 4
        alpha = torch.ones(2, M, 3, 3) / M  # uniform
        ent = entropy_loss(alpha)
        # After normalisation by log(M), uniform alpha should yield exactly 1.0
        assert abs(ent.item() - 1.0) < 0.01, (
            f"Expected normalised entropy ~1.0, got {ent.item():.4f}"
        )

    def test_entropy_one_hot_is_zero(self):
        """One-hot alpha (decisive) should give ~zero entropy."""
        alpha = torch.zeros(2, 3, 3, 3)
        alpha[:, 0, :, :] = 1.0  # all weight on kernel 0
        ent = entropy_loss(alpha)
        assert ent.item() < 0.001, f"Expected ~0 entropy, got {ent.item():.4f}"

    def test_entropy_gradient_flows(self):
        """Entropy loss should produce gradients on alpha."""
        alpha = torch.randn(2, 3, 4, 4).softmax(dim=1).requires_grad_(True)
        ent = entropy_loss(alpha)
        ent.backward()
        assert alpha.grad is not None

    def test_lambda_annealing(self):
        """Lambda should ramp from 0 to lambda_max linearly."""
        assert compute_lambda(0, 10, 0.5) == 0.0
        assert abs(compute_lambda(5, 10, 0.5) - 0.25) < 1e-6
        assert abs(compute_lambda(10, 10, 0.5) - 0.5) < 1e-6
        # After warmup, stays at max
        assert abs(compute_lambda(20, 10, 0.5) - 0.5) < 1e-6

    def test_lambda_zero_warmup(self):
        """Zero warmup should give lambda_max immediately."""
        assert compute_lambda(0, 0, 0.3) == 0.3


# =========================================================================
# 4. TeacherMoE model tests
# =========================================================================


class TestTeacherMoE:
    def _make_model(self, M=2, kernel_size=2, num_classes=2):
        """Helper to build a TeacherMoE from fake metadata."""
        kernel_names = ["kings", "horizontal", "ring", "cross"][:M]
        metadata = make_fake_metadata(kernel_names, kernel_size)
        return build_teacher_from_metadata(metadata, num_classes=num_classes)

    def test_forward_shape_2kern_2x2(self):
        """2 kernels, 2x2 grid (4 qubits) -> 8 channels."""
        model = self._make_model(M=2, kernel_size=2, num_classes=3)
        # Spatial size needs to be large enough for the CNN head
        # With 2x2 kernel, stride 1 on 28x28: h_out = 27, w_out = 27
        # But we're feeding quantum features directly. For 2x2 stride=1
        # on 28x28, h_out = (28-2)//1 + 1 = 27.
        # The head: Conv(2,s1) -> 26, MaxPool(2,s2) -> 13, Conv(2,s1) -> 12
        x = torch.randn(4, 8, 14, 14)
        # Initialize LazyLinear
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"

    def test_forward_shape_4kern_3x3(self):
        """4 kernels, 3x3 grid (9 qubits) -> 36 channels."""
        model = self._make_model(M=4, kernel_size=3, num_classes=2)
        # Need spatial >= 6 for the head to work:
        # Conv(2,s1)->5, MaxPool(2,s2)->2, Conv(2,s1)->1
        x = torch.randn(2, 36, 6, 6)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 2)

    def test_alpha_stored(self):
        """last_alpha should be populated after forward."""
        model = self._make_model(M=2, kernel_size=2)
        x = torch.randn(3, 8, 14, 14)
        with torch.no_grad():
            model(x)
        assert model.last_alpha is not None
        assert model.last_alpha.shape == (3, 2, 14, 14)

    def test_routing_stats(self):
        """get_routing_stats should return sensible values."""
        model = self._make_model(M=3, kernel_size=2, num_classes=2)
        x = torch.randn(4, 12, 8, 8)
        with torch.no_grad():
            model(x)
        stats = model.get_routing_stats()

        assert stats is not None
        assert set(stats["routing_ratio"].keys()) == set(model.kernel_names)
        assert set(stats["mean_alpha"].keys()) == set(model.kernel_names)
        assert set(stats["alpha_flat"].keys()) == set(model.kernel_names)

        # Routing ratios should sum to 1
        total_ratio = sum(stats["routing_ratio"].values())
        assert abs(total_ratio - 1.0) < 1e-5, f"Routing ratios sum to {total_ratio}"

        # Each alpha_flat should have B*H*W entries
        for name in model.kernel_names:
            expected_len = 4 * 8 * 8
            assert len(stats["alpha_flat"][name]) == expected_len

    def test_backward(self):
        """Gradients should flow through the full model."""
        model = self._make_model(M=2, kernel_size=2, num_classes=2)
        x = torch.randn(2, 8, 14, 14)
        # Init lazy layers
        with torch.no_grad():
            model(x)
        # Now train step
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient on {name}"

    def test_entropy_plus_ce_backward(self):
        """CE + entropy loss should both produce gradients on gate params."""
        model = self._make_model(M=2, kernel_size=2, num_classes=2)
        x = torch.randn(2, 8, 14, 14)
        labels = torch.tensor([0, 1])

        with torch.no_grad():
            model(x)

        logits = model(x)
        ce = nn.CrossEntropyLoss()(logits, labels)
        ent = entropy_loss(model.attention_block.last_alpha_live)
        loss = ce + 0.1 * ent
        loss.backward()

        # Gate parameters should have gradients from both terms
        for p in model.attention_block.gate.parameters():
            assert p.grad is not None, "Gate parameter missing gradient"
            assert p.grad.abs().sum() > 0, "Gate gradient is all zeros"

    def test_kernel_names_match(self):
        """Model's kernel_names should match what we passed in."""
        kernel_names = ["ring", "cross"]
        metadata = make_fake_metadata(kernel_names, kernel_size=3)
        model = build_teacher_from_metadata(metadata, num_classes=2)
        # get_kernel_names sorts by first channel index, which matches
        # the order in make_channel_kernel_map
        assert model.kernel_names == ["ring", "cross"]


# =========================================================================
# 5. Plotting tests (just check files get created)
# =========================================================================


class TestPlotting:
    def test_alpha_histogram_creates_file(self):
        """plot_alpha_histogram should create a PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "test_hist.png")
            vals = torch.rand(1000)
            plot_alpha_histogram(vals, "kings", epoch=5, save_path=path)
            assert os.path.isfile(path), f"Histogram file not created at {path}"

    def test_routing_ratio_creates_file(self):
        """plot_routing_ratio_over_epochs should create a PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "routing.png")
            history = [
                {"kings": 0.6, "ring": 0.4},
                {"kings": 0.55, "ring": 0.45},
                {"kings": 0.7, "ring": 0.3},
            ]
            plot_routing_ratio_over_epochs(history, ["kings", "ring"], path)
            assert os.path.isfile(path), f"Routing plot not created at {path}"


# =========================================================================
# 6. Mini training loop (end-to-end smoke test)
# =========================================================================


class TestMiniTraining:
    def test_training_loop_runs(self):
        """Full mini training loop: forward, CE+entropy loss, backward, step."""
        M, N = 2, 4  # 2 kernels, 2x2 grid
        num_classes = 3
        total_ch = M * N
        spatial = 14
        num_samples = 16
        batch_size = 8

        kernel_names = ["kings", "horizontal"]
        metadata = make_fake_metadata(kernel_names, kernel_size=2)
        model = build_teacher_from_metadata(metadata, num_classes=num_classes)

        # Create fake loaders
        train_loader = make_fake_loaders(
            num_samples, total_ch, spatial, num_classes, batch_size
        )

        # Init lazy layers
        with torch.no_grad():
            sample, _ = next(iter(train_loader))
            model(sample)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ce_fn = nn.CrossEntropyLoss()

        # Train for 3 epochs
        losses_over_epochs = []
        for epoch in range(3):
            model.train()
            epoch_loss = 0.0
            lam = compute_lambda(epoch, warmup_epochs=2, lambda_max=0.1)

            for features, labels in train_loader:
                optimizer.zero_grad()
                logits = model(features)
                ce = ce_fn(logits, labels)
                ent = entropy_loss(model.attention_block.last_alpha_live)
                loss = ce + lam * ent
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses_over_epochs.append(epoch_loss / len(train_loader))

        # Sanity: loss should be finite
        for l in losses_over_epochs:
            assert not (l != l), "Loss is NaN!"  # NaN check
            assert l < 1e6, f"Loss exploded: {l}"

        # Sanity: model should produce valid logits
        model.eval()
        with torch.no_grad():
            sample, _ = next(iter(train_loader))
            logits = model(sample)
            probs = torch.softmax(logits, dim=1)
            assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-4)

        # Sanity: alpha should be valid after eval
        alpha = model.last_alpha
        assert alpha.shape[1] == M
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_evaluate_compatibility(self):
        """The existing evaluate() function should work with TeacherMoE."""
        from src.utils.evaluate import evaluate

        M, N = 2, 4
        num_classes = 2
        metadata = make_fake_metadata(["kings", "horizontal"], kernel_size=2)
        model = build_teacher_from_metadata(metadata, num_classes=num_classes)

        loader = make_fake_loaders(16, M * N, 14, num_classes, batch_size=8)

        # Init lazy layers
        with torch.no_grad():
            sample, _ = next(iter(loader))
            model(sample)

        # evaluate() should work without errors
        loss, acc = evaluate(model, loader, "cpu", split_name="Test")
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

        # Full metrics too
        metrics = evaluate(
            model, loader, "cpu", split_name="Test", compute_full_metrics=True
        )
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "f1" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics
        assert "probs" in metrics
        assert "labels" in metrics

    def test_evaluate_teacher(self):
        """evaluate_teacher should return loss, acc, routing ratios, and alpha weights."""
        from src.models.teacher_moe_training import evaluate_teacher

        kernel_names = ["kings", "horizontal"]
        M, N = 2, 4
        num_samples = 12
        spatial = 8
        metadata = make_fake_metadata(kernel_names, kernel_size=2)
        model = build_teacher_from_metadata(metadata, num_classes=2)

        # Init LazyLinear
        loader = make_fake_loaders(num_samples, M * N, spatial, 2, batch_size=4)
        with torch.no_grad():
            sample, _ = next(iter(loader))
            model(sample)

        val_loss, val_acc, routing_ratio, alpha_vals = evaluate_teacher(
            model, loader, "cpu"
        )

        # Loss and accuracy are valid scalars
        assert isinstance(val_loss, float) and val_loss >= 0.0
        assert isinstance(val_acc, float) and 0.0 <= val_acc <= 1.0

        # Routing ratios cover all kernels and sum to 1
        assert set(routing_ratio.keys()) == set(kernel_names)
        assert abs(sum(routing_ratio.values()) - 1.0) < 1e-5
        for r in routing_ratio.values():
            assert 0.0 <= r <= 1.0

        # Alpha values are flat tensors with num_samples * spatial * spatial entries
        assert set(alpha_vals.keys()) == set(kernel_names)
        expected = num_samples * spatial * spatial
        for name, vals in alpha_vals.items():
            assert len(vals) == expected, (
                f"Kernel '{name}': expected {expected} values, got {len(vals)}"
            )


# =========================================================================
# Run all tests
# =========================================================================


def run_all_tests():
    """Run all test classes and report results."""
    test_classes = [
        TestKernelMapping,
        TestKernelChannelAttentionBlock,
        TestLosses,
        TestTeacherMoE,
        TestPlotting,
        TestMiniTraining,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for cls in test_classes:
        instance = cls()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]

        print(f"\n{'─' * 50}")
        print(f"  {cls.__name__}")
        print(f"{'─' * 50}")

        for method_name in sorted(test_methods):
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✓ {method_name}")
                total_passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                total_failed += 1
                failures.append((cls.__name__, method_name, str(e)))

    print(f"\n{'=' * 50}")
    print(f"  Results: {total_passed} passed, {total_failed} failed")
    print(f"{'=' * 50}")

    if failures:
        print("\nFailures:")
        for cls_name, method_name, msg in failures:
            print(f"  {cls_name}.{method_name}: {msg}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
