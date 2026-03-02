"""
Smoke and sanity tests for Phase 3: Sparse Reconstruction & Final Classifier.

Tests cover:
    1. build_sparse_tensor_fast — shape, zeroing, channel selection
    2. FinalClassifier forward pass and backward
    3. Routing analysis helper
    4. Mini training loop (end-to-end smoke test)

Run with:
    python -m pytest tests/test_final_classifier_phase3.py -v
    or
    python tests/test_final_classifier_phase3.py
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ts_moe_classification_head import (
    FinalClassifier,
    build_final_classifier_from_metadata,
)
from src.models.ts_moe_classification_head_training import (
    compute_routing_analysis,
    train_final_one_epoch,
)
from src.utils.evaluate import evaluate
from src.utils.kernel_mapping import build_kernel_to_channels_map, get_kernel_names
from src.utils.sparse_reconstruction import build_sparse_tensor_fast

# =========================================================================
# Helpers
# =========================================================================


def make_channel_kernel_map(kernel_names, channels_per_kernel):
    mapping = []
    ch = 0
    for name in kernel_names:
        for q in range(channels_per_kernel):
            mapping.append({"channel": ch, "kernel": name, "qubit": q})
            ch += 1
    return mapping


def make_fake_metadata(kernel_names, kernel_size):
    n_qubits = kernel_size * kernel_size
    total_channels = len(kernel_names) * n_qubits
    return {
        "channel_kernel_map": make_channel_kernel_map(kernel_names, n_qubits),
        "out_channels": total_channels,
        "kernel_size": kernel_size,
        "stride": kernel_size,
        "num_kernels": len(kernel_names),
    }


# =========================================================================
# Tests: Sparse Reconstruction
# =========================================================================


class TestSparseReconstruction:
    def _make_groups(self, kernel_names, N):
        """Helper: build channel_groups list from kernel names and N channels each."""
        return [list(range(k * N, (k + 1) * N)) for k in range(len(kernel_names))]

    def test_output_shape_matches_input(self):
        """Sparse tensor should have same shape as quantum features."""
        M, N, H, W, B = 2, 4, 7, 7, 3
        features = torch.randn(B, M * N, H, W)
        routing = torch.randint(0, M, (B, H, W))
        channel_groups = self._make_groups(["alpha", "beta"], N)

        sparse = build_sparse_tensor_fast(features, routing, channel_groups)
        assert sparse.shape == features.shape

    def test_zeros_for_unselected_kernel(self):
        """Channels for non-selected kernels should be zero."""
        M, N, H, W, B = 2, 4, 5, 5, 2
        features = torch.ones(B, M * N, H, W)
        # All patches choose kernel 0
        routing = torch.zeros(B, H, W, dtype=torch.long)
        channel_groups = self._make_groups(["alpha", "beta"], N)
        ch_beta = channel_groups[1]

        sparse = build_sparse_tensor_fast(features, routing, channel_groups)

        for ch in ch_beta:
            assert (sparse[:, ch, :, :] == 0).all(), f"Channel {ch} should be zero"

    def test_selected_kernel_channels_preserved(self):
        """Channels for the selected kernel should keep original values."""
        M, N, H, W, B = 2, 4, 5, 5, 2
        features = torch.randn(B, M * N, H, W)
        # All patches choose kernel 0
        routing = torch.zeros(B, H, W, dtype=torch.long)
        channel_groups = self._make_groups(["alpha", "beta"], N)
        ch_alpha = channel_groups[0]

        sparse = build_sparse_tensor_fast(features, routing, channel_groups)

        for ch in ch_alpha:
            assert torch.allclose(sparse[:, ch, :, :], features[:, ch, :, :]), (
                f"Channel {ch} should match original"
            )

    def test_mixed_routing(self):
        """Different patches can select different kernels."""
        M, N, H, W, B = 2, 4, 4, 4, 1
        features = torch.ones(B, M * N, H, W) * 5.0
        routing = torch.zeros(B, H, W, dtype=torch.long)
        # Patch (0, 0, 0) selects kernel 1
        routing[0, 0, 0] = 1
        channel_groups = self._make_groups(["alpha", "beta"], N)
        ch_alpha, ch_beta = channel_groups[0], channel_groups[1]

        sparse = build_sparse_tensor_fast(features, routing, channel_groups)

        # At (0,0): kernel 1 selected -> alpha channels zero, beta channels filled
        for ch in ch_alpha:
            assert sparse[0, ch, 0, 0] == 0.0
        for ch in ch_beta:
            assert sparse[0, ch, 0, 0] == 5.0

        # At (0,1): kernel 0 selected -> alpha channels filled, beta channels zero
        for ch in ch_alpha:
            assert sparse[0, ch, 0, 1] == 5.0
        for ch in ch_beta:
            assert sparse[0, ch, 0, 1] == 0.0

    def test_three_kernels(self):
        """Sparse reconstruction works with M=3."""
        M, N, H, W, B = 3, 4, 3, 3, 2
        features = torch.randn(B, M * N, H, W)
        routing = torch.randint(0, M, (B, H, W))
        channel_groups = self._make_groups(["a", "b", "c"], N)

        sparse = build_sparse_tensor_fast(features, routing, channel_groups)
        assert sparse.shape == features.shape

        # Verify that for each patch, exactly one kernel group is non-zero
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    selected_k = routing[b, i, j].item()
                    for k, chs in enumerate(channel_groups):
                        vals = sparse[b, chs, i, j]
                        if k == selected_k:
                            assert torch.allclose(vals, features[b, chs, i, j])
                        else:
                            assert (vals == 0).all()


# =========================================================================
# Tests: FinalClassifier Model
# =========================================================================


class TestFinalClassifier:
    def test_forward_shape(self):
        """Output shape should be (B, num_classes)."""
        in_channels = 18
        num_classes = 2
        B, H, W = 4, 13, 13
        model = FinalClassifier(in_channels, num_classes)
        x = torch.randn(B, in_channels, H, W)
        out = model(x)
        assert out.shape == (B, num_classes)

    def test_backward(self):
        """Gradients should flow through the model."""
        in_channels = 8
        num_classes = 2
        B, H, W = 2, 13, 13
        model = FinalClassifier(in_channels, num_classes)
        x = torch.randn(B, in_channels, H, W)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad

    def test_build_from_metadata(self):
        """Factory function should produce correct in_channels (M*N, no mask)."""
        kernel_names = ["kings", "horizontal"]
        metadata = make_fake_metadata(kernel_names, kernel_size=3)
        num_classes = 2

        model = build_final_classifier_from_metadata(metadata, num_classes)
        assert model.in_channels == 18  # 2 kernels * 9 channels each

    def test_different_activations(self):
        """Both relu and gelu activations should work."""
        for act in ("relu", "gelu"):
            model = FinalClassifier(8, 2, activation=act)
            x = torch.randn(2, 8, 13, 13)
            out = model(x)
            assert out.shape == (2, 2)


# =========================================================================
# Tests: Routing Analysis
# =========================================================================


class TestRoutingAnalysis:
    def test_single_class(self):
        """Routing analysis with one class."""
        kernel_names = ["a", "b"]
        routing = torch.zeros(10, 5, 5, dtype=torch.long)
        labels = torch.zeros(10, dtype=torch.long)

        analysis = compute_routing_analysis(routing, labels, kernel_names)
        assert 0 in analysis
        assert analysis[0]["a"] == 1.0
        assert analysis[0]["b"] == 0.0

    def test_two_classes_different_routing(self):
        """Different classes can have different routing distributions."""
        kernel_names = ["a", "b"]
        # Class 0 -> all kernel 0, Class 1 -> all kernel 1
        routing = torch.zeros(10, 3, 3, dtype=torch.long)
        labels = torch.zeros(10, dtype=torch.long)
        routing[5:, :, :] = 1
        labels[5:] = 1

        analysis = compute_routing_analysis(routing, labels, kernel_names)
        assert analysis[0]["a"] == 1.0
        assert analysis[1]["b"] == 1.0

    def test_fractions_sum_to_one(self):
        """Per-class routing fractions should sum to 1."""
        kernel_names = ["a", "b", "c"]
        routing = torch.randint(0, 3, (20, 4, 4))
        labels = torch.randint(0, 3, (20,))

        analysis = compute_routing_analysis(routing, labels, kernel_names)
        for c, per_kernel in analysis.items():
            total = sum(per_kernel.values())
            assert abs(total - 1.0) < 1e-6, f"Class {c} fractions sum to {total}"


# =========================================================================
# Tests: Mini Training Loop
# =========================================================================


class TestMiniTraining:
    def _make_sparse_loaders(
        self, in_channels, num_classes, spatial, n_train=32, n_val=16
    ):
        """Create fake sparse feature loaders."""
        train_x = torch.randn(n_train, in_channels, spatial, spatial)
        train_y = torch.randint(0, num_classes, (n_train,))
        val_x = torch.randn(n_val, in_channels, spatial, spatial)
        val_y = torch.randint(0, num_classes, (n_val,))

        train_loader = DataLoader(
            TensorDataset(train_x, train_y), batch_size=8, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_y), batch_size=8, shuffle=False
        )
        return train_loader, val_loader

    def test_training_loop_runs(self):
        """One epoch of training should run without errors."""
        in_channels = 18
        num_classes = 2
        spatial = 13

        model = FinalClassifier(in_channels, num_classes)
        # Initialize lazy layers
        dummy = torch.randn(1, in_channels, spatial, spatial)
        model(dummy)

        train_loader, _ = self._make_sparse_loaders(in_channels, num_classes, spatial)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        from tqdm import tqdm

        pbar = tqdm(range(1), leave=False)
        loss, acc = train_final_one_epoch(
            model, train_loader, optimizer, "cpu", grad_clip=1.0, epoch_pbar=pbar
        )
        pbar.close()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_evaluate_compatibility(self):
        """FinalClassifier should work with the existing evaluate() utility."""
        in_channels = 18
        num_classes = 2
        spatial = 13

        model = FinalClassifier(in_channels, num_classes)
        dummy = torch.randn(1, in_channels, spatial, spatial)
        model(dummy)

        _, val_loader = self._make_sparse_loaders(in_channels, num_classes, spatial)

        # Simple evaluate (loss, acc)
        val_loss, val_acc = evaluate(model, val_loader, "cpu", split_name="Val")
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)

        # Full metrics evaluate
        metrics = evaluate(
            model, val_loader, "cpu", split_name="Test", compute_full_metrics=True
        )
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "f1" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics
        assert "probs" in metrics
        assert "labels" in metrics

    def test_loss_decreases(self):
        """Loss should generally decrease over a few epochs on tiny data."""
        in_channels = 8
        num_classes = 2
        spatial = 13

        model = FinalClassifier(in_channels, num_classes, dropout=0.0)
        dummy = torch.randn(1, in_channels, spatial, spatial)
        model(dummy)

        train_loader, _ = self._make_sparse_loaders(
            in_channels, num_classes, spatial, n_train=16
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        from tqdm import tqdm

        losses = []
        for _ in range(5):
            pbar = tqdm(range(1), leave=False)
            loss, _ = train_final_one_epoch(
                model, train_loader, optimizer, "cpu", grad_clip=0.0, epoch_pbar=pbar
            )
            pbar.close()
            losses.append(loss)

        # Loss at the end should be lower than at the start (on average)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# =========================================================================
# Tests: End-to-end sparse pipeline (no student, just reconstruction)
# =========================================================================


class TestEndToEndSparse:
    """Test the full pipeline: features + routing -> sparse -> classifier."""

    def test_sparse_to_classifier(self):
        """Sparse tensors should feed directly into FinalClassifier."""
        M, N = 2, 9
        B, H, W = 4, 13, 13
        num_classes = 2

        features = torch.randn(B, M * N, H, W)
        routing = torch.randint(0, M, (B, H, W))
        channel_groups = [list(range(N)), list(range(N, 2 * N))]

        sparse = build_sparse_tensor_fast(features, routing, channel_groups)

        model = FinalClassifier(M * N, num_classes)
        logits = model(sparse)
        assert logits.shape == (B, num_classes)

    def test_full_train_eval_cycle(self):
        """Full cycle: build sparse -> train classifier -> evaluate."""
        M, N = 2, 4
        B_train, B_val = 16, 8
        H, W = 13, 13
        num_classes = 2

        channel_groups = [list(range(N)), list(range(N, 2 * N))]

        # Build sparse training data
        train_feats = torch.randn(B_train, M * N, H, W)
        train_routing = torch.randint(0, M, (B_train, H, W))
        train_sparse = build_sparse_tensor_fast(
            train_feats, train_routing, channel_groups
        )
        train_labels = torch.randint(0, num_classes, (B_train,))

        val_feats = torch.randn(B_val, M * N, H, W)
        val_routing = torch.randint(0, M, (B_val, H, W))
        val_sparse = build_sparse_tensor_fast(val_feats, val_routing, channel_groups)
        val_labels = torch.randint(0, num_classes, (B_val,))

        train_loader = DataLoader(
            TensorDataset(train_sparse, train_labels), batch_size=8, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_sparse, val_labels), batch_size=8, shuffle=False
        )

        model = FinalClassifier(M * N, num_classes)
        dummy = torch.randn(1, M * N, H, W)
        model(dummy)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        from tqdm import tqdm

        # Train a couple of epochs
        for _ in range(2):
            pbar = tqdm(range(1), leave=False)
            train_final_one_epoch(
                model, train_loader, optimizer, "cpu", grad_clip=1.0, epoch_pbar=pbar
            )
            pbar.close()

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, "cpu", split_name="Val")
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)

        # Full metrics
        metrics = evaluate(
            model, val_loader, "cpu", split_name="Test", compute_full_metrics=True
        )
        assert "accuracy" in metrics
        assert "probs" in metrics


# =========================================================================
# Runner
# =========================================================================


def run_all_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestSparseReconstruction,
        TestMaskChannel,
        TestFinalClassifier,
        TestRoutingAnalysis,
        TestMiniTraining,
        TestEndToEndSparse,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            test_fn = getattr(instance, method_name)
            try:
                test_fn()
                passed += 1
                print(f"  PASS  {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e))
                print(f"  FAIL  {cls.__name__}.{method_name}: {e}")
                traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Phase 3 Tests: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")

    if errors:
        print("\nFailed tests:")
        for cls_name, method_name, e in errors:
            print(f"  {cls_name}.{method_name}: {e}")
        return False
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
