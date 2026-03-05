"""
benchmark_batch_size.py

Benchmarks the effect of BATCH_SIZE on quantum dataset generation speed,
using the best device/interface combination found by benchmark_devices.py
(lightning.gpu + autograd) on the 4x4 kernel.

What BATCH_SIZE actually means here
------------------------------------
BATCH_SIZE is the number of *images* fed to QuantumConv2d at once.
Inside the forward pass, each image is first unfolded into patches:

    patches_per_image = ((28 - kernel_size) // stride + 1) ** 2

So the actual tensor sent to the QNode has shape:

    (BATCH_SIZE * patches_per_image, n_qubits)

For a 4x4 kernel with stride=4 on 28x28 images:
    patches_per_image = 7 * 7 = 49
    n_qubits          = 16
    QNode input shape = (BATCH_SIZE * 49, 16)

The benchmark times wall-clock seconds for processing a fixed number of
images under each batch size, then reports:
  - total time
  - time per image
  - effective throughput (images/sec)
  - projected time to process all 3 splits of breast_mnist (546+78+156 = 780 images)

Usage:
    python experiments/benchmark_batch_size.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.layers.quantum_convolution import QuantumConv2d
from src.physics.kernel_topologies import get_4x4_kernel_set

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KERNEL_SIZE = 4
STRIDE = 4
TOPOLOGY_NAMES = ["kings"]  # single topology keeps runtime manageable;
# cost scales linearly with more topologies
SCALING_FACTOR = 1.0
EVOLUTION_TIME = 2.5
EVOLUTION_MODE = "trotter"
IMAGE_SIZE = 28  # MedMNIST standard

QUANTUM_DEVICE = "lightning.gpu"
INTERFACE = "autograd"

# Batch sizes to sweep
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

# How many images to time per batch size.
# Keep small so the full sweep finishes quickly; results scale linearly.
N_WARMUP_IMAGES = 2  # run these before starting the clock (device/JIT warm-up)
N_TIMED_IMAGES = 6  # images actually measured

# breast_mnist split sizes (for projected time calculation)
BREAST_MNIST_IMAGES = {"train": 546, "val": 78, "test": 156}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_random_images(n_images, image_size):
    """Return a random float32 tensor shaped (n_images, 1, H, W) in [0, 1]."""
    return torch.rand(n_images, 1, image_size, image_size, dtype=torch.float32)


def build_q_conv():
    """Instantiate a QuantumConv2d with the benchmark settings."""
    print("  Building QuantumConv2d...", end=" ", flush=True)
    q = QuantumConv2d(
        in_channels=1,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        kernel_topology_names=TOPOLOGY_NAMES,
        scaling_factor=SCALING_FACTOR,
        evolution_time=EVOLUTION_TIME,
        mode=EVOLUTION_MODE,
        quantum_device=QUANTUM_DEVICE,
        interface=INTERFACE,
    )
    print("done")
    return q


def run_images(q_conv, images, batch_size):
    """
    Push `images` through q_conv in batches of `batch_size`.
    Returns list of output tensors (one per batch).
    """
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        with torch.no_grad():
            out = q_conv(batch)  # (batch, out_channels, h_out, w_out)
        outputs.append(out.detach().cpu())
    return outputs


def patches_per_image(kernel_size, stride, image_size):
    side = (image_size - kernel_size) // stride + 1
    return side * side


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("  BATCH SIZE BENCHMARK")
    print("=" * 70)
    print(f"  Device        : {QUANTUM_DEVICE}")
    print(f"  Interface     : {INTERFACE}")
    print(
        f"  Kernel        : {KERNEL_SIZE}x{KERNEL_SIZE}  ({KERNEL_SIZE**2} qubits, "
        f"Hilbert dim = {2**KERNEL_SIZE**2:,})"
    )
    print(f"  Stride        : {STRIDE}")
    print(f"  Topologies    : {TOPOLOGY_NAMES}  (x{len(TOPOLOGY_NAMES)})")
    print(f"  Evolution     : {EVOLUTION_MODE}, T={EVOLUTION_TIME}")
    ppi = patches_per_image(KERNEL_SIZE, STRIDE, IMAGE_SIZE)
    print(
        f"  Patches/image : {ppi}  ({IMAGE_SIZE}x{IMAGE_SIZE} → "
        f"{(IMAGE_SIZE - KERNEL_SIZE) // STRIDE + 1}x{(IMAGE_SIZE - KERNEL_SIZE) // STRIDE + 1} grid)"
    )
    print(f"  Warmup images : {N_WARMUP_IMAGES}")
    print(f"  Timed images  : {N_TIMED_IMAGES}")
    print()

    # Build the layer once — shared across all batch sizes so device init
    # cost is paid only once (same as in create_quantum_dataset.py)
    print("Initializing quantum layer (shared across all batch sizes):")
    q_conv = build_q_conv()
    print()

    # Pre-generate all the images we'll need
    total_images_needed = N_WARMUP_IMAGES + N_TIMED_IMAGES
    all_images = make_random_images(total_images_needed, IMAGE_SIZE)

    warmup_images = all_images[:N_WARMUP_IMAGES]
    timed_images = all_images[N_WARMUP_IMAGES:]

    # -----------------------------------------------------------------------
    # Sweep
    # -----------------------------------------------------------------------
    results = []  # (batch_size, sec_per_image)

    for bs in BATCH_SIZES:
        print(
            f"  Batch size {bs:>3}  |  QNode input: ({bs * ppi}, {KERNEL_SIZE**2})",
            end="  ",
            flush=True,
        )

        # warm-up
        try:
            run_images(q_conv, warmup_images, bs)
        except Exception as e:
            print(f"WARMUP FAILED: {e}")
            results.append((bs, None))
            continue

        # timed
        try:
            t0 = time.perf_counter()
            run_images(q_conv, timed_images, bs)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"TIMED RUN FAILED: {e}")
            results.append((bs, None))
            continue

        sec_per_image = elapsed / N_TIMED_IMAGES
        results.append((bs, sec_per_image))
        print(
            f"→  {sec_per_image * 1000:7.1f} ms/image  "
            f"({1.0 / sec_per_image:6.2f} img/s)"
        )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    valid = [(bs, s) for bs, s in results if s is not None]
    if not valid:
        print("\n  No valid results. Exiting.")
        return

    valid_sorted = sorted(valid, key=lambda x: x[1])
    fastest_bs, fastest_sec = valid_sorted[0]

    total_breast_mnist = sum(BREAST_MNIST_IMAGES.values())
    n_topos = len(TOPOLOGY_NAMES)

    print()
    print("=" * 70)
    print("  SUMMARY  (ranked fastest → slowest)")
    print("=" * 70)
    print(
        f"  {'Rank':<5} {'Batch':>6} {'ms/image':>10} {'img/s':>8} "
        f"{'×slower':>8}  {'Proj. time (all topos, all splits)':>34}"
    )
    print(f"  {'─' * 5} {'─' * 6} {'─' * 10} {'─' * 8} {'─' * 8}  {'─' * 34}")

    for rank, (bs, sec) in enumerate(valid_sorted, 1):
        slower = sec / fastest_sec
        # Scale projected time to all topologies (linear) and all 3 splits
        proj_sec = sec * total_breast_mnist * n_topos
        proj_min = proj_sec / 60.0
        throughput = 1.0 / sec
        print(
            f"  {rank:<5} {bs:>6} {sec * 1000:>10.1f} {throughput:>8.2f} "
            f"{slower:>8.1f}x  {proj_min:>32.1f} min"
        )

    # Show skipped
    skipped = [(bs,) for bs, s in results if s is None]
    if skipped:
        print(f"\n  Failed batch sizes: {[b for (b,) in skipped]}")

    print()
    print("=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)
    print(f"\n  Fastest batch size : {fastest_bs}")
    print(
        f"  Speed              : {fastest_sec * 1000:.1f} ms/image  "
        f"({1.0 / fastest_sec:.2f} img/s)"
    )
    proj_full = (fastest_sec * total_breast_mnist * n_topos) / 60.0
    proj_scaled = proj_full * (len(get_4x4_kernel_set()))  # all 5 topologies
    print(
        f"\n  Projected generation time for breast_mnist ({total_breast_mnist} images):"
    )
    print(f"    {n_topos} topology  : {proj_full:.1f} min")
    print(f"    5 topologies : {proj_scaled:.1f} min  (kings + chains + block_2x2)")
    print()
    print(f"  Set in create_quantum_dataset.py:")
    print(f"    BATCH_SIZE = {fastest_bs}")
    print()


if __name__ == "__main__":
    main()
