"""
create_quantum_dataset.py

Pass a classical image dataset through quantum kernels and cache the output
as a "quantum dataset" for later comparison and analysis.

Usage:
    python experiments/create_quantum_dataset.py

Modify the PARAMETERS section below to configure everything — dataset, kernel,
device, and interface — before running.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.layers.quantum_convolution import QuantumConv2d
from src.utils.color_conversion import rgb_to_grayscale_tensor, rgb_to_hsv_tensor
from src.utils.data import load_medmnist_dataset

# =============================================================================
# PARAMETERS - Edit these to configure the quantum dataset generation
# =============================================================================

# Which dataset to use — any key from DATASET_REGISTRY in src/config.py, including
# MedMNIST+ higher-resolution variants:
#   28×28 (standard): "pneumonia_mnist", "breast_mnist", "tissue_mnist", "oct_mnist", ...
#   64×64 (MedMNIST+): "pneumonia_mnist_64", "oct_mnist_64", "tissue_mnist_64", ...
#   128×128 (MedMNIST+): "pneumonia_mnist_128", "oct_mnist_128", ...
# The image resolution is derived automatically from the dataset name via
# DATASET_IMAGE_SIZE in src/config.py, or can be overridden with IMAGE_SIZE below.
DATASET_NAME = "tissue_mnist"

# Color space: "RGB", "HSV", or "GRAYSCALE"
# HSV: Only V (value) channel is processed with quantum kernels; H and S are passed classically
# GRAYSCALE: RGB is converted to single grayscale channel
COLOR_SPACE = "GRAYSCALE"

# Image resolution passed to the MedMNIST dataset constructor as `size=`.
# Set to None to infer automatically from DATASET_NAME (recommended):
#   "breast_mnist"    -> 28
#   "oct_mnist_64"    -> 64
#   "oct_mnist_128"   -> 128
# Override here only if you want a resolution that differs from the name convention.
IMAGE_SIZE = None  # None = auto-infer from DATASET_NAME

# Kernel size: 2, 3, or 4
KERNEL_SIZE = 3

# Stride for the convolution (use kernel_size for non-overlapping patches)
STRIDE = 3

# Which topologies to use
# For 2x2: ["kings", "horizontal", "vertical", "u_shape"]
# For 3x3: ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
# For 4x4: ["kings", "horizontal_chains", "vertical_chains", "diagonal_chains", "block_2x2"]
KERNEL_TOPOLOGY_NAMES = [
    "kings",
    "horizontal",
    "vertical",
    "cross",
    "ring",
    "chain",
    "star",
    "grid",
]

# Scaling factor for Rydberg Hamiltonian interaction strength
SCALING_FACTOR = 1

# Evolution time for quantum dynamics
EVOLUTION_TIME = 2.5

# Evolution mode: "trotter" (discrete steps, faster) or "exact" (ODE solver, slower)
EVOLUTION_MODE = "trotter"

# -----------------------------------------------------------------------------
# Device & interface — benchmark results (breast_mnist train, single topology):
#
#   Kernel 2x2  (16-dim  Hilbert space) — overhead-dominated, GPU not worth it
#     lightning.qubit | autograd  →   5.6 ms/patch  ~10 min   ★ fastest
#     lightning.qubit | torch     →   5.7 ms/patch  ~10 min
#     default.qubit   | autograd  →   9.2 ms/patch  ~16 min
#     default.qubit   | torch     →  11.9 ms/patch  ~21 min
#     lightning.gpu   | autograd  →  32.5 ms/patch  ~58 min
#
#   Kernel 3x3  (512-dim Hilbert space) — both CPU and GPU competitive
#     lightning.qubit | torch     →  10.4 ms/patch   ~8 min   ★ fastest
#     lightning.gpu   | autograd  →  12.1 ms/patch   ~9 min
#     lightning.qubit | autograd  →  22.9 ms/patch  ~17 min
#     default.qubit   | autograd  →  32.5 ms/patch  ~24 min
#
#   Kernel 4x4  (65536-dim Hilbert space) — GPU wins clearly
#     lightning.gpu   | autograd  →  29.9 ms/patch  ~13 min   ★ fastest
#     lightning.qubit | torch     →  68.2 ms/patch  ~30 min
#     lightning.qubit | autograd  →  81.9 ms/patch  ~37 min
#     default.qubit   | torch     → 244.9 ms/patch ~109 min
#
#   NOTE: JAX (jit or no-jit) was consistently slow across all sizes — avoid.
#   NOTE: lightning.gpu only supports "autograd" interface, not "torch".
#   NOTE: lightning.qubit does not support "jax" interface.
# -----------------------------------------------------------------------------

# PennyLane device to use for simulation.
# Options: "default.qubit", "lightning.qubit", "lightning.gpu"
QUANTUM_DEVICE = "default.qubit"

# Interface connecting PennyLane to the rest of the pipeline.
# Options: "autograd", "torch"
# lightning.gpu  → must use "autograd"
# lightning.qubit → "torch" is fastest (see table above)
# default.qubit  → "autograd" is fastest
INTERFACE = "torch"

# Batch size for processing (adjust based on memory).
# Benchmark results (lightning.gpu + autograd, 4x4 kernel, breast_mnist):
#   Batch  1 → 4073 ms/image  (baseline)
#   Batch  8 → 3771 ms/image  ★ fastest — sweet spot, no gain beyond this
#   Batch 16 → 3773 ms/image  (effectively tied with 8)
#   Batch 32 → 3778 ms/image  (effectively tied with 8)
# The bottleneck is statevector simulation, not data movement, so larger
# batches only marginally amortise Python/device overhead. 8 is the sweet spot.
BATCH_SIZE = 8

# Which splits to process
SPLITS = ["train", "val", "test"]


# =============================================================================
# Helper functions
# =============================================================================


def generate_output_filename(params):
    """
    Generate a descriptive filename for the quantum dataset.
    Format: {dataset}__res{size}_k{kernel}_s{stride}_t{topologies}_ev{evo}_sc{scale}[_gray|_hsv].npz

    The ``res{size}`` segment is omitted for legacy 28×28 datasets so that
    existing cached files are not renamed.
    """
    # Shorten topology names for filename
    topo_short = "-".join([t[:3] for t in params["kernel_topology_names"]])

    image_size = params.get("image_size", 28)
    size_segment = f"res{image_size}_" if image_size != 28 else ""

    filename = (
        f"{params['dataset_name']}__"
        f"{size_segment}"
        f"k{params['kernel_size']}_"
        f"s{params['stride']}_"
        f"t{topo_short}_"
        f"ev{params['evolution_time']:.2f}_"
        f"sc{params['scaling_factor']:.0f}"
    )

    # Add color space suffix
    color_space = params.get("color_space", "RGB")
    if color_space == "HSV":
        filename += "_hsv"
    elif color_space == "GRAYSCALE":
        filename += "_gray"

    filename += ".npz"
    return filename


def process_dataset_through_quantum(
    dataset, q_conv, batch_size, color_space="RGB", desc="Processing"
):
    """
    Run all images in a dataset through the quantum convolution layer.

    Args:
        dataset: Dataset to process
        q_conv: Quantum convolution layer (or None if using HSV with classical H,S)
        batch_size: Batch size for processing
        color_space: "RGB", "HSV", or "GRAYSCALE"
        desc: Progress bar description

    Returns:
        quantum_features: np.array of shape (N, out_channels, H_out, W_out)
        labels: np.array of shape (N,)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: keep order for reproducibility
        num_workers=0,  # Quantum stuff doesn't play nice with multiprocessing
    )

    all_features = []
    all_labels = []

    for images, labels in tqdm(loader, desc=desc):
        # images: (B, C, H, W), labels: (B, 1) or (B,)
        with torch.no_grad():
            if color_space == "HSV" and images.shape[1] == 3:
                # Convert RGB to HSV
                hsv_images = rgb_to_hsv_tensor(images)  # (B, 3, H, W)

                # Extract V channel for quantum processing
                v_channel = hsv_images[:, 2:3, :, :]  # (B, 1, H, W)

                # Process V channel through quantum kernels
                q_out_v = q_conv(v_channel)  # (B, out_channels, H_out, W_out)

                # Downsample H and S channels to match quantum output spatial dimensions
                h_out = q_out_v.shape[2]
                w_out = q_out_v.shape[3]

                # Use average pooling to downsample H and S channels
                from torch.nn.functional import adaptive_avg_pool2d

                h_channel = adaptive_avg_pool2d(
                    hsv_images[:, 0:1, :, :], (h_out, w_out)
                )
                s_channel = adaptive_avg_pool2d(
                    hsv_images[:, 1:2, :, :], (h_out, w_out)
                )

                # Concatenate: [H, S, quantum-V]
                q_out = torch.cat([h_channel, s_channel, q_out_v], dim=1)
            elif color_space == "GRAYSCALE" and images.shape[1] == 3:
                # Convert RGB to grayscale
                gray_images = rgb_to_grayscale_tensor(images)  # (B, 1, H, W)
                # Process grayscale through quantum kernels
                q_out = q_conv(gray_images)  # (B, out_channels, H_out, W_out)
            else:
                # RGB: process through quantum kernels normally
                q_out = q_conv(images)  # (B, out_channels, H_out, W_out)

        all_features.append(q_out.cpu().numpy())

        # Labels might be (B, 1) in medmnist, flatten to (B,)
        lbl = labels.numpy()
        if lbl.ndim > 1:
            lbl = lbl.squeeze(-1)
        all_labels.append(lbl)

    quantum_features = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    return quantum_features, labels_array


def main():
    print("=" * 60)
    print("QUANTUM DATASET GENERATOR")
    print("=" * 60)

    # Print current parameters
    print(f"\nParameters:")
    # Resolve image size: explicit IMAGE_SIZE overrides auto-detection from name
    from src.config import DATASET_IMAGE_SIZE

    resolved_image_size = (
        IMAGE_SIZE
        if IMAGE_SIZE is not None
        else DATASET_IMAGE_SIZE.get(DATASET_NAME, 28)
    )

    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Image size:       {resolved_image_size}x{resolved_image_size}")
    print(f"  Color space:      {COLOR_SPACE}")
    print(f"  Kernel size:      {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"  Stride:           {STRIDE}")
    print(f"  Topologies:       {KERNEL_TOPOLOGY_NAMES}")
    print(f"  Scaling factor:   {SCALING_FACTOR}")
    print(f"  Evolution time:   {EVOLUTION_TIME}")
    print(f"  Evolution mode:   {EVOLUTION_MODE}")
    print(f"  Quantum device:   {QUANTUM_DEVICE}")
    print(f"  Interface:        {INTERFACE}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print()

    # Setup paths
    data_root = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "data" / "quantum_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load one sample to detect number of input channels
    sample_ds = load_medmnist_dataset(
        DATASET_NAME, "train", data_root, image_size=resolved_image_size
    )
    sample_img, _ = sample_ds[0]
    in_channels = sample_img.shape[0]  # (C, H, W)
    print(f"Detected {in_channels} input channel(s)")

    # Determine quantum processing channels based on color space
    if COLOR_SPACE == "HSV" and in_channels == 3:
        # For HSV, only process V channel with quantum kernels
        quantum_in_channels = 1
        print(f"Using HSV: processing only V channel with quantum kernels")
        print(f"H and S channels will be passed classically")
    elif COLOR_SPACE == "GRAYSCALE" and in_channels == 3:
        # For GRAYSCALE, convert RGB to single channel
        quantum_in_channels = 1
        print(f"Using GRAYSCALE: converting RGB to grayscale for quantum processing")
    else:
        # For RGB or already grayscale, process all channels
        quantum_in_channels = in_channels

    # Build the quantum convolution layer
    # This is the fixed (non-trainable) feature extractor
    print("Initializing quantum convolution layer...")
    q_conv = QuantumConv2d(
        in_channels=quantum_in_channels,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        kernel_topology_names=KERNEL_TOPOLOGY_NAMES,
        scaling_factor=SCALING_FACTOR,
        evolution_time=EVOLUTION_TIME,
        mode=EVOLUTION_MODE,
        quantum_device=QUANTUM_DEVICE,
        interface=INTERFACE,
    )

    # Store the output channels info
    quantum_out_channels = q_conv.out_channels

    # For HSV, we add 2 classical channels (H and S)
    if COLOR_SPACE == "HSV" and in_channels == 3:
        out_channels = 2 + quantum_out_channels  # H, S, + quantum V
        print(f"Quantum layer outputs {quantum_out_channels} channels for V channel")
        print(
            f"Total output: {out_channels} channels (2 classical H,S + {quantum_out_channels} quantum V)"
        )
    else:
        out_channels = quantum_out_channels
        print(f"Quantum layer outputs {out_channels} channels per image")

    # Build channel-to-kernel mapping: for each output channel, record which kernel produced it
    n_qubits = KERNEL_SIZE * KERNEL_SIZE
    channel_kernel_map = []
    for topo_name in KERNEL_TOPOLOGY_NAMES:
        for qubit_idx in range(n_qubits):
            channel_kernel_map.append(
                {
                    "channel": len(channel_kernel_map),
                    "kernel": topo_name,
                    "qubit": qubit_idx,
                }
            )
    print(f"Channel-kernel mapping: {len(channel_kernel_map)} entries")
    print()

    # Process each split
    results = {}

    for split in SPLITS:
        print(f"--- Processing {split} split ---")

        # Load the classical dataset
        ds = load_medmnist_dataset(
            DATASET_NAME, split, data_root, image_size=resolved_image_size
        )
        print(f"Loaded {len(ds)} images")

        # Process through quantum layer
        features, labels = process_dataset_through_quantum(
            ds, q_conv, BATCH_SIZE, color_space=COLOR_SPACE, desc=f"  {split}"
        )

        results[f"{split}_features"] = features
        results[f"{split}_labels"] = labels

        print(f"  Output shape: {features.shape}")
        print()

    # Build metadata dict (will be saved as JSON string in the npz)
    metadata = {
        "dataset_name": DATASET_NAME,
        "image_size": resolved_image_size,
        "in_channels": in_channels,
        "color_space": COLOR_SPACE,
        "kernel_size": KERNEL_SIZE,
        "stride": STRIDE,
        "kernel_topology_names": KERNEL_TOPOLOGY_NAMES,
        "num_kernels": len(KERNEL_TOPOLOGY_NAMES),
        "scaling_factor": SCALING_FACTOR,
        "evolution_time": EVOLUTION_TIME,
        "out_channels": out_channels,
        "quantum_out_channels": quantum_out_channels,
        "channel_kernel_map": channel_kernel_map,
        "created_at": datetime.now().isoformat(),
        "train_samples": len(results["train_labels"]),
        "val_samples": len(results["val_labels"]),
        "test_samples": len(results["test_labels"]),
    }

    # Generate filename and save
    output_filename = generate_output_filename(metadata)
    output_path = output_dir / output_filename

    print(f"Saving quantum dataset to: {output_path}")

    # Save everything: features, labels, and metadata
    np.savez_compressed(
        output_path,
        train_features=results["train_features"],
        train_labels=results["train_labels"],
        val_features=results["val_features"],
        val_labels=results["val_labels"],
        test_features=results["test_features"],
        test_labels=results["test_labels"],
        metadata=json.dumps(metadata),  # Store metadata as JSON string
    )

    # Also save a human-readable metadata file alongside
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_path}")
    print()
    print("Done!")
    print(
        f"Quantum dataset saved with {metadata['train_samples']} train, "
        f"{metadata['val_samples']} val, {metadata['test_samples']} test samples."
    )


if __name__ == "__main__":
    main()
