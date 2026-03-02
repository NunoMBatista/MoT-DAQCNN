"""
create_quantum_dataset.py

Pass a classical image dataset through quantum kernels and cache the output
as a "quantum dataset" for later comparison and analysis.

Usage:
    python experiments/create_quantum_dataset.py

Modify the parameters below to change dataset, kernel size, topologies, etc.
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

# Which dataset to use: "pneumonia_mnist", "breast_mnist", "path_mnist", "derma_mnist", "tissue_mnist"
DATASET_NAME = "tissue_mnist"

# Color space: "RGB", "HSV", or "GRAYSCALE"
# HSV: Only V (value) channel is processed with quantum kernels; H and S are passed classically
# GRAYSCALE: RGB is converted to single grayscale channel
COLOR_SPACE = "GRAYSCALE"

# Kernel size: 2 for 2x2 or 3 for 3x3
KERNEL_SIZE = 3

# Stride for the convolution (use kernel_size for non-overlapping patches)
STRIDE = 3

# Which topologies to use
# For 2x2: ["kings", "horizontal", "vertical", "u_shape"]
# For 3x3: ["kings", "horizontal", "vertical", "cross", "ring"]
# KERNEL_TOPOLOGY_NAMES = ["kings", "horizontal", "vertical", "u_shape"]
KERNEL_TOPOLOGY_NAMES = ["kings", "horizontal", "cross", "ring"]
# KERNEL_TOPOLOGY_NAMES = ["kings"]

# Scaling factor for Rydberg Hamiltonian interaction strength
SCALING_FACTOR = 1
# Evolution time for quantum dynamics
EVOLUTION_TIME = 2.5

# Batch size for processing (adjust based on your memory)
BATCH_SIZE = 1024

# Which splits to process
SPLITS = ["train", "val", "test"]


# =============================================================================
# Helper functions
# =============================================================================


def generate_output_filename(params):
    """
    Generate a descriptive filename for the quantum dataset.
    Format: {dataset}__k{kernel}_s{stride}_t{topologies}_ev{evo}_sc{scale}[_hsv].npz
    """
    # Shorten topology names for filename
    topo_short = "-".join([t[:3] for t in params["kernel_topology_names"]])

    filename = (
        f"{params['dataset_name']}__"
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
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Color space:      {COLOR_SPACE}")
    print(f"  Kernel size:      {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"  Stride:           {STRIDE}")
    print(f"  Topologies:       {KERNEL_TOPOLOGY_NAMES}")
    print(f"  Scaling factor:   {SCALING_FACTOR}")
    print(f"  Evolution time:   {EVOLUTION_TIME}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print()

    # Setup paths
    data_root = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "data" / "quantum_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load one sample to detect number of input channels
    sample_ds = load_medmnist_dataset(DATASET_NAME, "train", data_root)
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
        mode="trotter",
        quantum_device="default.qubit",
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
        ds = load_medmnist_dataset(DATASET_NAME, split, data_root)
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
