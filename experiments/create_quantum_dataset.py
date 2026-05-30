"""
create_quantum_dataset.py

Pass a classical image dataset through quantum kernels and cache the output
as a "quantum dataset" for later comparison and analysis.

Usage:
    python experiments/create_quantum_dataset.py

Modify the PARAMETERS section below to configure everything — dataset, kernel,
device, and interface — before running.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
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
# DEFAULT PARAMETERS - These are overridden if --config is provided
# =============================================================================

# Which dataset to use
DATASET_NAME = "breast_mnist"

# Color space: "RGB", "HSV", or "GRAYSCALE"
COLOR_SPACE = "GRAYSCALE"

# Image resolution (None = auto-infer)
IMAGE_SIZE = None

# Kernel size: 2, 3, or 4
KERNEL_SIZE = 3

# Stride for the convolution
STRIDE = 3

# Which topologies to use
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
SCALING_FACTOR = 1.0

# Evolution time for quantum dynamics
EVOLUTION_TIME = 2.5

# Evolution mode: "trotter" or "exact"
EVOLUTION_MODE = "trotter"

# Encoding mode: "digital" (RY gates) or "analog" (local detuning)
ENCODING_MODE = "digital"

# Include ZZ correlator measurements (⟨Z_i Z_j⟩)
INCLUDE_CORRELATORS = False

# ---------------------------------------------------------------------------
# Noise model parameters (see docs/noise_model.md and src/physics/noise_model.py)
# When NOISE_ENABLED=True the device is forced to "default.mixed" (density-matrix
# simulation).  This is only practical for kernel_size <= 3 (9 qubits).
# ---------------------------------------------------------------------------
NOISE_ENABLED  = False   # master switch
NOISE_T1_US    = 200.0   # Rydberg-state T1 relaxation time [μs]
NOISE_T2_US    = 100.0   # qubit T2 dephasing time [μs]
NOISE_P_GATE   = 0.002   # per-gate single-qubit depolarizing error
NOISE_OMEGA_MHZ = 1.0    # Rabi frequency [MHz] for time-unit conversion

# PennyLane device ("default.qubit", "lightning.qubit", etc.)
QUANTUM_DEVICE = "default.qubit"

# Interface ("autograd", "torch")
INTERFACE = "torch"

# Batch size for processing
BATCH_SIZE = 8

# Which splits to process
SPLITS = ["train", "val", "test"]


def load_params_from_config(config_path):
    """Load and map YAML config keys to the global parameter names."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Use globals() to update the module-level constants
    g = globals()
    
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})

    if "name" in dataset_cfg: g["DATASET_NAME"] = dataset_cfg["name"]
    if "color_space" in dataset_cfg: g["COLOR_SPACE"] = dataset_cfg["color_space"]
    if "image_size" in dataset_cfg: g["IMAGE_SIZE"] = dataset_cfg["image_size"]
    if "batch_size" in dataset_cfg: g["BATCH_SIZE"] = dataset_cfg["batch_size"]
    
    if "kernel_size" in model_cfg: g["KERNEL_SIZE"] = model_cfg["kernel_size"]
    if "stride" in model_cfg: g["STRIDE"] = model_cfg["stride"]
    if "kernel_topology_names" in model_cfg: g["KERNEL_TOPOLOGY_NAMES"] = model_cfg["kernel_topology_names"]
    if "scaling_factor" in model_cfg: g["SCALING_FACTOR"] = model_cfg["scaling_factor"]
    if "evolution_time" in model_cfg: g["EVOLUTION_TIME"] = model_cfg["evolution_time"]
    if "mode" in model_cfg: g["EVOLUTION_MODE"] = model_cfg["mode"]
    if "encoding_mode" in model_cfg: g["ENCODING_MODE"] = model_cfg["encoding_mode"]
    if "include_correlators" in model_cfg: g["INCLUDE_CORRELATORS"] = model_cfg["include_correlators"]
    if "quantum_device" in model_cfg: g["QUANTUM_DEVICE"] = model_cfg["quantum_device"]

    # Noise block (optional; omitting the section leaves noise disabled)
    noise_cfg = cfg.get("noise", {})
    if noise_cfg.get("enabled", False):
        g["NOISE_ENABLED"]   = True
        g["NOISE_T1_US"]     = noise_cfg.get("T1_us",      g["NOISE_T1_US"])
        g["NOISE_T2_US"]     = noise_cfg.get("T2_us",      g["NOISE_T2_US"])
        g["NOISE_P_GATE"]    = noise_cfg.get("p_gate_1q",  g["NOISE_P_GATE"])
        g["NOISE_OMEGA_MHZ"] = noise_cfg.get("omega_mhz",  g["NOISE_OMEGA_MHZ"])

    print(f"Loaded parameters from {config_path}")

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

    # Add ZZ correlator suffix
    if params.get("include_correlators", False):
        filename += "_zz"

    # Add encoding mode suffix
    if params.get("encoding_mode", "digital") == "analog":
        filename += "_analog"

    # Add noise suffix with T1/T2 values so the regime is visible at a glance
    if params.get("noise_enabled", False):
        t1 = int(round(params.get("noise_T1_us", 200)))
        t2 = int(round(params.get("noise_T2_us", 100)))
        filename += f"_noisy-T1={t1}-T2={t2}"

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
    parser = argparse.ArgumentParser(description="QUANTUM DATASET GENERATOR")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    # Chunk mode: process a contiguous index range of ONE split and write a
    # partial ".chunk.npz" file. Used to parallelise a large cache across a job
    # array; merge_quantum_chunks.py reassembles the chunks into a normal cache.
    # Omitting --split keeps the original monolithic behaviour unchanged.
    parser.add_argument("--split", type=str, default=None,
                        help="Chunk mode: process only this split (train/val/test)")
    parser.add_argument("--start", type=int, default=None,
                        help="Chunk mode: first image index (default 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="Chunk mode: one-past-last image index (default = split size)")
    parser.add_argument("--chunk-dir", type=str, default=None,
                        help="Chunk mode: directory to write the chunk file into")
    args = parser.parse_args()

    if args.config:
        load_params_from_config(args.config)

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
    print(f"  ZZ correlators:   {INCLUDE_CORRELATORS}")
    print(f"  Quantum device:   {QUANTUM_DEVICE}")
    print(f"  Interface:        {INTERFACE}")
    print(f"  Batch size:       {BATCH_SIZE}")
    if NOISE_ENABLED:
        print(f"  Noise:            ENABLED")
        print(f"    T1:             {NOISE_T1_US} μs")
        print(f"    T2:             {NOISE_T2_US} μs")
        print(f"    p_gate_1q:      {NOISE_P_GATE}")
        print(f"    omega:          {NOISE_OMEGA_MHZ} MHz")
        print(f"    device forced:  default.mixed")
    else:
        print(f"  Noise:            disabled")
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

    # Build the quantum convolution layer (the fixed, non-trainable feature extractor).
    # When NOISE_ENABLED=True a qml.NoiseModel is created and injected; the device
    # is then forced to "default.mixed" inside DAQKLayer automatically.
    print("Initializing quantum convolution layer...")

    noise_model = None
    if NOISE_ENABLED:
        from src.physics.noise_model import make_rydberg_noise_model
        noise_model, noise_meta = make_rydberg_noise_model(
            T1_us=NOISE_T1_US,
            T2_us=NOISE_T2_US,
            p_gate_1q=NOISE_P_GATE,
            omega_mhz=NOISE_OMEGA_MHZ,
        )
        print(f"  Noise model built: T1_sim={noise_meta['T1_sim_units']:.1f}, "
              f"T2_sim={noise_meta['T2_sim_units']:.1f} sim-units")

    q_conv = QuantumConv2d(
        in_channels=in_channels,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        kernel_topology_names=KERNEL_TOPOLOGY_NAMES,
        scaling_factor=SCALING_FACTOR,
        evolution_time=EVOLUTION_TIME,
        mode=EVOLUTION_MODE,
        quantum_device=QUANTUM_DEVICE,
        interface=INTERFACE,
        include_correlators=INCLUDE_CORRELATORS,
        encoding_mode=ENCODING_MODE,
        noise_model=noise_model,
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
    from itertools import combinations

    n_qubits = KERNEL_SIZE * KERNEL_SIZE
    channel_kernel_map = []
    for topo_name in KERNEL_TOPOLOGY_NAMES:
        # Single-qubit Z measurements
        for qubit_idx in range(n_qubits):
            channel_kernel_map.append(
                {
                    "channel": len(channel_kernel_map),
                    "kernel": topo_name,
                    "qubit": qubit_idx,
                    "measurement": "Z",
                }
            )
        # ZZ correlator measurements
        if INCLUDE_CORRELATORS:
            for qi, qj in combinations(range(n_qubits), 2):
                channel_kernel_map.append(
                    {
                        "channel": len(channel_kernel_map),
                        "kernel": topo_name,
                        "qubit_pair": [qi, qj],
                        "measurement": "ZZ",
                    }
                )
    print(f"Channel-kernel mapping: {len(channel_kernel_map)} entries")
    print()

    # Base metadata — everything that defines the cache identity, shared by the
    # monolithic and chunk paths. Per-split sample counts and created_at are
    # added later (monolithic) or by the merge script (chunk mode).
    base_metadata = {
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
        "include_correlators": INCLUDE_CORRELATORS,
        "encoding_mode": ENCODING_MODE,
        # Noise parameters — stored even when disabled so the cache matcher can
        # unambiguously reject a noisy cache when a clean run is requested.
        "noise_enabled": NOISE_ENABLED,
        "noise_T1_us": NOISE_T1_US    if NOISE_ENABLED else None,
        "noise_T2_us": NOISE_T2_US    if NOISE_ENABLED else None,
        "noise_p_gate_1q": NOISE_P_GATE if NOISE_ENABLED else None,
        "noise_omega_mhz": NOISE_OMEGA_MHZ if NOISE_ENABLED else None,
        "channel_kernel_map": channel_kernel_map,
    }

    # -------------------------------------------------------------------------
    # CHUNK MODE: process one contiguous slice of one split, write a partial file.
    # -------------------------------------------------------------------------
    if args.split is not None:
        split = args.split
        ds = load_medmnist_dataset(
            DATASET_NAME, split, data_root, image_size=resolved_image_size
        )
        split_total = len(ds)
        start = 0 if args.start is None else args.start
        end = split_total if args.end is None else args.end
        assert 0 <= start < end <= split_total, (
            f"Bad range [{start}, {end}) for split '{split}' of size {split_total}"
        )

        from torch.utils.data import Subset
        sub = Subset(ds, range(start, end))
        print(f"--- CHUNK: {split}[{start}:{end}] of {split_total} ---")
        features, labels = process_dataset_through_quantum(
            sub, q_conv, BATCH_SIZE, color_space=COLOR_SPACE,
            desc=f"  {split}[{start}:{end}]",
        )
        # Sanity: produced exactly (end - start) rows, in order.
        assert features.shape[0] == end - start == labels.shape[0]

        chunk_meta = {
            **base_metadata,
            "split": split,
            "start": start,
            "end": end,
            "split_total": split_total,
            "created_at": datetime.now().isoformat(),
        }

        chunk_dir = Path(args.chunk_dir) if args.chunk_dir else output_dir
        chunk_dir.mkdir(parents=True, exist_ok=True)
        cache_base = generate_output_filename(base_metadata)[:-len(".npz")]
        chunk_name = f"{cache_base}__{split}_{start:06d}-{end:06d}.chunk.npz"
        chunk_path = chunk_dir / chunk_name

        np.savez_compressed(
            chunk_path,
            features=features,
            labels=labels,
            chunk_meta=json.dumps(chunk_meta),
        )
        print(f"Saved chunk to: {chunk_path}")
        print(f"  shape {features.shape}, covers {split}[{start}:{end}] / {split_total}")
        return

    # -------------------------------------------------------------------------
    # MONOLITHIC MODE: process all splits and write one full cache (original).
    # -------------------------------------------------------------------------
    results = {}
    for split in SPLITS:
        print(f"--- Processing {split} split ---")
        ds = load_medmnist_dataset(
            DATASET_NAME, split, data_root, image_size=resolved_image_size
        )
        print(f"Loaded {len(ds)} images")
        features, labels = process_dataset_through_quantum(
            ds, q_conv, BATCH_SIZE, color_space=COLOR_SPACE, desc=f"  {split}"
        )
        results[f"{split}_features"] = features
        results[f"{split}_labels"] = labels
        print(f"  Output shape: {features.shape}")
        print()

    metadata = {
        **base_metadata,
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
