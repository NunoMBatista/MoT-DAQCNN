import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DATASET_REGISTRY
from src.layers.quantum_convolution import QuantumConv2d

DATASET = "path_mnist"  # Any key from DATASET_REGISTRY
IMAGE_INDEX = 0  # Index of image to visualize from the dataset
KERNEL_NAMES = [
    "kings",
    "horizontal",
    "ring",
    "cross",
]  # Available: kings, horizontal, vertical, u_shape (2x2) or kings, horizontal, vertical, cross, ring (3x3)
KERNEL_SIZE = 2  # 2 for 2x2 or 3 for 3x3
STRIDE = 2  # Stride for convolution (use kernel_size for no overlap)
SCALING_FACTOR = 1000.0  # Interaction strength for Rydberg Hamiltonian
EVOLUTION_TIME = 6.28  # Time interval for quantum evolution
COMBINE_RGB = True  # If True, stack R/G/B outputs into one RGB image per cell


def load_dataset(dataset_name):
    data_root = PROJECT_ROOT / "data"
    os.makedirs(data_root, exist_ok=True)

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    _, class_name = DATASET_REGISTRY[dataset_name]

    import medmnist

    dataset_class = getattr(medmnist, class_name)
    ds = dataset_class(
        split="train",
        download=True,
        root=str(data_root),
        transform=transforms.ToTensor(),
    )
    return ds


def compute_kernel_similarity(quantum_channels, num_kernels, n_qubits, in_channels):
    """Compute mean absolute difference between every pair of kernels.

    For RGB, the quantum output has shape (out_channels * in_channels, H, W)
    laid out as [R_kernel0_q0 ... R_kernelN_qM, G_kernel0_q0 ... , B_...].
    We aggregate across all color channels so the similarity reflects the
    full kernel behaviour, not just one color slice.
    """
    out_channels = num_kernels * n_qubits  # channels per color
    kernel_outputs = []

    for k in range(num_kernels):
        parts = []
        for c in range(in_channels):
            offset = c * out_channels
            start = offset + k * n_qubits
            end = start + n_qubits
            parts.append(quantum_channels[start:end].flatten().cpu().numpy())
        kernel_outputs.append(np.concatenate(parts))

    similarities = np.zeros((num_kernels, num_kernels))
    for i in range(num_kernels):
        for j in range(num_kernels):
            similarities[i, j] = np.mean(np.abs(kernel_outputs[i] - kernel_outputs[j]))

    return similarities


def main():
    print(f"Loading {DATASET} dataset...")
    dataset = load_dataset(DATASET)
    image, label = dataset[IMAGE_INDEX]
    in_channels = image.shape[0]  # 1 for grayscale, 3 for RGB
    print(
        f"Selected image {IMAGE_INDEX} with label {label.item()}, "
        f"channels={in_channels}"
    )

    print(f"Initializing quantum convolution with kernels: {KERNEL_NAMES}")
    q_conv = QuantumConv2d(
        in_channels=in_channels,
        kernel_size=KERNEL_SIZE,
        kernel_topology_names=KERNEL_NAMES,
        stride=STRIDE,
        scaling_factor=SCALING_FACTOR,
        evolution_time=EVOLUTION_TIME,
        mode="trotter",
        quantum_device="default.qubit",
    )

    print("Processing image through quantum kernels...")
    image_batch = image.unsqueeze(0)
    quantum_output = q_conv(image_batch)
    print(f"Quantum processing complete! Output shape: {quantum_output.shape}")

    num_kernels = len(KERNEL_NAMES)
    n_qubits = KERNEL_SIZE * KERNEL_SIZE
    out_channels = num_kernels * n_qubits  # channels produced per color channel
    quantum_channels = quantum_output.squeeze(0).detach()

    # ------------------------------------------------------------------ #
    # For RGB the output layout after torch.cat in QuantumConv2d is:
    #   [R: kernel0_q0 .. kernelN_qM | G: kernel0_q0 .. | B: kernel0_q0 ..]
    #
    # COMBINE_RGB=True  → stack R,G,B into one RGB image per (kernel, qubit)
    #                     → num_kernels rows, same as grayscale
    # COMBINE_RGB=False → one row per (color_channel, kernel)
    #                     → in_channels * num_kernels rows
    # For grayscale the flag is ignored.
    # ------------------------------------------------------------------ #
    combine = COMBINE_RGB and in_channels == 3
    color_names = [""] if in_channels == 1 else ["R", "G", "B"]

    if combine:
        total_kernel_rows = num_kernels
    else:
        total_kernel_rows = in_channels * num_kernels
    total_rows = 1 + total_kernel_rows  # 1 for the original image row

    fig, axes = plt.subplots(
        total_rows,
        n_qubits,
        figsize=(n_qubits * 2.2, total_rows * 2.2),
    )
    if total_rows == 1:
        axes = axes.reshape(1, -1)

    # --- Row 0: original image ---
    for i in range(n_qubits):
        ax = axes[0, i]
        if i == 0:
            if in_channels == 1:
                ax.imshow(image.squeeze().numpy(), cmap="gray")
            else:
                # (C, H, W) -> (H, W, C)
                ax.imshow(image.permute(1, 2, 0).numpy())
            ax.set_title(f"Original\n(Label: {label.item()})")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Quantum output rows ---
    if combine:
        # One row per kernel; each cell is an RGB image built from the 3
        # color-channel outputs for that (kernel, qubit).
        for k_idx in range(num_kernels):
            row = 1 + k_idx
            for q_idx in range(n_qubits):
                rgb_planes = []
                for c_idx in range(3):
                    ch = c_idx * out_channels + k_idx * n_qubits + q_idx
                    plane = quantum_channels[ch].cpu().numpy()
                    rgb_planes.append(plane)
                # Stack to (H, W, 3) and normalise to [0, 1] for imshow
                rgb = np.stack(rgb_planes, axis=-1)
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

                ax = axes[row, q_idx]
                ax.imshow(rgb)
                if q_idx == 0:
                    ax.set_ylabel(
                        KERNEL_NAMES[k_idx], rotation=0, ha="right", va="center"
                    )
                if row == 1:
                    ax.set_title(f"Qubit {q_idx}")
                ax.set_xticks([])
                ax.set_yticks([])
    else:
        # Separate rows per (color_channel, kernel)
        row = 1
        for c_idx in range(in_channels):
            color_offset = c_idx * out_channels
            for k_idx in range(num_kernels):
                for q_idx in range(n_qubits):
                    channel_idx = color_offset + k_idx * n_qubits + q_idx
                    ax = axes[row, q_idx]
                    ax.imshow(
                        quantum_channels[channel_idx].cpu().numpy(), cmap="viridis"
                    )

                    # Row label: "kings" or "R · kings"
                    if q_idx == 0:
                        prefix = f"{color_names[c_idx]} · " if in_channels > 1 else ""
                        ax.set_ylabel(
                            f"{prefix}{KERNEL_NAMES[k_idx]}",
                            rotation=0,
                            ha="right",
                            va="center",
                        )

                    # Column header on the very first kernel row only
                    if row == 1:
                        ax.set_title(f"Qubit {q_idx}")

                    ax.set_xticks([])
                    ax.set_yticks([])
                row += 1

    print("Creating visualization...")
    plt.tight_layout()

    output_path = (
        PROJECT_ROOT / "outputs" / f"quantum_viz_{DATASET}_img{IMAGE_INDEX}.png"
    )
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # --- Kernel similarity heatmap ---
    if num_kernels > 1:
        print("Computing kernel similarity...")
        similarities = compute_kernel_similarity(
            quantum_channels, num_kernels, n_qubits, in_channels
        )
        all_identical = similarities.max() == 0.0

        fig_sim, ax_sim = plt.subplots(figsize=(8, 6))
        cmap = "Greys" if all_identical else "coolwarm_r"
        im = ax_sim.imshow(similarities, cmap=cmap, aspect="auto")

        ax_sim.set_xticks(range(num_kernels))
        ax_sim.set_yticks(range(num_kernels))
        ax_sim.set_xticklabels(KERNEL_NAMES, rotation=45, ha="right")
        ax_sim.set_yticklabels(KERNEL_NAMES)

        max_val = similarities.max()
        num_decimals = (
            max(3, int(np.ceil(-np.log10(max_val))) + 2) if max_val > 0 else 4
        )

        for i in range(num_kernels):
            for j in range(num_kernels):
                text_color = "white" if similarities[i, j] > max_val * 0.5 else "black"
                ax_sim.text(
                    j,
                    i,
                    f"{similarities[i, j]:.{num_decimals}f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        title = "Kernel Difference (Mean Absolute Difference)"
        if all_identical:
            title += "\n⚠ WARNING: All kernels produce identical outputs!"
        ax_sim.set_title(title)
        plt.colorbar(im, ax=ax_sim, label="Distance")
        plt.tight_layout()

        similarity_path = (
            PROJECT_ROOT
            / "outputs"
            / f"kernel_similarity_{DATASET}_img{IMAGE_INDEX}.png"
        )
        plt.savefig(similarity_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {similarity_path}")

    plt.show()


if __name__ == "__main__":
    main()
