import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.layers.quantum_convolution import QuantumConv2d

DATASET = "pneumonia_mnist"  # "pneumonia_mnist" or "breast_mnist"
IMAGE_INDEX = 0  # Index of image to visualize from the dataset
KERNEL_NAMES = [
    "kings",
    "horizontal",
    "vertical",
    "u_shape",
]  # Available: kings, horizontal, vertical, u_shape (2x2) or kings, horizontal, vertical, cross, ring (3x3)
KERNEL_SIZE = 2  # 2 for 2x2 or 3 for 3x3
STRIDE = 2  # Stride for convolution (use kernel_size for no overlap)
SCALING_FACTOR = 1.0  # Interaction strength for Rydberg Hamiltonian
EVOLUTION_TIME = 0.2  # Time interval for quantum evolution


def load_dataset(dataset_name):
    data_root = PROJECT_ROOT / "data"
    os.makedirs(data_root, exist_ok=True)

    if dataset_name == "pneumonia_mnist":
        from medmnist import PneumoniaMNIST

        ds = PneumoniaMNIST(
            split="train",
            download=True,
            root=str(data_root),
            transform=transforms.ToTensor(),
        )
    elif dataset_name == "breast_mnist":
        from medmnist import BreastMNIST

        ds = BreastMNIST(
            split="train",
            download=True,
            root=str(data_root),
            transform=transforms.ToTensor(),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ds


def compute_kernel_similarity(quantum_channels, num_kernels, n_qubits, kernel_names):
    similarities = np.zeros((num_kernels, num_kernels))
    kernel_outputs = []

    for k in range(num_kernels):
        start_idx = k * n_qubits
        end_idx = start_idx + n_qubits
        kernel_out = quantum_channels[start_idx:end_idx].flatten().cpu().numpy()
        kernel_outputs.append(kernel_out)

    for i in range(num_kernels):
        for j in range(num_kernels):
            diff = np.mean(np.abs(kernel_outputs[i] - kernel_outputs[j]))
            similarities[i, j] = diff

    all_identical = similarities.max() == 0.0

    return similarities


def main():
    print(f"Loading {DATASET} dataset...")
    dataset = load_dataset(DATASET)
    image, label = dataset[IMAGE_INDEX]
    print(f"Selected image {IMAGE_INDEX} with label {label.item()}")

    print(f"Initializing quantum convolution with kernels: {KERNEL_NAMES}")
    q_conv = QuantumConv2d(
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
    print("Quantum processing complete!")

    num_kernels = len(KERNEL_NAMES)
    n_qubits = KERNEL_SIZE * KERNEL_SIZE

    fig, axes = plt.subplots(
        num_kernels + 1, n_qubits, figsize=(n_qubits * 2, (num_kernels + 1) * 2)
    )

    if num_kernels == 0:
        axes = axes.reshape(1, -1)

    for i in range(n_qubits):
        ax = axes[0, i] if num_kernels > 0 else axes[i]
        if i == 0:
            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(f"Original\n(Label: {label.item()})")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    quantum_channels = quantum_output.squeeze(0).detach()

    for k_idx in range(num_kernels):
        for q_idx in range(n_qubits):
            channel_idx = k_idx * n_qubits + q_idx
            ax = axes[k_idx + 1, q_idx]
            ax.imshow(quantum_channels[channel_idx].cpu(), cmap="viridis")
            if q_idx == 0:
                ax.set_ylabel(KERNEL_NAMES[k_idx], rotation=0, ha="right", va="center")
            if k_idx == 0:
                ax.set_title(f"Qubit {q_idx}")
            ax.set_xticks([])
            ax.set_yticks([])

    print("Creating visualization...")
    plt.tight_layout()

    output_path = (
        PROJECT_ROOT / "outputs" / f"quantum_viz_{DATASET}_img{IMAGE_INDEX}.png"
    )
    os.makedirs(output_path.parent, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if num_kernels > 1:
        print("Computing kernel similarity...")
        similarities = compute_kernel_similarity(
            quantum_channels, num_kernels, n_qubits, KERNEL_NAMES
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
                text_color = (
                    "white"
                    if similarities[i, j] > similarities.max() * 0.5
                    else "black"
                )
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
