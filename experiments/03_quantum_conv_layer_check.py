import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.layers.quantum_convolution import QuantumConv2d


def main(quantum_device, mode):
    torch.manual_seed(0)

    kernel_names = ["kings", "horizontal"]
    q_conv = QuantumConv2d(
        kernel_size=2,
        kernel_topology_names=kernel_names,
        quantum_device=quantum_device,
        mode=mode,
        stride=3,
    )

    # Dummy grayscale image scaled to [0, pi]
    # Input shape is (batch_size, channels, height, width)
    dummy_img = torch.rand(1, 1, 28, 28) * torch.pi
    # THE STRIDE CAN BE TWEAKED TO NOT HAVE ANY OVERLAPS

    output = q_conv(dummy_img)

    print("Input Shape: ", tuple(dummy_img.shape))
    print("Output Shape:", tuple(output.shape))
    print("Expected channels (= num_kernels * k^2):", q_conv.out_channels)


if __name__ == "__main__":
    start_time = time.time()
    main("default.qubit", "trotter")
    end_time = time.time()
    print(f"Execution time for default.qubit: {end_time - start_time:.2f} seconds")

    # start_time = time.time()
    # main("lightning.gpu")
    # end_time = time.time()
    # print(f"Execution time for lightning.gpu: {end_time - start_time:.2f} seconds")
