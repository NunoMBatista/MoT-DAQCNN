import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.layers.quantum_convolution import QuantumConv2d


def main():
    torch.manual_seed(0)

    q_conv = QuantumConv2d(kernel_size=2, stride=1)

    # Dummy grayscale image scaled to [0, pi]
    # Input shape is (batch_size, channels, height, width)
    dummy_img = torch.rand(1, 1, 6, 6) * torch.pi

    output = q_conv(dummy_img)

    print("Input Shape: ", tuple(dummy_img.shape))
    print("Output Shape:", tuple(output.shape))
    print("Expected channels (= num_kernels * k^2):", q_conv.out_channels)

if __name__ == "__main__":
    main()
