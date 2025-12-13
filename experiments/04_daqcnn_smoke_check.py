import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.daqcnn import DAQCNN


def main():
    print("--- DAQCNN Smoke Check ---")
    torch.manual_seed(0)

    num_classes = 3
    kernel_names = ["kings", "horizontal"]

    model = DAQCNN(
        num_classes=num_classes,
        kernel_size=2,
        stride=1,
        kernel_topology_names=kernel_names,
        scaling_factor=1.0,
        mode="trotter",
        quantum_device="lightning.gpu",
        quantum_device_kwargs={"shots": None},
        dropout=0.0,
        classical_device="cuda",
    )
    model.eval()

    batch_size = 1
    dummy_img = torch.rand(batch_size, 1, 28, 28) * torch.pi

    with torch.no_grad():
        logits = model(dummy_img)

    print(f"Quantum out channels: {model.quantum.out_channels}")
    print(f"Input shape: {tuple(dummy_img.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")

    expected_shape = (batch_size, num_classes)
    assert logits.shape == expected_shape, f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "Found non-finite values in logits"

    print("DAQCNN forward pass succeeded.")


if __name__ == "__main__":
    main()
