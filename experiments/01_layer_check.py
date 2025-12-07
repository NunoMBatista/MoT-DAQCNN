import sys
from pathlib import Path


# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
from src.layers.daqk import DAQKLayer


def main():
    print("--- DAQK Layer Smoke Test ---")

    batch_size = 10
    n_qubits = 4

    # Random input angles in [0, pi]
    x = torch.rand(batch_size, n_qubits) * torch.pi

    layer = DAQKLayer(n_qubits=n_qubits, grid_size=2, scaling_factor=1.0, mode="trotter")

    with torch.no_grad():
        out = layer(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    assert out.shape == (batch_size, n_qubits), "Output shape mismatch"
    print("Layer forward pass succeeded.")


if __name__ == "__main__":
    main()
