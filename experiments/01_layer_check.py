import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
from src.layers.daqk import DAQKLayer


def main():
    print("--- DAQK Layer Test ---")

    batch_size = 1
    n_qubits = 4

    # Random input angles in [0, pi]
    x = torch.rand(batch_size, n_qubits) * torch.pi

    kernel_topology_names = ["kings", "horizontal", "vertical"]
    n_kernels = len(kernel_topology_names)

    layer = DAQKLayer(
        n_qubits=n_qubits, 
        grid_size=2, 
        scaling_factor=1.0, 
        mode="trotter",
        kernel_topology_names=kernel_topology_names
    )

    # Forward pass without training
    with torch.no_grad():
        out = layer(x)
        
    print(x)
    print(out)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    assert out.shape == (batch_size, n_qubits*n_kernels), "Output shape mismatch"
    print("Layer forward pass succeeded.")


if __name__ == "__main__":
    main()
