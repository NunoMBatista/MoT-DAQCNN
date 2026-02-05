import torch
import torch.nn as nn

from src.layers.quantum_convolution import QuantumConv2d


class DAQCNN(nn.Module):
    """DAQCNN: quantum convolution front-end + paper's classical head.

    Args:
        num_classes: Number of output classes.
        kernel_size: Quantum patch size (e.g., 2 or 3). Each patch maps to ``kernel_size**2`` qubits.
        stride: Stride for the quantum convolution unfold.
        kernel_topology_names: Iterable of kernel topology names; defaults per grid size if None.
        scaling_factor: Interaction strength fed to the Rydberg Hamiltonian.
        evolution_time: The time interval for quantum evolution.
        mode: "trotter" (gate-based, GPU-friendly) or "exact" (ODE solver).
        dropout: Dropout probability applied in the classical head.
        activation: Activation function name ("relu" or "gelu").
        quantum_device: Pennylane device name (e.g., "default.qubit", "lightning.gpu").
        quantum_device_kwargs: Extra kwargs for the quantum device (e.g., shots/batch_size).
        classical_device: Torch device to place the full model (e.g., "cuda" or "cpu").
        interface: Interface for quantum execution ("torch", "autograd", "jax").

    Notes:
        - The classical head mirrors the paper: Conv-BN-ReLU -> MaxPool -> Conv-ReLU -> Dropout ->
            Flatten -> Dropout -> Linear.
                - The classical head now uses a lazy Linear so spatial dimensions from the quantum front-end
                    (dependent on kernel_size/stride/input size) are inferred at runtime.
        - Batch size is not a constructor argument; pass batches of shape (B, C, H, W) to forward.
            Use a DataLoader to control how many samples are processed per step.
    """

    def __init__(
        self,
        num_classes: int,
        kernel_size: int = 2,
        stride: int = 1,
        kernel_topology_names=None,
        scaling_factor: float = 1.0,
        evolution_time: float = 0.2,
        mode: str = "trotter",
        dropout: float = 0.1,
        activation: str = "relu",
        quantum_device: str = "default.qubit",
        quantum_device_kwargs=None,
        classical_device=None,
        in_channels: int = 1,
        interface: str = "torch",
        use_jit: bool = False,
    ):
        super().__init__()

        self.quantum_convolutional_layer = QuantumConv2d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            kernel_topology_names=kernel_topology_names,
            scaling_factor=scaling_factor,
            evolution_time=evolution_time,
            mode=mode,
            quantum_device=quantum_device,
            quantum_device_kwargs=quantum_device_kwargs,
            interface=interface,
            use_jit=use_jit,
        )
        self.in_channels = in_channels
        # Backward compatibility alias
        self.quantum = self.quantum_convolutional_layer

        out_ch = self.quantum_convolutional_layer.out_channels
        # For RGB, the quantum layer outputs 3x more channels
        if in_channels == 3:
            out_ch *= 3

        # Select activation function
        act_fn = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        # Paper-like classical head:
        # Conv2D (out_ch -> 64, 2x2, stride=1, no padding) -> BN -> Activation
        # MaxPool2D (2x2, stride=2)
        # Conv2D (64 -> 64, 2x2, stride=1) -> Activation
        # Dropout -> Flatten -> Dropout -> Dense
        self.head = nn.Sequential(
            nn.Conv2d(out_ch, 64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(64),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
            # nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            act_fn,
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.LazyLinear(num_classes),
        )

        if classical_device is not None:
            self.to(classical_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantum_convolutional_layer(x).float()
        # Ensure quantum outputs live on the same device as the classical head
        target_device = next(self.head.parameters()).device
        if x.device != target_device:
            x = x.to(target_device)
        return self.head(x)
