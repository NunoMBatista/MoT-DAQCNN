import torch
import torch.nn as nn

from src.layers.quantum_convolution import QuantumConv2d
from src.utils.training_utils import build_classification_head


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
        override_quantum_out_channels: Override the quantum layer output channels (used for cached HSV datasets).

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
        override_quantum_out_channels: int = None,
        include_correlators: bool = False,
        head_hidden_channels: int = 64,
    ):
        super().__init__()

        self.bypass_quantum = False

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
            include_correlators=include_correlators,
        )
        self.in_channels = in_channels
        # Backward compatibility alias
        self.quantum = self.quantum_convolutional_layer

        # Use override if provided (for cached HSV datasets), otherwise compute from quantum layer
        if override_quantum_out_channels is not None:
            out_ch = override_quantum_out_channels
        else:
            out_ch = self.quantum_convolutional_layer.out_channels
            # For RGB, the quantum layer outputs 3x more channels
            if in_channels == 3:
                out_ch *= 3

        self.head = build_classification_head(
            out_ch, num_classes, dropout, activation, hidden_channels=head_hidden_channels
        )

        if classical_device is not None:
            self.to(classical_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.bypass_quantum:
            x = self.quantum_convolutional_layer(x).float()
        # Ensure inputs live on the same device as the classical head
        target_device = next(self.head.parameters()).device
        if x.device != target_device:
            x = x.to(target_device)
        return self.head(x)
