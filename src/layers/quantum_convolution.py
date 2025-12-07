import torch
import torch.nn as nn

from src.layers.daqk import DAQKLayer
from src.physics.kernel_topologies import get_2x2_kernel_set, get_3x3_kernel_set


class QuantumConv2d(nn.Module):
    """
    A sliding quantum convolution built from DAQKLayer.

    Each patch of size kernel_size * kernel_size is normalized to [0, pi] and encoded into
    kernel_size^2 qubits and processed by multiple DAQKLayer kernels, each with its own
    atom geometry (topology) encoded via coordinates. The outputs per patch are treated as
    channels in the resulting feature map. With num_kernels = M the output channels are
    M * kernel_size^2
    
    Attributes:
        in_channels (int): Number of input channels (1 for grayscale).
        out_channels (int): Number of output channels (must equal num_kernels * kernel_size^2).
        kernel_size (int): Size of the convolutional kernel (2 or 3).
        stride (int): Stride for the sliding window.
        num_kernels (int | None): Must match the number of topology coordinate sets.
        kernel_topology_names (Iterable[str] | None): Topologies to use; defaults to manual set per grid size (see kernel_topologies).
        scaling_factor (float): The 's' parameter for interaction strength.
        mode (str): 'trotter' (default) or 'exact'.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=None,
        kernel_size=2,
        stride=1,
        num_kernels=None,
        kernel_topology_names=None,
        scaling_factor=1.0,
        mode="trotter",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = kernel_size * kernel_size
        if kernel_topology_names is None:
            if kernel_size == 2:
                default_names = list(get_2x2_kernel_set().keys())
            elif kernel_size == 3:
                default_names = list(get_3x3_kernel_set().keys())
            else:
                default_names = ["kings"]
        else:
            default_names = kernel_topology_names

        self.quantum_kernel = DAQKLayer(
            n_qubits=self.n_qubits,
            grid_size=kernel_size,
            scaling_factor=scaling_factor,
            mode=mode,
            kernel_topology_names=default_names,
        )

        self.num_kernels = self.quantum_kernel.num_kernels if num_kernels is None else num_kernels

        expected_out_channels = self.num_kernels * self.n_qubits
        self.out_channels = out_channels if out_channels is not None else expected_out_channels

        if self.out_channels != expected_out_channels:
            raise ValueError(
                f"QuantumConv2d out_channels must equal num_kernels * kernel_size^2 ({expected_out_channels}), got {self.out_channels}"
            )

        # Sliding window extractor
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)


    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input patches to [0, pi]."""
        x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)  # avoid div by zero
        return x_norm * torch.pi


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        # Normalize inputs to [0, pi]
        x = self._normalize_inputs(x)
        
        # Extract patches of size (ks*ks) from input
        patches = self.unfold(x)  # (B, C*ks*ks, n_patches)
        # The number of patches is the last dimension
        n_patches = patches.shape[-1]

        # reshape to (B*n_patches, n_qubits)
        patches = patches.transpose(1, 2).reshape(-1, self.n_qubits)

        # Compute output spatial dimensions
        h_out = (H - self.kernel_size) // self.stride + 1
        w_out = (W - self.kernel_size) // self.stride + 1

        # Apply quantum kernel layer
        q_out = self.quantum_kernel(patches)  # (B*n_patches, num_kernels * n_qubits)
        
        # Reshape back to (B, out_channels, h_out, w_out) group outputs by image and by patch
        q_out = q_out.reshape(B, n_patches, self.num_kernels * self.n_qubits)
        
        # Transpose to (B, out_channels, n_patches) move channels ahead of patches
        q_out = q_out.transpose(1, 2)  # (B, out_channels, n_patches)
        
        # Reshape to (B, out_channels, h_out, w_out) map the n_patches into spacial h_out w_out
        return q_out.reshape(B, self.out_channels, h_out, w_out)
