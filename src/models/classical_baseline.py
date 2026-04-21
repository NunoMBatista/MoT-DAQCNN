import math
import torch
import torch.nn as nn
from src.utils.training_utils import build_classification_head

class ClassicalBaselineCNN(nn.Module):
    """
    Classical Baseline CNN designed to match the output shape of the Quantum Conv layer
    for a fair comparison.
    
    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (e.g., 1 for GRAYSCALE, 3 for RGB).
        kernel_size: The spatial size of the convolution filter (e.g., 3 for 3x3 patches).
        stride: The stride of the convolution (e.g., 3).
        out_channels: The number of channels output by the convolution.
                      (e.g., 45 for 1-kernel ZZ, 180 for 4-kernel ZZ).
        dropout: Dropout probability.
        activation: Activation function ("relu" or "gelu").
        head_hidden_channels: Number of hidden channels in the classical head.
        fixed_random_filters: If True, the conv weights are initialized randomly and frozen.
                              If False, they are trainable via backprop.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 3,
        out_channels: int = 180,
        dropout: float = 0.1,
        activation: str = "relu",
        head_hidden_channels: int = 64,
        fixed_random_filters: bool = False,
        raw_features: bool = False,
    ):
        super().__init__()
        
        self.raw_features = raw_features
        self.fixed_random_filters = fixed_random_filters
        self.out_channels = out_channels if not raw_features else in_channels
        
        if not self.raw_features:
            # Classical equivalent of the quantum layer
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False  # The quantum layer effectively acts without a trainable bias
            )
            
            if fixed_random_filters:
                # Initialize with Kaiming uniform
                nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
                # Freeze the weights to simulate a fixed non-trainable feature extractor
                for param in self.conv.parameters():
                    param.requires_grad = False
        
        # The exact same classical head used in DAQCNN
        self.head = build_classification_head(
            self.out_channels, num_classes, dropout, activation, hidden_channels=head_hidden_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.raw_features:
            # Pass through the "quantum-equivalent" classical convolution
            x = self.conv(x).float()
        
        # Ensure inputs live on the same device as the classical head
        target_device = next(self.head.parameters()).device
        if x.device != target_device:
            x = x.to(target_device)
        return self.head(x)
