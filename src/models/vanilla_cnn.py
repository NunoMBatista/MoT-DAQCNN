import torch
import torch.nn as nn
from src.utils.training_utils import build_classification_head

class VanillaCNN(nn.Module):
    """
    A standard Classical CNN that bypasses any custom kernel expansion.
    It feeds the raw image directly into the paper's classification head.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        head_hidden_channels: int = 64,
    ):
        super().__init__()
        
        # Feed raw pixels (1 channel for grayscale) directly into the head
        self.head = build_classification_head(
            in_channels, 
            num_classes, 
            dropout, 
            activation, 
            hidden_channels=head_hidden_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs live on the same device as the parameters
        target_device = next(self.head.parameters()).device
        if x.device != target_device:
            x = x.to(target_device)
        return self.head(x)
