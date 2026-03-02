from .daqk import DAQKLayer
from .kernel_channel_attention_block import KernelChannelAttentionBlock
from .quantum_convolution import QuantumConv2d

# Backward compatibility alias
GroupedSEBlock = KernelChannelAttentionBlock

__all__ = [
    "DAQKLayer",
    "KernelChannelAttentionBlock",
    "GroupedSEBlock",
    "QuantumConv2d",
]
