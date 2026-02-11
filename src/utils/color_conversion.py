"""
Color space conversion utilities for MoT-DAQCNN.

Provides centralized functions for converting between color spaces:
- RGB to HSV
- RGB to Grayscale

These conversions are used across dataset creation, visualization, and comparison scripts.
"""

import numpy as np
import torch


def rgb_to_hsv_tensor(rgb_tensor):
    """
    Convert RGB tensor to HSV.

    Args:
        rgb_tensor: torch.Tensor of shape (B, 3, H, W) with values in [0, 1]

    Returns:
        hsv_tensor: torch.Tensor of shape (B, 3, H, W)
                    H in [0, 1], S in [0, 1], V in [0, 1]
    """
    from torchvision.transforms import functional as F

    batch_size = rgb_tensor.shape[0]
    hsv_batch = []

    for i in range(batch_size):
        pil_img = F.to_pil_image(rgb_tensor[i])
        pil_hsv = pil_img.convert("HSV")
        hsv_tensor_single = F.to_tensor(pil_hsv)
        hsv_tensor_single = hsv_tensor_single / 255.0
        hsv_batch.append(hsv_tensor_single)

    hsv = torch.stack(hsv_batch, dim=0)
    return hsv


def rgb_to_grayscale_tensor(rgb_tensor):
    """
    Convert RGB tensor to grayscale using ITU-R BT.601 conversion.

    Args:
        rgb_tensor: torch.Tensor of shape (B, 3, H, W) with values in [0, 1]

    Returns:
        grayscale_tensor: torch.Tensor of shape (B, 1, H, W)
    """
    grayscale = (
        0.299 * rgb_tensor[:, 0:1, :, :]
        + 0.587 * rgb_tensor[:, 1:2, :, :]
        + 0.114 * rgb_tensor[:, 2:3, :, :]
    )
    return grayscale


def rgb_to_grayscale_numpy(rgb_array):
    """
    Convert RGB numpy array to grayscale using ITU-R BT.601 conversion.

    Args:
        rgb_array: np.ndarray of shape (C, H, W) with C=3, values in [0, 1]

    Returns:
        grayscale_array: np.ndarray of shape (1, H, W)
    """
    if rgb_array.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {rgb_array.shape[0]}")

    gray = (
        0.299 * rgb_array[0:1, :, :]
        + 0.587 * rgb_array[1:2, :, :]
        + 0.114 * rgb_array[2:3, :, :]
    )
    return gray


def apply_color_conversion(images, color_space):
    """
    Apply color space conversion to a batch of images.

    Args:
        images: torch.Tensor of shape (B, C, H, W)
        color_space: str, one of "RGB", "HSV", or "GRAYSCALE"

    Returns:
        converted: torch.Tensor with appropriate shape based on color_space
    """
    if color_space == "RGB":
        return images
    elif color_space == "HSV" and images.shape[1] == 3:
        return rgb_to_hsv_tensor(images)
    elif color_space == "GRAYSCALE" and images.shape[1] == 3:
        return rgb_to_grayscale_tensor(images)
    else:
        return images
