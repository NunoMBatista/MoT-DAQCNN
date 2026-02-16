"""
Loss functions for the TS-MoE (Teacher-Student Mixture of Experts) pipeline.

The key addition is entropy regularization, which pushes the SE block's
alpha weights toward decisive (0 or 1) routing instead of 0uniform mixing.
"""

import torch


def entropy_loss(alpha):
    """Compute the mean entropy of routing weights across all patches.

    Entropy is maximized when alpha is uniform (bad — indecisive routing)
    and minimized when alpha is one-hot (good — decisive routing).

    By adding ``lambda * entropy_loss(alpha)`` to the classification loss,
    we penalize indecisive routing and encourage the SE block to commit
    to a single kernel per patch.

    Args:
        alpha: Tensor of shape (B, M, H, W) with softmax-normalized
            routing weights. M is the number of kernels.

    Returns:
        Scalar tensor: mean entropy across all patches in the batch.
            Range is [0, log(M)] where M = alpha.shape[1].
    """
    # Small epsilon to avoid log(0)
    eps = 1e-8
    # Entropy per patch: -sum_k alpha_k * log(alpha_k), summed over M kernels
    # alpha shape: (B, M, H, W)
    h = -torch.sum(alpha * torch.log(alpha + eps), dim=1)  # (B, H, W)
    return h.mean()


def compute_lambda(epoch, warmup_epochs, lambda_max):
    """Compute the entropy regularization weight with linear annealing.

    Lambda starts at 0 (no entropy penalty — let the network explore freely)
    and ramps linearly to ``lambda_max`` over ``warmup_epochs`` epochs.
    After that it stays at ``lambda_max``.

    Args:
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of epochs over which to anneal lambda.
            If 0, lambda is always lambda_max.
        lambda_max: Maximum value of lambda.

    Returns:
        float: Current lambda value.
    """
    if warmup_epochs <= 0:
        return lambda_max
    progress = min(epoch / warmup_epochs, 1.0)
    return lambda_max * progress
