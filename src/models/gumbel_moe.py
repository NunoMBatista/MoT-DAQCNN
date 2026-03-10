"""
Gumbel-Softmax Mixture-of-Experts (GS-MoE) model.

Uses a lightweight RouterNetwork to produce per-patch routing logits over
kernel topologies, then applies Gumbel-Softmax (optionally with straight-through
hard routing) to select one kernel per patch. A shared classification head
operates on the masked quantum features.

This model works with pre-computed quantum features loaded from cached datasets.
It never runs quantum circuits itself — it just learns how to route and classify
the quantum outputs.

Architecture:
    Cached quantum features (B, M*N, H, W)
        -> RouterNetwork (MLP producing per-patch logits over K kernels)
        -> GumbelRouterBlock (Gumbel-Softmax sampling, soft or hard)
        -> Masked features (B, M*N, H, W) — unselected kernel groups zeroed
        -> Classification head (same as original DAQCNN head)
        -> Logits (B, num_classes)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.training_utils import build_classification_head


class RouterNetwork(nn.Module):
    """Lightweight MLP that produces per-patch routing logits over K kernels.

    The router operates on flat quantum feature vectors (one per spatial patch
    position). It outputs raw logits — no softmax is applied here.

    Architecture:
        LayerNorm(M*N)                          # scale-invariant input
        Linear(M*N, hidden_dim) + LayerNorm + GELU
        Linear(hidden_dim, hidden_dim // 2) + GELU
        Linear(hidden_dim // 2, K)              # raw logits

    Args:
        total_channels: M * N, the full channel count from the cache.
        num_kernels: K, number of kernel topologies.
        hidden_dim: Hidden dimension for the MLP. If None, defaults to
            max(total_channels, 32).
    """

    def __init__(
        self, total_channels: int, num_kernels: int, hidden_dim: Optional[int] = None
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(total_channels, 32)

        self.net = nn.Sequential(
            nn.LayerNorm(total_channels),
            nn.Linear(total_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce routing logits.

        Args:
            x: Tensor of shape (B*H*W, M*N) — one feature vector per patch.

        Returns:
            Tensor of shape (B*H*W, K) — raw logits over kernel topologies.
        """
        return self.net(x)


class GumbelRouterBlock(nn.Module):
    """Applies Gumbel-Softmax sampling to routing logits.

    Attributes:
        temperature: Current Gumbel temperature (annealed externally by the
            training loop). Higher = softer, lower = more discrete.
        hard: If True, use straight-through estimator (one-hot forward,
            soft backward). Set externally by the training loop.

    Args:
        temperature: Initial Gumbel temperature (default 1.0).
        hard: Whether to use hard (one-hot) routing from the start.
    """

    def __init__(self, temperature: float = 1.0, hard: bool = False):
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits: torch.Tensor) -> tuple:
        """Apply Gumbel-Softmax to routing logits.

        Args:
            logits: Tensor of shape (B*H*W, K) — raw router output.

        Returns:
            Tuple of (mask, soft_probs):
                - mask: Tensor of shape (B*H*W, K). If hard=False, this is the
                  soft Gumbel sample. If hard=True, this is a one-hot mask with
                  straight-through gradients.
                - soft_probs: Tensor of shape (B*H*W, K). Always the soft Gumbel
                  sample (used for budget loss regardless of hard/soft mode).
        """
        # Always compute soft sample first (needed for budget loss)
        soft_mask = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)

        if self.hard:
            # Straight-through estimator: forward with hard mask, backward
            # through soft_mask
            hard_mask = (soft_mask == soft_mask.max(dim=-1, keepdim=True)[0]).float()
            mask = hard_mask - soft_mask.detach() + soft_mask  # STE
        else:
            mask = soft_mask

        return mask, soft_mask


class GumbelMoE(nn.Module):
    """Gumbel-Softmax Mixture-of-Experts model for quantum kernel routing.

    Uses a RouterNetwork + GumbelRouterBlock to select one kernel topology
    per spatial patch, then passes the masked features through a shared
    classification head.

    Args:
        num_classes: Number of output classes.
        total_channels: M * N, the full channel count from the cache.
        num_kernels: K, number of kernel topologies.
        channels_per_kernel: N = kernel_size^2, channels per kernel.
        kernel_names: Ordered list of kernel topology names (sorted by first
            channel index, matching cache ordering).
        temperature: Initial Gumbel temperature.
        hard: Whether to start with hard routing.
        sparsity_weight: Lambda for the L1 budget loss on routing probabilities.
        dropout: Dropout probability for the classification head.
        activation: Activation function name ("relu" or "gelu").
        router_hidden_dim: Hidden dim for RouterNetwork. None = auto.
    """

    def __init__(
        self,
        num_classes: int,
        total_channels: int,
        num_kernels: int,
        channels_per_kernel: int,
        kernel_names: list,
        temperature: float = 1.0,
        hard: bool = False,
        sparsity_weight: float = 0.01,
        dropout: float = 0.1,
        activation: str = "relu",
        router_hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.total_channels = total_channels
        self.num_kernels = num_kernels
        self.channels_per_kernel = channels_per_kernel
        self.kernel_names = list(kernel_names)
        self.sparsity_weight = sparsity_weight

        # Router: MLP producing per-patch logits over K kernels
        self.router = RouterNetwork(total_channels, num_kernels, router_hidden_dim)

        # Gumbel-Softmax block: samples from logits
        self.router_block = GumbelRouterBlock(temperature, hard)

        # Classification head: shared CNN head (same as DAQCNN / TeacherMoE)
        self.head = build_classification_head(
            total_channels, num_classes, dropout, activation
        )

        # Stored after each forward pass for logging
        self.last_routing_probs = None

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass: route, mask, classify.

        Args:
            x: Cached quantum features of shape (B, M*N, H, W).

        Returns:
            Tuple of (class_logits, routing_probs):
                - class_logits: Tensor of shape (B, num_classes).
                - routing_probs: Tensor of shape (B*H*W, K) — soft Gumbel
                  probabilities for the budget loss (always soft, even when
                  hard routing is active).
        """
        B, C, H, W = x.shape
        assert C == self.total_channels, (
            f"Expected {self.total_channels} channels, got {C}"
        )

        # --- Router: per-patch routing decision ---
        # Reshape to (B*H*W, M*N): each patch gets its own routing decision
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, M*N)

        # Get routing logits
        logits = self.router(x_flat)  # (B*H*W, K)

        # Apply Gumbel-Softmax
        mask, soft_probs = self.router_block(logits)  # both (B*H*W, K)

        # Store soft probs for logging (detached)
        self.last_routing_probs = soft_probs.detach()

        # --- Masking: zero out unselected kernel groups ---
        # Reshape mask to (B, H, W, K) then permute to (B, K, H, W)
        mask_spatial = mask.reshape(B, H, W, self.num_kernels).permute(0, 3, 1, 2)
        # mask_spatial shape: (B, K, H, W)

        # For each kernel k, multiply all N channels of kernel k by mask[:, k:k+1, :, :]
        # Channel ordering follows kernel_names (sorted by first channel index)
        N = self.channels_per_kernel
        masked_x = x.clone()
        for k in range(self.num_kernels):
            ch_start = k * N
            ch_end = (k + 1) * N
            # mask[:, k:k+1, :, :] broadcasts across N channels
            masked_x[:, ch_start:ch_end, :, :] = (
                x[:, ch_start:ch_end, :, :] * mask_spatial[:, k : k + 1, :, :]
            )

        # --- Classification head ---
        class_logits = self.head(masked_x)  # (B, num_classes)

        return class_logits, soft_probs

    def compute_loss(
        self,
        class_logits: torch.Tensor,
        routing_probs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple:
        """Compute combined task + budget loss.

        Args:
            class_logits: Tensor of shape (B, num_classes) from forward().
            routing_probs: Tensor of shape (B*H*W, K) soft probs from forward().
            labels: Ground truth labels of shape (B,).
            criterion: Loss function (e.g. nn.CrossEntropyLoss).

        Returns:
            Tuple of (total_loss, task_loss_value, budget_loss_value):
                - total_loss: Scalar tensor (task + sparsity_weight * budget).
                - task_loss_value: Float, detached task loss for logging.
                - budget_loss_value: Float, detached budget loss for logging.
        """
        loss_task = criterion(class_logits, labels)

        # Budget loss: L1 penalty on total kernel activations per patch
        # routing_probs shape: (B*H*W, K)
        # sum over kernels per patch, then mean over all patches
        loss_budget = torch.mean(torch.sum(routing_probs, dim=-1))

        total_loss = loss_task + self.sparsity_weight * loss_budget

        return total_loss, loss_task.item(), loss_budget.item()

    def get_routing_stats(self):
        """Compute routing statistics from the last forward pass.

        Returns:
            dict with:
                - "routing_ratio": dict mapping kernel name -> fraction of patches
                  where this kernel had the highest probability.
                - "mean_prob": dict mapping kernel name -> mean soft probability.
            Returns None if no forward pass has been done yet.
        """
        probs = self.last_routing_probs  # (B*H*W, K)
        if probs is None:
            return None

        # Which kernel won at each patch
        winners = probs.argmax(dim=-1)  # (B*H*W,)
        total_patches = winners.numel()

        routing_ratio = {}
        mean_prob = {}

        for k, name in enumerate(self.kernel_names):
            routing_ratio[name] = (winners == k).float().sum().item() / total_patches
            mean_prob[name] = probs[:, k].mean().item()

        return {
            "routing_ratio": routing_ratio,
            "mean_prob": mean_prob,
        }
