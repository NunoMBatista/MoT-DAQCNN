"""
Ablation Study: Does MoE Routing Actually Help?

This experiment compares multiple strategies:
    1. Single Kernel Baseline: Use only ONE kernel for all patches.
    2. All Kernels Concatenated: Use ALL kernels (no routing, just concat).
    3. Soft MoE Routing (Teacher-style): Soft attention over kernels.
    4. Hard MoE Routing (TS-MoE-style): Hard per-patch routing with sparse tensors.
    5. Real TS-MoE Pipeline: Full Teacher → Student → Final Classifier pipeline.

For each strategy, we vary the amount of training data (10%, 25%, 50%, 100%)
to measure sample complexity — the key question being:

    "Does intelligent routing provide a sample-efficiency advantage,
     or does a powerful CNN head learn equally well from any kernel?"

The Hard MoE variant simulates what the real TS-MoE Final Classifier sees:
sparse tensors where only the selected kernel's channels are non-zero at each
spatial position. This is a closer approximation to the actual pipeline.

The Real TS-MoE runs the complete pipeline including Teacher training,
Student distillation, and Final Classifier on sparse routed features.

Usage:
    python experiments/ablation_kernel_routing.py --dataset pneumonia_mnist_3kern
    python experiments/ablation_kernel_routing.py --dataset breast_mnist_3kern
    python experiments/ablation_kernel_routing.py --dataset breast_mnist_3kern --include-real-tsmoe

The script expects a cached quantum dataset in data/quantum_datasets/.
"""

import argparse
import copy
import json
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

# Make sure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import QUANTUM_DATASETS_DIR
from src.utils.kernel_mapping import build_kernel_to_channels_map, get_kernel_names
from src.utils.training_utils import build_classification_head, entropy_loss


# =============================================================================
# Dataset configurations
# =============================================================================

DATASET_CONFIGS = {
    "pneumonia_mnist_2kern": {
        "npz": "pneumonia_mnist__k2_s2_tkin-hor-ver-u_s_ev2.50_sc1_gray.npz",
        "json": "pneumonia_mnist__k2_s2_tkin-hor-ver-u_s_ev2.50_sc1_gray.json",
    },
    "pneumonia_mnist_3kern": {
        "npz": "pneumonia_mnist__k3_s3_tkin-hor-cro-rin_ev2.50_sc1_gray.npz",
        "json": "pneumonia_mnist__k3_s3_tkin-hor-cro-rin_ev2.50_sc1_gray.json",
    },
    "breast_mnist_3kern": {
        "npz": "breast_mnist__k3_s3_tkin-hor-cro-rin_ev2.50_sc1_gray.npz",
        "json": "breast_mnist__k3_s3_tkin-hor-cro-rin_ev2.50_sc1_gray.json",
    },
}


# =============================================================================
# Models
# =============================================================================


class SingleKernelClassifier(nn.Module):
    """Classifier that uses only ONE kernel's channels.

    This is the baseline: "what if we just picked the best kernel and
    ignored the rest?" If this matches MoE performance, routing is useless.
    """

    def __init__(
        self,
        total_channels,
        kernel_to_channels,
        kernel_idx,
        num_classes,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.kernel_names = get_kernel_names(kernel_to_channels)
        self.kernel_name = self.kernel_names[kernel_idx]
        self.channel_indices = kernel_to_channels[self.kernel_name]
        self.in_channels = len(self.channel_indices)

        self.head = build_classification_head(
            self.in_channels, num_classes, dropout, activation
        )

    def forward(self, x):
        # Extract only the selected kernel's channels
        x_subset = x[:, self.channel_indices, :, :]
        return self.head(x_subset)


class AllKernelsClassifier(nn.Module):
    """Classifier that uses ALL kernels concatenated (no routing).

    This tests: "is a powerful CNN head able to implicitly learn which
    channels are useful, without explicit routing?"
    """

    def __init__(self, total_channels, num_classes, dropout=0.1, activation="relu"):
        super().__init__()
        self.head = build_classification_head(
            total_channels, num_classes, dropout, activation
        )

    def forward(self, x):
        return self.head(x)


class AllKernelsLargeClassifier(nn.Module):
    """Classifier with ALL kernels + extra capacity to match MoE param count.

    This is a parameter-matched baseline for fair comparison with SoftMoE/HardMoE.
    The extra parameters come from:
      1. Per-group BatchNorm (same as MoE models)
      2. An additional MLP layer before the classification head

    This answers: "Is MoE better because of intelligent routing, or just
    because it has more parameters?"
    """

    def __init__(
        self,
        total_channels,
        kernel_to_channels,
        num_classes,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.kernel_names = get_kernel_names(kernel_to_channels)
        self.num_kernels = len(self.kernel_names)
        self.channel_groups = [kernel_to_channels[k] for k in self.kernel_names]
        self.channels_per_kernel = len(self.channel_groups[0])
        self.total_channels = total_channels

        # Per-group BatchNorm (same as MoE models for fair comparison)
        self.group_norms = nn.ModuleList(
            [nn.BatchNorm2d(self.channels_per_kernel) for _ in range(self.num_kernels)]
        )

        # Extra capacity: 1x1 conv to add parameters roughly matching the gate network
        # MoE gate has ~2.5k params, so we add a similar-sized projection
        self.extra_capacity = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, total_channels, kernel_size=1),
        )

        # Classification head
        self.head = build_classification_head(
            total_channels, num_classes, dropout, activation
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Normalize each kernel group (same as MoE)
        normalized = torch.zeros_like(x)
        for k, (channels, norm) in enumerate(zip(self.channel_groups, self.group_norms)):
            normalized[:, channels, :, :] = norm(x[:, channels, :, :])
        x = normalized

        # Apply extra capacity (instead of routing, just transform)
        x = x + self.extra_capacity(x)  # Residual connection

        return self.head(x)


class SoftMoEClassifier(nn.Module):
    """Simplified Teacher-style soft MoE routing.

    Uses per-patch attention to weight kernel contributions, then classifies
    the soft-mixed features.
    """

    def __init__(
        self,
        total_channels,
        kernel_to_channels,
        num_classes,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.kernel_names = get_kernel_names(kernel_to_channels)
        self.num_kernels = len(self.kernel_names)
        self.channel_groups = [kernel_to_channels[k] for k in self.kernel_names]
        self.channels_per_kernel = len(self.channel_groups[0])
        self.total_channels = total_channels

        # Per-group BatchNorm (like Teacher)
        self.group_norms = nn.ModuleList(
            [nn.BatchNorm2d(self.channels_per_kernel) for _ in range(self.num_kernels)]
        )

        # Simple attention gate: global pool -> linear -> softmax
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_channels, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_kernels),
        )

        # Classification head
        self.head = build_classification_head(
            total_channels, num_classes, dropout, activation
        )

        self.last_alpha = None

    def forward(self, x):
        B, C, H, W = x.shape

        # Normalize each kernel group
        normalized = torch.zeros_like(x)
        for k, (channels, norm) in enumerate(zip(self.channel_groups, self.group_norms)):
            normalized[:, channels, :, :] = norm(x[:, channels, :, :])
        x = normalized

        # Compute attention weights (global, not per-patch for simplicity)
        alpha = torch.softmax(self.gate(x), dim=1)  # (B, M)
        self.last_alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)

        # Weight each kernel group's channels
        weighted = torch.zeros_like(x)
        for k, channels in enumerate(self.channel_groups):
            weight = alpha[:, k].view(B, 1, 1, 1)
            weighted[:, channels, :, :] = x[:, channels, :, :] * weight

        return self.head(weighted)


class SoftMoEPatchClassifier(nn.Module):
    """Per-patch soft MoE routing (matches real Teacher behavior).

    Unlike SoftMoEClassifier which uses global (image-wise) attention,
    this uses per-spatial-position soft attention via 1x1 convolutions.
    This is the closest simulation of the real Teacher's soft routing.
    """

    def __init__(
        self,
        total_channels,
        kernel_to_channels,
        num_classes,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.kernel_names = get_kernel_names(kernel_to_channels)
        self.num_kernels = len(self.kernel_names)
        self.channel_groups = [kernel_to_channels[k] for k in self.kernel_names]
        self.channels_per_kernel = len(self.channel_groups[0])
        self.total_channels = total_channels

        # Per-group BatchNorm (like Teacher)
        self.group_norms = nn.ModuleList(
            [nn.BatchNorm2d(self.channels_per_kernel) for _ in range(self.num_kernels)]
        )

        # Per-PATCH attention gate (like real Teacher)
        # Uses 1x1 conv to produce per-patch routing logits
        self.gate = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_kernels, kernel_size=1),
        )

        # Classification head
        self.head = build_classification_head(
            total_channels, num_classes, dropout, activation
        )

        self.last_alpha = None

    def forward(self, x):
        B, C, H, W = x.shape

        # Normalize each kernel group
        normalized = torch.zeros_like(x)
        for k, (channels, norm) in enumerate(zip(self.channel_groups, self.group_norms)):
            normalized[:, channels, :, :] = norm(x[:, channels, :, :])
        x = normalized

        # Compute per-patch routing logits
        routing_logits = self.gate(x)  # (B, M, H, W)
        alpha = torch.softmax(routing_logits, dim=1)  # (B, M, H, W)
        self.last_alpha = alpha

        # SOFT per-patch weighting (gradients flow through)
        weighted = torch.zeros_like(x)
        for k, channels in enumerate(self.channel_groups):
            # (B, 1, H, W) weight for kernel k at each spatial position
            weight = alpha[:, k:k+1, :, :]  # (B, 1, H, W)
            for ch in channels:
                weighted[:, ch, :, :] = x[:, ch, :, :] * weight.squeeze(1)

        return self.head(weighted)


class HardMoEClassifier(nn.Module):
    """Hard MoE routing that simulates TS-MoE Final Classifier behavior.

    Like SoftMoEPatchClassifier, but uses argmax (hard) routing decisions and
    builds sparse tensors where only the selected kernel's channels are
    non-zero at each spatial position. This matches what the real TS-MoE
    Final Classifier sees after Student routing.

    Key difference from soft: gradients don't flow through routing decisions,
    and the classifier must learn to interpret zeros as "not selected".
    """

    def __init__(
        self,
        total_channels,
        kernel_to_channels,
        num_classes,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.kernel_names = get_kernel_names(kernel_to_channels)
        self.num_kernels = len(self.kernel_names)
        self.channel_groups = [kernel_to_channels[k] for k in self.kernel_names]
        self.channels_per_kernel = len(self.channel_groups[0])
        self.total_channels = total_channels

        # Per-group BatchNorm (like Teacher)
        self.group_norms = nn.ModuleList(
            [nn.BatchNorm2d(self.channels_per_kernel) for _ in range(self.num_kernels)]
        )

        # Per-PATCH attention gate (more like real Teacher)
        # Uses 1x1 conv to produce per-patch routing logits
        self.gate = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_kernels, kernel_size=1),
        )

        # Classification head
        self.head = build_classification_head(
            total_channels, num_classes, dropout, activation
        )

        self.last_alpha = None
        self.last_routing = None

    def forward(self, x):
        B, C, H, W = x.shape

        # Normalize each kernel group
        normalized = torch.zeros_like(x)
        for k, (channels, norm) in enumerate(zip(self.channel_groups, self.group_norms)):
            normalized[:, channels, :, :] = norm(x[:, channels, :, :])
        x = normalized

        # Compute per-patch routing logits
        routing_logits = self.gate(x)  # (B, M, H, W)
        alpha = torch.softmax(routing_logits, dim=1)  # (B, M, H, W)
        self.last_alpha = alpha

        # HARD routing: argmax (no gradient flow through selection)
        routing = alpha.argmax(dim=1)  # (B, H, W)
        self.last_routing = routing

        # Build SPARSE tensor: only selected kernel's channels are non-zero
        sparse = torch.zeros_like(x)
        for k, channels in enumerate(self.channel_groups):
            # (B, 1, H, W) mask where kernel k was selected
            mask = (routing == k).unsqueeze(1).float()
            for ch in channels:
                sparse[:, ch, :, :] = x[:, ch, :, :] * mask.squeeze(1)

        return self.head(sparse)


# =============================================================================
# Real TS-MoE Pipeline Integration
# =============================================================================


def build_tsmoe_config(dataset_name, metadata, fraction=1.0, seed=42):
    """Build a config dict for the real TS-MoE pipeline.
    
    Creates a minimal config that matches the dataset and can be passed
    to run_ts_moe_pipeline().
    """
    # Map dataset names to actual names
    dataset_map = {
        "pneumonia_mnist_2kern": "pneumonia_mnist",
        "pneumonia_mnist_3kern": "pneumonia_mnist",
        "breast_mnist_3kern": "breast_mnist",
    }
    
    actual_dataset = dataset_map.get(dataset_name, dataset_name.split("_")[0] + "_mnist")
    
    cfg = {
        "dataset": {
            "name": actual_dataset,
            "data_root": "./data",
            "batch_size": 32,
            "num_workers": 2,
            "download": True,
            "color_space": metadata.get("color_space", "GRAYSCALE"),
            # For data fraction experiments, we'll handle this separately
            "train_fraction": fraction,
        },
        "model": {
            "architecture": "TS-MoE",
            "num_classes": 2,  # Will be updated based on dataset
            "kernel_size": metadata["kernel_size"],
            "stride": metadata["stride"],
            "kernel_topology_names": metadata["kernel_topology_names"],
            "scaling_factor": metadata["scaling_factor"],
            "evolution_time": metadata["evolution_time"],
            "mode": "trotter",
            "dropout": 0.5,
            "activation": "relu",
            "classical_device": "auto",
        },
        "ts_moe": {
            # Teacher settings
            "lambda_entropy_start": 0.0,
            "lambda_max": 0.1,
            "lambda_warmup_epochs": 30,
            "attention_hidden_dim": 64,
            "attention_gate_zero_init": True,
            # Student settings
            "student_hidden_dims": [128, 64],
            "student_epochs": 30,
            "student_lr": 1e-3,
            "student_batch_size": 256,
            "student_agreement_threshold": 0.90,
            "student_agreement_patience": 5,
            "student_weighted_ce": True,
            "student_balanced_sampler": False,
            "confidence_threshold": 0.3,
            "student_kd_alpha": 0.5,
            "student_kd_temperature": 2.0,
            "student_augment_features": True,
            # Final classifier settings
            "final_epochs": 50,
            "final_lr": 1e-3,
        },
        "training": {
            "epochs": 50,  # Teacher epochs
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
        },
        "teacher": {
            "epochs": 50,  # Explicit Teacher epochs
        },
        "misc": {
            "seed": seed,
        },
    }
    
    return cfg


def run_real_tsmoe(dataset_name, metadata, fraction, seed, output_dir, device):
    """Run the real TS-MoE pipeline and return test metrics.
    
    This calls the actual Teacher → Student → Final Classifier pipeline
    from src/models/train_ts_moe.py.
    
    NOTE: The real pipeline doesn't support train_fraction directly.
    For fraction < 1.0, we create a temporary subset of the quantum dataset
    and point the pipeline to it.
    """
    from src.models.train_ts_moe import run_ts_moe_pipeline, set_seed
    
    # Build config
    cfg = build_tsmoe_config(dataset_name, metadata, fraction, seed)
    
    # Determine number of classes from dataset
    if "pneumonia" in dataset_name:
        cfg["model"]["num_classes"] = 2
    elif "breast" in dataset_name:
        cfg["model"]["num_classes"] = 2
    else:
        cfg["model"]["num_classes"] = 2  # Default binary
    
    # Create a temporary output directory for this run
    run_dir = os.path.join(output_dir, f"tsmoe_frac{fraction}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Handle data fraction by creating a subset quantum dataset
    temp_datasets_dir = None
    if fraction < 1.0:
        # Create a temporary directory with subsetted quantum data
        temp_datasets_dir = Path(run_dir) / "temp_quantum_data"
        temp_datasets_dir.mkdir(exist_ok=True)
        
        # Load original quantum dataset
        config = DATASET_CONFIGS[dataset_name]
        npz_path = QUANTUM_DATASETS_DIR / config["npz"]
        json_path = QUANTUM_DATASETS_DIR / config["json"]
        
        data = np.load(npz_path, allow_pickle=True)
        
        # Subset train data
        np.random.seed(seed)
        train_features = data["train_features"]
        train_labels = data["train_labels"]
        n_train = len(train_labels)
        n_subset = max(1, int(n_train * fraction))
        indices = np.random.permutation(n_train)[:n_subset]
        
        # Create subsetted dataset
        subset_data = {
            "train_features": train_features[indices],
            "train_labels": train_labels[indices],
            "val_features": data["val_features"],
            "val_labels": data["val_labels"],
            "test_features": data["test_features"],
            "test_labels": data["test_labels"],
            "metadata": data["metadata"],
        }
        
        # Save subset
        subset_npz = temp_datasets_dir / config["npz"]
        np.savez(subset_npz, **subset_data)
        
        # Copy metadata JSON
        import shutil
        shutil.copy(json_path, temp_datasets_dir / config["json"])
    
    # Set seed
    set_seed(seed)
    
    try:
        # Run the full pipeline
        result = run_ts_moe_pipeline(
            cfg=cfg,
            seed=seed,
            output_dir=run_dir,
            verbose=False,  # Suppress output for cleaner ablation logs
            datasets_dir=temp_datasets_dir if temp_datasets_dir else None,
        )
        
        # Extract final classifier metrics
        final_result = result.get("final", {})
        
        return {
            "test_acc": final_result.get("test_acc", 0.0),
            "test_auc": final_result.get("test_auc", 0.0),
            "test_loss": final_result.get("test_loss", 0.0),
            "best_val_auc": final_result.get("test_auc", 0.0),  # Use test as proxy
            "teacher_acc": result.get("teacher", {}).get("test_acc", 0.0),
            "teacher_auc": result.get("teacher", {}).get("test_auc", 0.0),
            "student_agreement": result.get("student", {}).get("agreement", 0.0),
        }
    except Exception as e:
        print(f"    TS-MoE pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "test_acc": 0.0,
            "test_auc": 0.0,
            "test_loss": 0.0,
            "best_val_auc": 0.0,
            "error": str(e),
        }


# =============================================================================
# Training utilities
# =============================================================================


def train_epoch(model, loader, optimizer, criterion, device, moe_lambda=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)

        # Entropy regularization for MoE models (soft and hard)
        if moe_lambda > 0.0 and hasattr(model, "last_alpha") and model.last_alpha is not None:
            loss = loss + moe_lambda * entropy_loss(model.last_alpha)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # ROC-AUC
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")

    return total_loss / total, correct / total, auc


def create_subset_loader(full_dataset, fraction, batch_size, shuffle=True):
    """Create a DataLoader with a random subset of the data."""
    n_total = len(full_dataset)
    n_subset = max(1, int(n_total * fraction))
    indices = np.random.permutation(n_total)[:n_subset]
    subset = Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# Main experiment
# =============================================================================


def run_single_experiment(
    model_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    n_classes,
    device,
    epochs=50,
    lr=1e-3,
    moe_lambda=0.1,
):
    """Train a model and return final test metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_state = None

    pbar = tqdm(range(epochs), desc=model_name, leave=False)
    for epoch in pbar:
        # Apply entropy regularization for both soft_moe and hard_moe
        lam = moe_lambda if ("moe" in model_name.lower()) else 0.0
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, moe_lambda=lam
        )
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, n_classes
        )
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix(
            {"train_acc": f"{train_acc:.3f}", "val_auc": f"{val_auc:.3f}"}
        )

    # Restore best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_loss, test_acc, test_auc = evaluate(
        model, test_loader, criterion, device, n_classes
    )

    return {
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_loss": test_loss,
        "best_val_auc": best_val_auc,
    }


def run_ablation(dataset_name, data_fractions, seeds, epochs, device, output_dir, include_real_tsmoe=False):
    """Run the full ablation study."""
    # Load dataset config
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    npz_path = QUANTUM_DATASETS_DIR / config["npz"]
    json_path = QUANTUM_DATASETS_DIR / config["json"]

    if not npz_path.exists():
        raise FileNotFoundError(f"Quantum dataset not found: {npz_path}")

    # Load data
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*60}")

    data = np.load(npz_path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))

    train_features = torch.from_numpy(data["train_features"]).float()
    train_labels = torch.from_numpy(data["train_labels"]).long()
    val_features = torch.from_numpy(data["val_features"]).float()
    val_labels = torch.from_numpy(data["val_labels"]).long()
    test_features = torch.from_numpy(data["test_features"]).float()
    test_labels = torch.from_numpy(data["test_labels"]).long()

    total_channels = train_features.shape[1]
    n_classes = len(set(train_labels.tolist()))

    kernel_to_channels = build_kernel_to_channels_map(metadata["channel_kernel_map"])
    kernel_names = get_kernel_names(kernel_to_channels)
    num_kernels = len(kernel_names)

    print(f"  Total channels: {total_channels}")
    print(f"  Kernels: {kernel_names}")
    print(f"  Classes: {n_classes}")
    print(f"  Train samples: {len(train_labels)}")

    # Print parameter counts for each model type (for fair comparison verification)
    print(f"\n  Model parameter counts:")
    # Get spatial dimensions from data
    _, _, H, W = train_features.shape
    dummy_input = torch.zeros(1, total_channels, H, W)
    
    dummy_models = {
        "single_kernel": SingleKernelClassifier(total_channels, kernel_to_channels, 0, n_classes),
        "all_kernels": AllKernelsClassifier(total_channels, n_classes),
        "all_large": AllKernelsLargeClassifier(total_channels, kernel_to_channels, n_classes),
        "soft_moe_img": SoftMoEClassifier(total_channels, kernel_to_channels, n_classes),
        "soft_moe_patch": SoftMoEPatchClassifier(total_channels, kernel_to_channels, n_classes),
        "hard_moe": HardMoEClassifier(total_channels, kernel_to_channels, n_classes),
    }
    for name, m in dummy_models.items():
        # Run a dummy forward pass to initialize lazy modules
        with torch.no_grad():
            try:
                m(dummy_input)
            except Exception:
                pass  # Some models may fail on first pass, that's OK
        param_count = sum(p.numel() for p in m.parameters() if p.numel() > 0)
        print(f"    {name}: {param_count:,} params")
    del dummy_models

    # Create full datasets
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    # Full val/test loaders (always use all data for evaluation)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Results storage
    results = {
        "dataset": dataset_name,
        "metadata": metadata,
        "kernel_names": kernel_names,
        "data_fractions": data_fractions,
        "seeds": seeds,
        "experiments": [],
    }

    # Run experiments
    for fraction in data_fractions:
        print(f"\n{'='*60}")
        print(f"Data fraction: {fraction*100:.0f}%")
        print(f"{'='*60}")

        fraction_results = {"fraction": fraction, "models": {}}

        # Collect results for each model type
        model_types = []

        # 1. Single kernel baselines (one per kernel)
        for k_idx, k_name in enumerate(kernel_names):
            model_types.append((f"single_{k_name}", "single", k_idx))

        # 2. All kernels concatenated (baseline)
        model_types.append(("all_kernels", "all", None))

        # 3. All kernels with extra capacity (parameter-matched baseline)
        model_types.append(("all_large", "all_large", None))

        # 4. Soft MoE routing - image-wise (simplified)
        model_types.append(("soft_moe_img", "soft_moe_img", None))

        # 5. Soft MoE routing - patch-wise (matches real Teacher)
        model_types.append(("soft_moe_patch", "soft_moe_patch", None))

        # 6. Hard MoE routing (TS-MoE Final Classifier style)
        model_types.append(("hard_moe", "hard_moe", None))

        # 7. Real TS-MoE pipeline (full Teacher→Student→Final Classifier)
        # NOTE: Real TS-MoE only runs at fraction=1.0 because the pipeline
        # doesn't support data subsetting (MedMNIST and quantum data must align)
        if include_real_tsmoe and fraction == 1.0:
            model_types.append(("real_tsmoe", "real_tsmoe", None))

        for model_name, model_type, k_idx in model_types:
            print(f"\n  Model: {model_name}")
            seed_results = []

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create subset train loader
                train_loader = create_subset_loader(
                    train_dataset, fraction, batch_size=32, shuffle=True
                )

                # Build model
                if model_type == "single":
                    model = SingleKernelClassifier(
                        total_channels=total_channels,
                        kernel_to_channels=kernel_to_channels,
                        kernel_idx=k_idx,
                        num_classes=n_classes,
                    )
                elif model_type == "all":
                    model = AllKernelsClassifier(
                        total_channels=total_channels, num_classes=n_classes
                    )
                elif model_type == "all_large":
                    model = AllKernelsLargeClassifier(
                        total_channels=total_channels,
                        kernel_to_channels=kernel_to_channels,
                        num_classes=n_classes,
                    )
                elif model_type == "soft_moe_img":
                    model = SoftMoEClassifier(
                        total_channels=total_channels,
                        kernel_to_channels=kernel_to_channels,
                        num_classes=n_classes,
                    )
                elif model_type == "soft_moe_patch":
                    model = SoftMoEPatchClassifier(
                        total_channels=total_channels,
                        kernel_to_channels=kernel_to_channels,
                        num_classes=n_classes,
                    )
                elif model_type == "hard_moe":
                    model = HardMoEClassifier(
                        total_channels=total_channels,
                        kernel_to_channels=kernel_to_channels,
                        num_classes=n_classes,
                    )
                elif model_type == "real_tsmoe":
                    # Real TS-MoE has its own training pipeline
                    model = None

                # Train and evaluate
                if model_type == "real_tsmoe":
                    # Run the real TS-MoE pipeline
                    metrics = run_real_tsmoe(
                        dataset_name=dataset_name,
                        metadata=metadata,
                        fraction=fraction,
                        seed=seed,
                        output_dir=output_dir,
                        device=device,
                    )
                else:
                    metrics = run_single_experiment(
                        model_name=f"{model_name}_seed{seed}",
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        n_classes=n_classes,
                        device=device,
                        epochs=epochs,
                    )

                seed_results.append(metrics)
                # Handle None values from failed runs
                test_auc = metrics.get('test_auc') or 0.0
                test_acc = metrics.get('test_acc') or 0.0
                print(
                    f"    Seed {seed}: test_auc={test_auc:.4f}, "
                    f"test_acc={test_acc:.4f}"
                )

            # Aggregate across seeds (handle None values)
            test_aucs = [r.get("test_auc") or 0.0 for r in seed_results]
            test_accs = [r.get("test_acc") or 0.0 for r in seed_results]

            fraction_results["models"][model_name] = {
                "seed_results": seed_results,
                "test_auc_mean": float(np.mean(test_aucs)),
                "test_auc_std": float(np.std(test_aucs)),
                "test_acc_mean": float(np.mean(test_accs)),
                "test_acc_std": float(np.std(test_accs)),
            }

            print(
                f"    >> Mean: test_auc={np.mean(test_aucs):.4f}±{np.std(test_aucs):.4f}"
            )

        results["experiments"].append(fraction_results)

    # Save results
    output_path = output_dir / f"ablation_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print_summary_table(results)

    return results


def print_summary_table(results):
    """Print a nice summary table of results."""
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Dataset: {results['dataset']}")
    print(f"Kernels: {results['kernel_names']}")
    print()

    # Get all model names (consistent across fractions)
    model_names = list(results["experiments"][0]["models"].keys())

    # Header
    header = f"{'Data %':<10}"
    for name in model_names:
        # Truncate long names
        short_name = name[:12] if len(name) > 12 else name
        header += f"{short_name:>14}"
    print(header)
    print("-" * len(header))

    # Rows
    for exp in results["experiments"]:
        fraction = exp["fraction"]
        row = f"{fraction*100:>6.0f}%   "
        for name in model_names:
            m = exp["models"][name]
            row += f"  {m['test_auc_mean']:.3f}±{m['test_auc_std']:.2f}"
        print(row)

    print()

    # Best model per fraction
    print("Best model per data fraction:")
    for exp in results["experiments"]:
        fraction = exp["fraction"]
        best_name = max(
            exp["models"].keys(), key=lambda n: exp["models"][n]["test_auc_mean"]
        )
        best_auc = exp["models"][best_name]["test_auc_mean"]
        print(f"  {fraction*100:>3.0f}%: {best_name} (AUC={best_auc:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: Does MoE routing actually help?"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pneumonia_mnist_3kern",
        choices=list(DATASET_CONFIGS.keys()),
        help="Which quantum dataset to use.",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
        help="Training data fractions to test.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for reproducibility.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs per run."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--include-real-tsmoe",
        action="store_true",
        help="Include the real TS-MoE pipeline (expensive, runs full Teacher→Student→Final).",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Output directory
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "ablation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run ablation
    run_ablation(
        dataset_name=args.dataset,
        data_fractions=args.fractions,
        seeds=args.seeds,
        epochs=args.epochs,
        device=device,
        output_dir=output_dir,
        include_real_tsmoe=args.include_real_tsmoe,
    )


if __name__ == "__main__":
    main()
