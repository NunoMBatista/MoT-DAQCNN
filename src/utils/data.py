import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


# Mapping from dataset name to (medmnist_flag, DatasetClass)
DATASET_REGISTRY = {
    "pneumonia_mnist": ("pneumoniamnist", "PneumoniaMNIST"),
    "breast_mnist": ("breastmnist", "BreastMNIST"),
    "path_mnist": ("pathmnist", "PathMNIST"),
    "derma_mnist": ("dermamnist", "DermaMNIST"),
}


def get_medmnist_loaders(cfg, dataset_name):
    """Generic loader for any MedMNIST dataset.
    
    Args:
        cfg: Configuration dict with dataset.data_root, batch_size, num_workers, download
        dataset_name: Name of dataset from DATASET_REGISTRY keys
        
    Returns:
        (train_loader, val_loader, test_loader, n_classes)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    data_flag, class_name = DATASET_REGISTRY[dataset_name]
    
    try:
        from medmnist import INFO
        dataset_class = getattr(__import__("medmnist", fromlist=[class_name]), class_name)
    except ImportError as exc:
        raise ImportError(
            f"medmnist is required for {dataset_name}. Install with `pip install medmnist`."
        ) from exc
    
    info = INFO[data_flag]
    n_classes = len(info["label"])
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    def build_split(split):
        root = cfg["dataset"].get("data_root", "./data")
        os.makedirs(root, exist_ok=True)
        return dataset_class(
            split=split,
            download=cfg["dataset"].get("download", True),
            root=root,
            transform=transform,
        )
    
    train_ds = build_split("train")
    val_ds = build_split("val")
    test_ds = build_split("test")
    
    batch_size = cfg["dataset"].get("batch_size", 32)
    num_workers = cfg["dataset"].get("num_workers", 2)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, n_classes


def get_dataloaders(cfg):
    """Get dataloaders based on config. Auto-selects from DATASET_REGISTRY."""
    dataset_name = cfg.get("dataset", {}).get("name", "pneumonia_mnist")
    return get_medmnist_loaders(cfg, dataset_name)
