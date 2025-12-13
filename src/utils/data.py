import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def get_pneumonia_mnist_loaders(cfg):
    """Create train/val/test loaders for PneumoniaMNIST via medmnist.

    Expects cfg sections:
        dataset.data_root, dataset.batch_size, dataset.num_workers, dataset.download
    Returns:
        (train_loader, val_loader, test_loader, n_classes)
    """
    try:
        from medmnist import PneumoniaMNIST, INFO
    except ImportError as exc:
        raise ImportError("medmnist is required for PneumoniaMNIST. Install with `pip install medmnist`." ) from exc

    data_flag = "pneumoniamnist"
    info = INFO[data_flag]
    n_classes = len(info["label"])

    transform = transforms.Compose([transforms.ToTensor()])

    def build_split(split):
        root = cfg["dataset"].get("data_root", "./data")
        os.makedirs(root, exist_ok=True)
        return PneumoniaMNIST(
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, n_classes


def get_dataloaders(cfg):
    name = cfg.get("dataset", {}).get("name", "pneumonia_mnist")
    if name == "pneumonia_mnist":
        return get_pneumonia_mnist_loaders(cfg)
    raise ValueError(f"Unsupported dataset: {name}")
