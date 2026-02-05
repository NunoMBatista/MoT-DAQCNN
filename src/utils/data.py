import os

from torch.utils.data import DataLoader
from torchvision import transforms

# Mapping from dataset name to (medmnist_flag, DatasetClass)
DATASET_REGISTRY = {
    "pneumonia_mnist": ("pneumoniamnist", "PneumoniaMNIST"),
    "breast_mnist": ("breastmnist", "BreastMNIST"),
    "path_mnist": ("pathmnist", "PathMNIST"),
    "derma_mnist": ("dermamnist", "DermaMNIST"),
}

# Dataset channel information (RGB vs Grayscale)
DATASET_CHANNELS = {
    "pneumonia_mnist": 1,  # Grayscale
    "breast_mnist": 1,  # Grayscale
    "path_mnist": 3,  # RGB
    "derma_mnist": 3,  # RGB
}


def get_dataset_channels(dataset_name: str) -> int:
    """
    Get the number of channels for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Number of channels (1 for grayscale, 3 for RGB)
    """
    return DATASET_CHANNELS.get(dataset_name, 1)


def check_model_dataset_compatibility(
    model_in_channels: int, dataset_name: str
) -> tuple:
    """
    Check if a model is compatible with a dataset based on input channels.

    Args:
        model_in_channels: Number of input channels the model expects
        dataset_name: Name of the dataset to test on

    Returns:
        Tuple of (is_compatible: bool, error_message: str or None)
    """
    dataset_channels = get_dataset_channels(dataset_name)

    if model_in_channels != dataset_channels:
        model_type = "RGB" if model_in_channels == 3 else "Grayscale"
        dataset_type = "RGB" if dataset_channels == 3 else "Grayscale"
        error_msg = (
            f"Model/Dataset mismatch: Model was trained on {model_type} "
            f"({model_in_channels} channels) but {dataset_name} is {dataset_type} "
            f"({dataset_channels} channels). Please select a compatible dataset."
        )
        return False, error_msg

    return True, None


def get_medmnist_loaders(cfg, dataset_name):
    """Generic loader for any MedMNIST dataset.

    Args:
        cfg: Configuration dict with dataset.data_root, batch_size, num_workers, download
        dataset_name: Name of dataset from DATASET_REGISTRY keys

    Returns:
        (train_loader, val_loader, test_loader, n_classes)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}"
        )

    data_flag, class_name = DATASET_REGISTRY[dataset_name]

    try:
        from medmnist import INFO

        dataset_class = getattr(
            __import__("medmnist", fromlist=[class_name]), class_name
        )
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
