import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.config import DATA_DIR, DATASET_CHANNELS, DATASET_IMAGE_SIZE, DATASET_REGISTRY


def get_dataset_channels(dataset_name: str) -> int:
    """
    Get the number of channels for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Number of channels (1 for grayscale, 3 for RGB)
    """
    return DATASET_CHANNELS.get(dataset_name, 1)


def get_dataset_image_size(dataset_name: str) -> int:
    """
    Get the image resolution for a dataset.

    Standard MedMNIST datasets are 28×28.  Higher-resolution MedMNIST+ variants
    use the ``_64``, ``_128``, or ``_224`` suffix convention and return the
    corresponding size, which is passed as ``size=`` to the MedMNIST constructor.

    Args:
        dataset_name: Name of the dataset (key in DATASET_REGISTRY)

    Returns:
        Integer image size (e.g. 28, 64, 128, 224)
    """
    return DATASET_IMAGE_SIZE.get(dataset_name, 28)


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
        cfg: Configuration dict with dataset.data_root, batch_size, num_workers, download, subset_percent
        dataset_name: Name of dataset from DATASET_REGISTRY keys

    Config options:
        subset_percent: Optional percentage (0-100) to use only a subset of train/val data

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

    # Resolve image size: config key takes priority, falls back to registry default.
    # Standard MedMNIST is 28; MedMNIST+ _64/_128/_224 names carry their size.
    image_size = cfg["dataset"].get("image_size", get_dataset_image_size(dataset_name))

    transform = transforms.Compose([transforms.ToTensor()])

    def build_split(split):
        root = cfg["dataset"].get("data_root", str(DATA_DIR))
        os.makedirs(root, exist_ok=True)
        return dataset_class(
            split=split,
            download=cfg["dataset"].get("download", True),
            root=root,
            transform=transform,
            size=image_size,
        )

    train_ds = build_split("train")
    val_ds = build_split("val")
    test_ds = build_split("test")

    subset_percent = cfg["dataset"].get("subset_percent", None)
    if subset_percent is not None:
        if not (0 < subset_percent <= 100):
            raise ValueError(
                f"subset_percent must be between 0 and 100, got {subset_percent}"
            )

        train_size = int(len(train_ds) * subset_percent / 100.0)
        val_size = int(len(val_ds) * subset_percent / 100.0)

        torch.manual_seed(cfg.get("misc", {}).get("seed", 42))
        train_indices = torch.randperm(len(train_ds))[:train_size].tolist()
        val_indices = torch.randperm(len(val_ds))[:val_size].tolist()

        train_ds = Subset(train_ds, train_indices)
        val_ds = Subset(val_ds, val_indices)

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


def load_medmnist_dataset(
    dataset_name, split, data_root=None, download=True, image_size=None
):
    """Load a raw MedMNIST dataset for a given split.

    This is the centralized loader used across the codebase for loading
    raw MedMNIST datasets (not DataLoaders). Returns the dataset object
    directly for use cases that need direct access to samples.

    Args:
        dataset_name: Name of dataset from DATASET_REGISTRY keys
            (e.g., "pneumonia_mnist", "breast_mnist", "oct_mnist_64", etc.)
        split: Split name - "train", "val", or "test"
        data_root: Root directory for data storage. If None, uses DATA_DIR
        download: Whether to download the dataset if not present
        image_size: Override the image resolution (e.g. 28, 64, 128).
            If None, the size is inferred from DATASET_IMAGE_SIZE using the
            dataset name, so passing "oct_mnist_64" automatically uses 64.

    Returns:
        MedMNIST dataset object with .transform = ToTensor()

    Raises:
        ValueError: If dataset_name is not in DATASET_REGISTRY
        ImportError: If medmnist package is not installed

    Example:
        >>> train_ds = load_medmnist_dataset("pneumonia_mnist", "train")
        >>> img, label = train_ds[0]
        >>> print(img.shape, label)  # torch.Size([1, 28, 28])

        >>> train_ds_64 = load_medmnist_dataset("oct_mnist_64", "train")
        >>> img, label = train_ds_64[0]
        >>> print(img.shape, label)  # torch.Size([1, 64, 64])
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    _, class_name = DATASET_REGISTRY[dataset_name]

    try:
        import medmnist

        dataset_class = getattr(medmnist, class_name)
    except ImportError as exc:
        raise ImportError(
            f"medmnist is required for {dataset_name}. "
            "Install with `pip install medmnist`."
        ) from exc

    if data_root is None:
        data_root = str(DATA_DIR)

    os.makedirs(data_root, exist_ok=True)

    # Resolve size: explicit arg > registry default derived from name suffix
    size = (
        image_size if image_size is not None else get_dataset_image_size(dataset_name)
    )

    transform = transforms.ToTensor()
    return dataset_class(
        split=split,
        download=download,
        root=data_root,
        transform=transform,
        size=size,
    )


def get_dataloaders(cfg):
    """Get dataloaders based on config. Auto-selects from DATASET_REGISTRY."""
    dataset_name = cfg.get("dataset", {}).get("name", "pneumonia_mnist")
    return get_medmnist_loaders(cfg, dataset_name)


def get_dataset_info(dataset_name: str) -> dict:
    """Return a summary dict of key properties for a dataset name.

    Useful for logging and sanity-checking configs before training.

    Returns:
        dict with keys: name, image_size, channels, medmnist_flag, class_name
    """
    return {
        "name": dataset_name,
        "image_size": get_dataset_image_size(dataset_name),
        "channels": get_dataset_channels(dataset_name),
        "medmnist_flag": DATASET_REGISTRY[dataset_name][0],
        "class_name": DATASET_REGISTRY[dataset_name][1],
    }
