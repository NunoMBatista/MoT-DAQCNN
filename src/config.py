"""
Global project paths, directories, and dataset constants.

Every path is anchored to the repository root so scripts work regardless of
the current working directory.
"""

from pathlib import Path

# Repository root: src/ lives one level below it
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Top-level directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Quantum dataset cache
QUANTUM_DATASETS_DIR = DATA_DIR / "quantum_datasets"

# Mapping from dataset name to (medmnist_flag, DatasetClass)
DATASET_REGISTRY = {
    "pneumonia_mnist": ("pneumoniamnist", "PneumoniaMNIST"),
    "breast_mnist": ("breastmnist", "BreastMNIST"),
    "path_mnist": ("pathmnist", "PathMNIST"),
    "derma_mnist": ("dermamnist", "DermaMNIST"),
    "tissue_mnist": ("tissuemnist", "TissueMNIST"),
}

# Dataset channel information (RGB vs Grayscale)
DATASET_CHANNELS = {
    "pneumonia_mnist": 1,  # Grayscale
    "breast_mnist": 1,  # Grayscale
    "path_mnist": 3,  # RGB
    "derma_mnist": 3,  # RGB
    "tissue_mnist": 1,  # Grayscale
}
