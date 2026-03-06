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
# Names ending in _64 / _128 / _224 refer to MedMNIST+ higher-resolution
# variants loaded via the `size=` kwarg of each MedMNIST dataset class.
DATASET_REGISTRY = {
    # --- 28×28 (standard MedMNIST) ---
    "pneumonia_mnist": ("pneumoniamnist", "PneumoniaMNIST"),
    "breast_mnist": ("breastmnist", "BreastMNIST"),
    "path_mnist": ("pathmnist", "PathMNIST"),
    "derma_mnist": ("dermamnist", "DermaMNIST"),
    "tissue_mnist": ("tissuemnist", "TissueMNIST"),
    "oct_mnist": ("octmnist", "OCTMNIST"),
    "organ_a_mnist": ("organamnist", "OrganAMNIST"),
    "organ_c_mnist": ("organcmnist", "OrganCMNIST"),
    "organ_s_mnist": ("organsmnist", "OrganSMNIST"),
    # --- 64×64 (MedMNIST+) ---
    "pneumonia_mnist_64": ("pneumoniamnist", "PneumoniaMNIST"),
    "breast_mnist_64": ("breastmnist", "BreastMNIST"),
    "tissue_mnist_64": ("tissuemnist", "TissueMNIST"),
    "oct_mnist_64": ("octmnist", "OCTMNIST"),
    "organ_a_mnist_64": ("organamnist", "OrganAMNIST"),
    "organ_c_mnist_64": ("organcmnist", "OrganCMNIST"),
    "organ_s_mnist_64": ("organsmnist", "OrganSMNIST"),
    # --- 128×128 (MedMNIST+) ---
    "pneumonia_mnist_128": ("pneumoniamnist", "PneumoniaMNIST"),
    "tissue_mnist_128": ("tissuemnist", "TissueMNIST"),
    "oct_mnist_128": ("octmnist", "OCTMNIST"),
    "organ_a_mnist_128": ("organamnist", "OrganAMNIST"),
}

# Image resolution for each dataset name.
# 28 is the default (standard MedMNIST); _64 / _128 / _224 suffixes select
# the corresponding MedMNIST+ size= argument.
DATASET_IMAGE_SIZE = {
    # 28×28
    "pneumonia_mnist": 28,
    "breast_mnist": 28,
    "path_mnist": 28,
    "derma_mnist": 28,
    "tissue_mnist": 28,
    "oct_mnist": 28,
    "organ_a_mnist": 28,
    "organ_c_mnist": 28,
    "organ_s_mnist": 28,
    # 64×64
    "pneumonia_mnist_64": 64,
    "breast_mnist_64": 64,
    "tissue_mnist_64": 64,
    "oct_mnist_64": 64,
    "organ_a_mnist_64": 64,
    "organ_c_mnist_64": 64,
    "organ_s_mnist_64": 64,
    # 128×128
    "pneumonia_mnist_128": 128,
    "tissue_mnist_128": 128,
    "oct_mnist_128": 128,
    "organ_a_mnist_128": 128,
}

# Dataset channel information (RGB vs Grayscale)
DATASET_CHANNELS = {
    # 28×28
    "pneumonia_mnist": 1,  # Grayscale
    "breast_mnist": 1,  # Grayscale
    "path_mnist": 3,  # RGB
    "derma_mnist": 3,  # RGB
    "tissue_mnist": 1,  # Grayscale
    "oct_mnist": 1,  # Grayscale
    "organ_a_mnist": 1,  # Grayscale
    "organ_c_mnist": 1,  # Grayscale
    "organ_s_mnist": 1,  # Grayscale
    # 64×64
    "pneumonia_mnist_64": 1,
    "breast_mnist_64": 1,
    "tissue_mnist_64": 1,
    "oct_mnist_64": 1,
    "organ_a_mnist_64": 1,
    "organ_c_mnist_64": 1,
    "organ_s_mnist_64": 1,
    # 128×128
    "pneumonia_mnist_128": 1,
    "tissue_mnist_128": 1,
    "oct_mnist_128": 1,
    "organ_a_mnist_128": 1,
}
