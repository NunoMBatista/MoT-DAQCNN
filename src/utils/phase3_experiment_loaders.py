import torch
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from src.utils.data import get_medmnist_loaders

class QuantumFeatureDataset(Dataset):
    def __init__(self, npz_path, split="train"):
        data = np.load(npz_path, allow_pickle=True)
        self.features = torch.from_numpy(data[f"{split}_features"]).float()
        self.labels = torch.from_numpy(data[f"{split}_labels"]).long()
        if self.labels.ndim > 1:
            self.labels = self.labels.squeeze()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_synced_stratified_loaders(cfg, quantum_npz_path, fraction):
    """
    Returns synced classical and quantum loaders with stratified sampling 
    to handle class imbalance across data fractions.
    """
    seed = cfg.get("misc", {}).get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Create a LOCAL copy so we don't overwrite the global subset_percent config
    local_cfg = copy.deepcopy(cfg)
    local_cfg["dataset"]["subset_percent"] = 100 
    
    # Load Classical Datasets
    c_train_full, c_val, c_test, n_classes = get_medmnist_loaders(local_cfg, local_cfg["dataset"]["name"])
    
    # 2. Load Quantum Dataset
    q_train_ds = QuantumFeatureDataset(quantum_npz_path, split="train")
    q_val_ds = QuantumFeatureDataset(quantum_npz_path, split="val")
    q_test_ds = QuantumFeatureDataset(quantum_npz_path, split="test")

    # 3. Stratified Subsetting for Training Data
    if fraction < 1.0:
        # Robustly handle MedMNIST label structure
        full_ds = c_train_full.dataset
        if hasattr(full_ds, 'labels'):
            labels = np.array(full_ds.labels).flatten()
        else:
            labels = np.array([y for _, y in full_ds]).flatten()
        
        indices = np.arange(len(labels))
        train_idx, _ = train_test_split(
            indices, 
            train_size=fraction, 
            stratify=labels, 
            random_state=seed
        )
    else:
        train_idx = np.arange(len(q_train_ds))

    # 4. Create Subsets
    c_train_subset = Subset(c_train_full.dataset, train_idx)
    q_train_subset = Subset(q_train_ds, train_idx)
    
    # 5. Create Loaders
    batch_size = cfg["dataset"]["batch_size"]
    
    c_train_loader = DataLoader(c_train_subset, batch_size=batch_size, shuffle=True)
    q_train_loader = DataLoader(q_train_subset, batch_size=batch_size, shuffle=True)
    
    c_val_loader = DataLoader(c_val.dataset, batch_size=batch_size, shuffle=False)
    q_val_loader = DataLoader(q_val_ds, batch_size=batch_size, shuffle=False)
    
    c_test_loader = DataLoader(c_test.dataset, batch_size=batch_size, shuffle=False)
    q_test_loader = DataLoader(q_test_ds, batch_size=batch_size, shuffle=False)

    return (c_train_loader, c_val_loader, c_test_loader), \
           (q_train_loader, q_val_loader, q_test_loader), n_classes