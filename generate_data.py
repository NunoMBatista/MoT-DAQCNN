import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

# Setup project paths
PROJECT_ROOT = Path("/scratch/sp7007/MoT-DAQCNN")
sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.quantum_convolution import QuantumConv2d
from src.utils.color_conversion import rgb_to_grayscale_tensor
from src.utils.data import load_medmnist_dataset

# Environment Optimization for 128 cores
os.environ["OMP_NUM_THREADS"] = "128"
os.environ["MKL_NUM_THREADS"] = "128"

def run_extraction():
    # Load Config
    CONFIG_PATH = PROJECT_ROOT / "configs" / "breast_mnist" / "cache_generation" / "digital_zz.yml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    data_root = PROJECT_ROOT / "data"
    SELECTED_TOPOLOGIES = ["kings", "horizontal", "vertical", "ring"]

    print("🏗️ Initializing Quantum Engine...")
    q_conv = QuantumConv2d(
        in_channels=1,
        kernel_size=model_cfg['kernel_size'],
        stride=model_cfg['stride'],
        kernel_topology_names=SELECTED_TOPOLOGIES,
        scaling_factor=model_cfg['scaling_factor'],
        evolution_time=model_cfg['evolution_time'],
        mode=model_cfg.get("mode", "trotter"),
        quantum_device="lightning.qubit",
        interface="torch",
        include_correlators=True,
        encoding_mode="digital",
    )
    q_conv.eval()

    results = {}
    H_OUT, W_OUT = 9, 9 

    for split in ["train", "val", "test"]:
        print(f"🔄 Processing {split}...")
        ds = load_medmnist_dataset(dataset_cfg['name'], split, data_root)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        
        split_features = np.zeros((len(ds), q_conv.out_channels, H_OUT, W_OUT), dtype=np.float32)
        split_labels = np.zeros(len(ds), dtype=np.int64)

        idx = 0
        for i, (images, labels) in enumerate(loader):
            with torch.no_grad():
                if images.shape[1] == 3:
                    images = rgb_to_grayscale_tensor(images)
                q_out = q_conv(images)
            
            batch_curr = images.shape[0]
            split_features[idx : idx + batch_curr] = q_out.numpy()
            split_labels[idx : idx + batch_curr] = labels.numpy().squeeze()
            idx += batch_curr
            
            if i % 10 == 0:
                print(f"   > {split}: Batch {i}/{len(loader)} done.")

        results[f"{split}_features"] = split_features
        results[f"{split}_labels"] = split_labels

    # FINAL SAVE
    topo_short = "-".join([t[:3] for t in SELECTED_TOPOLOGIES])
    save_path = data_root / "quantum_datasets" / f"breastmnist_A100_final_t{topo_short}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        save_path,
        train_features=results["train_features"], train_labels=results["train_labels"],
        val_features=results["val_features"], val_labels=results["val_labels"],
        test_features=results["test_features"], test_labels=results["test_labels"],
        metadata=json.dumps({"topologies": SELECTED_TOPOLOGIES, "hardware": "128_core_sbatch"})
    )
    print(f"✨ SAVED ALL TO {save_path}")

if __name__ == "__main__":
    run_extraction()