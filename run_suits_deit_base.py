import sys
import os
import torch

# FIX: Add the current directory to path so 'src' is found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.experiment_loaders import get_synced_loaders
from src.train_experiments import run_experiment # Matched filename

# 1. SETUP BASE HYPERPARAMETERS
base_cfg = {
    "dataset": {
        "name": "breast_mnist",
        "batch_size": 32,
        "n_classes": 2,
        "class_names": ["Healthy", "Affected"]
    },
    "model": {
        "from_scratch": False,        # False means use pretrained deit tiny pretrained
        "backbone": "deit_base_patch16_224",  # other available : DeiT-Small: "deit_small_patch16_224" #DeiT-Base: "deit_base_patch16_224"       
        "num_layers": 5,            
        "nhead": 4,                 
        "dim_feedforward": 128      
    },
    "quantum": {
        "n_qubits": 9,
        "topologies": ["kings", "horizontal", "vertical", "ring"] 
    },
    "train": {
        "warmup_epochs": 30,         
        "epochs": 100,                
        "warmup_lr": 1e-3,          
        "finetune_lr": 1e-5,        
        "scratch_lr": 1e-4          
    },
    "wandb": {
        "project": "Quantum-ViT-deit_base_patch16_224"
    }
}

# 2. DEFINE THE EXPERIMENT MATRIX
data_fractions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
quantum_npz = "data/quantum_datasets/breast_mnist__k3_s3_tkin-hor-ver-rin_ev2.50_sc1_gray.npz"

# 3. THE EXECUTION LOOP
for fraction in data_fractions:
    print(f"\n Starting Suite for {int(fraction*100)}% Data")
    
    # Get the synced loaders
    (c_loaders), (q_loaders), n_classes = get_synced_loaders(base_cfg, quantum_npz, fraction)
    base_cfg['dataset']['subset_percent'] = int(fraction * 100)

    print("Running Classical Baseline...")
    run_experiment(base_cfg, *c_loaders, is_hybrid=False)

    print("Running Hybrid Model...")
    run_experiment(base_cfg, *q_loaders, is_hybrid=True)