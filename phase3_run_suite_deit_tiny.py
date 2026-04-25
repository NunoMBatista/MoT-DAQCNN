import sys
import os
from pathlib import Path
import copy
import wandb

# 1. FIX PATHS: Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 2. Corrected Imports
from src.utils.phase3_experiment_loaders import get_synced_stratified_loaders
from src.phase3_train_experiments import run_phase3_experiment

# Configuration for Pretrained DeiT-Tiny
base_cfg = {
    "dataset": {
        "name": "breast_mnist",
        "batch_size": 32,
        "n_classes": 2,
        "class_names": ["Healthy", "Affected"]
    },
    "model": {
        "from_scratch": False,           # Critical: tells the model to load weights
        "backbone": "deit_tiny_patch16_224",
        "num_layers": 12,
        "embed_dim": 192,
        "nhead": 3
    },
    "train": {
        "epochs": 400,                   # UPDATED TO 400
        "lr": 2e-5,                      # Lower learning rate for fine-tuning
        "weight_decay": 0.05
    },
    "wandb": {
        "project": "Phase3_DeiT_Ablation"
    }
}

data_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
QUANTUM_NPZ = "data/quantum_datasets/breastmnist_A100_final_tkin-hor-ver-rin.npz"

def main():
    print(f"🚀 Starting Phase 3: DeiT-Tiny Ablation Study with Delta Logging", flush=True)
    print(f"📍 Project Root: {PROJECT_ROOT}", flush=True)

    for fraction in data_fractions:
        percent = int(fraction * 100)
        
        print(f"\n" + "="*60, flush=True)
        print(f"📈 CURRENT PHASE: {percent}% Data Strategy (DeiT-Tiny)", flush=True)
        print("="*60, flush=True)
        
        # 3. CREATE ISOLATED CONFIG
        current_cfg = copy.deepcopy(base_cfg)
        current_cfg["dataset"]["subset_percent"] = percent
        
        try:
            # 4. GET STRATIFIED LOADERS
            print(f"📥 Loading and stratifying data...", flush=True)
            loaders = get_synced_stratified_loaders(current_cfg, QUANTUM_NPZ, fraction)
            c_loaders, q_loaders, _ = loaders

            # 5. RUN CLASSICAL BASELINE
            print(f"📉 Running Classical DeiT-Tiny ({percent}% data)...", flush=True)
            c_acc, c_auc = run_phase3_experiment(
                cfg=current_cfg, 
                train_loader=c_loaders[0], 
                val_loader=c_loaders[1], 
                test_loader=c_loaders[2], 
                is_hybrid=False, 
                experiment_type="deit_tiny"
            )

            # 6. RUN HYBRID (QUANTUM) MODEL
            print(f"⚛️ Running Hybrid DeiT-Tiny ({percent}% data)...", flush=True)
            h_acc, h_auc = run_phase3_experiment(
                cfg=current_cfg, 
                train_loader=q_loaders[0], 
                val_loader=q_loaders[1], 
                test_loader=q_loaders[2], 
                is_hybrid=True, 
                experiment_type="deit_tiny"
            )

            # 7. CALCULATE AND LOG DELTAS (-5%, +3% etc)
            # This logs to a specific group for the "Advantage Bar Plot"
            acc_delta = (h_acc - c_acc) * 100
            auc_delta = (h_auc - c_auc)

            wandb.init(
                project=current_cfg['wandb']['project'],
                name=f"Delta_DeiT_{percent}%",
                group="DeiT_Performance_Deltas",
                reinit=True
            )
            wandb.log({
                "fraction": percent,
                "advantage/accuracy_pct": acc_delta,
                "advantage/auc_points": auc_delta
            })
            wandb.finish()

            print(f"📊 {percent}% Result: Hybrid Acc Advantage is {acc_delta:.2f}%", flush=True)
            
        except Exception as e:
            print(f"❌ Error encountered at {percent}% data: {str(e)}", flush=True)
            continue

    print("\n✅ All DeiT-Tiny Ablation Experiments Completed!", flush=True)

if __name__ == "__main__":
    main()