import sys
import os
from pathlib import Path
import copy
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.phase3_experiment_loaders import get_synced_stratified_loaders
from src.phase3_train_experiments import run_phase3_experiment

cfg = {
    "dataset": {"name": "breast_mnist", "batch_size": 32, "n_classes": 2},
    "model": {"from_scratch": True, "num_layers": 5, "backbone": "none", "nhead": 4, "embed_dim": 192},
    "train": {"epochs": 400, "lr": 1e-4, "weight_decay": 0.01},
    "wandb": {"project": "Phase3_Scratch_Ablation"}
}

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
q_npz = "data/quantum_datasets/breastmnist_A100_final_tkin-hor-ver-rin.npz"

def main():
    print("🚀 Starting Phase 3 Suite with Delta Logging", flush=True)
    
    for f in fractions:
        percent = int(f * 100)
        current_cfg = copy.deepcopy(cfg)
        current_cfg["dataset"]["subset_percent"] = percent
        
        try:
            loaders = get_synced_stratified_loaders(current_cfg, q_npz, f)
            c_loaders, q_loaders, _ = loaders
            
            # 1. CLASSICAL RUN
            c_acc, c_auc = run_phase3_experiment(current_cfg, *c_loaders, is_hybrid=False, experiment_type="scratch")
            
            # 2. HYBRID RUN
            h_acc, h_auc = run_phase3_experiment(current_cfg, *q_loaders, is_hybrid=True, experiment_type="scratch")
            
            # 3. CALCULATE DELTAS
            acc_delta = (h_acc - c_acc) * 100
            auc_delta = (h_auc - c_auc)
            
            # 4. LOG DELTA TO SEPARATE WANDB RUN
            # This creates the bar chart where Y is the advantage
            wandb.init(
                project=cfg['wandb']['project'],
                name=f"Delta_Analysis_{percent}%",
                group="Performance_Deltas",
                reinit=True
            )
            wandb.log({
                "fraction": percent,
                "advantage/accuracy_pct": acc_delta, # This is your -5%, +3% etc.
                "advantage/auc_points": auc_delta
            })
            wandb.finish()
            
            print(f"📊 {percent}%: Hybrid Acc Advantage: {acc_delta:.2f}%")

        except Exception as e:
            print(f"❌ Error at {percent}%: {e}")
            continue

if __name__ == "__main__":
    main()