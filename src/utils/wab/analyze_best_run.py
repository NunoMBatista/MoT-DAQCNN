import wandb
import pandas as pd

api = wandb.Api()
entity = "nunombatista-university-of-coimbra"
project = "MoT-DAQCNN"

# Target the best run specifically
run_id = "hp_search_original_breastmnist_find_best_kernels_20260311_142054"
# Actually I need the 12-char ID, let me find it by name
runs = api.runs(f"{entity}/{project}")
target_run = None
for r in runs:
    if "find_best_kernels" in r.name:
        target_run = r
        break

if target_run:
    print(f"Analyzing run: {target_run.name}")
    s = target_run.summary
    c = target_run.config
    
    # Check for anything related to disagreement or diversity
    metrics = {k: v for k, v in s.items() if any(x in k for x in ['acc', 'iou', 'disagree', 'diff', 'oracle'])}
    print("Summary Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Check if there's an Optuna study summary (often used in your project)
    if 'optuna.best_params' in s:
        print("\nBest Parameters (Best Kernels):")
        best_p = s['optuna.best_params']
        for k, v in best_p.items():
            if 'kernel' in k or 'topology' in k:
                print(f"  {k}: {v}")
else:
    print("Could not find the 'find_best_kernels' run.")
