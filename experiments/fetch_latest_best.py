import wandb
import json

def fetch_latest_best():
    api = wandb.Api()
    entity = "nunombatista-university-of-coimbra"
    projects = ["MoT-DAQCNN-ZZ-Search", "MoT-DAQCNN"]
    
    for proj in projects:
        print(f"\n--- Project: {proj} ---")
        runs = api.runs(f"{entity}/{proj}")
        for run in runs:
            if run.state in ["running", "finished"]:
                best = run.summary.get("study/best_objective_so_far")
                if best is None:
                    best = run.summary.get("best_val_acc")
                if best is None:
                    best = run.summary.get("test_acc")
                
                # Try to identify what it is (Random vs Quantum)
                name = run.name
                arch = run.config.get("architecture", "N/A")
                if arch == "N/A":
                    arch = run.config.get("cfg.base", {}).get("model", {}).get("architecture", "N/A")
                
                fixed = run.config.get("cfg.search", {}).get("model.fixed_random_filters", {}).get("value")
                if fixed is None:
                    fixed = run.config.get("cfg.base", {}).get("model", {}).get("fixed_random_filters", False)
                
                label = "Quantum"
                if arch == "classical_baseline":
                    label = "Classical (Random)" if fixed else "Classical (Trainable)"
                
                print(f"Run: {name:45s} | Type: {label:20s} | Best Objective: {best}")

if __name__ == "__main__":
    fetch_latest_best()
