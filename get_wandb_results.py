import wandb
import os

api = wandb.Api()
entity = "nunombatista-university-of-coimbra"

print("Fetching projects...")
projects = api.projects(entity)
for p in projects:
    print(f"Project: {p.name}")
    try:
        runs = api.runs(f"{entity}/{p.name}")
        for run in runs:
            if run.state == "finished" or run.state == "running":
                # We can just print the name, state, and key metrics
                acc = run.summary.get('test_acc', 'N/A')
                auc = run.summary.get('test_auc', 'N/A')
                if acc == 'N/A' and auc == 'N/A':
                    acc = run.summary.get('attr/test_acc', 'N/A')
                    auc = run.summary.get('attr/test_auc', 'N/A')
                print(f"  - Run: {run.name} | State: {run.state} | Acc: {acc} | AUC: {auc}")
    except Exception as e:
        print(f"  Error fetching runs: {e}")

