import wandb
import pandas as pd

def fetch_history_max():
    api = wandb.Api()
    entity = "nunombatista-university-of-coimbra"
    project = "MoT-DAQCNN"
    
    names = [
        'hp_search:ORIGINAL:breast_mnist:DIGITAL:1KERN:ZZ:20260329_120734',
        'hp_search:breast_mnist:classical_baseline:1TRAINKERN:20260421_123140',
        'hp_search:breast_mnist:classical_baseline:1RANDKERN:20260421_123428'
    ]
    
    for n in names:
        print(f"\nChecking history for: {n}")
        try:
            run = api.run(f"{entity}/{project}/{n}")
            # Get the history of the objective metric
            # Using scan=True to get all trials
            history = run.history(keys=["trial/objective", "study/best_objective_so_far"], pandas=False)
            
            if not history:
                print("  No history found yet.")
                continue
                
            objs = [h.get("trial/objective") for h in history if h.get("trial/objective") is not None]
            bests = [h.get("study/best_objective_so_far") for h in history if h.get("study/best_objective_so_far") is not None]
            
            if objs:
                print(f"  Max Trial Objective: {max(objs)}")
            if bests:
                print(f"  Max Best Objective:  {max(bests)}")
            print(f"  Total History Points: {len(history)}")
        except Exception as e:
            print(f"  Error fetching {n}: {e}")

if __name__ == "__main__":
    fetch_history_max()
