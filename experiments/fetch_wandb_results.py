import wandb
import json
import os

def fetch_results():
    api = wandb.Api()
    entity = "nunombatista-university-of-coimbra"
    projects = ["MoT-DAQCNN", "MoT-DAQCNN-ZZ-Search"]
    
    all_results = {}
    
    for project in projects:
        print(f"Fetching from {project}...")
        runs = api.runs(f"{entity}/{project}")
        project_runs = []
        for run in runs:
            # We want either finished runs or the best results from hyperparameter searches
            run_summary = {}
            for k, v in run.summary.items():
                if k.startswith("_"): continue
                # Handle non-serializable objects
                if isinstance(v, (int, float, str, bool, list)) or v is None:
                    run_summary[k] = v
                else:
                    try:
                        run_summary[k] = float(v)
                    except:
                        run_summary[k] = str(v)

            res = {
                "name": run.name,
                "state": run.state,
                "config": {k: v for k, v in run.config.items() if not k.startswith("_")},
                "summary": run_summary
            }
            project_runs.append(res)
        all_results[project] = project_runs
        
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/wandb_results_dump.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Done! Saved to outputs/wandb_results_dump.json")

if __name__ == "__main__":
    fetch_results()
