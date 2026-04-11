import wandb
import pandas as pd

api = wandb.Api()
entity = "nunombatista-university-of-coimbra"
project = "MoT-DAQCNN"

print(f"Fetching Oracle-Router runs from {entity}/{project}...")
runs = api.runs(f"{entity}/{project}", filters={"config.model.architecture": "oracle-routed"})

data = []
for run in runs:
    s = run.summary
    c = run.config
    
    # Try to find the router's accuracy (Stage 3 in your pipeline)
    # Different versions might have different keys
    router_acc = s.get('router_acc', s.get('oracle_router_acc', s.get('best_router_val_acc', None)))
    final_acc = s.get('test_acc', s.get('final_acc', None))
    
    data.append({
        'run_name': run.name,
        'router_acc': router_acc,
        'final_test_acc': final_acc,
        'dataset': c.get('dataset.name', 'unknown'),
        'kernels': c.get('model.kernel_topology_names', [])
    })

df = pd.DataFrame(data)
if not df.empty:
    print(df.to_string())
else:
    print("No 'oracle-routed' runs found in WandB.")
