import wandb

def find_ids():
    api = wandb.Api()
    entity = "nunombatista-university-of-coimbra"
    project = "MoT-DAQCNN"
    
    targets = [
        'hp_search:ORIGINAL:breast_mnist:DIGITAL:1KERN:ZZ:20260329_120734',
        'hp_search:breast_mnist:classical_baseline:1TRAINKERN:20260421_123140',
        'hp_search:breast_mnist:classical_baseline:1RANDKERN:20260421_123428'
    ]
    
    print("Searching for internal Run IDs...")
    runs = api.runs(f"{entity}/{project}")
    for run in runs:
        if run.name in targets:
            print(f"Name: {run.name:50s} | ID: {run.id} | State: {run.state}")

if __name__ == "__main__":
    find_ids()
