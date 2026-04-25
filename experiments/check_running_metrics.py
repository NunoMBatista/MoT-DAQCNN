import json

def check_running():
    with open('outputs/wandb_results_dump.json', 'r') as f:
        data = json.load(f)
        
    runs = {r['name']: r for r in data['MoT-DAQCNN']}
    names = [
        'hp_search:breast_mnist:classical_baseline:1TRAINKERN:20260421_123140',
        'hp_search:breast_mnist:classical_baseline:1RANDKERN:20260421_123428'
    ]
    
    for n in names:
        r = runs.get(n)
        if not r: continue
        s = r['summary']
        print(f"Run: {n}")
        print(f"  State: {r['state']}")
        print(f"  Best Objective So Far: {s.get('study/best_objective_so_far')}")
        print(f"  Current Trials: {s.get('trial/number')}")
        print("-" * 30)

if __name__ == "__main__":
    check_running()
