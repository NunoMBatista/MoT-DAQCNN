import json

def analyze_1kern():
    with open('outputs/wandb_results_dump.json', 'r') as f:
        data = json.load(f)
        
    runs = {r['name']: r for r in data['MoT-DAQCNN']}
    names = [
        'hp_search:ORIGINAL:breast_mnist:DIGITAL:1KERN:ZZ:20260329_120734',
        'hp_search:breast_mnist:classical_baseline:1TRAINKERN:20260421_123140',
        'hp_search:breast_mnist:classical_baseline:1RANDKERN:20260421_123428'
    ]
    
    for n in names:
        r = runs.get(n)
        if not r: 
            print(f"Run {n} not found.")
            continue
        s = r['summary']
        c = r['config']
        print(f"Run: {n}")
        print(f"  Trials Completed: {s.get('search/n_trials_completed')}")
        print(f"  Trial Seeds:      {c.get('trial_seeds')}")
        print(f"  Best Objective:   {s.get('search/best_value')}")
        print("-" * 30)

if __name__ == "__main__":
    analyze_1kern()
