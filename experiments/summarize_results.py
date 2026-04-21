import json

def summarize():
    with open('outputs/wandb_results_dump.json', 'r') as f:
        data = json.load(f)
        
    for proj, runs in data.items():
        print(f"\n--- Project: {proj} ---")
        # Filter for relevant runs (finished or best in search)
        relevant_runs = [r for run in runs if (r := parse_run(run)) is not None]
        # Sort by accuracy
        relevant_runs.sort(key=lambda x: x['acc'] if isinstance(x['acc'], (int, float)) else 0, reverse=True)
        
        for r in relevant_runs[:15]:
            print(f"Name: {r['name']:40s} | Arch: {r['arch']:10s} | ZZ: {str(r['zz']):5s} | Acc: {r['acc']} | AUC: {r['auc']}")

def parse_run(run):
    s = run['summary']
    c = run['config']
    
    # In Optuna runs, metrics are often in attr/ or search/
    acc = s.get('attr/test_acc')
    if acc is None: acc = s.get('test_acc')
    if acc is None: acc = s.get('best_test_acc')
    if acc is None: acc = s.get('search/best_value') # If objective was test_acc
    
    auc = s.get('attr/test_auc')
    if auc is None: auc = s.get('test_auc')
    if auc is None: auc = s.get('best_test_auc')
    
    arch = c.get('architecture')
    if arch is None: 
        # Check inside nested base cfg
        base = c.get('cfg.base', {})
        arch = base.get('model', {}).get('architecture')
    
    zz = c.get('cfg.base', {}).get('model', {}).get('include_correlators')
    if zz is None: zz = s.get('param/model.include_correlators')
    if zz is None: zz = False
    
    return {
        "name": run['name'],
        "arch": str(arch),
        "zz": zz,
        "acc": acc,
        "auc": auc
    }

if __name__ == "__main__":
    summarize()
