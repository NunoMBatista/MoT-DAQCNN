import wandb

api = wandb.Api()
entity = "nunombatista-university-of-coimbra"
project = "MoT-DAQCNN"

def fetch_baseline_results():
    print(f"--- Fetching Original DAQCNN (Baseline) runs from {entity}/{project} ---")
    
    runs = api.runs(f"{entity}/{project}")
    
    data = []
    for run in runs:
        # We look for 'original' in the name and make sure it's NOT the ZZ version
        # Based on the names I saw: hp_search:original:breast_mnist:...
        if "original" in run.name.lower() and "zz" not in run.name.lower():
            s = run.summary
            
            # Based on the summary keys I saw: 'attr/test_acc', 'attr/test_auc'
            acc = s.get('attr/test_acc')
            auc = s.get('attr/test_auc')
            
            # Parameters
            kernels = s.get('param/model.kernel_topology_names', [])
            num_kernels = len(kernels) if isinstance(kernels, list) else 1
            
            if acc is not None:
                # These might be single values if logged per trial, 
                # or mean if aggregated in summary.
                # Assuming 'attr/test_acc' is the mean.
                data.append({
                    'name': run.name,
                    'type': f"{num_kernels}-Kernel",
                    'acc': f"{float(acc)*100:.1f}%" if float(acc) < 1 else f"{float(acc):.1f}%",
                    'auc': f"{float(auc):.3f}" if auc is not None else "N/A",
                    'topo': kernels
                })

    if data:
        print(f"{'Run Name':<50} | {'Type':<10} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'Topology'}")
        print("-" * 120)
        # Sort by type then by acc (descending)
        for item in sorted(data, key=lambda x: (x['type'], x['acc']), reverse=True):
            print(f"{item['name']:<50} | {item['type']:<10} | {item['acc']:<10} | {item['auc']:<10} | {item['topo']}")
    else:
        print(f"\n[!] No 'original' (No ZZ) runs found in WandB matching the criteria.")

if __name__ == "__main__":
    fetch_baseline_results()
