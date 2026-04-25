import wandb

def fetch_by_id():
    api = wandb.Api()
    entity = "nunombatista-university-of-coimbra"
    project = "MoT-DAQCNN"
    
    ids = {
        'Quantum': '5n0fhnjb',
        'Trainable': 'zy06pdtj',
        'Random': 'dbjewy4w'
    }
    
    for label, rid in ids.items():
        print(f"\nAnalyzing {label} (ID: {rid})")
        try:
            run = api.run(f"{entity}/{project}/{rid}")
            # Scan history for objective
            history = run.history(keys=["trial/objective", "study/best_objective_so_far"], samples=2000)
            
            if history.empty:
                print("  No history found.")
                continue
                
            max_obj = history["trial/objective"].max()
            last_best = history["study/best_objective_so_far"].dropna().iloc[-1] if "study/best_objective_so_far" in history and not history["study/best_objective_so_far"].dropna().empty else "N/A"
            
            print(f"  Max Objective Ever Found: {max_obj}")
            print(f"  Best Mean (Last Logged): {last_best}")
            print(f"  Trials logged in W&B:    {len(history)}")
            
            # Specifically check Trial 800 if it exists
            if len(history) >= 800:
                t800 = history.iloc[799]
                print(f"  Trial 800 Objective:     {t800.get('trial/objective', 'N/A')}")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    fetch_by_id()
