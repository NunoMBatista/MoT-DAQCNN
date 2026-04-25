import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from src.models.phase3_vit_comparison import Phase3ExperimentModel

def run_phase3_experiment(cfg, train_loader, val_loader, test_loader, is_hybrid, experiment_type):
    """
    Runs a single training/evaluation session for either Classical or Hybrid models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if wandb.run is not None:
        wandb.finish()

    subset_p = cfg['dataset'].get('subset_percent', 'unknown')
    group_name = f"phase3_{experiment_type}_ablation"
    run_name = f"{'Hybrid' if is_hybrid else 'Classical'}_data_{subset_p}%"

    model = Phase3ExperimentModel(
        is_hybrid=is_hybrid,
        from_scratch=cfg['model']['from_scratch'],
        model_name=cfg['model']['backbone'],
        num_layers=cfg['model']['num_layers'],
        num_classes=cfg['dataset']['n_classes'],
        q_channels=180  
    ).to(device)

    wandb.init(
        project=cfg['wandb']['project'], 
        group=group_name, 
        name=run_name, 
        config=cfg,
        reinit=True
    )
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(cfg['train']['lr']), 
        weight_decay=cfg['train'].get('weight_decay', 0.01)
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n" + "-"*40)
    print(f"🚀 RUN START: {run_name}")
    print(f"📦 Training Samples: {len(train_loader.dataset)}")
    print(f"-"*40, flush=True)

    for epoch in range(cfg['train']['epochs']):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.ndim > 1: y = y.squeeze()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.append(y.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(np.concatenate(train_targets), np.concatenate(train_preds))

        model.eval()
        val_loss = 0.0
        all_y, all_p = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if y.ndim > 1: y = y.squeeze()
                outputs = model(x)
                val_loss += criterion(outputs, y).item()
                all_y.append(y.cpu().numpy())
                all_p.append(torch.softmax(outputs, dim=1).cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_probs = np.concatenate(all_p)
        val_labels = np.concatenate(all_y)
        val_acc = accuracy_score(val_labels, np.argmax(val_probs, axis=1))
        val_auc = roc_auc_score(val_labels, val_probs[:, 1]) if cfg['dataset']['n_classes'] == 2 else 0

        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/acc": train_acc,
            "val/loss": avg_val_loss,
            "val/acc": val_acc,
            "val/auc": val_auc
        })

    # FINAL EVALUATION
    model.eval()
    test_y, test_p = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if y.ndim > 1: y = y.squeeze()
            outputs = model(x)
            test_y.append(y.cpu().numpy())
            test_p.append(torch.softmax(outputs, dim=1).cpu().numpy())

    test_probs = np.concatenate(test_p)
    test_labels = np.concatenate(test_y)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs[:, 1]) if cfg['dataset']['n_classes'] == 2 else 0

    # LOGGING BOTH INDIVIDUAL AND COMPARISON METRICS
    wandb.log({
        "final/test_acc": test_acc,
        "final/test_auc": test_auc,
        "comparison/auc": test_auc, # For the screenshot style bar chart
        "comparison/acc": test_acc, # For the screenshot style bar chart
        "visuals/roc_curve": wandb.plot.roc_curve(test_labels, test_probs, labels=["Healthy", "Affected"]),
        "visuals/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=test_labels, preds=test_preds, class_names=["Healthy", "Affected"]
        )
    })
    
    wandb.run.summary["final_acc"] = test_acc
    wandb.run.summary["final_auc"] = test_auc
    wandb.run.summary["is_hybrid"] = int(is_hybrid)
    wandb.run.summary["subset_percent"] = subset_p

    print(f"✅ Run Complete. Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}\n", flush=True)
    
    wandb.finish()
    del model
    torch.cuda.empty_cache()
    
    return test_acc, test_auc # Return results for suite delta calculation