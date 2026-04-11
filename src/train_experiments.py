import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.models.vit_comparison import ExperimentModel

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_probs, all_labels = 0, [], []
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        if labels.ndim > 1:
            labels = labels.squeeze()
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    
    return total_loss / len(loader), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if labels.ndim > 1:
                labels = labels.squeeze()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    
    return total_loss / len(loader), acc, probs, labels

def plot_binary_roc(y_true, y_probs, run_name):
    """Creates a clean, single-line ROC plot for the positive class."""
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {run_name}')
    plt.legend(loc="lower right")
    
    img = wandb.Image(plt)
    plt.close()
    return img

def run_experiment(cfg, train_loader, val_loader, test_loader, is_hybrid=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.models.vit_comparison import ExperimentModel 
    
    model = ExperimentModel(
        is_hybrid=is_hybrid,
        from_scratch=cfg['model'].get('from_scratch', True),
        num_layers=cfg['model'].get('num_layers', 5),
        nhead=cfg['model'].get('nhead', 4),
        dim_feedforward=cfg['model'].get('dim_feedforward', 128),
        model_name=cfg['model'].get('backbone', "deit_tiny_patch16_224"),
        num_classes=cfg['dataset']['n_classes'],
        num_topologies=len(cfg['quantum']['topologies']),
        qubits_per_kernel=cfg['quantum']['n_qubits']
    ).to(device)
    
    is_scratch = cfg.get("model", {}).get("from_scratch", True)
    mode_str = "Scratch-ViT" if is_scratch else f"Standard-{cfg['model']['backbone']}"
    run_name = f"{'Hybrid' if is_hybrid else 'Classical'}_{mode_str}_{cfg['dataset']['subset_percent']}%"
    
    wandb.init(project=cfg['wandb']['project'], name=run_name, config=cfg)
    criterion = nn.CrossEntropyLoss()

    # --- TRAINING ---
    if not is_scratch:
        print(f"--- Phase 1: Warmup ---")
        model.freeze_backbone(True)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['train'].get('warmup_lr', 1e-3))
        for epoch in range(cfg['train'].get('warmup_epochs', 5)):
            t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device)
            wandb.log({"epoch": epoch, "train/loss": t_loss, "train/acc": t_acc, "val/loss": v_loss, "val/acc": v_acc})

        print(f"--- Phase 2: Full Fine-tuning ---")
        model.freeze_backbone(False)
        optimizer = optim.AdamW(model.parameters(), lr=cfg['train'].get('finetune_lr', 1e-5))
        start_epoch = cfg['train'].get('warmup_epochs', 5)
    else:
        print(f"--- Training Custom Scratch Model ---")
        optimizer = optim.AdamW(model.parameters(), lr=cfg['train'].get('scratch_lr', 1e-4))
        start_epoch = 0

    # Main Training Loop
    for epoch in range(cfg['train']['epochs']):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Log training and validation metrics per epoch
        wandb.log({
            "epoch": epoch + start_epoch, 
            "train/loss": t_loss, 
            "train/acc": t_acc,
            "val/loss": v_loss, 
            "val/acc": v_acc
        })

    # --- FINAL EVALUATION ---
    print("--- Final Evaluation on Test Set ---")
    test_loss, test_acc, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    test_preds = np.argmax(test_probs, axis=1)
    test_auc = roc_auc_score(test_labels, test_probs[:, 1]) if cfg['dataset']['n_classes'] == 2 else 0

    wandb.log({
        "final/test_acc": test_acc,
        "final/test_auc": test_auc,
        "visuals/clean_roc_curve": plot_binary_roc(test_labels, test_probs, run_name),
        "visuals/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=test_labels, preds=test_preds, 
            class_names=cfg['dataset'].get('class_names', ["Healthy", "Affected"])),
        "visuals/detailed_roc": wandb.plot.roc_curve(test_labels, test_probs)
    })
    
    print(f"Final Test Acc: {test_acc:.4f} | Final Test AUC: {test_auc:.4f}")
    wandb.finish()