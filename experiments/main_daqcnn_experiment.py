import argparse
import os
import random
import sys
import time
import yaml

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.daqcnn import DAQCNN
from src.utils.data import get_dataloaders


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, optim, device, grad_clip):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optim.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, labels) * bs
        total_count += bs

    return total_loss / total_count, total_acc / total_count


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, total_count = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long().to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, labels) * bs
            total_count += bs
    return total_loss / total_count, total_acc / total_count


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DAQCNN experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "pneumonia_mnist.yml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = cfg.get("misc", {}).get("seed", 0)
    set_seed(seed)

    # Determine device: use config preference but fall back to CPU if CUDA unavailable
    requested_device = cfg.get("model", {}).get("classical_device", "auto")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = requested_device
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(cfg)

    # Model
    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", n_classes)
    model = DAQCNN(
        num_classes=num_classes,
        kernel_size=model_cfg.get("kernel_size", 2),
        stride=model_cfg.get("stride", 1),
        kernel_topology_names=model_cfg.get("kernel_topology_names", None),
        scaling_factor=model_cfg.get("scaling_factor", 1.0),
        mode=model_cfg.get("mode", "trotter"),
        dropout=model_cfg.get("dropout", 0.1),
        quantum_device=model_cfg.get("quantum_device", "default.qubit"),
        quantum_device_kwargs=model_cfg.get("quantum_device_kwargs", None),
        classical_device=device,
    )

    model.to(device)

    # Optimizer
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    epochs = optim_cfg.get("epochs", 1)
    grad_clip = optim_cfg.get("grad_clip", 0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_every = cfg.get("misc", {}).get("log_every", 50)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s"
        )

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
