"""Head-capacity sweep on frozen features.

For each (head_type, feature_source) cell, run a small Optuna study over
lr/weight_decay/dropout, then evaluate the best HPs over multiple seeds
on the test set. Results: one CSV row per cell (mean ± std) and one row
per (cell, seed).

The sweep is designed to answer: how does the quantum-vs-classical AUC
gap evolve as a function of downstream classifier capacity?

See docs/plans/capacity_sweep.md for the full design.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data import get_dataloaders
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)
from src.utils.training_utils import (
    build_classification_head,
    build_linear_head,
    build_mlp1_head,
    build_mlp2_head,
)
from src.utils.classical_nonlinear_features import poly2_features, rff_features


# ---------------------------------------------------------------------------
# Defaults (kept module-level so they show up in --help-style introspection)
# ---------------------------------------------------------------------------
DATASET = "breast_mnist"
NUM_CLASSES = 2
ACTIVATION = "gelu"
EPOCHS = 100
PATIENCE = 30
GRAD_CLIP = 1.0
BATCH_SIZE = 32
RANDOM_FILTER_SEED = 42  # fixed across all heads/seeds for the classical baseline


# ---------------------------------------------------------------------------
# Head ladder
# Each entry: (builder_fn(num_classes, dropout, activation, in_channels) -> Module,
#              use_scheduler: bool, label: str)
# in_channels is needed by the CNN heads; MLP/Linear heads ignore it.
# ---------------------------------------------------------------------------

def _head_factory(name):
    if name == "linear":
        return (lambda nc, do, act, ic: build_linear_head(nc), False)
    if name == "mlp1":
        return (lambda nc, do, act, ic: build_mlp1_head(nc, hidden=32, dropout=do, activation=act), True)
    if name == "mlp2":
        return (lambda nc, do, act, ic: build_mlp2_head(nc, h1=64, h2=32, dropout=do, activation=act), True)
    if name == "cnn_small":
        return (lambda nc, do, act, ic: build_classification_head(ic, nc, do, act, hidden_channels=16), True)
    if name == "cnn_large":
        return (lambda nc, do, act, ic: build_classification_head(ic, nc, do, act, hidden_channels=64), True)
    raise ValueError(f"Unknown head: {name}")


HEAD_NAMES = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]


# ---------------------------------------------------------------------------
# Feature sources
# ---------------------------------------------------------------------------

# All possible source names (static, for argparse choices). The actual specs
# (quantum config paths) are built per-dataset by build_feature_sources().
SOURCE_NAMES = [
    "raw", "random_9", "random_45", "random_180",
    "digital_z_1k", "digital_zz_1k", "analog_z_1k", "analog_zz_1k",
    "digital_zz_4k", "analog_zz_4k",
    "poly2_45", "rff_45", "rff_180",          # classical nonlinear controls
]


def build_feature_sources(dataset):
    """Feature-source specs for a dataset (quantum configs follow configs/<dataset>/)."""
    c = f"configs/{dataset}"
    return {
        "raw":           {"kind": "raw"},
        "random_9":      {"kind": "random", "n_filters": 9,   "kernel_size": 3, "stride": 3},
        "random_45":     {"kind": "random", "n_filters": 45,  "kernel_size": 3, "stride": 3},
        "random_180":    {"kind": "random", "n_filters": 180, "kernel_size": 3, "stride": 3},
        "digital_z_1k":  {"kind": "quantum", "config": f"{c}/digital_z_best.yml"},
        "digital_zz_1k": {"kind": "quantum", "config": f"{c}/digital_zz_best.yml"},
        "analog_z_1k":   {"kind": "quantum", "config": f"{c}/analog_z_best.yml"},
        "analog_zz_1k":  {"kind": "quantum", "config": f"{c}/analog_zz_best.yml"},
        "digital_zz_4k": {"kind": "quantum", "config": f"{c}/digital_zz_4k_best.yml"},
        "analog_zz_4k":  {"kind": "quantum", "config": f"{c}/analog_zz_4k_best.yml"},
        "poly2_45":      {"kind": "poly2"},
        "rff_45":        {"kind": "rff", "n_output": 45},
        "rff_180":       {"kind": "rff", "n_output": 180},
    }


# Populated in main() from --dataset; module-level default keeps imports working.
FEATURE_SOURCES = build_feature_sources(DATASET)


def standardize_features(train, val, test):
    """Per-feature z-scoring (mean 0, unit variance), fit on train only and
    applied to all splits, matching the LinearSVC probe's StandardScaler. Removes
    the feature-scale confound so the capacity comparison is about information,
    not magnitude. Operates on the flattened (C*H*W) feature vector and reshapes
    back, so spatial structure is preserved for the CNN heads.
    """
    (Xtr, ytr), (Xv, yv), (Xt, yt) = train, val, test
    mean = Xtr.reshape(Xtr.shape[0], -1).mean(0, keepdim=True)
    std = Xtr.reshape(Xtr.shape[0], -1).std(0, keepdim=True) + 1e-6

    def z(X):
        s = X.shape
        return ((X.reshape(s[0], -1) - mean) / std).reshape(s)

    return (z(Xtr), ytr), (z(Xv), yv), (z(Xt), yt)


def _stack(loader):
    """Concatenate all batches from a DataLoader into (X, y) tensors."""
    xs, ys = [], []
    for x, y in loader:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x))
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(np.asarray(y))
        xs.append(x.float())
        ys.append(y.long().squeeze())
    return torch.cat(xs), torch.cat(ys)


def load_features(source_name, source_spec):
    """Return ((X_train, y_train), (X_val, y_val), (X_test, y_test), in_channels)."""
    if source_spec["kind"] == "quantum":
        with open(source_spec["config"]) as f:
            cfg = yaml.safe_load(f)
        cache_path = find_cached_quantum_dataset(cfg)
        if cache_path is None:
            raise FileNotFoundError(f"No quantum cache for source '{source_name}'")
        topos = cfg["model"]["kernel_topology_names"]
        train_loader, val_loader, test_loader, _, _ = load_cached_quantum_dataset(
            cache_path, batch_size=1000, num_workers=0, requested_kernels=topos
        )
        (Xtr, ytr) = _stack(train_loader)
        (Xv,  yv)  = _stack(val_loader)
        (Xt,  yt)  = _stack(test_loader)
        return (Xtr, ytr), (Xv, yv), (Xt, yt), Xtr.shape[1]

    # Both raw and random use the same dataloader path
    cfg = {"dataset": {"name": DATASET, "data_root": "./data", "batch_size": 1000,
                       "num_workers": 0, "download": True, "color_space": "GRAYSCALE"}}
    train_loader, val_loader, test_loader, _ = get_dataloaders(cfg)

    if source_spec["kind"] == "raw":
        (Xtr, ytr) = _stack(train_loader)
        (Xv,  yv)  = _stack(val_loader)
        (Xt,  yt)  = _stack(test_loader)
        return (Xtr, ytr), (Xv, yv), (Xt, yt), Xtr.shape[1]

    if source_spec["kind"] == "random":
        rng = np.random.default_rng(source_spec.get("seed", RANDOM_FILTER_SEED))
        N = source_spec["n_filters"]
        ks = source_spec["kernel_size"]
        st = source_spec["stride"]
        w = rng.standard_normal((N, 1, ks, ks)).astype(np.float32)
        norms = np.linalg.norm(w.reshape(N, -1), axis=1, keepdims=True)
        w /= (norms.reshape(N, 1, 1, 1) + 1e-8)
        filters = torch.from_numpy(w)

        def apply_filters(loader):
            xs, ys = [], []
            for x, y in loader:
                out = F.conv2d(x.float(), filters, stride=st)
                xs.append(out)
                ys.append(y.long().squeeze())
            return torch.cat(xs), torch.cat(ys)

        (Xtr, ytr) = apply_filters(train_loader)
        (Xv,  yv)  = apply_filters(val_loader)
        (Xt,  yt)  = apply_filters(test_loader)
        return (Xtr, ytr), (Xv, yv), (Xt, yt), Xtr.shape[1]

    if source_spec["kind"] in ("poly2", "rff"):
        # Classical nonlinear controls: per-patch degree-2 / RFF maps on the
        # raw [0,1] image, same patches the random-conv baseline sees.
        if source_spec["kind"] == "poly2":
            fmap = lambda x: poly2_features(x)
        else:
            D = source_spec["n_output"]
            seed = source_spec.get("seed", RANDOM_FILTER_SEED)
            fmap = lambda x: rff_features(x, D, seed=seed)

        def apply_map(loader):
            xs, ys = [], []
            for x, y in loader:
                xs.append(fmap(x.float()))
                ys.append(y.long().squeeze())
            return torch.cat(xs), torch.cat(ys)

        (Xtr, ytr) = apply_map(train_loader)
        (Xv,  yv)  = apply_map(val_loader)
        (Xt,  yt)  = apply_map(test_loader)
        return (Xtr, ytr), (Xv, yv), (Xt, yt), Xtr.shape[1]

    raise ValueError(f"Unknown source kind: {source_spec['kind']}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_head(head, train, val, test, lr, weight_decay, use_scheduler,
               epochs=None, patience=None, batch_size=None,
               grad_clip=GRAD_CLIP, device="cpu", seed=0):
    """Train one head with given HPs. Returns dict of val/test metrics from
    the lowest-val-loss checkpoint.
    """
    # Read module globals at call time so --epochs/--batch-size overrides apply.
    if epochs is None: epochs = EPOCHS
    if patience is None: patience = PATIENCE
    if batch_size is None: batch_size = BATCH_SIZE
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = head.to(device)
    Xtr, ytr = train[0].to(device), train[1].to(device)
    Xv,  yv  = val[0].to(device),   val[1].to(device)
    Xt,  yt  = test[0].to(device),  test[1].to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if use_scheduler else None)
    criterion = nn.CrossEntropyLoss()

    n_train = Xtr.shape[0]
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            logits = head(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(Xv)
            val_loss = criterion(val_logits, yv).item()
            val_acc = (val_logits.argmax(dim=1) == yv).float().mean().item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_logits = head(Xt)
        preds = test_logits.argmax(dim=1).cpu().numpy()
        if NUM_CLASSES == 2:
            scores = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        else:
            scores = torch.softmax(test_logits, dim=1).cpu().numpy()

    yt_np = yt.cpu().numpy()
    test_acc = accuracy_score(yt_np, preds)
    if NUM_CLASSES == 2:
        test_auc = roc_auc_score(yt_np, scores)
    else:
        test_auc = roc_auc_score(yt_np, scores, multi_class="ovr", average="macro")
    test_f1 = f1_score(yt_np, preds, average="macro")

    return {
        "val_acc": best_val_acc, "val_loss": best_val_loss,
        "test_acc": test_acc, "test_auc": test_auc, "test_f1": test_f1,
    }


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------

def run_cell(head_name, source_name, train, val, test, in_channels,
             cell_dir, n_trials, seeds, device):
    cell_dir.mkdir(parents=True, exist_ok=True)
    builder, use_scheduler = _head_factory(head_name)

    # Optuna study (one DB per cell, supports resume)
    db_path = cell_dir / "study.db"
    study = optuna.create_study(
        study_name=f"{head_name}_{source_name}",
        direction="maximize",
        storage=f"sqlite:///{db_path}",
        sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e-3, log=True)
        do = trial.suggest_float("dropout", 0.0, 0.7)
        head = builder(NUM_CLASSES, do, ACTIVATION, in_channels)
        metrics = train_head(
            head, train, val, test, lr=lr, weight_decay=wd,
            use_scheduler=use_scheduler, device=device, seed=0,
        )
        return metrics["val_acc"]

    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best = study.best_params
    best_value = study.best_value

    # Validate top-1 over the provided seeds
    val_dir = cell_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    per_seed = []
    for seed in seeds:
        head = builder(NUM_CLASSES, best["dropout"], ACTIVATION, in_channels)
        metrics = train_head(
            head, train, val, test,
            lr=best["lr"], weight_decay=best["wd"],
            use_scheduler=use_scheduler, device=device, seed=seed,
        )
        with open(val_dir / f"seed_{seed}.json", "w") as f:
            json.dump({"seed": seed, **metrics, "params": best}, f, indent=2)
        per_seed.append({"seed": seed, **metrics})

    summary = {
        "head": head_name, "source": source_name,
        "in_channels": int(in_channels),
        "n_trials_completed": len(study.trials),
        "best_value_val_acc": float(best_value),
        "best_params": {k: float(v) for k, v in best.items()},
        "test_acc_mean": float(np.mean([m["test_acc"] for m in per_seed])),
        "test_acc_std":  float(np.std([m["test_acc"] for m in per_seed])),
        "test_auc_mean": float(np.mean([m["test_auc"] for m in per_seed])),
        "test_auc_std":  float(np.std([m["test_auc"] for m in per_seed])),
        "test_f1_mean":  float(np.mean([m["test_f1"] for m in per_seed])),
        "test_f1_std":   float(np.std([m["test_f1"] for m in per_seed])),
    }
    with open(cell_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary, per_seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _param_count(builder, num_classes, in_channels, dummy_input):
    """Count params of a freshly built head, with LazyLinear materialised."""
    head = builder(num_classes, 0.0, ACTIVATION, in_channels)
    with torch.no_grad():
        head(dummy_input)  # materialise LazyLinear
    return sum(p.numel() for p in head.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="breast_mnist")
    p.add_argument("--heads", nargs="+", default=HEAD_NAMES, choices=HEAD_NAMES)
    p.add_argument("--sources", nargs="+", default=SOURCE_NAMES, choices=SOURCE_NAMES)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--output-dir", type=str,
                   default="outputs/paper_results/capacity_sweep/breast_mnist")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' | 'cuda' | 'cpu'")
    p.add_argument("--standardize", action=argparse.BooleanOptionalAction,
                   default=True, help="per-feature z-score each source (fit on train)")
    p.add_argument("--num-classes", type=int, default=2,
                   help="2 for binary MedMNIST; 8 for TissueMNIST (sets multiclass AUC)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override batch size (default 32; use larger for big datasets)")
    p.add_argument("--epochs", type=int, default=None, help="override max epochs")
    args = p.parse_args()
    globals()["NUM_CLASSES"] = args.num_classes
    if args.batch_size is not None: globals()["BATCH_SIZE"] = args.batch_size
    if args.epochs is not None: globals()["EPOCHS"] = args.epochs

    # Dataset-aware: set the module globals used by load_features and build the
    # feature-source specs (quantum config paths) for the chosen dataset.
    globals()["DATASET"] = args.dataset
    globals()["FEATURE_SOURCES"] = build_feature_sources(args.dataset)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)

    # Load all feature sources upfront (cached, fast)
    print("\nLoading feature sources...")
    features = {}
    for src in args.sources:
        spec = FEATURE_SOURCES[src]
        t0 = time.time()
        (train, val, test, in_ch) = load_features(src, spec)
        if args.standardize:
            train, val, test = standardize_features(train, val, test)
        features[src] = (train, val, test, in_ch)
        dim = int(np.prod(train[0].shape[1:]))
        print(f"  {src:<16s}  shape={tuple(train[0].shape[1:])}  dim={dim}  "
              f"train={train[0].shape[0]}  val={val[0].shape[0]}  test={test[0].shape[0]}  "
              f"({time.time()-t0:.1f}s)")

    # Run all cells
    summary_rows = []
    per_seed_rows = []
    t_sweep0 = time.time()
    for src in args.sources:
        train, val, test, in_ch = features[src]
        dummy = train[0][:1].to(device)
        for head in args.heads:
            cell_dir = cells_dir / f"{head}__{src}"
            print(f"\n=== {head} on {src} ===")
            t0 = time.time()
            summary, per_seed = run_cell(
                head, src, train, val, test, in_ch,
                cell_dir, args.n_trials, args.seeds, device,
            )

            # Param count (materialise LazyLinear)
            builder, _ = _head_factory(head)
            try:
                n_params = _param_count(builder, NUM_CLASSES, in_ch, dummy.cpu())
            except Exception:
                n_params = -1
            summary["param_count"] = n_params

            print(f"  best val_acc={summary['best_value_val_acc']:.4f}  "
                  f"test_acc={summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}  "
                  f"test_auc={summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f}  "
                  f"params={n_params}  ({time.time()-t0:.1f}s)")

            summary_rows.append({
                "head": head, "source": src,
                "in_channels": in_ch, "param_count": n_params,
                "n_trials": summary["n_trials_completed"],
                "best_val_acc": summary["best_value_val_acc"],
                "best_lr": summary["best_params"]["lr"],
                "best_wd": summary["best_params"]["wd"],
                "best_dropout": summary["best_params"]["dropout"],
                "test_acc_mean": summary["test_acc_mean"], "test_acc_std": summary["test_acc_std"],
                "test_auc_mean": summary["test_auc_mean"], "test_auc_std": summary["test_auc_std"],
                "test_f1_mean":  summary["test_f1_mean"],  "test_f1_std":  summary["test_f1_std"],
            })
            for m in per_seed:
                per_seed_rows.append({"head": head, "source": src, **m})

    # Write aggregated CSVs
    summary_csv = out_dir / "summary.csv"
    per_seed_csv = out_dir / "per_seed.csv"
    with open(summary_csv, "w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    with open(per_seed_csv, "w", newline="") as f:
        if per_seed_rows:
            w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
            w.writeheader()
            w.writerows(per_seed_rows)

    print(f"\nSweep complete in {(time.time()-t_sweep0)/60:.1f} min")
    print(f"  summary:  {summary_csv}")
    print(f"  per_seed: {per_seed_csv}")


if __name__ == "__main__":
    main()
