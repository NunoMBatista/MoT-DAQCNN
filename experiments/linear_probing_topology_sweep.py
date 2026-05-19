"""Linear probing sweep over (topology, encoding_mode).

For each (topology, encoding) pair we build an in-memory config, locate
the matching quantum feature cache, extract the corresponding channels,
and fit a LinearSVC probe. The sweep covers:

  - 1-kernel scale: 8 topologies x 3 encoding modes (Digital-Z,
    Digital-ZZ, Analog-ZZ) = 24 cells.
  - 4-kernel scale: 5 topology sets (g+c+h+{kings, vertical, cross,
    ring, star}) x 3 encoding modes = 15 cells.

Output CSV columns: scale, encoding, topology_set, dim, acc, auc.

The point of this sweep is to make the encoding comparison topology-
controlled: at matched topology, does analog encoding produce more
linearly-separable features than digital? Earlier probing fixed
topology=kings for all encodings (fair but arbitrary); the per-encoding
CNN-best topology run is unfair (confounds topology with encoding).
This sweep gives us both views from the same data.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)


TOPOLOGIES_1K = ["kings", "horizontal", "vertical", "cross",
                 "ring", "chain", "star", "grid"]

ENCODINGS = [
    {"name": "Digital-Z",  "encoding_mode": "digital", "include_correlators": False},
    {"name": "Digital-ZZ", "encoding_mode": "digital", "include_correlators": True},
    {"name": "Analog-ZZ",  "encoding_mode": "analog",  "include_correlators": True},
]

# 4-kernel sets — same ones used in the HP search (1h / 1i)
TOPOLOGIES_4K = [
    ["grid", "chain", "horizontal", "kings"],
    ["grid", "chain", "horizontal", "vertical"],
    ["grid", "chain", "horizontal", "cross"],
    ["grid", "chain", "horizontal", "ring"],
    ["grid", "chain", "horizontal", "star"],
]


def collect(loader):
    xs, ys = [], []
    for x, y in loader:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        xs.append(x.reshape(x.shape[0], -1))
        ys.append(np.asarray(y).squeeze())
    return np.concatenate(xs), np.concatenate(ys)


def probe(X_train, y_train, X_test, y_test, C=1.0):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    svm = LinearSVC(C=C, max_iter=50_000, dual="auto")
    svm.fit(X_tr, y_train)
    preds = svm.predict(X_te)
    acc = accuracy_score(y_test, preds)
    scores = svm.decision_function(X_te)
    if len(np.unique(y_train)) == 2:
        auc = roc_auc_score(y_test, scores)
    else:
        auc = roc_auc_score(y_test, scores, multi_class="ovr", average="macro")
    return acc, auc


def make_cfg(topology_names, encoding_mode, include_correlators,
             kernel_size=3, stride=3, dataset_name="breast_mnist"):
    return {
        "dataset": {
            "name": dataset_name,
            "data_root": "./data",
            "batch_size": 1000,
            "num_workers": 0,
            "download": True,
            "color_space": "GRAYSCALE",
        },
        "model": {
            "num_classes": 2,
            "kernel_size": kernel_size,
            "stride": stride,
            "kernel_topology_names": list(topology_names),
            "scaling_factor": 1.0,
            "evolution_time": 2.5,
            "mode": "trotter",
            "encoding_mode": encoding_mode,
            "include_correlators": include_correlators,
            "quantum_device": "default.qubit",
            "classical_device": "auto",
        },
    }


def probe_cfg(cfg, label, C=1.0):
    cache_path = find_cached_quantum_dataset(cfg)
    if cache_path is None:
        print(f"  [SKIP] {label:<40s}  no cache")
        return None
    topos = cfg["model"]["kernel_topology_names"]
    train_loader, _, test_loader, _, _ = load_cached_quantum_dataset(
        cache_path, batch_size=1000, num_workers=0, requested_kernels=topos
    )
    Xtr, ytr = collect(train_loader)
    Xte, yte = collect(test_loader)
    acc, auc = probe(Xtr, ytr, Xte, yte, C=C)
    dim = Xtr.shape[1]
    print(f"  {label:<40s}  dim={dim:>5d}  acc={acc:.4f}  AUC={auc:.4f}")
    return acc, auc, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["1k", "4k", "both"], default="both")
    ap.add_argument("--dataset", default="breast_mnist")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--output-csv",
                    default="outputs/paper_results/linear_probing/breast_mnist_topology_sweep.csv")
    args = ap.parse_args()

    rows = []

    if args.scale in ("1k", "both"):
        print("\n=== 1-kernel sweep: 8 topologies x 3 encodings ===")
        for enc in ENCODINGS:
            print(f"\n[{enc['name']}]")
            for t in TOPOLOGIES_1K:
                cfg = make_cfg([t], enc["encoding_mode"], enc["include_correlators"],
                               dataset_name=args.dataset)
                label = f"{enc['name']} 1k {t}"
                result = probe_cfg(cfg, label, C=args.C)
                if result is None:
                    continue
                acc, auc, dim = result
                rows.append({
                    "scale": "1k", "encoding": enc["name"],
                    "topology_set": t, "dim": dim,
                    "acc": f"{acc:.4f}", "auc": f"{auc:.4f}",
                })

    if args.scale in ("4k", "both"):
        print("\n=== 4-kernel sweep: 5 topology sets x 3 encodings ===")
        for enc in ENCODINGS:
            print(f"\n[{enc['name']}]")
            for topos in TOPOLOGIES_4K:
                cfg = make_cfg(topos, enc["encoding_mode"], enc["include_correlators"],
                               dataset_name=args.dataset)
                set_label = "+".join(topos)
                label = f"{enc['name']} 4k {set_label}"
                result = probe_cfg(cfg, label, C=args.C)
                if result is None:
                    continue
                acc, auc, dim = result
                rows.append({
                    "scale": "4k", "encoding": enc["name"],
                    "topology_set": set_label, "dim": dim,
                    "acc": f"{acc:.4f}", "auc": f"{auc:.4f}",
                })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
