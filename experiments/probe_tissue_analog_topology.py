"""Val-selected 1k topology for the analog TissueMNIST rows.

The binary-dataset probe (probe_val_vs_test_topology.py) hardcodes 2-class AUC and
Breast/Pneumonia. TissueMNIST is 8-class, so this trains a multinomial LogReg per
topology and scores macro one-vs-rest AUC on BOTH val and test, selecting the
topology on VAL (the fair criterion) for Analog-Z and Analog-ZZ at single-kernel
scale. Run after the analog-Z cache is derived.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset, load_cached_quantum_dataset,
)

TOPOS = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
ENCODINGS = [("Analog-Z", "analog", False), ("Analog-ZZ", "analog", True)]


def collect(loader):
    xs, ys = [], []
    for x, y in loader:
        x = x.numpy() if isinstance(x, torch.Tensor) else x
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        xs.append(x.reshape(x.shape[0], -1))
        ys.append(np.asarray(y).squeeze())
    return np.concatenate(xs), np.concatenate(ys)


def cfg(topo, enc, corr):
    return {"dataset": {"name": "tissue_mnist", "data_root": "./data", "batch_size": 2000,
                        "num_workers": 0, "download": False, "color_space": "GRAYSCALE"},
            "model": {"num_classes": 8, "kernel_size": 3, "stride": 3,
                      "kernel_topology_names": [topo], "scaling_factor": 1.0,
                      "evolution_time": 2.5, "mode": "trotter", "encoding_mode": enc,
                      "include_correlators": corr, "quantum_device": "default.qubit",
                      "classical_device": "auto"}}


for name, enc, corr in ENCODINGS:
    print(f"\n===== {name} =====", flush=True)
    rows = []
    for t in TOPOS:
        cp = find_cached_quantum_dataset(cfg(t, enc, corr))
        if cp is None:
            print(f"  {t:<11s} no cache", flush=True)
            continue
        tr, va, te, _, _ = load_cached_quantum_dataset(
            cp, batch_size=2000, num_workers=0, requested_kernels=[t])
        Xtr, ytr = collect(tr); Xva, yva = collect(va); Xte, yte = collect(te)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(sc.transform(Xtr), ytr)
        va_auc = roc_auc_score(yva, clf.predict_proba(sc.transform(Xva)),
                               multi_class="ovr", average="macro")
        te_auc = roc_auc_score(yte, clf.predict_proba(sc.transform(Xte)),
                               multi_class="ovr", average="macro")
        rows.append((t, va_auc, te_auc))
        print(f"  {t:<11s} val={va_auc:.4f} test={te_auc:.4f}", flush=True)
        del Xtr, Xva, Xte
    if rows:
        vb = max(rows, key=lambda r: r[1])
        tb = max(rows, key=lambda r: r[2])
        print(f"  => {name} VAL-selected={vb[0]} (val={vb[1]:.4f} test={vb[2]:.4f}) | "
              f"test-selected={tb[0]} (test={tb[2]:.4f}) | "
              f"inflation={tb[2]-vb[2]:+.4f}", flush=True)
