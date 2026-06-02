"""Fair topology selection for the capacity sweep.

The existing topology sweep (linear_probing_topology_sweep.py) reports TEST AUC
and discards val, so picking the max-AUC topology = selecting on the test set
(optimistically biased). This script trains a LinearSVC on train and scores on
BOTH val and test for every topology, so we can:
  (a) select the topology on VAL (the fair criterion) and report its TEST AUC, and
  (b) quantify the selection inflation = test-selected best minus val-selected best.

The val-selected topology is what we freeze in the probe-best capacity re-run.
1k scale, encodings Digital-Z / Digital-ZZ / Analog-ZZ, both datasets.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset, load_cached_quantum_dataset,
)

TOPOS = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
ENCODINGS = [("Digital-Z", "digital", False),
             ("Digital-ZZ", "digital", True),
             ("Analog-ZZ", "analog", True)]


def collect(loader):
    xs, ys = [], []
    for x, y in loader:
        x = x.numpy() if isinstance(x, torch.Tensor) else x
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        xs.append(x.reshape(x.shape[0], -1))
        ys.append(np.asarray(y).squeeze())
    return np.concatenate(xs), np.concatenate(ys)


def cfg(topo, enc, corr, ds):
    return {"dataset": {"name": ds, "data_root": "./data", "batch_size": 1000,
                        "num_workers": 0, "download": True, "color_space": "GRAYSCALE"},
            "model": {"num_classes": 2, "kernel_size": 3, "stride": 3,
                      "kernel_topology_names": [topo], "scaling_factor": 1.0,
                      "evolution_time": 2.5, "mode": "trotter", "encoding_mode": enc,
                      "include_correlators": corr, "quantum_device": "default.qubit",
                      "classical_device": "auto"}}


for ds in ["breast_mnist", "pneumonia_mnist"]:
    print(f"\n===== {ds} =====", flush=True)
    for name, enc, corr in ENCODINGS:
        rows = []
        for t in TOPOS:
            cp = find_cached_quantum_dataset(cfg(t, enc, corr, ds))
            if cp is None:
                continue
            tr, va, te, _, _ = load_cached_quantum_dataset(
                cp, batch_size=1000, num_workers=0, requested_kernels=[t])
            Xtr, ytr = collect(tr); Xva, yva = collect(va); Xte, yte = collect(te)
            sc = StandardScaler().fit(Xtr)
            svm = LinearSVC(C=1.0, max_iter=50_000, dual="auto").fit(sc.transform(Xtr), ytr)
            va_auc = roc_auc_score(yva, svm.decision_function(sc.transform(Xva)))
            te_auc = roc_auc_score(yte, svm.decision_function(sc.transform(Xte)))
            rows.append((t, va_auc, te_auc))
        if not rows:
            print(f"  {name}: no caches", flush=True); continue
        val_best = max(rows, key=lambda r: r[1])
        test_best = max(rows, key=lambda r: r[2])
        print(f"  {name}:", flush=True)
        for t, v, te in rows:
            m = (" <val-best" if t == val_best[0] else "") + (" <test-best" if t == test_best[0] else "")
            print(f"    {t:<11s} val={v:.4f} test={te:.4f}{m}", flush=True)
        print(f"    => VAL-selected={val_best[0]} (test={val_best[2]:.4f}) | "
              f"TEST-selected={test_best[0]} (test={test_best[2]:.4f}) | "
              f"selection inflation={test_best[2]-val_best[2]:+.4f}", flush=True)
