"""Does quantum's linear-probe lead survive a regularization-matched classifier?

The default probe uses LinearSVC(C=1.0) un-tuned, which overfits high-dimensional
classical maps (rff_180=14580-dim) on tiny train sets and understates them. Here
we tune C on VAL per source and report TEST AUC, isolating the estimator effect
from topology. Quantum sources use their fair (val-selected) topology.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.quantum_dataset_cache import find_cached_quantum_dataset, load_cached_quantum_dataset

CS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0]
DS = "breast_mnist"


def collect(loader):
    xs, ys = [], []
    for x, y in loader:
        x = x.numpy() if isinstance(x, torch.Tensor) else x
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        xs.append(x.reshape(x.shape[0], -1)); ys.append(np.asarray(y).squeeze())
    return np.concatenate(xs), np.concatenate(ys)


def quantum(topo, enc, corr):
    cfg = {"dataset": {"name": DS, "data_root": "./data", "batch_size": 1000, "num_workers": 0,
                       "download": True, "color_space": "GRAYSCALE"},
           "model": {"num_classes": 2, "kernel_size": 3, "stride": 3, "kernel_topology_names": [topo],
                     "scaling_factor": 1.0, "evolution_time": 2.5, "mode": "trotter",
                     "encoding_mode": enc, "include_correlators": corr,
                     "quantum_device": "default.qubit", "classical_device": "auto"}}
    cp = find_cached_quantum_dataset(cfg)
    tr, va, te, _, _ = load_cached_quantum_dataset(cp, batch_size=1000, num_workers=0, requested_kernels=[topo])
    return collect(tr), collect(va), collect(te)


def classical(name):
    z = np.load(f"data/quantum_datasets/{DS}__classical_{name}.npz", allow_pickle=True)
    def flat(s):
        X = z[f"{s}_features"]; return X.reshape(X.shape[0], -1), np.asarray(z[f"{s}_labels"]).squeeze()
    return flat("train"), flat("val"), flat("test")


def tuned(tr, va, te):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = tr, va, te
    sc = StandardScaler().fit(Xtr)
    best = (-1, None, None)
    for C in CS:
        svm = LinearSVC(C=C, max_iter=50_000, dual="auto").fit(sc.transform(Xtr), ytr)
        v = roc_auc_score(yva, svm.decision_function(sc.transform(Xva)))
        t = roc_auc_score(yte, svm.decision_function(sc.transform(Xte)))
        if v > best[0]:
            best = (v, t, C)
    return best  # (val_auc, test_auc, C)


SOURCES = [
    ("Digital-ZZ (cross)",  lambda: quantum("cross", "digital", True)),
    ("Analog-ZZ (horizontal)", lambda: quantum("horizontal", "analog", True)),
    ("Digital-Z (horizontal)", lambda: quantum("horizontal", "digital", False)),
    ("poly2_45",  lambda: classical("poly2")),
    ("rff_45",    lambda: classical("rff45")),
    ("rff_180",   lambda: classical("rff180")),
]

print(f"===== {DS}: C-tuned-on-val LinearSVC, report TEST AUC =====", flush=True)
for label, load in SOURCES:
    v, t, C = tuned(*load())
    print(f"  {label:<24s} test={t:.4f}  (val={v:.4f}, C={C})", flush=True)
