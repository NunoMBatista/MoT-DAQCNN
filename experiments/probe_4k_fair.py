"""Fair 4-kernel (180-dim ensemble) probe: val-select the topology SET, then
C-tune on val, report test. Same two-stage fairness as the 1k analysis. Compare
to the dimension-matched classical-nonlinear map rff_180 (also C-tuned).

4k quantum = 4 topologies x 45 = 180 channels x 81 patches = 14580 flat dims,
matching rff_180 (180 RFF features x 81 patches).
"""
import os, sys, gc, argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.quantum_dataset_cache import find_cached_quantum_dataset, load_cached_quantum_dataset

CS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0]
SETS = [["grid", "chain", "horizontal", "kings"],
        ["grid", "chain", "horizontal", "vertical"],
        ["grid", "chain", "horizontal", "cross"],
        ["grid", "chain", "horizontal", "ring"],
        ["grid", "chain", "horizontal", "star"]]
ENCODINGS = [("Digital-Z", "digital", False),
             ("Digital-ZZ", "digital", True),
             ("Analog-ZZ", "analog", True)]


def collect(loader):
    xs, ys = [], []
    for x, y in loader:
        x = x.numpy() if isinstance(x, torch.Tensor) else x
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        xs.append(x.reshape(x.shape[0], -1)); ys.append(np.asarray(y).squeeze())
    return np.concatenate(xs), np.concatenate(ys)


def load_set(topo_set, enc, corr, ds):
    cfg = {"dataset": {"name": ds, "data_root": "./data", "batch_size": 1000, "num_workers": 0,
                       "download": True, "color_space": "GRAYSCALE"},
           "model": {"num_classes": 2, "kernel_size": 3, "stride": 3, "kernel_topology_names": topo_set,
                     "scaling_factor": 1.0, "evolution_time": 2.5, "mode": "trotter",
                     "encoding_mode": enc, "include_correlators": corr,
                     "quantum_device": "default.qubit", "classical_device": "auto"}}
    cp = find_cached_quantum_dataset(cfg)
    if cp is None:
        return None
    tr, va, te, _, _ = load_cached_quantum_dataset(cp, batch_size=1000, num_workers=0, requested_kernels=topo_set)
    return collect(tr), collect(va), collect(te)


def fit_auc(tr, va, te, C):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = tr, va, te
    sc = StandardScaler().fit(Xtr)
    svm = LinearSVC(C=C, max_iter=30_000, dual="auto").fit(sc.transform(Xtr), ytr)
    return (roc_auc_score(yva, svm.decision_function(sc.transform(Xva))),
            roc_auc_score(yte, svm.decision_function(sc.transform(Xte))))


def ctune(tr, va, te):
    best = (-1, None, None)
    for C in CS:
        v, t = fit_auc(tr, va, te, C)
        if v > best[0]:
            best = (v, t, C)
    return best


def load_rff180(ds):
    z = np.load(f"data/quantum_datasets/{ds}__classical_rff180.npz", allow_pickle=True)
    def flat(s):
        X = z[f"{s}_features"]; return X.reshape(X.shape[0], -1), np.asarray(z[f"{s}_labels"]).squeeze()
    return flat("train"), flat("val"), flat("test")


_ap = argparse.ArgumentParser()
_ap.add_argument("--datasets", default="breast_mnist,pneumonia_mnist")
_ap.add_argument("--only-enc", default="", help="comma list of encoding labels and/or 'rff_180'")
_args = _ap.parse_args()
DATASETS = _args.datasets.split(",")
ONLY_ENC = set(x for x in _args.only_enc.split(",") if x)

for ds in DATASETS:
    print(f"\n===== {ds} : 4-kernel fair probe =====", flush=True)
    for name, enc, corr in ENCODINGS:
        if ONLY_ENC and name not in ONLY_ENC:
            continue
        # stage 1: val-select the topology set at C=1.0, keeping ONLY the best
        # set's arrays in memory (holding all five 14580-dim copies can OOM).
        best = None  # (set, untuned_test, data)
        best_val = -1.0
        for s in SETS:
            d = load_set(s, enc, corr, ds)
            if d is None:
                continue
            v, t = fit_auc(*d, 1.0)
            if v > best_val:
                best_val, best = v, (s, t, d)
            else:
                del d
            gc.collect()
        if best is None:
            print(f"  {name}: no caches", flush=True); continue
        # stage 2: C-tune the val-selected set
        vb, tb, Cb = ctune(*best[2])
        setname = "+".join(x[:3] for x in best[0])
        print(f"  {name:<11s} val-set={setname:<20s} untuned_test={best[1]:.4f}  "
              f"C-tuned_test={tb:.4f} (C={Cb})", flush=True)
        del best; gc.collect()
    # classical 4k anchor
    if not ONLY_ENC or "rff_180" in ONLY_ENC:
        vr, tr_, Cr = ctune(*load_rff180(ds))
        print(f"  {'rff_180':<11s} {'(classical)':<29s} C-tuned_test={tr_:.4f} (C={Cr})", flush=True)
