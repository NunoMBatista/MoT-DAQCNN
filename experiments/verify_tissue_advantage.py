"""Fairness audit of the apparent low-capacity quantum advantage on TissueMNIST.

The capacity sweep showed digital_zz beating classical nonlinear maps at the
linear head. Two confounds could inflate that: the quantum topology was selected
on TEST (probe_tissue_topologies.py uses no val split), and RFF used a fixed
gamma=1/9 heuristic (never tuned). This script removes both:

  - quantum: probe EVERY 1k topology and 4k ensemble, select on a held-out VAL
    subsample, report the val-selected pick's TEST AUC.
  - classical: poly-2, and RFF at a SWEPT bandwidth (gamma), each C- and
    gamma-tuned on VAL, reported on TEST. Matched dimension (45 and 180).

All at the linear-probe level (regularized multinomial LogReg, the fair matched
estimator), multiclass macro-OVR AUC on a subsample with train/val/test split.
If val-selected quantum still beats gamma-tuned RFF at matched dim, the edge is
real; if a tuned RFF closes it, it was a fairness artifact.
"""
import os, sys, time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.quantum_dataset_cache import load_cached_quantum_dataset
from src.utils.data import get_dataloaders
from src.utils.classical_nonlinear_features import poly2_features, rff_features

torch.manual_seed(0)   # reproducible loader shuffles / subsample draws

CACHE = "data/quantum_datasets/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz.npz"
TOPOS = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
ENSEMBLES = [["grid", "chain", "horizontal", t] for t in
             ["kings", "vertical", "cross", "ring", "star"]]
N_TR, N_VAL, N_TE = 20000, 8000, 15000
CS = [1e-2, 1e-1, 1.0]
GAMMAS_REL = [0.25, 0.5, 1.0, 2.0, 4.0]   # multiples of the 1/9 heuristic
NCLS = 8


def macro_auc(clf, X, y):
    P = clf.predict_proba(X)
    Yb = label_binarize(y, classes=list(range(NCLS)))
    return roc_auc_score(Yb, P, average="macro", multi_class="ovr")


def fit_eval(Xtr, ytr, Xv, yv, Xte, yte):
    """C-tuned-on-val LogReg; return (val_auc, test_auc, C) at the best C."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xv, Xte = sc.transform(Xtr), sc.transform(Xv), sc.transform(Xte)
    best = (-1, None, None)
    for C in CS:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Xtr, ytr)
        v = macro_auc(clf, Xv, yv); t = macro_auc(clf, Xte, yte)
        if v > best[0]:
            best = (v, t, C)
    return best


def stack(loader, n):
    xs, ys, got = [], [], 0
    for x, y in loader:
        xs.append(x.float()); ys.append(y.long().squeeze()); got += len(x)
        if got >= n:
            break
    X = torch.cat(xs)[:n]; Y = torch.cat(ys)[:n]
    return X, Y.numpy()


def flat(X):
    return X.reshape(X.shape[0], -1).numpy()


print("loading quantum ZZ cache subsample...", flush=True)
tr_loader, _, te_loader, _, _ = load_cached_quantum_dataset(
    CACHE, batch_size=2000, num_workers=0)
Xq_all, yq_all = stack(tr_loader, N_TR + N_VAL)
Xq_te, yq_te = stack(te_loader, N_TE)
Xq_tr, ytr = Xq_all[:N_TR], yq_all[:N_TR]
Xq_v, yv = Xq_all[N_TR:N_TR + N_VAL], yq_all[N_TR:N_TR + N_VAL]
# channel block for topology k is [k*45:(k+1)*45]
def topo_block(X, names):
    idx = []
    for nm in names:
        k = TOPOS.index(nm); idx += list(range(k * 45, (k + 1) * 45))
    return X[:, idx, :, :]

print("\n=== QUANTUM digital_zz, per topology (val-selected) ===", flush=True)
q1k = []
for t in TOPOS:
    v, te, C = fit_eval(flat(topo_block(Xq_tr, [t])), ytr,
                        flat(topo_block(Xq_v, [t])), yv,
                        flat(topo_block(Xq_te, [t])), yq_te)
    q1k.append((t, v, te)); print(f"  1k {t:<11s} val={v:.4f} test={te:.4f}", flush=True)
b1 = max(q1k, key=lambda r: r[1])
q4k = []
for ens in ENSEMBLES:
    v, te, C = fit_eval(flat(topo_block(Xq_tr, ens)), ytr,
                        flat(topo_block(Xq_v, ens)), yv,
                        flat(topo_block(Xq_te, ens)), yq_te)
    q4k.append(("+".join(x[:3] for x in ens), v, te)); print(f"  4k {q4k[-1][0]:<20s} val={v:.4f} test={te:.4f}", flush=True)
b4 = max(q4k, key=lambda r: r[1])

# free quantum mem
del Xq_all, Xq_tr, Xq_v, Xq_te

print("\nloading tissue patches for classical maps...", flush=True)
cfg = {"dataset": {"name": "tissue_mnist", "data_root": "./data", "batch_size": 2000,
                   "num_workers": 0, "download": False, "color_space": "GRAYSCALE"}}
ctr, _, cte, _ = get_dataloaders(cfg)
# Use the classical loader's OWN labels. get_dataloaders and the quantum cache
# loader both shuffle the train split independently (data.py:144,
# quantum_dataset_cache.py:268), so the classical features must be paired with
# the labels from the same loader -- pairing them with the quantum ytr/yv yields
# a sub-chance AUC. The classical subsample is therefore a different random draw
# from the same official splits than the quantum one; this is a fair comparison
# (same distribution, same train/val/test splits, identical estimator).
Pall, yc_all = stack(ctr, N_TR + N_VAL)   # raw images [N,1,28,28] + their labels
Pte, yc_te = stack(cte, N_TE)
Ptr, Pv = Pall[:N_TR], Pall[N_TR:N_TR + N_VAL]
yc_tr, yc_v = yc_all[:N_TR], yc_all[N_TR:N_TR + N_VAL]
g0 = 1.0 / 9

print("\n=== CLASSICAL (matched estimator, val-tuned) ===", flush=True)
# poly-2 (45)
v, te, C = fit_eval(flat(poly2_features(Ptr)), yc_tr, flat(poly2_features(Pv)), yc_v,
                    flat(poly2_features(Pte)), yc_te)
print(f"  poly2_45            test={te:.4f}", flush=True)
# RFF at swept gamma, both dims
for D in (45, 180):
    rows = []
    for gr in GAMMAS_REL:
        g = g0 * gr
        v, te, C = fit_eval(flat(rff_features(Ptr, D, gamma=g)), yc_tr,
                            flat(rff_features(Pv, D, gamma=g)), yc_v,
                            flat(rff_features(Pte, D, gamma=g)), yc_te)
        rows.append((gr, v, te))
    bg = max(rows, key=lambda r: r[1])
    print(f"  rff_{D} gamma-tuned   val={bg[1]:.4f} test={bg[2]:.4f} (gamma={bg[0]}x heuristic)", flush=True)

print("\n=== VERDICT (linear-probe, fair) ===", flush=True)
print(f"  quantum digital_zz 1k val-selected: {b1[0]} -> test {b1[2]:.4f}", flush=True)
print(f"  quantum digital_zz 4k val-selected: {b4[0]} -> test {b4[2]:.4f}", flush=True)
