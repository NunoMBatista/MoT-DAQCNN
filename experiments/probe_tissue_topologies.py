"""TissueMNIST linear-probe topology sweep (multiclass), to pick the best
topology for each capacity-sweep quantum source — same role the linear probe
plays for Breast/Pneumonia, plus it is the section-5.1 representational result
for Tissue.

Loads the 8-topology digital cache once, subsamples the train/test split (40k/20k;
topology RANKING is stable under subsampling and 40k gives stable LinearSVC AUC),
and probes each topology's Z-only and Z+ZZ features with a multiclass LinearSVC.
Channel layout is sequential: topology k = channels [k*45:(k+1)*45], Z = first 9.

Usage: python experiments/probe_tissue_topologies.py
"""
import os, sys, json, csv
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.quantum_dataset_cache import load_cached_quantum_dataset

CACHE = "data/quantum_datasets/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz.npz"
TOPOS = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
N_TR, N_TE = 15000, 8000
MAX_STACK_TR, MAX_STACK_TE = 30000, 15000  # cap the materialized cache (covers 8 classes)
SEED = 0


def stack(loader, max_n=None):
    """Materialize the loader into numpy, stopping early once max_n samples are
    collected. Stacking the full 165k-sample Tissue cache (~25 GB torch.cat) is
    what made earlier probe runs exceed the walltime; a capped stack is enough
    because topology RANKING (all topologies see the same subset) is stable."""
    xs, ys, n = [], [], 0
    for x, y in loader:
        xs.append(x.float()); ys.append(y.long().squeeze()); n += len(x)
        if max_n and n >= max_n:
            break
    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


def subsample(X, y, n, rng):
    if len(y) <= n:
        return X, y
    idx = rng.choice(len(y), n, replace=False)
    return X[idx], y[idx]


def probe(Xtr, ytr, Xte, yte):
    Xtr = Xtr.reshape(Xtr.shape[0], -1); Xte = Xte.reshape(Xte.shape[0], -1)
    sc = StandardScaler().fit(Xtr)
    # LogisticRegression (lbfgs multinomial) converges in seconds on these
    # standardized low-dim blocks, unlike LinearSVC with max_iter=20000 which
    # timed out. predict_proba gives calibrated probabilities, so macro-OVR AUC
    # is computed directly without the label-binarize workaround.
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))
    return roc_auc_score(yte, proba, multi_class="ovr", average="macro")


def main():
    import time
    t0 = time.time()
    print("loading cache (all 8 topologies)...", flush=True)
    tr, _, te, _, meta = load_cached_quantum_dataset(CACHE, batch_size=4096, num_workers=0,
                                                     requested_kernels=TOPOS)
    Xtr, ytr = stack(tr, MAX_STACK_TR); Xte, yte = stack(te, MAX_STACK_TE)
    print(f"  stacked train{Xtr.shape} test{Xte.shape} in {time.time()-t0:.0f}s "
          f"classes={sorted(set(ytr.tolist()))}", flush=True)
    rng = np.random.default_rng(SEED)
    Xtr, ytr = subsample(Xtr, ytr, N_TR, rng)
    Xte, yte = subsample(Xte, yte, N_TE, rng)
    print(f"  subsampled: train{Xtr.shape} test{Xte.shape}", flush=True)

    rows = []
    print(f"\n{'topology':12} {'Z(9)':>8} {'ZZ(45)':>8}")
    for k, topo in enumerate(TOPOS):
        blk = slice(k * 45, (k + 1) * 45)
        z = probe(Xtr[:, k * 45:k * 45 + 9], ytr, Xte[:, k * 45:k * 45 + 9], yte)
        zz = probe(Xtr[:, blk], ytr, Xte[:, blk], yte)
        print(f"  {topo:12} {z:8.4f} {zz:8.4f}")
        rows.append({"topology": topo, "auc_z": round(z, 4), "auc_zz": round(zz, 4)})

    # 4-kernel sets: a few candidates (best-singles + standard set)
    best_zz = sorted(rows, key=lambda r: -r["auc_zz"])[:4]
    sets = {
        "best4_zz": [r["topology"] for r in best_zz],
        "g_c_h_kings": ["grid", "chain", "horizontal", "kings"],
    }
    print(f"\n4-kernel ZZ sets:")
    for label, ts in sets.items():
        idx = np.concatenate([np.arange(TOPOS.index(t) * 45, (TOPOS.index(t) + 1) * 45) for t in ts])
        zz4 = probe(Xtr[:, idx], ytr, Xte[:, idx], yte)
        print(f"  {label:14} {ts} -> {zz4:.4f}")
        rows.append({"topology": f"4k:{label}:{'+'.join(ts)}", "auc_z": "", "auc_zz": round(zz4, 4)})

    out = "outputs/paper_results/linear_probing/tissue_mnist_topology_probe.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topology", "auc_z", "auc_zz"]); w.writeheader(); w.writerows(rows)
    bz = max((r for r in rows if r["auc_z"] != ""), key=lambda r: r["auc_z"])
    bzz = max((r for r in rows if not str(r["topology"]).startswith("4k")), key=lambda r: r["auc_zz"])
    print(f"\n>>> BEST digital_z topology: {bz['topology']} ({bz['auc_z']})")
    print(f">>> BEST digital_zz topology: {bzz['topology']} ({bzz['auc_zz']})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
