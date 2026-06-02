"""Seed-averaged linear-SVM probe of the classical baselines and nonlinear
controls (random projection, poly-2, RFF), on the same patches and probe
protocol as the quantum topology sweep.

Random and RFF feature maps are random by construction, so we report them as a
mean over N projection seeds (the honest summary of "what a random map of this
dimension achieves"). Poly-2 and raw are deterministic. Quantum is deterministic
too and is not re-seeded; its values come from the topology sweep CSV.

Cross-check: random_45 seed 42 reproduces the paper's hardcoded 0.766; the seed
MEAN is higher (~0.78 on Breast), which is the value the paper should use.

Usage: python experiments/probe_classical_nonlinear.py --dataset breast_mnist --seeds 10
"""
import argparse, os, sys, csv
import numpy as np, torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.data import get_dataloaders
from src.utils.classical_nonlinear_features import poly2_features, rff_features


def stack(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.float()); ys.append(y.long().squeeze())
    return torch.cat(xs), torch.cat(ys)


def random_conv(images, n_filters, seed, k=3, s=3):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n_filters, 1, k, k)).astype(np.float32)
    w /= (np.linalg.norm(w.reshape(n_filters, -1), axis=1).reshape(-1, 1, 1, 1) + 1e-8)
    return F.conv2d(images, torch.from_numpy(w), stride=s)


def probe(Xtr, ytr, Xte, yte, C=1.0):
    Xtr = Xtr.reshape(Xtr.shape[0], -1).numpy()
    Xte = Xte.reshape(Xte.shape[0], -1).numpy()
    sc = StandardScaler().fit(Xtr)
    svm = LinearSVC(C=C, max_iter=50_000, dual="auto").fit(sc.transform(Xtr), ytr.numpy())
    s = svm.decision_function(sc.transform(Xte))
    return accuracy_score(yte.numpy(), (s > 0).astype(int)), roc_auc_score(yte.numpy(), s)


def seeded_mean(fn_tr, fn_te, ytr, yte, seeds):
    """Probe AUC averaged over projection seeds for a random feature map."""
    accs, aucs = [], []
    for sd in seeds:
        a, u = probe(fn_tr(sd), ytr, fn_te(sd), yte)
        accs.append(a); aucs.append(u)
    return np.mean(accs), np.mean(aucs), np.std(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_mnist")
    ap.add_argument("--seeds", type=int, default=10, help="projection seeds for random/RFF")
    ap.add_argument("--seeds-180", type=int, default=5, help="fewer seeds for slow 180-dim probes")
    args = ap.parse_args()
    cfg = {"dataset": {"name": args.dataset, "data_root": "./data", "batch_size": 1000,
                       "num_workers": 0, "download": True, "color_space": "GRAYSCALE"}}
    tr, _, te, _ = get_dataloaders(cfg)
    Xtr, ytr = stack(tr); Xte, yte = stack(te)
    s_lo = list(range(args.seeds)); s_hi = list(range(args.seeds_180))
    print(f"[{args.dataset}] train={Xtr.shape[0]} test={Xte.shape[0]}  seeds45={len(s_lo)} seeds180={len(s_hi)}")
    rows = []

    def add(name, dim, acc, auc, std):
        print(f"  {name:12} dim={dim:6d}  acc={acc:.4f}  auc={auc:.4f} +-{std:.4f}")
        rows.append({"source": name, "dim": dim, "acc": round(acc, 4),
                     "auc": round(auc, 4), "auc_std": round(std, 4)})

    # deterministic
    a, u = probe(Xtr, ytr, Xte, yte); add("raw", int(np.prod(Xtr.shape[1:])), a, u, 0.0)
    a, u = probe(poly2_features(Xtr), ytr, poly2_features(Xte), yte)
    add("poly2_45", poly2_features(Xtr[:1]).shape[1] * 81, a, u, 0.0)
    # seeded
    add("random_45", 3645, *seeded_mean(lambda s: random_conv(Xtr, 45, s),
        lambda s: random_conv(Xte, 45, s), ytr, yte, s_lo))
    add("rff_45", 3645, *seeded_mean(lambda s: rff_features(Xtr, 45, seed=s),
        lambda s: rff_features(Xte, 45, seed=s), ytr, yte, s_lo))
    add("random_180", 14580, *seeded_mean(lambda s: random_conv(Xtr, 180, s),
        lambda s: random_conv(Xte, 180, s), ytr, yte, s_hi))
    add("rff_180", 14580, *seeded_mean(lambda s: rff_features(Xtr, 180, seed=s),
        lambda s: rff_features(Xte, 180, seed=s), ytr, yte, s_hi))

    out = f"outputs/paper_results/linear_probing/{args.dataset}_classical_nonlinear.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "dim", "acc", "auc", "auc_std"])
        w.writeheader(); w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
