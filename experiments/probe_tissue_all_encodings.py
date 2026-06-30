"""Matched full-dataset TissueMNIST linear probe for one encoding cache.

Builds the Tissue rows for the merged 3-dataset probing table on the SAME
footing across encodings, fixing the earlier mismatch where the digital probe
subsampled to 15k/8k while the analog probe used the full splits. Every
encoding x topology is probed on the FULL train/test splits with one protocol:
per-topology StandardScaler + multinomial LogisticRegression(C=1.0), scored as
8-class macro one-vs-rest AUC (test split).

One cache (digital or analog) per invocation via --cache, so the two run as
independent parallel jobs. Results are written INCREMENTALLY after each
topology, so a walltime kill never loses completed rows. The big ZZ cache
(digital 49 GB, analog 23 GB on disk) is read once via np.load (train + test
features only); needs a high-mem node (--mem=200G). Do NOT run locally.

Z vs ZZ: a topology's 45 channels are 9 single-qubit <Z_i> followed by 36
pairwise <Z_iZ_j>. Z uses the first 9 channels, ZZ uses all 45.
"""
import os, sys, json, csv, gc, time, argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.kernel_mapping import build_kernel_to_channels_map

QD = "data/quantum_datasets"
CACHES = {
    "digital": f"{QD}/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz.npz",
    "analog":  f"{QD}/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz_analog.npz",
}
TOPOS = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
OUTDIR = "outputs/paper_results/linear_probing"


def probe(Xtr, ytr, Xte, yte):
    """Standardise, fit multinomial LogReg, return macro-OVR test AUC."""
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1).fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))
    return roc_auc_score(yte, proba, multi_class="ovr", average="macro")


def write_csv(rows, enc, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topology", f"{enc}_z", f"{enc}_zz"])
        w.writeheader()
        w.writerows(rows)


def run(enc):
    path = CACHES[enc]
    out_csv = f"{OUTDIR}/tissue_mnist_probe_{enc}.csv"
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    print(f"===== {enc} :: {os.path.basename(path)} =====", flush=True)
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["metadata"]))
    k2c = build_kernel_to_channels_map(meta["channel_kernel_map"])
    Xtr_all = data["train_features"]
    ytr = np.asarray(data["train_labels"]).squeeze()
    Xte_all = data["test_features"]
    yte = np.asarray(data["test_labels"]).squeeze()
    print(f"  loaded train{Xtr_all.shape} test{Xte_all.shape} in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for topo in TOPOS:
        ch = k2c[topo]                        # 45 sorted channel indices
        z_ch, zz_ch = ch[:9], ch              # first 9 = <Z_i>; all 45 = Z+ZZ
        tt = time.time()
        z = probe(Xtr_all[:, z_ch], ytr, Xte_all[:, z_ch], yte)
        zz = probe(Xtr_all[:, zz_ch], ytr, Xte_all[:, zz_ch], yte)
        rows.append({"topology": topo, f"{enc}_z": round(z, 4), f"{enc}_zz": round(zz, 4)})
        write_csv(rows, enc, out_csv)         # incremental: survive a walltime kill
        print(f"  {topo:12} z={z:.4f} zz={zz:.4f}  "
              f"[{len(rows)}/{len(TOPOS)} done, {time.time()-tt:.0f}s]", flush=True)

    del Xtr_all, Xte_all, data
    gc.collect()
    print(f"\n{enc} means: "
          f"z={np.mean([r[f'{enc}_z'] for r in rows]):.4f} "
          f"zz={np.mean([r[f'{enc}_zz'] for r in rows]):.4f}")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", choices=list(CACHES), required=True)
    run(ap.parse_args().cache)
