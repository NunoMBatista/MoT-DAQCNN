"""Matched full-dataset TissueMNIST linear probe for all four encodings.

Builds the Tissue rows for the merged 3-dataset probing table on the SAME
footing across encodings, fixing the earlier mismatch where the digital probe
subsampled to 15k/8k while the analog probe used the full splits. Here every
encoding x topology is probed on the FULL train/test splits with one protocol:
per-topology StandardScaler + multinomial LogisticRegression(C=1.0), scored as
8-class macro one-vs-rest AUC (test split).

Memory: each ZZ cache (digital 49 GB, analog 23 GB on disk) is read ONCE via
np.load (train + test features only), then every topology's channels are sliced
in memory. This avoids re-reading the cache per topology. Run on a high-mem node
(--mem=200G); do NOT run locally.

Z vs ZZ: a topology's 45 channels are 9 single-qubit <Z_i> followed by 36
pairwise <Z_iZ_j>. Z uses the first 9 channels, ZZ uses all 45 (same convention
as derive_z_from_zz.py and probe_tissue_topologies.py).
"""
import os, sys, json, csv, gc, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.kernel_mapping import build_kernel_to_channels_map

QD = "data/quantum_datasets"
# (encoding label, ZZ cache filename) — Z is sliced from the same ZZ cache
CACHES = [
    ("digital", f"{QD}/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz.npz"),
    ("analog",  f"{QD}/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz_analog.npz"),
]
TOPOS = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
OUT = "outputs/paper_results/linear_probing/tissue_mnist_all_encodings_probe.csv"


def probe(Xtr, ytr, Xte, yte):
    """Standardise, fit multinomial LogReg, return macro-OVR test AUC."""
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1).fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))
    return roc_auc_score(yte, proba, multi_class="ovr", average="macro")


# results[topology] = {"digital_z": .., "digital_zz": .., "analog_z": .., "analog_zz": ..}
results = {t: {} for t in TOPOS}

for enc, path in CACHES:
    t0 = time.time()
    print(f"\n===== {enc} :: {os.path.basename(path)} =====", flush=True)
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["metadata"]))
    k2c = build_kernel_to_channels_map(meta["channel_kernel_map"])

    # Read full train/test features once (val is unused for the table).
    Xtr_all = data["train_features"]          # (N, C, H, W)
    ytr = np.asarray(data["train_labels"]).squeeze()
    Xte_all = data["test_features"]
    yte = np.asarray(data["test_labels"]).squeeze()
    print(f"  loaded train{Xtr_all.shape} test{Xte_all.shape} in {time.time()-t0:.0f}s", flush=True)

    print(f"  {'topology':12} {'Z(9)':>8} {'ZZ(45)':>8}")
    for topo in TOPOS:
        ch = k2c[topo]                        # 45 sorted channel indices
        z_ch, zz_ch = ch[:9], ch              # first 9 = <Z_i>; all 45 = Z+ZZ
        z = probe(Xtr_all[:, z_ch], ytr, Xte_all[:, z_ch], yte)
        zz = probe(Xtr_all[:, zz_ch], ytr, Xte_all[:, zz_ch], yte)
        results[topo][f"{enc}_z"] = round(z, 4)
        results[topo][f"{enc}_zz"] = round(zz, 4)
        print(f"  {topo:12} {z:8.4f} {zz:8.4f}", flush=True)

    del Xtr_all, Xte_all, data
    gc.collect()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cols = ["topology", "digital_z", "digital_zz", "analog_z", "analog_zz"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for topo in TOPOS:
        w.writerow({"topology": topo, **results[topo]})

# Per-encoding means for the table's Mean row
print("\n===== means (test macro-OVR AUC) =====")
for c in cols[1:]:
    vals = [results[t][c] for t in TOPOS if c in results[t]]
    print(f"  {c:12} {np.mean(vals):.4f}")
print(f"\nsaved {OUT}")
