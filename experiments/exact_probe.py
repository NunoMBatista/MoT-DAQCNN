"""R6 (camera-ready): does the linear probe change when the Trotter error is removed?

For every (dataset, topology, encoding, T) that has an exact cache from
exact_feature_cache.py, run the A9 probe (standardized LinearSVC, C = 1) on the
production delta_t = 0.05 features and on the exact-evolution features, Z+ZZ and
Z-only, and report both AUCs and their difference. Both arms go through the same
loader, the same channel slice and the same probe call, so the only difference is
the time integration. The LinearSVC is unseeded; its run-to-run noise is ~1e-3
(A9 control), so differences below ~0.005 are noise.

Control: the Trotter-arm AUCs must agree with outputs/camera_ready/tsweep_probe.csv
(the A9 source of record) to that noise level; the script prints the max gap.

Usage:
  python experiments/exact_probe.py --datasets breast_mnist pneumonia_mnist
"""
import argparse, csv, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.linear_probing_topology_sweep import collect, probe, make_cfg   # noqa: E402
from src.utils.quantum_dataset_cache import (                                    # noqa: E402
    find_cached_quantum_dataset, load_cached_quantum_dataset)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "camera_ready")
EXACT_DIR = os.path.join(OUT_DIR, "exact_caches")


def features(path, topo, z_only):
    """Train/test matrices for one topology from a ZZ cache; Z-only by channel slice."""
    tr, _, te, _, meta = load_cached_quantum_dataset(path, batch_size=1000, num_workers=0,
                                                     requested_kernels=[topo])
    Xtr, ytr = collect(tr)
    Xte, yte = collect(te)
    if z_only:
        # collect() flattens (N, C, 9, 9) -> (N, C*81) channel-major; keep the Z channels
        z_ch = [e["channel"] for e in meta["channel_kernel_map"] if e["measurement"] == "Z"]
        cols = np.concatenate([np.arange(c * 81, (c + 1) * 81) for c in z_ch])
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    return Xtr, ytr, Xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["breast_mnist", "pneumonia_mnist"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "exact_probe.csv"))
    args = ap.parse_args()

    # A9 source of record, for the control on the Trotter arm
    a9 = {}
    for r in csv.DictReader(open(os.path.join(OUT_DIR, "tsweep_probe.csv"))):
        a9[(r["dataset"], r["topology"], r["encoding"], float(r["T"]), r["correlators"] == "True")] = float(r["auc"])

    rows = []
    for exact_path in sorted(glob.glob(os.path.join(EXACT_DIR, "*__exact.npz"))):
        meta = json.load(open(exact_path[:-4] + ".json"))
        ds, enc, T = meta["dataset_name"], meta["encoding_mode"], float(meta["evolution_time"])
        if ds not in args.datasets:
            continue
        for topo in meta["kernel_topology_names"]:
            cfg = make_cfg([topo], enc, True, dataset_name=ds)
            cfg["model"]["evolution_time"] = T
            trotter_path = find_cached_quantum_dataset(cfg)
            assert trotter_path is not None, (ds, topo, enc, T)
            for corr in (True, False):
                auc = {}
                for arm, path in (("trotter", trotter_path), ("exact", exact_path)):
                    Xtr, ytr, Xte, yte = features(path, topo, z_only=not corr)
                    _, auc[arm] = probe(Xtr, ytr, Xte, yte, C=args.C)
                ref = a9.get((ds, topo, enc, T, corr), float("nan"))
                rows.append(dict(dataset=ds, topology=topo, encoding=enc, T=T, correlators=corr,
                                 dim=int(Xtr.shape[1]), auc_trotter=auc["trotter"], auc_exact=auc["exact"],
                                 exact_minus_trotter=auc["exact"] - auc["trotter"], auc_a9=ref,
                                 trotter_minus_a9=auc["trotter"] - ref))
                print(f"  {ds:16s} {topo:11s} {enc:8s} T={T:<4} corr={str(corr):5s} "
                      f"trotter={auc['trotter']:.4f} exact={auc['exact']:.4f} "
                      f"diff={auc['exact']-auc['trotter']:+.4f}  (A9 {ref:.4f})", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    d = np.array([r["exact_minus_trotter"] for r in rows])
    g = np.array([r["trotter_minus_a9"] for r in rows])
    print(f"\nwrote {args.out} ({len(rows)} probes)")
    print(f"control: max |trotter - A9| = {np.nanmax(np.abs(g)):.4f} (SVM noise ~0.001)")
    print(f"exact minus trotter: min {d.min():+.4f}, max {d.max():+.4f}, mean {d.mean():+.4f}, max |.| {np.max(np.abs(d)):.4f}")


if __name__ == "__main__":
    main()
