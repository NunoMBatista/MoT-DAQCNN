"""A9: is the digital-analog null specific to the tested operating point?

R4 #6 and R6 ask whether the encoding null (digital and analog perform
comparably) holds only at T = 2.5, and R6 separately asks about the original
tau = 0.2 operating point of Simen et al. This probes the frozen features at
five evolution times under both encodings, at matched topology and matched
dimensionality, so the only thing varying is T.

The probe is a standardized LinearSVC. Its penalty is untuned, which would be a
problem for a quantum-versus-classical claim but is not one here: both arms of
every contrast are quantum features of identical dimension, so the probe is
affected equally. This is a representational measure, not an operating point.

T = 2.5 is read from the existing eight-topology caches by channel slicing; the
other four evolution times come from the caches generated for this sweep.
"""
import argparse, csv, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.linear_probing_topology_sweep import (
    collect, probe, make_cfg)
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset, load_cached_quantum_dataset)

TOPOLOGIES = {
    "breast_mnist":    ["kings", "cross", "star"],
    "pneumonia_mnist": ["kings", "horizontal", "star"],
}
TS = [0.2, 0.5, 1.0, 2.5, 4.0]
OUT = "outputs/camera_ready"


def one(dataset, topo, enc, T, corr, num_classes, C):
    cfg = make_cfg([topo], enc, corr, dataset_name=dataset)
    cfg["model"]["evolution_time"] = T
    cfg["model"]["num_classes"] = num_classes
    path = find_cached_quantum_dataset(cfg)
    if path is None:
        return None
    tr, _, te, _, _ = load_cached_quantum_dataset(
        path, batch_size=1000, num_workers=0, requested_kernels=[topo])
    Xtr, ytr = collect(tr)
    Xte, yte = collect(te)
    acc, auc = probe(Xtr, ytr, Xte, yte, C=C)
    return dict(dataset=dataset, topology=topo, encoding=enc, T=T,
                correlators=corr, dim=int(Xtr.shape[1]), acc=acc, auc=auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["breast_mnist"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(OUT, "tsweep_probe.csv"))
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        for topo in TOPOLOGIES[ds]:
            for corr in (True, False):
                for enc in ("digital", "analog"):
                    for T in TS:
                        r = one(ds, topo, enc, T, corr, args.num_classes, args.C)
                        if r is None:
                            print(f"  [missing] {ds} {topo} {enc} T={T} corr={corr}")
                            continue
                        rows.append(r)
                        print(f"  {ds:16s} {topo:11s} {enc:8s} T={T:<4} "
                              f"corr={str(corr):5s} dim={r['dim']:3d} AUC={r['auc']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} probes)")

    # The A9 readout: the digital-minus-analog gap as a function of T.
    print("\nDigital minus analog AUC, at matched topology and dimension:")
    idx = {(r["dataset"], r["topology"], r["correlators"], r["encoding"], r["T"]): r["auc"]
           for r in rows}
    for ds in args.datasets:
        for corr in (True, False):
            label = "Z+ZZ" if corr else "Z only"
            print(f"\n  {ds}, {label}")
            print("    topology     " + "".join(f"T={t:<7}" for t in TS))
            spreads = []
            for topo in TOPOLOGIES[ds]:
                cells = []
                for T in TS:
                    d = idx.get((ds, topo, corr, "digital", T))
                    a = idx.get((ds, topo, corr, "analog", T))
                    cells.append(f"{d - a:+.4f} " if d is not None and a is not None else "   --   ")
                    if d is not None and a is not None:
                        spreads.append(d - a)
                print(f"    {topo:12s} " + "".join(f"{c:<9s}" for c in cells))
            if spreads:
                print(f"    -> largest |digital - analog| across all T and topologies: "
                      f"{max(abs(s) for s in spreads):.4f}")


if __name__ == "__main__":
    main()
