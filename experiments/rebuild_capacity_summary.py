"""Rebuild a capacity-sweep summary.csv from the per-cell summary.json files.

Parallel workers each overwrite summary.csv with only their own sources, so the
combined file must be reassembled from cells/*/summary.json after a fan-out run.
The cell JSONs do NOT store param_count (identical across MedMNIST datasets since
all images are 28x28), so we borrow the (head, source) -> param_count map from any
existing summary.csv (this dataset's stale partial, or another dataset's).

Usage: python experiments/rebuild_capacity_summary.py --dir outputs/.../<dataset>
"""
import os, json, csv, glob, argparse

COLUMNS = ["head", "source", "in_channels", "param_count", "n_trials",
           "best_val_acc", "best_lr", "best_wd", "best_dropout",
           "test_acc_mean", "test_acc_std", "test_auc_mean", "test_auc_std",
           "test_f1_mean", "test_f1_std"]


def borrow_param_counts(paths):
    """Build (head, in_channels) -> param_count from existing summary.csv files.
    param_count is fixed by the head architecture and input channel count, so
    sources of equal dim (e.g. random_45, poly2_45, rff_45) share it."""
    pc = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for row in csv.DictReader(open(p)):
            if row.get("param_count") and row.get("in_channels"):
                pc[(row["head"], row["in_channels"])] = row["param_count"]
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dataset dir holding cells/ and summary.csv")
    ap.add_argument("--borrow", nargs="*", default=[],
                    help="extra summary.csv paths to borrow param_count from")
    args = ap.parse_args()

    pc = borrow_param_counts([os.path.join(args.dir, "summary.csv")] + args.borrow)

    rows = []
    for jp in sorted(glob.glob(os.path.join(args.dir, "cells", "*", "summary.json"))):
        d = json.load(open(jp))
        bp = d.get("best_params", {})
        rows.append({
            "head": d["head"], "source": d["source"], "in_channels": d["in_channels"],
            "param_count": pc.get((d["head"], str(d["in_channels"])), 0),
            "n_trials": d.get("n_trials_completed", ""),
            "best_val_acc": d.get("best_value_val_acc", ""),
            "best_lr": bp.get("lr", ""), "best_wd": bp.get("wd", ""),
            "best_dropout": bp.get("dropout", ""),
            "test_acc_mean": d["test_acc_mean"], "test_acc_std": d["test_acc_std"],
            "test_auc_mean": d["test_auc_mean"], "test_auc_std": d["test_auc_std"],
            "test_f1_mean": d["test_f1_mean"], "test_f1_std": d["test_f1_std"],
        })

    out = os.path.join(args.dir, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    missing = sum(1 for r in rows if r["param_count"] == 0)
    print(f"wrote {len(rows)} cells to {out}  ({missing} missing param_count)")


if __name__ == "__main__":
    main()
