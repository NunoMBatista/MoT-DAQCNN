"""A2 extension: pick the end-to-end camp representative on validation, not test.

SUPERSEDED 2026-09-20 by val_selected_e2e_tenseed.py, which selects on ten-seed
validation metrics recomputed from the saved checkpoints and takes the four
Breast/Pneumonia Random rows from the frozen-filter re-validation. This script
and its CSV (outputs/camera_ready/val_selected_e2e.csv) still use the rows that
were validated with trainable filters; kept for the record only. QUANTUM below
remains the single camp definition imported by the ten-seed script.

The paper's headline "best quantum vs best classical" margin currently takes the
maximum test AUC over the rows of each camp, which is a selection on test of
exactly the kind the Fig. 3 margins were fixed for (A2). This script reapplies
the same principle to Table III.

Selector: `search_value` from each run's validation/validation_summary.json,
which is the Optuna objective (validation accuracy) of the selected trial. No
per-seed validation metrics were stored for these runs, so this is a single-run
val_acc rather than the seed-averaged rule used for the capacity sweep.
It never touches test.

Reads the run mapping from paired_seed_intervals.py so there is one definition.
"""
import json, os, sys, csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_seed_intervals import E2E, PAPER_TABLE

OUT = "outputs/camera_ready"

# Camp membership. Everything with a quantum front-end is "quantum"; the
# random/trainable projections, the classical nonlinear maps, and raw pixels
# are "classical".
QUANTUM = {"Digital-Z 1k", "Digital-ZZ 1k", "Analog-ZZ 1k",
           "Digital-ZZ 4k", "Analog-ZZ 4k", "Analog-Z 1k"}


def load_run(path):
    """Return (val_acc, test_auc_mean, test_acc_mean, n_trials) for a run dir."""
    vs = os.path.join(path, "validation", "validation_summary.json")
    if not os.path.exists(vs):
        vs = os.path.join(path, "validation_summary.json")
    top = json.load(open(vs))[0]
    # Trial count: how many trials the search actually ran, needed because a
    # max-over-200 val_acc is more winner's-cursed than a max-over-25.
    n_trials = None
    ss = os.path.join(path, "search_summary.json")
    if os.path.exists(ss):
        j = json.load(open(ss))
        for k in ("n_trials", "num_trials", "completed_trials", "n_completed"):
            if k in j:
                n_trials = j[k]
                break
    trials_dir = os.path.join(path, "trials")
    if n_trials is None and os.path.isdir(trials_dir):
        n_trials = len([d for d in os.listdir(trials_dir) if d.startswith("trial_")])
    return top["search_value"], top["test_auc"]["mean"], top["test_acc"]["mean"], n_trials


def main():
    os.makedirs(OUT, exist_ok=True)
    datasets = sorted({d for d, _ in E2E}, key=lambda x: ("Breast" not in x, x))
    rows = []
    for ds in datasets:
        recs = []
        for (d, model), path in E2E.items():
            if d != ds:
                continue
            if not os.path.exists(path):
                print(f"  MISSING {d} / {model}: {path}")
                continue
            val, auc, acc, nt = load_run(path)
            recs.append(dict(model=model, camp="Q" if model in QUANTUM else "C",
                             val_acc=val, test_auc=auc, test_acc=acc, n_trials=nt))
        print(f"\n=== {ds} ({len(recs)} rows) ===")
        print(f"{'model':16s} {'camp':4s} {'val_acc':>8s} {'test_auc':>9s} {'test_acc':>9s} {'trials':>6s}")
        for r in sorted(recs, key=lambda r: (r["camp"], -r["val_acc"])):
            print(f"{r['model']:16s} {r['camp']:4s} {r['val_acc']:8.4f} "
                  f"{r['test_auc']:9.4f} {r['test_acc']:9.4f} {str(r['n_trials']):>6s}")
        for camp in ("Q", "C"):
            camp_recs = [r for r in recs if r["camp"] == camp]
            if not camp_recs:
                continue
            # Validation accuracy is quantised by the validation-set size (1/78
            # on BreastMNIST, 1/524 on PneumoniaMNIST), so the top val_acc is
            # routinely a multi-way tie. Breaking that tie by test AUC would
            # reintroduce exactly the selection-on-test we are removing, and
            # breaking it by dict order is arbitrary, so the representative is
            # the MEAN test AUC over every row tied at the maximum val_acc.
            top_val = max(r["val_acc"] for r in camp_recs)
            tied = [r for r in camp_recs if r["val_acc"] == top_val]
            val_auc = sum(r["test_auc"] for r in tied) / len(tied)
            val_acc_t = sum(r["test_acc"] for r in tied) / len(tied)
            test_pick = max(camp_recs, key=lambda r: r["test_auc"])
            names = ", ".join(sorted(r["model"] for r in tied))
            print(f"  {camp}: val-selected = {names} (n={len(tied)} tied at "
                  f"val_acc {top_val:.4f}), mean test AUC {val_auc:.4f} "
                  f"[{min(r['test_auc'] for r in tied):.4f}, "
                  f"{max(r['test_auc'] for r in tied):.4f}] | "
                  f"test-selected = {test_pick['model']} "
                  f"(test AUC {test_pick['test_auc']:.4f})")
            rows.append(dict(dataset=ds, camp=camp,
                             val_selected=names,
                             n_tied=len(tied),
                             val_selected_auc=val_auc,
                             val_selected_auc_min=min(r["test_auc"] for r in tied),
                             val_selected_auc_max=max(r["test_auc"] for r in tied),
                             val_selected_acc=val_acc_t,
                             test_selected=test_pick["model"],
                             test_selected_auc=test_pick["test_auc"],
                             test_selected_acc=test_pick["test_acc"],
                             agree=test_pick["model"] in {r["model"] for r in tied}))
        q = [r for r in rows if r["dataset"] == ds and r["camp"] == "Q"]
        c = [r for r in rows if r["dataset"] == ds and r["camp"] == "C"]
        if q and c:
            print(f"  MARGIN (Q-C) AUC: val-selected {q[0]['val_selected_auc'] - c[0]['val_selected_auc']:+.4f}"
                  f"   test-selected {q[0]['test_selected_auc'] - c[0]['test_selected_auc']:+.4f}")

    path = os.path.join(OUT, "val_selected_e2e.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
