"""Cross-camp margins with the representative source selected on VALIDATION.

Reviewer 4 (#3) objects that the quantum-vs-classical margins in the capacity
figure pick the best source in each camp by *test* AUC, which is selection on
the test set. This script re-derives every margin by selecting the camp
representative on validation accuracy (the quantity the sweep actually
optimised) and then reporting that fixed choice's test AUC.

Selection rule, fixed in advance:
  for each (dataset, head, camp): representative = argmax over sources of the
  mean validation accuracy across validation seeds 1..9; ties broken by the
  source order listed in CAMPS. Seed 0 is excluded because it reproduces the
  Optuna best trial exactly in 90 of 195 cells, so its validation accuracy is
  the maximised value there rather than a fresh estimate. The reported number
  is the selected source's mean test AUC over all 10 seeds, matching the
  published table.

Caveat to state in the paper: the sweep stored val accuracy, not val AUC, so
selection and reporting use different metrics (see tracker item A8).

Everything is read from the per-seed JSONs, which are the primitive the summary
CSVs were aggregated from. No retraining.

Run from the repo root:
    python experiments/val_selected_margins.py
Writes outputs/camera_ready/val_selected_margins.csv (gitignored).
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]

# The two camps compared in Fig. 3. `raw` is the grey anchor and belongs to
# neither camp, exactly as in plot_capacity_grid.py.
QUANTUM = ["digital_z_1k", "digital_zz_1k", "digital_zz_4k",
           "analog_z_1k", "analog_zz_1k", "analog_zz_4k"]
CLASSICAL = ["random_9", "random_45", "random_180",
             "poly2_45", "rff_45", "rff_180"]
CAMPS = [("Q", QUANTUM), ("C", CLASSICAL)]

# Tissue ran on MCC; its per-seed files came back in the retrieval dump.
DATASETS = [
    ("BreastMNIST", "outputs/paper_results/capacity_sweep_std/breast_mnist"),
    ("PneumoniaMNIST", "outputs/paper_results/capacity_sweep_std/pneumonia_mnist"),
    ("TissueMNIST", "outputs/mcc_retrieved/capacity_sweep_std/tissue_mnist"),
]


def load_cells(sweep_dir):
    """(head, source) -> per-seed validation and test aggregates for one sweep."""
    cells = {}
    for cell_dir in sorted(glob.glob(os.path.join(ROOT, sweep_dir, "cells", "*"))):
        head, _, source = os.path.basename(cell_dir).partition("__")
        seed_files = sorted(glob.glob(os.path.join(cell_dir, "validation", "seed_*.json")))
        if not seed_files:
            continue
        val_accs, test_aucs = {}, []
        for path in seed_files:
            rec = json.load(open(path))
            val_accs[rec["seed"]] = rec["val_acc"]
            test_aucs.append(rec["test_auc"])
        # Selection metric drops the Optuna search seed; see module docstring.
        held_out = [v for seed, v in val_accs.items() if seed != 0]
        cells[(head, source)] = {
            "n_seeds": len(seed_files),
            "val_acc_mean": sum(held_out) / len(held_out),
            "val_acc_mean_all": sum(val_accs.values()) / len(val_accs),
            "test_auc_mean": sum(test_aucs) / len(test_aucs),
            "test_aucs": test_aucs,
        }
    return cells


def pick(cells, head, sources, key):
    """Best source in `sources` at `head`, ranked by `key`. None if no cell ran."""
    best_src, best_val = None, None
    for src in sources:
        cell = cells.get((head, src))
        if cell is None:
            continue
        if best_val is None or cell[key] > best_val:
            best_src, best_val = src, cell[key]
    return best_src


def tag_margin(margin):
    """Format a signed margin the way the figure labels it: Q.0136 / C.0061."""
    return ("Q" if margin >= 0 else "C") + f"{abs(margin):.4f}"[1:]


def main():
    rows = []
    disagreements = []
    for dname, sweep_dir in DATASETS:
        cells = load_cells(sweep_dir)
        print(f"\n=== {dname} ({len(cells)} cells) ===")
        print(f"{'head':<10} {'val-selected Q':<16} {'C':<12} {'margin':>8}   "
              f"{'test-selected (old)':<30} {'old':>8}")
        for head in HEADS:
            row = {"dataset": dname, "head": head}
            incomplete = False
            for tag, sources in CAMPS:
                # the fix: select on validation, report test
                val_src = pick(cells, head, sources, "val_acc_mean")
                # the bug being replaced: select on test, report test
                test_src = pick(cells, head, sources, "test_auc_mean")
                if val_src is None:
                    incomplete = True
                    break
                if pick(cells, head, sources, "val_acc_mean_all") != val_src:
                    disagreements.append(f"{dname}/{head}/{tag}")
                row[f"{tag}_val_source"] = val_src
                row[f"{tag}_val_test_auc"] = cells[(head, val_src)]["test_auc_mean"]
                row[f"{tag}_test_source"] = test_src
                row[f"{tag}_test_test_auc"] = cells[(head, test_src)]["test_auc_mean"]
            if incomplete:
                continue
            # signed margin, positive = quantum ahead
            new = row["Q_val_test_auc"] - row["C_val_test_auc"]
            old = row["Q_test_test_auc"] - row["C_test_test_auc"]
            row["val_selected_margin"] = new
            row["test_selected_margin"] = old
            row["margin_shift"] = new - old
            rows.append(row)
            print(f"{head:<10} {row['Q_val_source']:<16} {row['C_val_source']:<12} "
                  f"{tag_margin(new):>8}   "
                  f"{row['Q_test_source'] + ' / ' + row['C_test_source']:<30} "
                  f"{tag_margin(old):>8}")

    out_dir = os.path.join(ROOT, "outputs", "camera_ready")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "val_selected_margins.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")
    print(f"selection robustness: {len(disagreements)}/{2 * len(rows)} picks change "
          f"if the search seed is kept in the selection metric"
          + (f" -> {disagreements}" if disagreements else ""))


if __name__ == "__main__":
    main()
