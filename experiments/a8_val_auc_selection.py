"""A8: does selecting on validation accuracy change conclusions about AUC?

R4 #4 notes the Optuna objective is val_acc while every conclusion is about
test AUC, and asks either to optimize val AUC or to show the conclusions hold.
Re-running 200-trial end-to-end searches across 13 models and 3 datasets is
disproportionate, so this is the sensitivity check: BreastMNIST, the whole
capacity sweep, one search per cell, TWO selections from it.

Because all 30 trials run regardless, re-ranking them by a second recorded
metric is free. Each trial stores its validation AUC as an Optuna user
attribute; the cell is then validated twice over the 10 seeds, once for the
trial that maximizes val_acc and once for the trial that maximizes val AUC.
When both rules pick the same trial the second validation is skipped.

This is a WITHIN-RUN comparison by construction. Head initialisation was never
seed-controlled in the published sweep, so re-runs are not bit-identical to it
and must never be compared against the published values; both arms here come
from the same re-run and are directly comparable. Unlike the published sweep,
this driver seeds before building the head so the two arms differ only in which
hyperparameters were selected.

Writes to outputs/camera_ready/a8/ (never outputs/paper_results/).
"""
import argparse, csv, json, os, sys
from pathlib import Path

import optuna
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_capacity_sweep as hcs

optuna.logging.set_verbosity(optuna.logging.WARNING)

HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]


def run_cell(head_name, source_name, train, val, test, in_ch,
             cell_dir, n_trials, seeds, device):
    """One search, two selections. Returns a dict of per-rule results."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    builder, use_scheduler = hcs._head_factory(head_name)

    study = optuna.create_study(
        study_name=f"a8_{head_name}_{source_name}",
        direction="maximize",
        storage=f"sqlite:///{cell_dir / 'study.db'}",
        sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e-3, log=True)
        do = trial.suggest_float("dropout", 0.0, 0.7)
        torch.manual_seed(0)
        head = builder(hcs.NUM_CLASSES, do, hcs.ACTIVATION, in_ch)
        m = hcs.train_head(head, train, val, test, lr=lr, weight_decay=wd,
                           use_scheduler=use_scheduler, device=device, seed=0)
        # The second selection criterion, recorded but not optimized.
        trial.set_user_attr("val_auc", float(m["val_auc"]))
        return m["val_acc"]

    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    done = [t for t in study.trials if t.value is not None
            and t.user_attrs.get("val_auc") == t.user_attrs.get("val_auc")]  # drop NaN
    by_acc = max(done, key=lambda t: t.value)
    by_auc = max(done, key=lambda t: t.user_attrs["val_auc"])

    def validate(trial):
        aucs, accs = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            head = builder(hcs.NUM_CLASSES, trial.params["dropout"],
                           hcs.ACTIVATION, in_ch)
            m = hcs.train_head(head, train, val, test,
                               lr=trial.params["lr"], weight_decay=trial.params["wd"],
                               use_scheduler=use_scheduler, device=device, seed=seed)
            aucs.append(m["test_auc"])
            accs.append(m["test_acc"])
        n = len(aucs)
        mean = sum(aucs) / n
        return dict(test_auc_mean=mean, test_acc_mean=sum(accs) / n, test_auc_values=aucs)

    same = by_acc.number == by_auc.number
    res_acc = validate(by_acc)
    res_auc = res_acc if same else validate(by_auc)
    # Per-seed values are kept so the difference between the two rules can be
    # given a paired-over-seeds interval. Without them a margin moving from
    # +0.005 to -0.004 cannot be told apart from retraining noise.
    return dict(
        test_auc_values_sel_acc=res_acc["test_auc_values"],
        test_auc_values_sel_auc=res_auc["test_auc_values"],
        head=head_name, source=source_name, same_trial=same,
        trial_by_val_acc=by_acc.number, trial_by_val_auc=by_auc.number,
        val_acc_of_acc_pick=by_acc.value,
        val_auc_of_acc_pick=by_acc.user_attrs["val_auc"],
        val_acc_of_auc_pick=by_auc.value,
        val_auc_of_auc_pick=by_auc.user_attrs["val_auc"],
        auc_sel_acc=res_acc["test_auc_mean"], auc_sel_auc=res_auc["test_auc_mean"],
        acc_sel_acc=res_acc["test_acc_mean"], acc_sel_auc=res_auc["test_acc_mean"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breast_mnist")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default="outputs/camera_ready/a8")
    p.add_argument("--heads", nargs="+", default=HEADS)
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    globals()["hcs"].DATASET = args.dataset
    hcs.FEATURE_SOURCES = hcs.build_feature_sources(args.dataset)
    sources = args.sources or list(hcs.FEATURE_SOURCES)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sources:
        spec = hcs.FEATURE_SOURCES[source]
        train, val, test, in_ch = hcs.load_features(source, spec)
        train, val, test = hcs.standardize_features(train, val, test)
        for head in args.heads:
            cell = out / "cells" / f"{head}__{source}"
            cached = cell / "result.json"
            if cached.exists():
                rows.append(json.load(open(cached)))
                print(f"  SKIP {head:10s} {source:14s} (cached)")
                continue
            r = run_cell(head, source, train, val, test, in_ch, cell,
                         args.n_trials, args.seeds, args.device)
            json.dump(r, open(cached, "w"), indent=2)
            rows.append(r)
            flag = "same trial" if r["same_trial"] else "DIFFERENT trial"
            print(f"  {head:10s} {source:14s} {flag:15s} "
                  f"AUC(val_acc-sel)={r['auc_sel_acc']:.4f} "
                  f"AUC(val_auc-sel)={r['auc_sel_auc']:.4f} "
                  f"delta={r['auc_sel_auc'] - r['auc_sel_acc']:+.4f}")

    path = out / "a8_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
