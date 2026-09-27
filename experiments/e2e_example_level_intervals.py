"""A7, example level: bootstrap intervals over TEST EXAMPLES for the end-to-end
contrasts on the two binary datasets, from per-example test probabilities
recovered from the saved checkpoints (experiments/e2e_val_reinference.py) and
from the frozen-filter re-validation of the Random rows
(experiments/e2e_random_fixed_revalidate.py). TissueMNIST has no saved
per-example predictions and is not covered.

Three 95% intervals per contrast (A minus B, test AUC):
  seeds     paired t over the ten validation seeds with the test split fixed,
            i.e. A7's published interval: retraining variability only.
  examples  percentile bootstrap over test examples (B resamples) with the ten
            trained models fixed: test-sampling variability only. Every
            resample re-scores each seed of both arms on the same examples, so
            the pairing over examples is exact.
  both      seeds (paired A/B, with replacement) and examples resampled
            together.

Controls, run before any contrast is read:
  * a model against itself must give exactly [0, 0] for all three intervals;
  * with the resample replaced by the full index set, the 'examples'
    statistic must equal the seed-level mean difference to float precision.

    python experiments/e2e_example_level_intervals.py
"""
import csv
import math
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from paired_seed_intervals import E2E_CONTRASTS          # noqa: E402  one definition

REINF = os.path.join(ROOT, "outputs", "camera_ready", "e2e_val_reinference")
FIXED = os.path.join(ROOT, "outputs", "camera_ready", "e2e_random_fixed")
OUT = os.path.join(ROOT, "outputs", "camera_ready", "e2e_example_level_intervals.csv")
FIXED_ROWS = {(d, m) for d in ("BreastMNIST", "PneumoniaMNIST") for m in ("Random x45", "Random x180")}
N_SEEDS = 10
B = 2000

# The ten-seed validation-selected pairs (val_selected_e2e_tenseed.py, after
# the Random-row repair), added to the pre-specified E1-E4 contrasts.
SELECTOR_CONTRASTS = [
    ("S-breast-valauc", "BreastMNIST", "Digital-ZZ 4k", "RFF x45", "ten-seed val-AUC-selected representatives"),
    ("S-breast-valacc", "BreastMNIST", "Digital-ZZ 1k", "Random x45", "ten-seed val-acc-selected representatives"),
    ("S-pneumonia-valauc", "PneumoniaMNIST", "Analog-ZZ 4k", "Raw pixels", "ten-seed val-AUC-selected representatives"),
    ("S-pneumonia-valacc", "PneumoniaMNIST", "Analog-ZZ 4k", "Trainable x180", "ten-seed val-acc-selected representatives"),
]


def load_scores(dataset, model):
    """Positive-class test scores, shape (N_SEEDS, n_test), and the labels."""
    scores, labels = [], None
    if (dataset, model) in FIXED_ROWS:
        for k in range(N_SEEDS):
            z = np.load(os.path.join(FIXED, f"{dataset}__{model.replace(' ', '_')}", f"seed_{k}", "test_preds.npz"))
            scores.append(np.asarray(z["probs"])[:, 1])
            lab = np.asarray(z["labels"]).reshape(-1)
            assert labels is None or np.array_equal(labels, lab)
            labels = lab
    else:
        z = np.load(os.path.join(REINF, f"{dataset}__{model.replace(' ', '_').replace('/', '_')}_preds.npz"))
        for k in range(N_SEEDS):
            scores.append(np.asarray(z[f"seed_{k}_test_probs"])[:, 1])
            lab = np.asarray(z[f"seed_{k}_test_labels"]).reshape(-1)
            assert labels is None or np.array_equal(labels, lab)
            labels = lab
    return np.stack(scores).astype(np.float64), labels.astype(np.int64)


def auc_matrix(scores, labels, idx):
    """AUC of every seed on the example subset idx, shape (N_SEEDS,)."""
    y = labels[idx]
    return np.array([roc_auc_score(y, s[idx]) for s in scores])


def seed_t_interval(d):
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se


def bootstrap(sa, sb, labels, rng, resample_seeds):
    """Percentile bootstrap of the seed-mean AUC difference A minus B."""
    n = len(labels)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        d = auc_matrix(sa, labels, idx) - auc_matrix(sb, labels, idx)
        if resample_seeds:
            d = d[rng.integers(0, N_SEEDS, N_SEEDS)]
        diffs[b] = d.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_contrast(cid, dataset, a, b, sentence, rng):
    sa, la = load_scores(dataset, a)
    sb, lb = load_scores(dataset, b)
    assert np.array_equal(la, lb), (cid, "labels differ between arms")
    full = np.arange(len(la))
    d_full = auc_matrix(sa, la, full) - auc_matrix(sb, la, full)
    m, lo_s, hi_s = seed_t_interval(d_full)
    lo_e, hi_e = bootstrap(sa, sb, la, rng, resample_seeds=False)
    lo_b, hi_b = bootstrap(sa, sb, la, rng, resample_seeds=True)
    return dict(contrast=cid, dataset=dataset, arm_a=a, arm_b=b, n_test=len(la), n_pos=int(la.sum()),
                mean_diff=m, seeds_lo=lo_s, seeds_hi=hi_s, examples_lo=lo_e, examples_hi=hi_e,
                both_lo=lo_b, both_hi=hi_b,
                width_ratio_examples_over_seeds=(hi_e - lo_e) / (hi_s - lo_s),
                nonzero_seeds=not (lo_s <= 0 <= hi_s), nonzero_examples=not (lo_e <= 0 <= hi_e),
                nonzero_both=not (lo_b <= 0 <= hi_b), sentence=sentence)


def main():
    rng = np.random.default_rng(0)

    # ---- controls ----
    sa, la = load_scores("BreastMNIST", "Digital-ZZ 4k")
    full = np.arange(len(la))
    self_diff = auc_matrix(sa, la, full) - auc_matrix(sa, la, full)
    assert np.all(self_diff == 0.0)
    lo_e, hi_e = bootstrap(sa, sa, la, np.random.default_rng(1), resample_seeds=False)
    lo_b, hi_b = bootstrap(sa, sa, la, np.random.default_rng(1), resample_seeds=True)
    assert lo_e == hi_e == 0.0 and lo_b == hi_b == 0.0, (lo_e, hi_e, lo_b, hi_b)
    sb, lb = load_scores("BreastMNIST", "Random x45")
    d_full = (auc_matrix(sa, la, full) - auc_matrix(sb, lb, full)).mean()
    d_stat = (auc_matrix(sa, la, full) - auc_matrix(sb, lb, full)).mean()   # the bootstrap statistic on the full index set
    assert abs(d_full - d_stat) < 1e-12
    print(f"controls OK: self-contrast intervals are [0,0]; full-index statistic = seed-mean diff = {d_full:+.4f}")

    rows = []
    contrasts = [c for c in E2E_CONTRASTS if c[1] != "TissueMNIST"] + SELECTOR_CONTRASTS
    print(f"\n{'contrast':20s} {'diff':>8s} {'seeds 95%':>20s} {'examples 95%':>20s} {'both 95%':>20s} {'w_ex/w_seed':>11s}")
    for cid, dataset, a, b, sentence in contrasts:
        r = run_contrast(cid, dataset, a, b, sentence, rng)
        rows.append(r)
        flag = lambda k: "*" if r[k] else " "
        print(f"{cid:20s} {r['mean_diff']:+8.4f} [{r['seeds_lo']:+.4f},{r['seeds_hi']:+.4f}]{flag('nonzero_seeds')} "
              f"[{r['examples_lo']:+.4f},{r['examples_hi']:+.4f}]{flag('nonzero_examples')} "
              f"[{r['both_lo']:+.4f},{r['both_hi']:+.4f}]{flag('nonzero_both')} "
              f"{r['width_ratio_examples_over_seeds']:11.2f}", flush=True)
    print("(* = interval excludes zero)")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
