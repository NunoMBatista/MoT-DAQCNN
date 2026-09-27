"""A7, example level, head-capacity cells: bootstrap intervals over TEST EXAMPLES
for the TissueMNIST linear-head contrasts (the paper's low-capacity quantum
exception), from per-example test scores written by
experiments/a12_projection_seeds.py in frozen-draw mode, i.e. the published
protocol with both arms in one harness:
  outputs/camera_ready/a12/tissue_mnist_frozen/linear__<source>/seed_<k>_preds.npz
    scores  (n_test, n_classes) softmax probabilities (binary: (n_test,) positive-class)
    labels  (n_test,)

Contrasts are the pre-specified A7 ones (paired_seed_intervals.py):
  C2-tissue-4k       digital_zz_4k vs rff_180   (the +0.014 headline)
  C2-tissue-1k-rff   digital_zz_1k vs rff_45
  C2-tissue-1k-poly  digital_zz_1k vs poly2_45

Three 95% intervals per contrast (A minus B, macro one-vs-rest test AUC):
  seeds     paired t over the ten seeds with the test split fixed (retraining
            only; the A7 estimator, here on the A12-harness reruns rather than
            the published cells, so the two arms share one harness)
  examples  percentile bootstrap over test examples (B resamples) with the ten
            trained models fixed: test-sampling only
  both      seeds (with replacement) and examples resampled together

The AUC of a bootstrap resample is computed exactly from multiplicity weights:
one sort per (model, class) up front, then O(n) per resample, with sklearn's
0.5 tie rule. 2000 resamples x 20 models x 8 classes on 47k examples is a few
minutes on a laptop; roc_auc_score on every resample would take an hour.

Controls, run before any contrast is read:
  * weighted AUC with unit weights equals sklearn roc_auc_score (binary and
    macro-OVR) and with integer weights equals sklearn on the expanded multiset,
    both to 1e-12, on synthetic data with ties planted;
  * a model against itself gives exactly [0, 0] for all three intervals;
  * with the resample replaced by the full index set, the bootstrap statistic
    equals the seed-level mean difference.

    python experiments/a7_capacity_example_level.py
    python experiments/a7_capacity_example_level.py --dataset breast_mnist \
        --preds-dir <dir> --contrasts smoke:digital_zz_1k:poly2_45 --n-seeds 2
"""
import argparse
import csv
import math
import os
import time

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "camera_ready")

# pre-specified contrasts (A7 ids), all at the linear head
CONTRASTS = [
    ("C2-tissue-4k", "digital_zz_4k", "rff_180", "the +0.014 low-capacity exception, four-kernel scale"),
    ("C2-tissue-1k-rff", "digital_zz_1k", "rff_45", "single-kernel scale vs matched RFF"),
    ("C2-tissue-1k-poly", "digital_zz_1k", "poly2_45", "single-kernel scale vs matched poly-2"),
]


class ClassRanker:
    """One class column of one model, sorted once, so the AUC of any weighted
    resample is O(n): AUC = sum_g P_g (N_below_g + 0.5 N_g) / (P N), where g runs
    over tie groups of equal score, P_g / N_g are the positive / negative weight
    in the group and N_below_g the negative weight at strictly lower scores."""

    def __init__(self, scores, is_pos):
        order = np.argsort(scores, kind="stable")
        s = scores[order]
        new_group = np.concatenate([[True], s[1:] != s[:-1]])
        self.group = np.cumsum(new_group) - 1        # tie-group id per sorted position
        self.n_groups = int(self.group[-1]) + 1
        self.order = order
        self.pos = is_pos[order].astype(np.float64)   # 1 for positives, in sorted order
        self.neg = 1.0 - self.pos

    def auc(self, w):
        """w: multiplicity of every example (original index order)."""
        ws = w[self.order]
        P = np.bincount(self.group, weights=ws * self.pos, minlength=self.n_groups)
        N = np.bincount(self.group, weights=ws * self.neg, minlength=self.n_groups)
        below = np.cumsum(N) - N
        return float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))


class Model:
    """One trained model's test scores: a ranker per class (one for binary)."""

    def __init__(self, scores, labels):
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels).reshape(-1)
        if scores.ndim == 1:
            self.rankers = [ClassRanker(scores, labels == 1)]
        else:
            self.rankers = [ClassRanker(scores[:, c], labels == c) for c in range(scores.shape[1])]

    def auc(self, w):
        # macro one-vs-rest for multi-class, plain AUC for binary
        return float(np.mean([r.auc(w) for r in self.rankers]))


def load_models(preds_dir, source, n_seeds):
    """The n_seeds trained models of one linear cell and the (shared) test labels."""
    models, labels = [], None
    for k in range(n_seeds):
        z = np.load(os.path.join(preds_dir, f"linear__{source}", f"seed_{k}_preds.npz"))
        lab = np.asarray(z["labels"]).reshape(-1)
        assert labels is None or np.array_equal(labels, lab), (source, k, "labels differ across seeds")
        labels = lab
        models.append(Model(z["scores"], lab))
    return models, labels


def seed_t_interval(d):
    n = len(d)
    m = float(d.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se


def bootstrap(ma, mb, n, rng, n_boot, resample_seeds):
    """Percentile bootstrap of the seed-mean AUC difference A minus B."""
    n_seeds = len(ma)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        d = np.array([ma[k].auc(w) - mb[k].auc(w) for k in range(n_seeds)])
        if resample_seeds:
            d = d[rng.integers(0, n_seeds, n_seeds)]
        diffs[b] = d.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def controls():
    """Planted controls with known answers; every assert is a hard stop."""
    rng = np.random.default_rng(123)
    n, C = 3000, 8
    y = rng.integers(0, C, n)
    S = rng.random((n, C)) + 0.6 * np.eye(C)[y]
    S = np.round(S, 2)                                  # plant ties
    m = Model(S, y)
    ones = np.ones(n)
    # macro one-vs-rest AUC is the mean of the per-class binary AUCs
    ref = np.mean([roc_auc_score(y == c, S[:, c]) for c in range(C)])
    assert abs(m.auc(ones) - ref) < 1e-12, (m.auc(ones), ref)
    ref_sk = roc_auc_score(y, S / S.sum(1, keepdims=True), multi_class="ovr", average="macro")
    m_norm = Model(S / S.sum(1, keepdims=True), y)
    assert abs(m_norm.auc(ones) - ref_sk) < 1e-12, (m_norm.auc(ones), ref_sk)
    # integer weights == sklearn on the expanded multiset
    w = rng.integers(0, 4, n).astype(np.float64)
    rep = np.repeat(np.arange(n), w.astype(int))
    ref_w = np.mean([roc_auc_score(y[rep] == c, S[rep, c]) for c in range(C)])
    assert abs(m.auc(w) - ref_w) < 1e-12, (m.auc(w), ref_w)
    # binary path
    yb = (y < 3).astype(int)
    sb = np.round(rng.random(n) + 0.4 * yb, 2)
    mb = Model(sb, yb)
    assert abs(mb.auc(ones) - roc_auc_score(yb, sb)) < 1e-12
    assert abs(mb.auc(w) - roc_auc_score(yb[rep], sb[rep])) < 1e-12
    # self-contrast is exactly [0, 0]
    lo, hi = bootstrap([m, m], [m, m], n, np.random.default_rng(1), 50, resample_seeds=False)
    lo2, hi2 = bootstrap([m, m], [m, m], n, np.random.default_rng(1), 50, resample_seeds=True)
    assert lo == hi == 0.0 and lo2 == hi2 == 0.0, (lo, hi, lo2, hi2)
    print("controls OK: weighted AUC == sklearn (macro-OVR and binary, unit and integer weights); "
          "self-contrast [0, 0]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tissue_mnist")
    ap.add_argument("--preds-dir", default=None,
                    help="default outputs/camera_ready/a12/<dataset>_frozen")
    ap.add_argument("--contrasts", nargs="+", default=None,
                    help="id:source_a:source_b (default: the three pre-specified Tissue contrasts)")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    preds_dir = args.preds_dir or os.path.join(OUT_DIR, "a12", f"{args.dataset}_frozen")
    out = args.out or os.path.join(OUT_DIR, f"a7_capacity_example_level_{args.dataset}.csv")
    contrasts = CONTRASTS if args.contrasts is None else [
        tuple(c.split(":")) + ("ad hoc",) for c in args.contrasts]

    controls()
    rng = np.random.default_rng(0)
    rows = []
    for cid, a, b, sentence in contrasts:
        t0 = time.time()
        ma, la = load_models(preds_dir, a, args.n_seeds)
        mb, lb = load_models(preds_dir, b, args.n_seeds)
        assert np.array_equal(la, lb), (cid, "labels differ between arms")
        n = len(la)
        ones = np.ones(n)
        auc_a = np.array([m.auc(ones) for m in ma])
        auc_b = np.array([m.auc(ones) for m in mb])
        d_full = auc_a - auc_b
        m_diff, lo_s, hi_s = seed_t_interval(d_full)
        # control: the bootstrap statistic on the full index set is the seed-mean difference
        d_stat = np.mean([ma[k].auc(ones) - mb[k].auc(ones) for k in range(args.n_seeds)])
        assert abs(d_stat - m_diff) < 1e-12
        lo_e, hi_e = bootstrap(ma, mb, n, rng, args.n_boot, resample_seeds=False)
        lo_b, hi_b = bootstrap(ma, mb, n, rng, args.n_boot, resample_seeds=True)
        # nan when the seed interval is undefined (one seed) or degenerate (self-contrast)
        w_ratio = (hi_e - lo_e) / (hi_s - lo_s) if (hi_s == hi_s and hi_s != lo_s) else float("nan")
        rows.append(dict(contrast=cid, dataset=args.dataset, head="linear", arm_a=a, arm_b=b,
                         n_test=n, n_seeds=args.n_seeds, n_boot=args.n_boot,
                         mean_a=auc_a.mean(), mean_b=auc_b.mean(), mean_diff=m_diff,
                         seeds_lo=lo_s, seeds_hi=hi_s, examples_lo=lo_e, examples_hi=hi_e,
                         both_lo=lo_b, both_hi=hi_b, width_ratio_examples_over_seeds=w_ratio,
                         nonzero_seeds=not (lo_s <= 0 <= hi_s), nonzero_examples=not (lo_e <= 0 <= hi_e),
                         nonzero_both=not (lo_b <= 0 <= hi_b), sentence=sentence))
        print(f"{cid:18s} {a} {auc_a.mean():.4f} vs {b} {auc_b.mean():.4f}  diff {m_diff:+.4f}  "
              f"seeds [{lo_s:+.4f}, {hi_s:+.4f}]  examples [{lo_e:+.4f}, {hi_e:+.4f}]  "
              f"both [{lo_b:+.4f}, {hi_b:+.4f}]  ({time.time() - t0:.0f} s)", flush=True)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
