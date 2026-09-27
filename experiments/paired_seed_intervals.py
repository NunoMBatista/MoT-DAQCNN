"""Paired-over-seeds confidence intervals for every "tie" / "no difference" claim.

Reviewer 4 (#3) objects that overlapping standard deviations do not establish
equivalence, and asks for paired confidence intervals on AUC differences or an
equivalence test with a pre-specified margin. This script supplies both from
data that already exists: every reported cell was validated over 10 seeds (5 on
TissueMNIST end-to-end), and `set_seed(s)` seeds python/numpy/torch globally
before each run, so seed s is a shared nuisance factor across models and the
per-seed AUCs can be differenced within seed.

WHAT THIS DOES AND DOES NOT COVER. The pairing is over *training seeds*: it
answers "would this difference survive retraining?". It does NOT cover
test-example variance, which is what a DeLong or example-level bootstrap would
add, because no per-example predictions were ever saved (cells keep aggregates
plus per-seed scalars). Every model is scored on the same fixed MedMNIST test
split, so the test-sample noise is common to all arms and is invisible here.
An example-level interval needs re-inference at the already-known best
hyperparameters; the cost is printed at the end of this run so the trade can be
decided with a number rather than a guess.

Why a t interval and not a bootstrap over seeds: a percentile bootstrap over 10
values cannot produce an interval wider than the observed range, so with n=10 it
is anticonservative exactly where honesty matters. The paired t interval is the
right instrument at this sample size.

PRE-SPECIFIED EQUIVALENCE MARGIN: delta = 0.01 AUC (and 0.01 accuracy), fixed
here before any contrast was computed. It is one point of AUC, the granularity
at which the paper's tables report, and it is applied uniformly to every
contrast. Equivalence is declared by TOST: the two one-sided tests at alpha=0.05
pass exactly when the 90% CI of the paired difference lies inside (-delta,
+delta). A difference is called non-zero when the 95% CI excludes 0. The two
verdicts are independent, so a contrast can be both (a real difference smaller
than the margin) or neither (underpowered).

MULTIPLICITY: the contrast list below is pre-specified and each entry maps to a
named sentence in the paper. These are per-claim intervals, not a family-wise
corrected screen, and no contrast that is not tied to a sentence should be read
for significance.

Run from the repo root:
    python experiments/paired_seed_intervals.py
Writes outputs/camera_ready/paired_seed_intervals.csv (gitignored).
"""
import csv
import json
import math
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_selected_margins import (ROOT, HEADS, QUANTUM, CLASSICAL, DATASETS,
                                  load_cells, pick)

DELTA = 0.01          # pre-specified equivalence margin, see module docstring
CI_LEVEL = 0.95       # for the "is it non-zero" verdict
TOST_LEVEL = 0.90     # 90% CI <=> two one-sided tests at alpha=0.05

# ---------------------------------------------------------------------------
# End-to-end runs (Table III). Each row is identified by the winning trial's
# params, not by base_config.yml, which is stale in several run directories.
# PAPER_AUC/PAPER_ACC below re-derive the published cell from the same file, so
# a mis-mapped directory fails loudly instead of producing a plausible interval.
# ---------------------------------------------------------------------------
E2E = {
    ("BreastMNIST", "Digital-Z 1k"):     "outputs/paper_results/hp_search/outputs/hp_search_original_breastmnist_20260518_171036",
    ("BreastMNIST", "Digital-ZZ 1k"):    "outputs/paper_results/hp_search/hp_search_original_breastmnist_20260518_184203",
    ("BreastMNIST", "Analog-ZZ 1k"):     "outputs/paper_results/hp_search/hp_search_original_breastmnist_20260519_181050",
    ("BreastMNIST", "Digital-ZZ 4k"):    "outputs/paper_results/hp_search/outputs/hp_search_original_breastmnist_20260518_233451",
    ("BreastMNIST", "Analog-ZZ 4k"):     "outputs/paper_results/hp_search/hp_search_original_breastmnist_20260519_012532",
    ("BreastMNIST", "Random x45"):       "outputs/paper_results/hp_search/outputs/hp_search_classical_baseline_breastmnist_20260518_194114",
    ("BreastMNIST", "Trainable x45"):    "outputs/paper_results/hp_search/hp_search_classical_baseline_breastmnist_20260518_201522",
    ("BreastMNIST", "Random x180"):      "outputs/paper_results/hp_search/outputs/hp_search_classical_baseline_breastmnist_20260519_014949",
    ("BreastMNIST", "Trainable x180"):   "outputs/paper_results/hp_search/hp_search_classical_baseline_breastmnist_20260519_104726",
    ("BreastMNIST", "poly-2 x45"):       "outputs/mcc_retrieved/hp_search/classical_nonlinear/breast_mnist__poly2",
    ("BreastMNIST", "RFF x45"):          "outputs/mcc_retrieved/hp_search/classical_nonlinear/breast_mnist__rff_45",
    ("BreastMNIST", "RFF x180"):         "outputs/mcc_retrieved/hp_search/classical_nonlinear/breast_mnist__rff_180",
    ("BreastMNIST", "Raw pixels"):       "outputs/paper_results/hp_search/outputs/hp_search_classical_baseline_breastmnist_20260518_222901",

    ("PneumoniaMNIST", "Digital-Z 1k"):   "outputs/paper_results/hp_search/pneu_digital_z_1k",
    ("PneumoniaMNIST", "Digital-ZZ 1k"):  "outputs/paper_results/hp_search/pneu_digital_zz_1k",
    ("PneumoniaMNIST", "Analog-ZZ 1k"):   "outputs/paper_results/hp_search/pneu_analog_zz_1k",
    ("PneumoniaMNIST", "Digital-ZZ 4k"):  "outputs/paper_results/hp_search/pneu_digital_zz_4k",
    ("PneumoniaMNIST", "Analog-ZZ 4k"):   "outputs/paper_results/hp_search/pneu_analog_zz_4k",
    ("PneumoniaMNIST", "Random x45"):     "outputs/paper_results/hp_search/pneu_classical_random_1k",
    ("PneumoniaMNIST", "Trainable x45"):  "outputs/paper_results/hp_search/pneu_classical_trainable_1k",
    ("PneumoniaMNIST", "Random x180"):    "outputs/paper_results/hp_search/pneu_classical_random_4k",
    ("PneumoniaMNIST", "Trainable x180"): "outputs/paper_results/hp_search/pneu_classical_trainable_4k",
    ("PneumoniaMNIST", "poly-2 x45"):     "outputs/mcc_retrieved/hp_search/classical_nonlinear/pneumonia_mnist__poly2",
    ("PneumoniaMNIST", "RFF x45"):        "outputs/mcc_retrieved/hp_search/classical_nonlinear/pneumonia_mnist__rff_45",
    ("PneumoniaMNIST", "RFF x180"):       "outputs/mcc_retrieved/hp_search/classical_nonlinear/pneumonia_mnist__rff_180",
    ("PneumoniaMNIST", "Raw pixels"):     "outputs/paper_results/hp_search/pneu_raw",

    ("TissueMNIST", "Digital-Z 1k"):     "outputs/mcc_retrieved/hp_search/tissue_mnist/digital_z",
    ("TissueMNIST", "Digital-ZZ 1k"):    "outputs/mcc_retrieved/hp_search/tissue_mnist/digital_zz_1k",
    ("TissueMNIST", "Analog-ZZ 1k"):     "outputs/mcc_retrieved/hp_search/tissue_mnist/analog_zz_1k",
    ("TissueMNIST", "Digital-ZZ 4k"):    "outputs/mcc_retrieved/hp_search/tissue_mnist/digital_zz_4k",
    ("TissueMNIST", "Analog-ZZ 4k"):     "outputs/mcc_retrieved/hp_search/tissue_mnist/analog_zz_4k",
    ("TissueMNIST", "Random x45"):       "outputs/mcc_retrieved/hp_search/tissue_mnist/random_1k",
    ("TissueMNIST", "Trainable x45"):    "outputs/mcc_retrieved/hp_search/tissue_mnist/trainable_1k",
    ("TissueMNIST", "Random x180"):      "outputs/mcc_retrieved/hp_search/tissue_mnist/random_4k",
    ("TissueMNIST", "Trainable x180"):   "outputs/mcc_retrieved/hp_search/tissue_mnist/trainable_4k",
    ("TissueMNIST", "poly-2 x45"):       "outputs/mcc_retrieved/hp_search/tissue_mnist/poly2",
    ("TissueMNIST", "RFF x45"):          "outputs/mcc_retrieved/hp_search/tissue_mnist/rff_45",
    ("TissueMNIST", "RFF x180"):         "outputs/mcc_retrieved/hp_search/tissue_mnist/rff_180",
    ("TissueMNIST", "Raw pixels"):       "outputs/mcc_retrieved/hp_search/tissue_mnist/raw",
}

# Published Table III cells, transcribed from docs/paper/paper_extended.tex, as
# (acc, auc) rounded to three decimals. The control: every mapped directory must
# reproduce its own published cell.
PAPER_TABLE = {
    ("BreastMNIST", "Digital-Z 1k"): (0.848, 0.868),
    ("BreastMNIST", "Digital-ZZ 1k"): (0.855, 0.883),
    ("BreastMNIST", "Analog-ZZ 1k"): (0.844, 0.889),
    ("BreastMNIST", "Digital-ZZ 4k"): (0.852, 0.901),
    ("BreastMNIST", "Analog-ZZ 4k"): (0.847, 0.893),
    ("BreastMNIST", "Random x45"): (0.863, 0.896),
    ("BreastMNIST", "Trainable x45"): (0.862, 0.889),
    ("BreastMNIST", "Random x180"): (0.860, 0.896),
    ("BreastMNIST", "Trainable x180"): (0.835, 0.863),
    ("BreastMNIST", "poly-2 x45"): (0.819, 0.865),
    ("BreastMNIST", "RFF x45"): (0.844, 0.888),
    ("BreastMNIST", "RFF x180"): (0.853, 0.889),
    ("BreastMNIST", "Raw pixels"): (0.824, 0.843),
    ("PneumoniaMNIST", "Digital-Z 1k"): (0.865, 0.953),
    ("PneumoniaMNIST", "Digital-ZZ 1k"): (0.858, 0.948),
    ("PneumoniaMNIST", "Analog-ZZ 1k"): (0.858, 0.943),
    ("PneumoniaMNIST", "Digital-ZZ 4k"): (0.869, 0.955),
    ("PneumoniaMNIST", "Analog-ZZ 4k"): (0.859, 0.959),
    ("PneumoniaMNIST", "Random x45"): (0.856, 0.957),
    ("PneumoniaMNIST", "Trainable x45"): (0.857, 0.950),
    ("PneumoniaMNIST", "Random x180"): (0.851, 0.959),
    ("PneumoniaMNIST", "Trainable x180"): (0.857, 0.945),
    ("PneumoniaMNIST", "poly-2 x45"): (0.853, 0.948),
    ("PneumoniaMNIST", "RFF x45"): (0.845, 0.962),
    ("PneumoniaMNIST", "RFF x180"): (0.850, 0.952),
    ("PneumoniaMNIST", "Raw pixels"): (0.855, 0.951),
    ("TissueMNIST", "Digital-Z 1k"): (0.565, 0.866),
    ("TissueMNIST", "Digital-ZZ 1k"): (0.583, 0.880),
    ("TissueMNIST", "Analog-ZZ 1k"): (0.586, 0.880),
    ("TissueMNIST", "Digital-ZZ 4k"): (0.582, 0.878),
    ("TissueMNIST", "Analog-ZZ 4k"): (0.570, 0.876),
    ("TissueMNIST", "Random x45"): (0.609, 0.892),
    ("TissueMNIST", "Trainable x45"): (0.608, 0.893),
    ("TissueMNIST", "Random x180"): (0.609, 0.893),
    ("TissueMNIST", "Trainable x180"): (0.606, 0.892),
    ("TissueMNIST", "poly-2 x45"): (0.615, 0.896),
    ("TissueMNIST", "RFF x45"): (0.584, 0.883),
    ("TissueMNIST", "RFF x180"): (0.594, 0.890),
    ("TissueMNIST", "Raw pixels"): (0.598, 0.886),
}

# Pre-specified end-to-end contrasts, one per claim in Sec. results_endtoend.
# (id, dataset, model_a, model_b, the sentence it tests)
E2E_CONTRASTS = [
    ("E1-breast", "BreastMNIST", "Digital-ZZ 4k", "Random x45",
     "best quantum vs best classical: +0.005 AUC, -0.012 acc (frozen Random x45, 2026-09-20)"),
    ("E1-pneumonia", "PneumoniaMNIST", "Analog-ZZ 4k", "RFF x45",
     "best classical narrowly ahead by 0.003 AUC, 'within one standard deviation'"),
    ("E1-tissue", "TissueMNIST", "Analog-ZZ 1k", "poly-2 x45",
     "best classical ahead by 0.016 AUC: the claimed reversal"),
    ("E2-breast", "BreastMNIST", "Digital-ZZ 1k", "Analog-ZZ 1k", "encoding null, single-kernel"),
    ("E2-pneumonia", "PneumoniaMNIST", "Digital-ZZ 1k", "Analog-ZZ 1k", "encoding null, single-kernel"),
    ("E2-tissue", "TissueMNIST", "Digital-ZZ 1k", "Analog-ZZ 1k",
     "'digital and analog tied to within 0.001'"),
    ("E3-breast", "BreastMNIST", "Digital-ZZ 4k", "Analog-ZZ 4k", "encoding null, four-kernel"),
    ("E3-pneumonia", "PneumoniaMNIST", "Digital-ZZ 4k", "Analog-ZZ 4k", "encoding null, four-kernel"),
    ("E3-tissue", "TissueMNIST", "Digital-ZZ 4k", "Analog-ZZ 4k", "encoding null, four-kernel"),
    ("E4-breast", "BreastMNIST", "Digital-ZZ 1k", "Digital-Z 1k",
     "ZZ lifts end-to-end AUC 0.868 -> 0.883 on the dataset that gains at the probe"),
    ("E4-pneumonia", "PneumoniaMNIST", "Digital-ZZ 1k", "Digital-Z 1k",
     "'ZZ correlators give no end-to-end benefit here'"),
    ("E5-tissue", "TissueMNIST", "Analog-ZZ 1k", "Raw pixels",
     "'even raw pixels outperform every quantum model'"),
]

# Pre-specified capacity-sweep contrasts, one per claim in Sec. results_capacity.
# (id, dataset, head, source_a, source_b, sentence)
CAP_CONTRASTS = [
    ("C2-tissue-4k", "TissueMNIST", "linear", "digital_zz_4k", "rff_180",
     "the +0.014 low-capacity exception, the paper's one positive quantum result"),
    ("C2-tissue-1k-rff", "TissueMNIST", "linear", "digital_zz_1k", "rff_45",
     "+0.008 at single-kernel scale vs matched RFF"),
    ("C2-tissue-1k-poly", "TissueMNIST", "linear", "digital_zz_1k", "poly2_45",
     "+0.008 at single-kernel scale vs matched poly-2"),
    ("C3-breast-mlp1", "BreastMNIST", "mlp1", "digital_zz_1k", "random_45",
     "'+0.020 on BreastMNIST at MLP-1 ... above one standard deviation'"),
    ("C3-breast-mlp2", "BreastMNIST", "mlp2", "digital_zz_1k", "random_45",
     "'+0.040 on BreastMNIST at MLP-2 ... above one standard deviation'"),
    ("C4-pneumonia-mlp1", "PneumoniaMNIST", "mlp1", "digital_zz_1k", "random_45",
     "'+0.008 on PneumoniaMNIST at MLP-1 ... above one standard deviation'"),
    ("C4-pneumonia-mlp2", "PneumoniaMNIST", "mlp2", "digital_zz_1k", "random_45",
     "'+0.010 on PneumoniaMNIST at MLP-2 ... above one standard deviation'"),
    # Added 2026-09-27 (audit item FM-6): the paper already quoted this interval,
    # computed outside this list, so it was added after its value was known.
    ("C5-breast-mlp1", "BreastMNIST", "mlp1", "analog_zz_4k", "rff_180",
     "'Analog-ZZ: -0.030 to +0.022, the latter at BreastMNIST MLP-1, paired interval [+0.003,+0.042]'"),
]


def half_up3(x):
    """Round to three decimals half-up, the convention a printed table uses.

    Python's round() is half-to-even and its float arguments are rarely exact
    halves anyway, so comparing published cells needs this stated explicitly.
    """
    return float(Decimal(repr(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


# The BreastMNIST / PneumoniaMNIST "Random x45 / x180" validations were trained
# with trainable filters by mistake (tracker, 2026-09-20). Their per-seed values
# come from the frozen-filter re-validation instead of the original run dir.
FIXED_RANDOM = {
    E2E[(d, m)]: os.path.join("outputs", "camera_ready", "e2e_random_fixed",
                              f"{d}__{m.replace(' ', '_')}", "summary.json")
    for d in ("BreastMNIST", "PneumoniaMNIST") for m in ("Random x45", "Random x180")
}


def load_e2e(path):
    """Per-seed test accuracy and AUC for one end-to-end run, keyed by seed index."""
    if path in FIXED_RANDOM:
        j = json.load(open(os.path.join(ROOT, FIXED_RANDOM[path])))
        assert j["n_seeds"] == 10 and j["all_frozen"], FIXED_RANDOM[path]
        return {"acc": [r["test_acc"] for r in j["seeds"]],
                "auc": [r["test_auc"] for r in j["seeds"]]}
    rec = json.load(open(os.path.join(ROOT, path, "validation", "validation_summary.json")))[0]
    return {"acc": rec["test_acc"]["values"], "auc": rec["test_auc"]["values"]}


def paired(a, b):
    """Paired statistics for two equal-length per-seed vectors, a minus b.

    Returns the mean difference, the paired SD, the 95% CI (is it non-zero?),
    the 90% CI (TOST at alpha=0.05 against the pre-specified margin), and two
    diagnostics of whether the pairing bought anything: the across-seed
    correlation and the ratio of the paired SD to the unpaired SD it replaces.
    """
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t95 = stats.t.ppf(0.5 + CI_LEVEL / 2, n - 1)
    t90 = stats.t.ppf(0.5 + TOST_LEVEL / 2, n - 1)

    # what the unpaired (independent-samples) SD of the difference would be
    sd_a = math.sqrt(sum((x - sum(a) / n) ** 2 for x in a) / (n - 1))
    sd_b = math.sqrt(sum((y - sum(b) / n) ** 2 for y in b) / (n - 1))
    unpaired_sd = math.sqrt(sd_a ** 2 + sd_b ** 2)
    corr = stats.pearsonr(a, b)[0] if sd_a > 0 and sd_b > 0 else float("nan")

    lo95, hi95 = mean - t95 * se, mean + t95 * se
    lo90, hi90 = mean - t90 * se, mean + t90 * se
    return {
        "n_seeds": n, "mean_diff": mean, "paired_sd": sd,
        "ci95_lo": lo95, "ci95_hi": hi95, "ci90_lo": lo90, "ci90_hi": hi90,
        "nonzero_95": not (lo95 <= 0 <= hi95),
        "equivalent_tost": (lo90 > -DELTA) and (hi90 < DELTA),
        "seed_corr": corr,
        "pairing_sd_ratio": sd / unpaired_sd if unpaired_sd > 0 else float("nan"),
    }


def verdict(r):
    """One word for the pair of independent tests, for the console table."""
    if r["nonzero_95"] and r["equivalent_tost"]:
        return "real-but-small"          # detectable, yet inside the margin
    if r["nonzero_95"]:
        return "DIFFERENT"
    if r["equivalent_tost"]:
        return "EQUIVALENT"
    return "inconclusive"                # CI spans both 0 and a margin edge


def main():
    rows = []

    # --- control: every mapped run must reproduce its published table cell ---
    # Two failure modes to tell apart. A wrong directory lands on a different
    # model and misses by far more than the table's own resolution; a last-digit
    # transcription slip misses by at most one unit of the third decimal. The
    # first is fatal, the second is reported and carried forward.
    print("=== Table III mapping control ===")
    e2e_data, fatal, slips = {}, [], []
    for key, path in E2E.items():
        d = load_e2e(path)
        e2e_data[key] = d
        got = {"acc": sum(d["acc"]) / len(d["acc"]), "auc": sum(d["auc"]) / len(d["auc"])}
        want = dict(zip(("acc", "auc"), PAPER_TABLE[key]))
        for metric in ("acc", "auc"):
            gap = abs(got[metric] - want[metric])
            if gap > 0.001:
                fatal.append(f"{key} {metric}: got {got[metric]:.4f}, table says {want[metric]}")
            elif half_up3(got[metric]) != want[metric]:
                slips.append(f"{key} {metric}: {got[metric]:.6f} rounds to "
                             f"{half_up3(got[metric]):.3f}, table says {want[metric]:.3f}")
    if fatal:
        raise SystemExit("MAPPING FAILED, refusing to compute intervals:\n  " + "\n  ".join(fatal))
    print(f"all {len(E2E)} mapped runs land within 0.001 of their published (acc, AUC) cell")
    if slips:
        print(f"{len(slips)} published cells are one unit high in the last decimal "
              f"(4 dp then 3 dp double rounding), no claim depends on them:")
        for s in slips:
            print("   " + s)
    print()

    # --- end-to-end contrasts, both metrics ---
    print("=== End-to-end (Table III), paired over validation seeds ===")
    print(f"{'claim':<16} {'metric':<4} {'n':>2} {'diff':>8} {'95% CI':>19} {'verdict':<15}")
    for cid, dset, a_name, b_name, sentence in E2E_CONTRASTS:
        for metric in ("auc", "acc"):
            a = e2e_data[(dset, a_name)][metric]
            b = e2e_data[(dset, b_name)][metric]
            r = paired(a, b)
            rows.append({"table": "endtoend", "claim": cid, "dataset": dset,
                         "arm_a": a_name, "arm_b": b_name, "metric": metric,
                         "mean_a": sum(a) / len(a), "mean_b": sum(b) / len(b),
                         "verdict": verdict(r), "sentence": sentence, **r})
            print(f"{cid:<16} {metric:<4} {r['n_seeds']:>2} {r['mean_diff']:>+8.4f} "
                  f"[{r['ci95_lo']:>+7.4f},{r['ci95_hi']:>+7.4f}] {verdict(r):<15}")

    # --- capacity sweep contrasts ---
    print("\n=== Head-capacity sweep, paired over validation seeds (AUC) ===")
    cells = {dname: load_cells(sweep_dir) for dname, sweep_dir in DATASETS}
    print(f"{'claim':<20} {'head':<7} {'diff':>8} {'95% CI':>19} {'verdict':<15}")
    for cid, dset, head, a_src, b_src, sentence in CAP_CONTRASTS:
        a = cells[dset][(head, a_src)]["test_aucs"]
        b = cells[dset][(head, b_src)]["test_aucs"]
        r = paired(a, b)
        rows.append({"table": "capacity", "claim": cid, "dataset": dset, "head": head,
                     "arm_a": a_src, "arm_b": b_src, "metric": "auc",
                     "mean_a": sum(a) / len(a), "mean_b": sum(b) / len(b),
                     "verdict": verdict(r), "sentence": sentence, **r})
        print(f"{cid:<20} {head:<7} {r['mean_diff']:>+8.4f} "
              f"[{r['ci95_lo']:>+7.4f},{r['ci95_hi']:>+7.4f}] {verdict(r):<15}")

    # --- the validation-selected camp margins from Fig. 4, every head ---
    print("\n=== Validation-selected camp margins (Fig. 4), paired over seeds ===")
    print(f"{'dataset':<15} {'head':<10} {'Q source':<15} {'C source':<12} "
          f"{'diff':>8} {'95% CI':>19} {'verdict':<15}")
    for dname, _ in DATASETS:
        for head in HEADS:
            q_src = pick(cells[dname], head, QUANTUM, "val_acc_mean")
            c_src = pick(cells[dname], head, CLASSICAL, "val_acc_mean")
            if q_src is None or c_src is None:
                continue
            a = cells[dname][(head, q_src)]["test_aucs"]
            b = cells[dname][(head, c_src)]["test_aucs"]
            r = paired(a, b)
            rows.append({"table": "camp_margin", "claim": f"C1-{dname}-{head}",
                         "dataset": dname, "head": head, "arm_a": q_src, "arm_b": c_src,
                         "metric": "auc", "mean_a": sum(a) / len(a), "mean_b": sum(b) / len(b),
                         "verdict": verdict(r),
                         "sentence": "validation-selected quantum minus classical margin",
                         **r})
            print(f"{dname:<15} {head:<10} {q_src:<15} {c_src:<12} "
                  f"{r['mean_diff']:>+8.4f} [{r['ci95_lo']:>+7.4f},{r['ci95_hi']:>+7.4f}] "
                  f"{verdict(r):<15}")

    # --- pairing diagnostic: did differencing within seed actually help? ---
    ratios = [r["pairing_sd_ratio"] for r in rows if not math.isnan(r["pairing_sd_ratio"])]
    corrs = [r["seed_corr"] for r in rows if not math.isnan(r["seed_corr"])]
    helped = sum(1 for x in ratios if x < 1.0)
    print(f"\n=== Pairing diagnostic over {len(ratios)} contrasts ===")
    print(f"median across-seed correlation between the two arms: {sorted(corrs)[len(corrs)//2]:+.3f}")
    print(f"paired SD / unpaired SD: median {sorted(ratios)[len(ratios)//2]:.3f}, "
          f"below 1 (pairing helped) in {helped}/{len(ratios)} contrasts")
    print("A correlation near zero means the seed is not a shared nuisance factor "
          "and the pairing is nominal; say so rather than claiming a paired gain.")

    out_dir = os.path.join(ROOT, "outputs", "camera_ready")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "paired_seed_intervals.csv")
    fields = ["table", "claim", "dataset", "head", "arm_a", "arm_b", "metric",
              "mean_a", "mean_b", "n_seeds", "mean_diff", "paired_sd",
              "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "nonzero_95",
              "equivalent_tost", "verdict", "seed_corr", "pairing_sd_ratio", "sentence"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}  ({len(rows)} contrasts, delta = {DELTA})")

    # --- cost of the example-level interval R4 actually asked for ---
    # Capacity sweep: 30 Optuna trials + 10 validation seeds per cell. Re-running
    # only the validation seeds with predictions saved skips the search.
    print("\nCost of upgrading to example-level (DeLong / bootstrap over test cases):")
    print("  capacity sweep  10 of 40 trainings per cell = 25% of the original sweep")
    print("  end-to-end      10 of 210 trainings per model = 4.8% of the original search")
    print("  Only the seeds listed in the contrasts above actually need it.")


if __name__ == "__main__":
    main()
