"""A2 extension, second instrument: select the Table III camp representative on
a TEN-SEED-MEAN validation metric recomputed from the saved checkpoints
(experiments/e2e_val_reinference.py), instead of the single search run's
val_acc that val_selected_e2e.py uses.

RULE, FIXED BEFORE ANY RESULT WAS READ (2026-09-20 17:58 WEST):
  primary   representative = argmax over the camp's rows of the ten-seed-mean
            validation AUC. Continuous, so ties do not arise; matches the
            metric the paper reports.
  secondary representative = argmax of the ten-seed-mean validation accuracy,
            the search objective. Reported alongside; if the two disagree, both
            are stated in the paper.
  The representative's test AUC is its stored ten-seed mean. The Q-C margin
  gets a paired-over-seeds 95% t interval (nominal pairing, as in A7).
  All ten seeds are used: unlike the capacity sweep, validation seed 0 does
  NOT reproduce the search run (checked on the smoke rows), so there is no
  reason to drop it.

The script refuses any row whose test-AUC control failed in the re-inference.
TissueMNIST is not re-inferred (caches too large for this machine) and is
unaffected: validation and test already pick the same rows there.
"""
import csv
import glob
import json
import math
import os
import sys

from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from val_selected_e2e import QUANTUM          # noqa: E402  one camp definition

IN = os.path.join(ROOT, "outputs", "camera_ready", "e2e_val_reinference")
OUT = os.path.join(ROOT, "outputs", "camera_ready", "val_selected_e2e_tenseed.csv")


def paired_ci(a, b, level=0.95):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return m, m - t * sd / math.sqrt(n), m + t * sd / math.sqrt(n)


FIXED = os.path.join(ROOT, "outputs", "camera_ready", "e2e_random_fixed")


def load_fixed_random_rows():
    """The frozen-filter re-validation of the Random x45 / x180 rows (Breast and
    Pneumonia), in the same shape as a re-inference JSON. The published rows
    were trained with trainable filters by mistake (tracker, 2026-09-20)."""
    out = {}
    for f in glob.glob(os.path.join(FIXED, "*", "summary.json")):
        j = json.load(open(f))
        if j["n_seeds"] != 10 or not j["all_frozen"]:
            raise SystemExit(f"incomplete or unfrozen re-validation: {f}")
        out[(j["dataset"], j["row"])] = dict(
            dataset=j["dataset"], model=j["row"], search_value=float("nan"),
            all_controls_match=True,
            mean_val_acc=sum(s["val_acc"] for s in j["seeds"]) / 10,
            mean_val_auc=sum(s["val_auc"] for s in j["seeds"]) / 10,
            mean_test_auc=j["test_auc_mean"], seeds=j["seeds"])
    return out


def main():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(IN, "*.json")))]
    bad = [r for r in rows if not r["all_controls_match"]]
    if bad:
        raise SystemExit("control failed for: " + ", ".join(f"{r['dataset']}/{r['model']}" for r in bad))
    if "--random-fixed" in sys.argv:
        fixed = load_fixed_random_rows()
        rows = [fixed.get((r["dataset"], r["model"]), r) for r in rows]
        swapped = [k for k in fixed if any(r["dataset"] == k[0] and r["model"] == k[1] for r in rows)]
        print(f"--random-fixed: swapped {len(swapped)} rows: {swapped}")

    out = []
    for ds in sorted({r["dataset"] for r in rows}, key=lambda x: "Breast" not in x):
        recs = [r for r in rows if r["dataset"] == ds]
        print(f"\n=== {ds} ({len(recs)} rows) ===")
        print(f"{'model':16s} camp {'search_val':>10s} {'val_acc10':>9s} {'val_auc10':>9s} {'test_auc':>8s}")
        for r in sorted(recs, key=lambda r: (r["model"] in QUANTUM, -r["mean_val_auc"])):
            r["camp"] = "Q" if r["model"] in QUANTUM else "C"
            print(f"{r['model']:16s} {r['camp']:4s} {r['search_value']:10.4f} "
                  f"{r['mean_val_acc']:9.4f} {r['mean_val_auc']:9.4f} {r['mean_test_auc']:8.4f}")

        picks = {}
        for rule, key in (("val_auc10", "mean_val_auc"), ("val_acc10", "mean_val_acc")):
            for camp in ("Q", "C"):
                camp_recs = [r for r in recs if r["camp"] == camp]
                top = max(r[key] for r in camp_recs)
                tied = [r for r in camp_recs if r[key] == top]
                picks[(rule, camp)] = tied
            q, c = picks[(rule, "Q")], picks[(rule, "C")]
            # a tie is possible only under val_acc10; average over tied rows per seed
            qv = [sum(r["seeds"][i]["test_auc"] for r in q) / len(q) for i in range(10)]
            cv = [sum(r["seeds"][i]["test_auc"] for r in c) / len(c) for i in range(10)]
            m, lo, hi = paired_ci(qv, cv)
            qn = ", ".join(r["model"] for r in q)
            cn = ", ".join(r["model"] for r in c)
            print(f"  [{rule}] Q = {qn} ({sum(qv)/10:.4f})  C = {cn} ({sum(cv)/10:.4f})  "
                  f"margin Q-C {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
            out.append(dict(dataset=ds, rule=rule, q_rep=qn, c_rep=cn,
                            q_test_auc=sum(qv) / 10, c_test_auc=sum(cv) / 10,
                            margin=m, ci_lo=lo, ci_hi=hi))
        # reference: the two rules the paper already reports
        tq = max((r for r in recs if r["camp"] == "Q"), key=lambda r: r["mean_test_auc"])
        tc = max((r for r in recs if r["camp"] == "C"), key=lambda r: r["mean_test_auc"])
        print(f"  [test-selected]  Q = {tq['model']} ({tq['mean_test_auc']:.4f})  "
              f"C = {tc['model']} ({tc['mean_test_auc']:.4f})  margin "
              f"{tq['mean_test_auc'] - tc['mean_test_auc']:+.4f}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
