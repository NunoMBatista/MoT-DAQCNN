"""A8 readout: paired intervals on the selection-objective effect, against the
measured run-to-run reproducibility floor.

Two comparisons, and the second is what decides whether the first means
anything:

1. WITHIN-RUN. For each cell, both selection rules were validated over the same
   ten seeds on the same feature tensors, so the difference can be differenced
   within seed and given a paired-t interval. This isolates the selection
   objective.

2. BETWEEN-RUN. The same 65 cells were computed twice in separate processes.
   `load_features` drains a shuffled, unseeded train loader, so each process
   draws a different training row order. The spread between the two runs of the
   SAME quantity is the reproducibility floor, and any within-run effect
   smaller than it cannot be acted on.
"""
import glob, json, math, re, sys
import collections
from scipy import stats

QUANTUM = {"digital_z_1k", "digital_zz_1k", "digital_zz_4k",
           "analog_z_1k", "analog_zz_1k", "analog_zz_4k"}
HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
RUN1_LOG = ("/tmp/claude-1000/-home-nuno-Documents-MoT-DAQCNN/"
            "47d8636e-c4f3-4b58-9c3c-ee45dc6c4522/scratchpad/a8_run.log")


def paired_ci(a, b, level=0.95):
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    se = sd / math.sqrt(n)
    return m, m - t * se, m + t * se


cells = {}
for f in glob.glob("outputs/camera_ready/a8/cells/*/result.json"):
    r = json.load(open(f))
    cells[(r["head"], r["source"])] = r

# ---- 2. reproducibility floor, from run 1's log versus run 2's JSONs --------
run1 = {}
pat = re.compile(r"^\s+(\S+)\s+(\S+)\s+(?:DIFFERENT trial|same trial)\s+"
                 r"AUC\(val_acc-sel\)=([\d.]+)\s+AUC\(val_auc-sel\)=([\d.]+)")
for line in open(RUN1_LOG):
    m = pat.match(line)
    if m:
        run1[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))

floor = []
for k, r in cells.items():
    if k in run1:
        floor.append(abs(r["auc_sel_acc"] - run1[k][0]))
        floor.append(abs(r["auc_sel_auc"] - run1[k][1]))
floor.sort()
n = len(floor)
print(f"REPRODUCIBILITY FLOOR (same quantity, two independent runs, n={n})")
print(f"  median |diff| {floor[n//2]:.4f}   90th pct {floor[int(0.9*n)]:.4f}   "
      f"max {floor[-1]:.4f}")
print(f"  |diff| exceeds 0.005 in {sum(x > 0.005 for x in floor)}/{n} comparisons")

# ---- 1. within-run selection-objective effect ------------------------------
print(f"\nSELECTION OBJECTIVE, within run (val_auc-selected minus val_acc-selected)")
sig = 0
big = []
for k, r in sorted(cells.items()):
    m, lo, hi = paired_ci(r["test_auc_values_sel_auc"], r["test_auc_values_sel_acc"])
    nonzero = not (lo <= 0 <= hi)
    sig += nonzero
    if nonzero:
        big.append((abs(m), k, m, lo, hi))
print(f"  cells whose 95% paired interval excludes zero: {sig}/{len(cells)}")
print(f"  cells where the two rules chose the same trial: "
      f"{sum(r['same_trial'] for r in cells.values())}/{len(cells)}")
big.sort(reverse=True)
print("  largest resolved effects:")
for _, k, m, lo, hi in big[:5]:
    print(f"    {k[0]:10s} {k[1]:12s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]")

# ---- conclusions: camp margins per head ------------------------------------
print(f"\nCAMP MARGIN (best quantum minus best classical) per head")
print(f"  {'head':10s} {'val_acc-sel':>12s} {'val_auc-sel':>12s} {'shift':>8s} "
      f"{'> floor?':>9s}")
flips = 0
for h in HEADS:
    recs = [(s, r) for (hh, s), r in cells.items() if hh == h]
    def margin(key):
        q = max(r[key] for s, r in recs if s in QUANTUM)
        c = max(r[key] for s, r in recs if s not in QUANTUM)
        return q - c
    a, b = margin("auc_sel_acc"), margin("auc_sel_auc")
    flip = (a > 0) != (b > 0)
    flips += flip
    shift = b - a
    print(f"  {h:10s} {a:+12.4f} {b:+12.4f} {shift:+8.4f} "
          f"{'yes' if abs(shift) > floor[int(0.9*n)] else 'NO':>9s}"
          + ("   SIGN FLIP" if flip else ""))
print(f"\n  sign flips: {flips}/5, but see the floor column: a flip whose shift")
print(f"  is below the reproducibility floor is not an effect of the objective.")

# ---- the claim the paper actually states: a FIXED pair --------------------
# The paper's low-capacity numbers (+0.020 / +0.040 on BreastMNIST) are
# Digital-ZZ 1k minus Random x45, a pre-specified pair, not "best quantum".
# An earlier version of this block took max over TEST AUC to pick the quantum
# arm, which is the selection-on-test bias A2 removed; corrected 2026-09-20.
print(f"\nPAPER'S LOW-CAPACITY CLAIM (digital_zz_1k minus random_45, fixed pair):")
for h in ("mlp1", "mlp2"):
    q = cells[(h, "digital_zz_1k")]
    r45 = cells[(h, "random_45")]
    for key, label in (("test_auc_values_sel_acc", "val_acc-selected"),
                       ("test_auc_values_sel_auc", "val_auc-selected")):
        m, lo, hi = paired_ci(q[key], r45[key])
        print(f"  {h:6s} {label:16s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"{'  excludes 0' if not (lo <= 0 <= hi) else ''}")
