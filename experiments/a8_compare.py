"""A8 readout: do the conclusions change when selection uses val AUC instead of val_acc?

Reads outputs/camera_ready/a8/a8_summary.csv, which holds one row per
(head, source) with the mean test AUC obtained under each selection rule.
Conclusions at stake, in the order the paper makes them:
  1. the ordering of feature sources within each head,
  2. the sign of the best-quantum minus best-classical margin at each head,
  3. the specific low-capacity claims about random projections.
"""
import csv, collections
from scipy import stats

QUANTUM = {"digital_z_1k", "digital_zz_1k", "digital_zz_4k",
           "analog_z_1k", "analog_zz_1k", "analog_zz_4k"}
HEAD_ORDER = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]

rows = list(csv.DictReader(open("outputs/camera_ready/a8/a8_summary.csv")))
by_head = collections.defaultdict(list)
for r in rows:
    by_head[r["head"]].append((r["source"], float(r["auc_sel_acc"]), float(r["auc_sel_auc"])))

n_diff = sum(r["same_trial"] != "True" for r in rows)
deltas = [float(r["auc_sel_auc"]) - float(r["auc_sel_acc"]) for r in rows]
print(f"Cells where the two rules chose a different trial: {n_diff}/{len(rows)}")
print(f"Test-AUC delta (val_auc-selected minus val_acc-selected):")
print(f"  mean {sum(deltas)/len(deltas):+.4f}   "
      f"min {min(deltas):+.4f}   max {max(deltas):+.4f}   "
      f"|delta|>0.01 in {sum(abs(d) > 0.01 for d in deltas)}/{len(deltas)} cells")

print(f"\n{'head':10s} {'rank rho':>9s} {'Q-C (val_acc)':>14s} {'Q-C (val_auc)':>14s} {'sign flip':>10s}")
flips = 0
for h in HEAD_ORDER:
    recs = by_head[h]
    a = [x[1] for x in recs]
    b = [x[2] for x in recs]
    rho = stats.spearmanr(a, b).statistic
    def margin(idx):
        q = max(x[idx] for x in recs if x[0] in QUANTUM)
        c = max(x[idx] for x in recs if x[0] not in QUANTUM)
        return q - c
    m_acc, m_auc = margin(1), margin(2)
    flip = (m_acc > 0) != (m_auc > 0)
    flips += flip
    print(f"{h:10s} {rho:9.3f} {m_acc:+14.4f} {m_auc:+14.4f} {'YES' if flip else 'no':>10s}")
print(f"\nCamp-margin sign flips: {flips}/5 heads")

# The paper's specific low-capacity claim: quantum over the linear random
# projection at MLP-1 and MLP-2 on BreastMNIST.
print("\nQuantum minus random_45, the paper's low-capacity claim:")
for h in ("mlp1", "mlp2"):
    recs = {x[0]: (x[1], x[2]) for x in by_head[h]}
    r45 = recs["random_45"]
    bq = max((v for k, v in recs.items() if k in QUANTUM), key=lambda t: t[0])
    bq2 = max((v for k, v in recs.items() if k in QUANTUM), key=lambda t: t[1])
    print(f"  {h}: val_acc-selected {bq[0] - r45[0]:+.4f}   "
          f"val_auc-selected {bq2[1] - r45[1]:+.4f}")
