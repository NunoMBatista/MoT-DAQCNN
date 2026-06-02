"""Is the capacity-sweep random baseline (fixed at projection seed 42) biased at
the TRAINED heads, the way it is at the probe? Retrain the linear and CNN-64
heads on random_45 features built from several projection seeds, using the
stored best HPs of the random_45 cell, and report the spread of the 10-seed-mean
test AUC across projection seeds. Small spread -> the capacity figure is robust
to the projection seed; large spread -> the random baseline needs seed-averaging.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import experiments.head_capacity_sweep as hcs

DATASET = sys.argv[1] if len(sys.argv) > 1 else "breast_mnist"
PROJ_SEEDS = [42, 0, 1, 2, 3]
TRAIN_SEEDS = list(range(10))
HEADS = ["linear", "cnn_large"]

hcs.DATASET = DATASET
hcs.FEATURE_SOURCES = hcs.build_feature_sources(DATASET)
cells = f"outputs/paper_results/capacity_sweep/{DATASET}/cells"

print(f"[{DATASET}] random_45 trained-head seed sensitivity "
      f"(proj seeds {PROJ_SEEDS}, {len(TRAIN_SEEDS)} train seeds/cell)")
for head in HEADS:
    bp = json.load(open(f"{cells}/{head}__random_45/summary.json"))["best_params"]
    builder, use_sched = hcs._head_factory(head)
    proj_means = []
    for ps in PROJ_SEEDS:
        spec = {"kind": "random", "n_filters": 45, "kernel_size": 3, "stride": 3, "seed": ps}
        (train, val, test, in_ch) = hcs.load_features("random_45", spec)
        aucs = []
        for ts in TRAIN_SEEDS:
            h = builder(hcs.NUM_CLASSES, bp["dropout"], hcs.ACTIVATION, in_ch)
            m = hcs.train_head(h, train, val, test, lr=bp["lr"], weight_decay=bp["wd"],
                               use_scheduler=use_sched, device="cpu", seed=ts)
            aucs.append(m["test_auc"])
        proj_means.append(np.mean(aucs))
        print(f"  {head:10} proj_seed={ps:2d}  meanAUC={np.mean(aucs):.4f}")
    pm = np.array(proj_means)
    print(f"  {head:10} ACROSS PROJ SEEDS: mean={pm.mean():.4f}  std={pm.std():.4f}  "
          f"seed42={proj_means[0]:.4f}  (seed42 - mean = {proj_means[0]-pm.mean():+.4f})\n")
