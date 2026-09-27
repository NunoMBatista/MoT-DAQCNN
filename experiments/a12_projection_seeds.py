"""Re-run the classical projection cells with the random draw regenerated per seed.

Reviewer 4 (#6) asks how the RFF and random-filter seeds are handled and whether
the random maps are regenerated across the reported training seeds. The answer in
the published sweep is no: `RANDOM_FILTER_SEED = 42` is used for every head and
every training seed, so all ten "seeds" of a random_* or rff_* cell share one
projection. Their reported spread therefore contains head-initialisation and
shuffling noise but NO projection noise, which makes every quantum-minus-classical
margin against those arms look steadier than it is. CLAUDE.md already records that
seed 42 is a 5th-percentile low draw for random_45 at the probe, so the bias runs
in the quantum arm's favour.

The fix costs nothing extra. Rather than crossing several draws with ten seeds,
the draw is tied TO the seed: seed s drives the projection, the head
initialisation and the shuffling together. That is what "regenerated per seed"
means, and it keeps the protocol at ten runs per cell. The Optuna search is not
repeated; each cell reuses the best hyperparameters already found for it.

CONFOUND THIS SCRIPT AVOIDS. Its numbers must not be compared against the
published per-seed values, because the harness differs: in the original
`run_cell` the head was constructed from whatever ambient RNG state the 30-trial
Optuna study happened to leave, and `train_head` only reseeds afterwards, so head
initialisation was never controlled by the seed at all. Here the seed is set
before the head is built, which makes runs reproducible but not bit-identical to
the originals. The comparison that matters is therefore frozen-draw versus
per-seed-draw *within this script*, run with `--frozen-seed 42` and without it,
so the harness cancels and only the draw differs. Run both.

Writes one JSON per (head, source, seed) under the output directory, and
optionally the per-example test scores an example-level interval needs. Skips any
(cell, seed) already on disk, so it is safe to resume or top up.

    python experiments/a12_projection_seeds.py --dataset breast_mnist \
        --cells outputs/paper_results/capacity_sweep_std/breast_mnist \
        --out outputs/camera_ready/a12/breast_mnist_varying --save-predictions
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_capacity_sweep as hcs

# The five sources that draw a random projection. poly-2 has no free draw and
# the quantum kernels are deterministic, so neither is affected.
PROJECTION_SOURCES = ["random_9", "random_45", "random_180", "rff_45", "rff_180"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_mnist")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--cells", required=True,
                    help="existing sweep dir holding cells/<head>__<src>/summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", default=PROJECTION_SOURCES)
    ap.add_argument("--heads", nargs="+",
                    default=["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--frozen-seed", type=int, default=None,
                    help="control mode: use this one projection draw for every "
                         "training seed, reproducing the published protocol "
                         "inside this harness")
    ap.add_argument("--no-standardize", action="store_true",
                    help="skip z-scoring; the published sweep is standardised, "
                         "so this is for diagnosis only")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # head_capacity_sweep reads these as module globals at call time
    hcs.DATASET = args.dataset
    hcs.NUM_CLASSES = args.num_classes
    hcs.BATCH_SIZE = args.batch_size
    hcs.EPOCHS = args.epochs
    hcs.FEATURE_SOURCES = hcs.build_feature_sources(args.dataset)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    mode = "frozen" if args.frozen_seed is not None else "per-seed"
    print(f"{args.dataset}: {len(args.sources)} sources x {len(args.heads)} heads "
          f"x {len(args.seeds)} seeds, projection draw = {mode}", flush=True)

    built_draw, built = None, None   # cache the last feature build, see below

    for source in args.sources:
        spec = hcs.FEATURE_SOURCES[source]
        built_draw, built = None, None
        # best hyperparameters already found for each head on this source
        best = {}
        for head in args.heads:
            summary = Path(args.cells) / "cells" / f"{head}__{source}" / "summary.json"
            if not summary.exists():
                print(f"  SKIP {head}__{source}: no summary.json at {summary}", flush=True)
                continue
            best[head] = json.load(open(summary))["best_params"]
        if not best:
            continue

        for seed in args.seeds:
            # which draw this run uses: the frozen control, or the seed itself
            draw = args.frozen_seed if args.frozen_seed is not None else seed
            todo = [h for h in best
                    if not (out_root / f"{h}__{source}" / f"seed_{seed}.json").exists()]
            if not todo:
                continue

            # One feature build serves every head at this (source, draw), and
            # in frozen mode the draw never changes, so it is built once for all
            # ten seeds instead of ten times. On TissueMNIST that is 165k images
            # of conv or RFF per rebuild, so the cache is worth the four lines.
            # The published results are the STANDARDISED sweep, so z-scoring is
            # not optional here: without it the mlp1__random_45 cell lands at
            # 0.739 test AUC against a published 0.845.
            if draw != built_draw:
                train, val, test, in_ch = hcs.load_features(source, {**spec, "seed": draw})
                if not args.no_standardize:
                    train, val, test = hcs.standardize_features(train, val, test)
                built_draw, built = draw, (train, val, test, in_ch)
            train, val, test, in_ch = built
            print(f"  {source} draw={draw} seed={seed}: features {tuple(train[0].shape)}",
                  flush=True)

            for head_name in todo:
                builder, use_scheduler = hcs._head_factory(head_name)
                hp = best[head_name]
                # Seed BEFORE building the head so initialisation is controlled
                # too; the original protocol left it to the ambient RNG state.
                torch.manual_seed(seed)
                np.random.seed(seed)
                head = builder(args.num_classes, hp["dropout"], hcs.ACTIVATION, in_ch)
                metrics = hcs.train_head(
                    head, train, val, test,
                    lr=hp["lr"], weight_decay=hp["wd"],
                    use_scheduler=use_scheduler, device=args.device, seed=seed,
                    return_scores=args.save_predictions,
                )
                cell_dir = out_root / f"{head_name}__{source}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                if args.save_predictions:
                    np.savez_compressed(cell_dir / f"seed_{seed}_preds.npz",
                                        scores=metrics.pop("test_scores"),
                                        labels=metrics.pop("test_labels"))
                record = {"seed": seed, "projection_draw": draw, "head": head_name,
                          "source": source, "in_channels": int(in_ch),
                          "params": hp, **metrics}
                with open(cell_dir / f"seed_{seed}.json", "w") as fh:
                    json.dump(record, fh, indent=2)
                print(f"    {head_name:<10} val {metrics['val_acc']:.4f}  "
                      f"test AUC {metrics['test_auc']:.4f}", flush=True)

    print(f"done -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
