"""Re-run the ten-seed validation of the end-to-end "Random x45 / x180" rows with
the random filters actually FROZEN.

What went wrong (verified 2026-09-20 from the checkpoints). The search configs
for these rows set `model.fixed_random_filters: {type: fixed, value: true}`.
`build_trial_config` applies fixed entries during the 200-trial search, so the
search selected hyperparameters for a frozen-filter model. `validate_top_k`
then re-applies only `trial.params`, which never contains a fixed entry, and
builds the validation config from `base_config.yml`, which has no such key. The
ten reported seeds were therefore trained with `fixed_random_filters=False`:
every final checkpoint's Adam state holds an update slot for `conv.weight`.
TissueMNIST is unaffected (its base configs carry the flag).

The repair is the step that went wrong, nothing more: keep the searched
hyperparameters, set the flag, run the same ten seeds through the same runner.

Controls planted before any number is read:
  * `--control` first trains one seed WITHOUT the flag in this harness and
    asserts Adam DID update the filters, so the instrument can tell the two
    cases apart.
  * Every real seed asserts the opposite from its own checkpoint: no Adam slot
    of the conv weight's shape.

Writes under outputs/camera_ready/e2e_random_fixed/ (never paper_results).

    python experiments/e2e_random_fixed_revalidate.py --dataset BreastMNIST --row "Random x45" --control
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from paired_seed_intervals import E2E                                   # noqa: E402
from hyperparameter_search import set_seed                              # noqa: E402
from src.models.classical_baseline_training import run_classical_baseline  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "camera_ready", "e2e_random_fixed")


def adam_touched_conv(final_ckpt_path):
    """True if the optimizer state holds a slot shaped like conv.weight."""
    f = torch.load(final_ckpt_path, map_location="cpu", weights_only=False)
    conv_shape = tuple(f["model_state_dict"]["conv.weight"].shape)
    slots = {tuple(v["exp_avg"].shape) for v in f["optimizer_state_dict"]["state"].values()
             if "exp_avg" in v}
    return conv_shape in slots


def load_cfg(dataset, row):
    run_dir = os.path.join(ROOT, E2E[(dataset, row)])
    val_dir = glob.glob(os.path.join(run_dir, "validation", "validation_rank1_trial*"))
    assert len(val_dir) == 1, val_dir
    cfg = yaml.safe_load(open(os.path.join(val_dir[0], "config.yml")))
    cfg["dataset"]["num_workers"] = 0
    cfg["model"]["classical_device"] = "cpu"
    return cfg, run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--row", required=True, help='"Random x45" or "Random x180"')
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--control", action="store_true",
                    help="first train seed 0 WITHOUT the flag and assert Adam updated the filters")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg, run_dir = load_cfg(args.dataset, args.row)
    tag = f"{args.dataset}__{args.row.replace(' ', '_')}"
    out_dir = os.path.join(OUT, tag)
    os.makedirs(out_dir, exist_ok=True)

    if args.control:
        cdir = os.path.join(out_dir, "control_trainable_seed0")
        if not os.path.exists(os.path.join(cdir, "final_model_seed_0.pt")):
            ccfg = json.loads(json.dumps(cfg))
            ccfg["model"]["fixed_random_filters"] = False
            os.makedirs(cdir, exist_ok=True)
            r = run_classical_baseline(ccfg, 0, cdir, verbose=False, set_seed_fn=set_seed)
            json.dump({k: r[k] for k in ("test_acc", "test_auc", "val_acc", "val_auc")},
                      open(os.path.join(cdir, "result.json"), "w"))
        assert adam_touched_conv(os.path.join(cdir, "final_model_seed_0.pt")), \
            "control failed: trainable run shows no Adam slot for conv.weight"
        print(f"[{tag}] control OK: without the flag Adam updates conv.weight", flush=True)

    cfg["model"]["fixed_random_filters"] = True
    results = []
    for seed in args.seeds:
        sdir = os.path.join(out_dir, f"seed_{seed}")
        rpath = os.path.join(sdir, "result.json")
        if os.path.exists(rpath):
            results.append(json.load(open(rpath)))
            continue
        os.makedirs(sdir, exist_ok=True)
        r = run_classical_baseline(cfg, seed, sdir, verbose=False, set_seed_fn=set_seed)
        frozen = not adam_touched_conv(os.path.join(sdir, f"final_model_seed_{seed}.pt"))
        rec = {k: r[k] for k in ("seed", "val_acc", "val_auc", "test_acc", "test_auc", "test_f1")}
        rec["frozen_verified"] = bool(frozen)
        np.savez_compressed(os.path.join(sdir, "test_preds.npz"),
                            probs=np.asarray(r["test_probs"]), labels=np.asarray(r["test_labels"]))
        json.dump(rec, open(rpath, "w"), indent=2)
        results.append(rec)
        print(f"[{tag}] seed {seed}: test_acc {rec['test_acc']:.4f} test_auc {rec['test_auc']:.4f} "
              f"val_auc {rec['val_auc']:.4f} frozen={'yes' if frozen else 'NO'}", flush=True)

    acc = [r["test_acc"] for r in results]
    auc = [r["test_auc"] for r in results]
    summary = dict(dataset=args.dataset, row=args.row, source_run=run_dir, n_seeds=len(results),
                   all_frozen=all(r["frozen_verified"] for r in results),
                   test_acc_mean=float(np.mean(acc)), test_acc_std=float(np.std(acc)),
                   test_auc_mean=float(np.mean(auc)), test_auc_std=float(np.std(auc)),
                   seeds=results)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(f"[{tag}] DONE acc {summary['test_acc_mean']:.4f}+-{summary['test_acc_std']:.4f}  "
          f"auc {summary['test_auc_mean']:.4f}+-{summary['test_auc_std']:.4f}  "
          f"all_frozen={summary['all_frozen']}", flush=True)


if __name__ == "__main__":
    main()
