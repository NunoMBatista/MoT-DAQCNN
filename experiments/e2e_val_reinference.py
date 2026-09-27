"""Recompute validation metrics for every end-to-end validation seed from its
saved checkpoint, so the camp representative in Table III can be selected on a
TEN-SEED-MEAN validation metric instead of the single search run's val_acc.

Why. `val_selected_e2e.py` selects each camp's representative by
`search_value`, the validation accuracy of the one Optuna search run that won,
a max over 200 trials on 78 (Breast) or 524 (Pneumonia) images. That selector is
quantized (1/78) and routinely tied. But every validated seed left a
`best_model_seed_k.pt` checkpoint, and the validation split is deterministic, so
a ten-seed-mean validation AUC and accuracy are one forward pass away. That is
the same rule A2 applied to the capacity sweep.

Control planted before any number is read: the same checkpoint is also run on
the TEST split and its AUC must reproduce `validation_summary.json`'s stored
per-seed `test_auc.values` entry (the runner restored the best checkpoint before
its test evaluation, so they must agree to float precision). A row whose test
AUCs do not reproduce is flagged and the selector refuses it.

Bonus: per-example test probabilities are saved per seed, which is what the
example-level intervals R4 asked for (A7) need and which were never kept.

TissueMNIST is REFUSED here. Its quantum caches are 22-46 GB compressed and
must never be opened with np.load on this machine (OOM, 2026-09-19); the cache
loader used by the end-to-end runner does exactly that. Tissue's selection is
unambiguous under the existing rule (validation and test pick the same rows).

    python experiments/e2e_val_reinference.py --datasets BreastMNIST PneumoniaMNIST
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

from paired_seed_intervals import E2E                       # noqa: E402
from src.models.daqcnn import DAQCNN                        # noqa: E402
from src.models.classical_baseline import ClassicalBaselineCNN  # noqa: E402
from src.utils.data import get_dataloaders                  # noqa: E402
from src.utils.evaluate import evaluate                     # noqa: E402
from src.utils.quantum_dataset_cache import (               # noqa: E402
    find_cached_quantum_dataset, load_cached_quantum_dataset)

OUT = os.path.join(ROOT, "outputs", "camera_ready", "e2e_val_reinference")
DEVICE = "cpu"


def build_loaders_and_factory(cfg):
    """Return (val_loader, test_loader, make_model) for one validated config.

    Mirrors run_single_seed (quantum / cached rows) and run_classical_baseline
    (random, trainable, raw rows) in src/models, minus training.
    """
    cfg = json.loads(json.dumps(cfg))          # private copy
    cfg["dataset"]["num_workers"] = 0
    model_cfg = cfg["model"]
    # The May-2026 quantum runs predate the `architecture` key; a topology list
    # is present exactly when the run built a DAQCNN (cached path).
    is_daqcnn = model_cfg.get("architecture") == "original" or \
        model_cfg.get("kernel_topology_names") is not None

    if is_daqcnn:
        # Quantum rows and the poly-2 / RFF rows, which live in classical
        # caches with a synthetic topology name and go through the same path.
        cache = find_cached_quantum_dataset(cfg)
        if cache is None:
            raise FileNotFoundError(f"no cache for {model_cfg['kernel_topology_names']}")
        _, val_loader, test_loader, n_classes, meta = load_cached_quantum_dataset(
            cache, cfg["dataset"]["batch_size"], 0,
            requested_kernels=model_cfg.get("kernel_topology_names"))
        num_classes = model_cfg.get("num_classes", n_classes)

        def make_model():
            m = DAQCNN(
                num_classes=num_classes,
                kernel_size=model_cfg.get("kernel_size", 2),
                stride=model_cfg.get("stride", 1),
                kernel_topology_names=model_cfg.get("kernel_topology_names"),
                scaling_factor=model_cfg.get("scaling_factor", 1.0),
                evolution_time=model_cfg.get("evolution_time", 0.2),
                mode=model_cfg.get("mode", "trotter"),
                dropout=model_cfg.get("dropout", 0.1),
                activation=model_cfg.get("activation", "relu"),
                quantum_device=model_cfg.get("quantum_device", "default.qubit"),
                quantum_device_kwargs=model_cfg.get("quantum_device_kwargs"),
                classical_device=DEVICE,
                in_channels=model_cfg.get("in_channels", 1),
                interface=model_cfg.get("interface", "torch"),
                include_correlators=model_cfg.get("include_correlators", False),
                head_hidden_channels=model_cfg.get("head_hidden_channels", 64),
                override_quantum_out_channels=meta.get("out_channels"),
            )
            m.bypass_quantum = True
            return m
        return val_loader, test_loader, make_model

    # Classical baseline rows: random / trainable filters, raw pixels.
    _, val_loader, test_loader, n_classes = get_dataloaders(cfg)
    num_classes = model_cfg.get("num_classes", n_classes)

    def make_model():
        return ClassicalBaselineCNN(
            num_classes=num_classes,
            in_channels=model_cfg.get("in_channels", 1),
            kernel_size=model_cfg.get("kernel_size", 3),
            stride=model_cfg.get("stride", 3),
            out_channels=model_cfg.get("out_channels", 180),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            head_hidden_channels=model_cfg.get("head_hidden_channels", 64),
            fixed_random_filters=model_cfg.get("fixed_random_filters", False),
            raw_features=model_cfg.get("raw_features", False),
        )
    return val_loader, test_loader, make_model


def load_checkpoint_into(model, path, probe_batch):
    """Materialize any lazy layers with one forward pass, then load weights."""
    model.eval()
    with torch.no_grad():
        model(probe_batch)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    return model


def run_row(dataset, model_name, run_dir):
    val_dirs = glob.glob(os.path.join(run_dir, "validation", "validation_rank1_trial*"))
    assert len(val_dirs) == 1, val_dirs
    val_dir = val_dirs[0]
    cfg = yaml.safe_load(open(os.path.join(val_dir, "config.yml")))
    stored = json.load(open(os.path.join(run_dir, "validation", "validation_summary.json")))[0]
    stored_auc = stored["test_auc"]["values"]
    stored_acc = stored["test_acc"]["values"]

    val_loader, test_loader, make_model = build_loaders_and_factory(cfg)
    probe = next(iter(val_loader))[0]

    seed_dirs = sorted(glob.glob(os.path.join(val_dir, "seed_*")),
                       key=lambda p: int(p.rsplit("_", 1)[1]))
    rows = []
    preds = {}
    for i, sd in enumerate(seed_dirs):
        seed = int(sd.rsplit("_", 1)[1])
        ck = os.path.join(sd, f"best_model_seed_{seed}.pt")
        model = load_checkpoint_into(make_model(), ck, probe)
        v = evaluate(model, val_loader, DEVICE, "Val", compute_full_metrics=True)
        t = evaluate(model, test_loader, DEVICE, "Test", compute_full_metrics=True)
        # The stored per-seed lists are in seed order 0..n-1. The stored AUCs
        # were computed on a GPU; a CPU float32 conv can reorder one near-tied
        # pair, which moves a binary AUC by exactly 1/(n_pos*n_neg) (Breast:
        # 2.09e-4). Accept up to two such flips; accuracy must be exact.
        y = np.asarray(t["labels"])
        granule = 1.0 / (y.sum() * (len(y) - y.sum())) if len(np.unique(y)) == 2 else 1e-4
        auc_diff = abs(t["auc"] - stored_auc[i])
        match = auc_diff <= 2 * granule + 1e-9 and abs(t["accuracy"] - stored_acc[i]) < 1e-6
        rows.append(dict(seed=seed, val_acc=v["accuracy"], val_auc=v["auc"],
                         val_loss=v["loss"], test_acc=t["accuracy"], test_auc=t["auc"],
                         test_auc_stored=stored_auc[i], test_acc_stored=stored_acc[i],
                         auc_diff_granules=float(auc_diff / granule),
                         control_match=bool(match)))
        preds[f"seed_{seed}_test_probs"] = np.asarray(t["probs"])
        preds[f"seed_{seed}_test_labels"] = np.asarray(t["labels"])
        preds[f"seed_{seed}_val_probs"] = np.asarray(v["probs"])
        preds[f"seed_{seed}_val_labels"] = np.asarray(v["labels"])
        print(f"    seed {seed}: val_acc {v['accuracy']:.4f} val_auc {v['auc']:.4f} | "
              f"test_auc {t['auc']:.4f} stored {stored_auc[i]:.4f} "
              f"{'OK' if match else 'MISMATCH'}", flush=True)

    out = dict(dataset=dataset, model=model_name, run_dir=run_dir,
               search_value=stored["search_value"], n_seeds=len(rows),
               all_controls_match=all(r["control_match"] for r in rows),
               mean_val_acc=float(np.mean([r["val_acc"] for r in rows])),
               mean_val_auc=float(np.mean([r["val_auc"] for r in rows])),
               mean_test_auc=float(np.mean([r["test_auc"] for r in rows])),
               seeds=rows)
    tag = f"{dataset}__{model_name.replace(' ', '_').replace('/', '_')}"
    with open(os.path.join(OUT, tag + ".json"), "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(os.path.join(OUT, tag + "_preds.npz"), **preds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["BreastMNIST", "PneumoniaMNIST"])
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these row names")
    args = ap.parse_args()
    if any("Tissue" in d for d in args.datasets):
        raise SystemExit("TissueMNIST refused: its caches must not be loaded on this machine.")
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)

    for (dataset, model_name), run_dir in E2E.items():
        if dataset not in args.datasets:
            continue
        if args.models and model_name not in args.models:
            continue
        tag = f"{dataset}__{model_name.replace(' ', '_').replace('/', '_')}"
        if os.path.exists(os.path.join(OUT, tag + ".json")):
            print(f"skip {tag} (done)", flush=True)
            continue
        print(f"\n== {dataset} / {model_name}", flush=True)
        out = run_row(dataset, model_name, os.path.join(ROOT, run_dir))
        print(f"   mean val_acc {out['mean_val_acc']:.4f}  mean val_auc {out['mean_val_auc']:.4f}  "
              f"mean test_auc {out['mean_test_auc']:.4f}  controls "
              f"{'ALL OK' if out['all_controls_match'] else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
