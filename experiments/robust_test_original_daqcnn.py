import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.daqcnn_training import run_single_seed  # noqa: E402
from src.utils.plotting import (  # noqa: E402
    plot_multi_seed_loss_curves,
    plot_multi_seed_roc_curves,
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Robust multi-seed experiment runner for DAQCNN")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "breast_mnist", "original_daqcnn_best.yml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="MoT-DAQCNN")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=[])
    parser.add_argument("--wandb-notes", type=str, default=None)

    args = parser.parse_args()
    cfg = load_config(args.config)

    seed_spec = cfg.get("misc", {}).get("seed", 0)
    seeds = seed_spec if isinstance(seed_spec, list) else [seed_spec]

    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        if wandb is None:
            print("[wandb] wandb not installed. Continuing without W&B logging.")
        else:
            os.environ["WANDB_MODE"] = args.wandb_mode
            os.environ.setdefault("WANDB_SILENT", "true")
            run_name = f"robust_eval:original:{cfg.get('dataset', {}).get('name', 'unknown')}:{time.strftime('%Y%m%d_%H%M%S')}"
            try:
                wandb_run = wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=run_name,
                    tags=list(args.wandb_tags),
                    notes=args.wandb_notes,
                    config={"script": "robust_test_original_daqcnn.py", "cfg": cfg, "seeds": seeds},
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] Failed to init: {e}")
                wandb_run = None

    print("\n" + "=" * 60)
    print("Robust Experiment Runner — DAQCNN")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Seeds:  {seeds}")
    print("=" * 60 + "\n")

    os.makedirs("outputs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_slug = cfg.get("dataset", {}).get("name", "unknown").replace("_", "")
    output_dir = os.path.join("outputs", f"run_{dataset_slug}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    all_results = []
    for seed in seeds:
        result = run_single_seed(
            cfg, seed, output_dir, verbose=(len(seeds) == 1), set_seed_fn=set_seed
        )
        all_results.append(result)

        if wandb_run is not None and isinstance(result, dict):
            try:
                wandb_run.log(
                    {
                        "seed/seed": seed,
                        "seed/test_loss": result.get("test_loss"),
                        "seed/test_acc": result.get("test_acc"),
                        "seed/test_auc": result.get("test_auc"),
                        "seed/test_f1": result.get("test_f1"),
                        "seed/test_recall": result.get("test_recall"),
                    },
                    step=seeds.index(seed),
                )
            except Exception:
                pass

    # Save individual results
    with open(os.path.join(output_dir, "individual_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Aggregate metrics
    metric_keys = ["test_loss", "test_acc", "test_auc", "test_f1", "test_recall",
                   "final_train_loss", "final_val_loss", "final_train_acc", "final_val_acc"]
    aggregate_stats = {}
    for key in metric_keys:
        values = [r.get(key) for r in all_results if r.get(key) is not None]
        if values:
            aggregate_stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    with open(os.path.join(output_dir, "aggregate_metrics.json"), "w") as f:
        json.dump(aggregate_stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Performance Summary (n={len(seeds)} seeds)")
    print("=" * 60)
    for key, stats in aggregate_stats.items():
        print(f"{key:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("=" * 60)

    if len(seeds) > 1:
        all_train = [r["train_losses"] for r in all_results if r.get("train_losses")]
        all_val = [r["val_losses"] for r in all_results if r.get("val_losses")]
        if all_train and all_val:
            plot_multi_seed_loss_curves(
                all_train, all_val, os.path.join(output_dir, "loss_curve_multi_seed.png")
            )

        all_labels = [np.array(r["test_labels"]) for r in all_results if r.get("test_labels")]
        all_probs = [np.array(r["test_probs"]) for r in all_results if r.get("test_probs")]
        if all_labels and all_probs:
            plot_multi_seed_roc_curves(
                all_labels, all_probs,
                os.path.join(output_dir, "roc_curve_multi_seed.png"),
                num_classes=all_results[0].get("num_classes", 2),
            )

    if wandb_run is not None:
        try:
            for key, stats in aggregate_stats.items():
                wandb_run.summary[f"aggregate/{key}_mean"] = stats["mean"]
                wandb_run.summary[f"aggregate/{key}_std"] = stats["std"]
        except Exception:
            pass
        finally:
            try:
                wandb_run.finish()
            except Exception:
                pass

    print(f"\nAll outputs saved to: {output_dir}")
    print("Experiment complete!")


if __name__ == "__main__":
    main()
