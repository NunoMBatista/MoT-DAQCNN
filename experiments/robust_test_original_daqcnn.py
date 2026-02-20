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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_ts_moe_for_seed(cfg, seed, output_dir, verbose=True):
    """Run the full TS-MoE pipeline (Teacher → Student → Final Classifier)
    for a single seed and return a results dict compatible with the
    aggregation logic used for the original DAQCNN path."""

    from src.models.train_ts_moe import run_ts_moe_pipeline

    result = run_ts_moe_pipeline(cfg, seed, output_dir, verbose=verbose)

    # Flatten into the same shape the aggregation code expects
    final = result["final"]
    summary = result["summary"]

    return {
        "seed": seed,
        "architecture": "TS-MoE",
        "train_losses": final.get("train_losses", []),
        "val_losses": final.get("val_losses", []),
        "train_accs": final.get("train_accs", []),
        "val_accs": final.get("val_accs", []),
        "test_loss": final.get("test_loss", 0.0),
        "test_acc": final.get("test_acc", 0.0),
        "test_auc": final.get("test_auc", 0.0),
        "test_f1": final.get("test_f1", 0.0),
        "test_recall": final.get("test_recall", 0.0),
        "test_probs": final.get("test_probs", []),
        "test_labels": final.get("test_labels", []),
        "test_confusion_matrix": final.get("test_confusion_matrix", []),
        "num_classes": final.get("num_classes", 2),
        "final_train_loss": final.get("final_train_loss", 0.0),
        "final_val_loss": final.get("final_val_loss", 0.0),
        "final_train_acc": final.get("final_train_acc", 0.0),
        "final_val_acc": final.get("final_val_acc", 0.0),
        # Extra TS-MoE specific fields
        "teacher_test_acc": summary.get("teacher_test_acc"),
        "student_agreement": summary.get("student_agreement"),
        "speedup_factor": summary.get("speedup_factor"),
        "pipeline_time_s": summary.get("pipeline_time_s"),
    }


def main():
    parser = argparse.ArgumentParser(description="Robust experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "pneumonia_mnist.yml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Detect architecture
    architecture = cfg.get("model", {}).get("architecture", "original")

    # Get seeds - can be single value or list
    seed_spec = cfg.get("misc", {}).get("seed", 0)
    if isinstance(seed_spec, list):
        seeds = seed_spec
    else:
        seeds = [seed_spec]

    print("\n" + "=" * 60)
    print("Robust Experiment Runner")
    print("=" * 60)
    print(f"Architecture: {architecture}")
    print(f"Config: {args.config}")
    print(f"Seeds: {seeds}")
    print(f"Number of runs: {len(seeds)}")
    print(f"{'=' * 60}\n")

    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = "ts_moe" if architecture == "TS-MoE" else "run"
    output_dir = os.path.join("outputs", f"{prefix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Save config to output directory
    config_save_path = os.path.join(output_dir, "config.yml")
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}\n")

    # Run experiments for each seed
    all_results = []
    for seed in seeds:
        if architecture == "TS-MoE":
            result = run_ts_moe_for_seed(
                cfg, seed, output_dir, verbose=(len(seeds) == 1)
            )
        else:
            result = run_single_seed(
                cfg, seed, output_dir, verbose=(len(seeds) == 1), set_seed_fn=set_seed
            )
        all_results.append(result)

    # Aggregate results
    print(f"\n{'=' * 60}")
    print(f"Summary of All Runs ({architecture})")
    print(f"{'=' * 60}\n")

    # Save individual results
    individual_results_path = os.path.join(output_dir, "individual_results.json")
    with open(individual_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Individual results saved to: {individual_results_path}")

    # Compute and save aggregate metrics
    metric_keys = [
        "test_loss",
        "test_acc",
        "test_auc",
        "test_f1",
        "test_recall",
        "final_train_loss",
        "final_val_loss",
        "final_train_acc",
        "final_val_acc",
    ]

    metrics = {}
    for key in metric_keys:
        values = [r.get(key) for r in all_results if r.get(key) is not None]
        if values:
            metrics[key] = values

    aggregate_stats = {}
    for metric_name, values in metrics.items():
        aggregate_stats[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    # Add model metadata from first result if available
    if all_results and "model_metadata" in all_results[0]:
        aggregate_stats["model_metadata"] = all_results[0]["model_metadata"]

    # Add TS-MoE specific aggregate metrics
    if architecture == "TS-MoE":
        for key in ["teacher_test_acc", "student_agreement", "pipeline_time_s"]:
            values = [r.get(key) for r in all_results if r.get(key) is not None]
            if values:
                aggregate_stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }

    aggregate_stats_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(aggregate_stats_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Aggregate metrics saved to: {aggregate_stats_path}\n")

    # Print summary
    print(f"{'=' * 60}")
    print(f"Performance Summary (n={len(seeds)} runs)")
    print(f"{'=' * 60}")
    for metric_name, stats in aggregate_stats.items():
        # Skip non-metric entries
        if metric_name in ("model_metadata",):
            continue
        if not isinstance(stats, dict) or "mean" not in stats:
            continue
        print(f"{metric_name:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"{'=' * 60}\n")

    # Plot multi-seed loss curves if multiple seeds
    if len(seeds) > 1:
        all_train_losses = [
            r["train_losses"] for r in all_results if r.get("train_losses")
        ]
        all_val_losses = [r["val_losses"] for r in all_results if r.get("val_losses")]
        if all_train_losses and all_val_losses:
            multi_seed_plot_path = os.path.join(output_dir, "loss_curve_multi_seed.png")
            plot_multi_seed_loss_curves(
                all_train_losses, all_val_losses, multi_seed_plot_path
            )
            print(f"Multi-seed loss curve saved to: {multi_seed_plot_path}\n")

        # Plot multi-seed ROC curves
        all_test_labels = [
            np.array(r["test_labels"]) for r in all_results if r.get("test_labels")
        ]
        all_test_probs = [
            np.array(r["test_probs"]) for r in all_results if r.get("test_probs")
        ]
        if all_test_labels and all_test_probs:
            num_classes = all_results[0].get("num_classes", 2)
            multi_roc_plot_path = os.path.join(output_dir, "roc_curve_multi_seed.png")
            plot_multi_seed_roc_curves(
                all_test_labels,
                all_test_probs,
                multi_roc_plot_path,
                num_classes=num_classes,
            )
            print(f"Multi-seed ROC curve saved to: {multi_roc_plot_path}\n")

    print(f"All outputs saved to: {output_dir}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
