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


def main():
    parser = argparse.ArgumentParser(description="Robust DAQCNN experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "pneumonia_mnist.yml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Get seeds - can be single value or list
    seed_spec = cfg.get("misc", {}).get("seed", 0)
    if isinstance(seed_spec, list):
        seeds = seed_spec
    else:
        seeds = [seed_spec]

    print(f"\n{'=' * 60}")
    print(f"Robust DAQCNN Experiment")
    print(f"{'=' * 60}")
    print(f"Config: {args.config}")
    print(f"Seeds: {seeds}")
    print(f"Number of runs: {len(seeds)}")
    print(f"{'=' * 60}\n")

    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"run_{timestamp}")
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
        result = run_single_seed(
            cfg, seed, output_dir, verbose=(len(seeds) == 1), set_seed_fn=set_seed
        )
        all_results.append(result)

    # Aggregate results
    print(f"\n{'=' * 60}")
    print(f"Summary of All Runs")
    print(f"{'=' * 60}\n")

    # Save individual results
    individual_results_path = os.path.join(output_dir, "individual_results.json")
    with open(individual_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Individual results saved to: {individual_results_path}")

    # Compute and save aggregate metrics
    metrics = {
        "test_loss": [r["test_loss"] for r in all_results],
        "test_acc": [r["test_acc"] for r in all_results],
        "test_auc": [r["test_auc"] for r in all_results],
        "test_f1": [r["test_f1"] for r in all_results],
        "test_recall": [r["test_recall"] for r in all_results],
        "final_train_loss": [r["final_train_loss"] for r in all_results],
        "final_val_loss": [r["final_val_loss"] for r in all_results],
        "final_train_acc": [r["final_train_acc"] for r in all_results],
        "final_val_acc": [r["final_val_acc"] for r in all_results],
    }

    aggregate_stats = {}
    for metric_name, values in metrics.items():
        aggregate_stats[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    # Add model metadata (same for all seeds, so use first result)
    if all_results and "model_metadata" in all_results[0]:
        aggregate_stats["model_metadata"] = all_results[0]["model_metadata"]

    aggregate_stats_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(aggregate_stats_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Aggregate metrics saved to: {aggregate_stats_path}\n")

    # Print summary
    print(f"{'=' * 60}")
    print(f"Performance Summary (n={len(seeds)} runs)")
    print(f"{'=' * 60}")
    for metric_name, stats in aggregate_stats.items():
        # Skip model_metadata (it's not a metric)
        if metric_name == "model_metadata":
            continue
        print(f"{metric_name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"{'=' * 60}\n")

    # Plot multi-seed loss curves if multiple seeds
    if len(seeds) > 1:
        all_train_losses = [r["train_losses"] for r in all_results]
        all_val_losses = [r["val_losses"] for r in all_results]
        multi_seed_plot_path = os.path.join(output_dir, "loss_curve_multi_seed.png")
        plot_multi_seed_loss_curves(
            all_train_losses, all_val_losses, multi_seed_plot_path
        )
        print(f"Multi-seed loss curve saved to: {multi_seed_plot_path}\n")

        # Plot multi-seed ROC curves
        all_test_labels = [np.array(r["test_labels"]) for r in all_results]
        all_test_probs = [np.array(r["test_probs"]) for r in all_results]
        num_classes = all_results[0]["num_classes"]
        multi_roc_plot_path = os.path.join(output_dir, "roc_curve_multi_seed.png")
        plot_multi_seed_roc_curves(
            all_test_labels,
            all_test_probs,
            multi_roc_plot_path,
            num_classes=num_classes,
        )
        print(f"Multi-seed ROC curve saved to: {multi_roc_plot_path}\n")

    print(f"All outputs saved to: {output_dir}")
    print(f"\nExperiment complete!")


if __name__ == "__main__":
    main()
