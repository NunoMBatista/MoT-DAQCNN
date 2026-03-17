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
from src.models.train_ts_moe import run_ts_moe_pipeline  # noqa: E402
from src.utils.plotting import (  # noqa: E402
    plot_multi_seed_loss_curves,
    plot_multi_seed_roc_curves,
)

# Optional W&B (kept optional so this script works without wandb installed)
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
    parser = argparse.ArgumentParser(description="Robust experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "pneumonia_mnist.yml"),
        help="Path to YAML config file",
    )

    # --- W&B (optional) ---
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable real-time logging to Weights & Biases (optional).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (user or team).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="MoT-DAQCNN",
        help="W&B project name (default: MoT-DAQCNN).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (online/offline/disabled).",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=[],
        help="Optional W&B tags.",
    )
    parser.add_argument(
        "--wandb-notes",
        type=str,
        default=None,
        help="Optional W&B notes/description.",
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

    # --- Initialize W&B (optional) ---
    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        if wandb is None:
            print(
                "[wandb] wandb is not installed. Install with: pip install wandb\n"
                "Continuing without W&B logging."
            )
        else:
            os.environ["WANDB_MODE"] = args.wandb_mode
            os.environ.setdefault("WANDB_SILENT", "true")

            # Build a small, safe config payload (W&B config prefers JSON-serializable types)
            wandb_config = {
                "script": "experiments/robust_test_original_daqcnn.py",
                "config_path": args.config,
                "architecture": architecture,
                "seeds": seeds,
                "n_seeds": len(seeds),
                "dataset.name": cfg.get("dataset", {}).get("name", "unknown"),
                "model.architecture": architecture,
            }

            # Try to add the full YAML config; if it contains non-serializable types,
            # W&B will raise, so we keep it nested and tolerate failures.
            wandb_config["cfg"] = cfg

            run_name = f"robust_eval:{architecture}:{wandb_config['dataset.name']}:{time.strftime('%Y%m%d_%H%M%S')}"
            try:
                wandb_run = wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=run_name,
                    tags=list(args.wandb_tags),
                    notes=args.wandb_notes,
                    config=wandb_config,
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] Failed to init W&B run: {e}\nContinuing without W&B.")
                wandb_run = None

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
    if architecture == "TS-MoE":
        prefix = "ts_moe"
    elif architecture == "gumbel":
        prefix = "gumbel"
    else:
        prefix = "run"
    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    dataset_slug = dataset_name.replace("_", "")
    output_dir = os.path.join("outputs", f"{prefix}_{dataset_slug}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    if wandb_run is not None:
        try:
            wandb_run.config.update({"output_dir": output_dir}, allow_val_change=True)
        except Exception:
            pass

    # Save config to output directory
    config_save_path = os.path.join(output_dir, "config.yml")
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}\n")

    # Run experiments for each seed
    all_results = []
    for seed in seeds:
        if wandb_run is not None:
            try:
                wandb_run.log(
                    {"seed/current": seed, "seed/index": seeds.index(seed)},
                    step=seeds.index(seed),
                )
            except Exception:
                pass

        if architecture == "TS-MoE":
            ts_result = run_ts_moe_pipeline(
                cfg, seed, output_dir, verbose=(len(seeds) == 1)
            )
            # Flatten into the same shape the aggregation code expects
            teacher = ts_result["teacher"]
            student = ts_result["student"]
            final = ts_result["final"]
            summary = ts_result["summary"]
            result = {
                "seed": seed,
                "architecture": "TS-MoE",
                # Final classifier metrics (kept for compatibility)
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
                # Per-phase histories for TS-MoE multi-seed plots
                "teacher_train_losses": teacher.get("train_losses", []),
                "teacher_val_losses": teacher.get("val_losses", []),
                "teacher_ce_losses": teacher.get("train_ce_losses", []),
                "teacher_ent_losses": teacher.get("train_ent_losses", []),
                "student_train_losses": student.get("train_losses", []),
                "student_val_losses": student.get("val_losses", []),
                "final_train_losses": final.get("train_losses", []),
                "final_val_losses": final.get("val_losses", []),
                # Explicit skip flags (helps explain null/empty entries)
                "student_skipped": bool(student.get("skipped", False)),
                "student_skip_reason": student.get("skip_reason"),
                "final_skipped": bool(final.get("skipped", False)),
                "final_skip_reason": final.get("skip_reason"),
                # Extra TS-MoE specific fields
                "teacher_test_acc": summary.get("teacher_test_acc"),
                "student_agreement": summary.get("student_agreement"),
                "speedup_factor": summary.get("speedup_factor"),
                "pipeline_time_s": summary.get("pipeline_time_s"),
            }
        elif architecture == "gumbel":
            from src.models.gumbel_moe_training import run_gumbel_moe

            result = run_gumbel_moe(
                cfg, seed, output_dir, verbose=(len(seeds) == 1), set_seed_fn=set_seed
            )
        else:
            result = run_single_seed(
                cfg, seed, output_dir, verbose=(len(seeds) == 1), set_seed_fn=set_seed
            )

        all_results.append(result)

        # Log per-seed summary metrics (real-time at seed granularity)
        if wandb_run is not None and isinstance(result, dict):
            try:
                seed_step = seeds.index(seed)
                wandb_run.log(
                    {
                        "seed/seed": seed,
                        "seed/test_loss": result.get("test_loss"),
                        "seed/test_acc": result.get("test_acc"),
                        "seed/test_auc": result.get("test_auc"),
                        "seed/test_f1": result.get("test_f1"),
                        "seed/test_recall": result.get("test_recall"),
                        "seed/final_train_loss": result.get("final_train_loss"),
                        "seed/final_val_loss": result.get("final_val_loss"),
                        "seed/final_train_acc": result.get("final_train_acc"),
                        "seed/final_val_acc": result.get("final_val_acc"),
                        "seed/pipeline_time_s": result.get("pipeline_time_s"),
                        "seed/speedup_factor": result.get("speedup_factor"),
                        "seed/student_agreement": result.get("student_agreement"),
                        "seed/teacher_test_acc": result.get("teacher_test_acc"),
                    },
                    step=seed_step,
                )
            except Exception:
                pass

            # Optional: log learning curves for this seed (epoch granularity)
            try:
                train_losses = result.get("train_losses") or []
                val_losses = result.get("val_losses") or []
                max_len = max(len(train_losses), len(val_losses))
                for epoch_idx in range(max_len):
                    wandb_run.log(
                        {
                            "epoch/seed": seed,
                            "epoch/epoch": epoch_idx + 1,
                            "epoch/train_loss": train_losses[epoch_idx]
                            if epoch_idx < len(train_losses)
                            else None,
                            "epoch/val_loss": val_losses[epoch_idx]
                            if epoch_idx < len(val_losses)
                            else None,
                        }
                    )
            except Exception:
                pass

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

        # TS-MoE: also save per-phase multi-seed loss curves
        if architecture == "TS-MoE":
            phase_specs = [
                (
                    "teacher",
                    "teacher_train_losses",
                    "teacher_val_losses",
                    "teacher_loss_curve_multi_seed.png",
                ),
                (
                    "student",
                    "student_train_losses",
                    "student_val_losses",
                    "student_loss_curve_multi_seed.png",
                ),
                (
                    "final classifier",
                    "final_train_losses",
                    "final_val_losses",
                    "final_loss_curve_multi_seed.png",
                ),
            ]

            for phase_name, train_key, val_key, filename in phase_specs:
                phase_train = [r[train_key] for r in all_results if r.get(train_key)]
                phase_val = [r[val_key] for r in all_results if r.get(val_key)]
                if phase_train and phase_val:
                    phase_plot_path = os.path.join(output_dir, filename)
                    # For the teacher phase, also pass CE and entropy components
                    extra_kwargs = {}
                    if phase_name == "teacher":
                        phase_ce = [
                            r["teacher_ce_losses"]
                            for r in all_results
                            if r.get("teacher_ce_losses")
                        ]
                        phase_ent = [
                            r["teacher_ent_losses"]
                            for r in all_results
                            if r.get("teacher_ent_losses")
                        ]
                        if phase_ce:
                            extra_kwargs["all_ce_losses"] = phase_ce
                        if phase_ent:
                            extra_kwargs["all_ent_losses"] = phase_ent
                    plot_multi_seed_loss_curves(
                        phase_train, phase_val, phase_plot_path, **extra_kwargs
                    )
                    print(
                        f"TS-MoE {phase_name} multi-seed loss curve saved to: "
                        f"{phase_plot_path}\n"
                    )

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

    # Finalize W&B run (if enabled)
    if wandb_run is not None:
        try:
            # Put aggregate metrics into W&B summary for easy comparison
            for metric_name, stats in aggregate_stats.items():
                if metric_name in ("model_metadata",):
                    continue
                if not isinstance(stats, dict) or "mean" not in stats:
                    continue
                wandb_run.summary[f"aggregate/{metric_name}_mean"] = stats.get("mean")
                wandb_run.summary[f"aggregate/{metric_name}_std"] = stats.get("std")

            # Upload key result files and plots from the output directory
            wandb_run.save(os.path.join(output_dir, "config.yml"))
            wandb_run.save(os.path.join(output_dir, "individual_results.json"))
            wandb_run.save(os.path.join(output_dir, "aggregate_metrics.json"))
            for fname in os.listdir(output_dir):
                if fname.endswith(".png"):
                    wandb_run.save(os.path.join(output_dir, fname))
        except Exception:
            pass
        finally:
            try:
                wandb_run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
