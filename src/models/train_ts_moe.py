"""
Unified TS-MoE training script.

Chains all three phases of the Teacher-Student Mixture-of-Topologies pipeline:
    Phase 1: Train Teacher (Grouped SE routing on cached quantum features)
    Phase 2: Train Student Gatekeeper (distillation from Teacher)
    Phase 3: Train Final Classifier (on sparse routed features)

Usage:
    python -m src.models.train_ts_moe --config configs/pneumonia_mnist_ts_moe.yml
"""

import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.student_training import run_student_training  # noqa: E402
from src.models.teacher_moe_training import run_teacher_training  # noqa: E402
from src.models.ts_moe_classification_head_training import (
    run_final_classifier_training,  # noqa: E402
)
from src.utils.plotting import plot_multi_seed_roc_curves  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_ts_moe_pipeline(
    cfg,
    seed,
    output_dir,
    verbose=True,
    datasets_dir=None,
    raw_image_datasets=None,
):
    """Run the full TS-MoE pipeline for a single seed.

    Chains Teacher → Student → Final Classifier, passing checkpoint paths
    between phases. All outputs are saved under ``output_dir/seed_{seed}/``.

    Args:
        cfg: Loaded YAML config dict.
        seed: Random seed.
        output_dir: Root output directory.
        verbose: Whether to print progress.
        datasets_dir: Optional path to search for cached quantum datasets.
        raw_image_datasets: Optional dict ``{"train": ds, "val": ds, "test": ds}``
            of raw image datasets (used in tests to skip MedMNIST download).

    Returns:
        dict with results from all three phases and a summary comparison.
    """
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    t_pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Teacher
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# Phase 1: Teacher Training (seed={seed})")
        print(f"{'#' * 60}")

    teacher_dir = os.path.join(seed_dir, "teacher")
    os.makedirs(teacher_dir, exist_ok=True)

    teacher_result = run_teacher_training(
        cfg,
        seed,
        teacher_dir,
        verbose=verbose,
        set_seed_fn=set_seed,
        datasets_dir=datasets_dir,
    )

    # Find the best teacher checkpoint (fall back to final)
    teacher_best = os.path.join(teacher_dir, f"teacher_best_seed_{seed}.pt")
    teacher_final = os.path.join(teacher_dir, f"teacher_final_seed_{seed}.pt")
    teacher_ckpt = teacher_best if os.path.exists(teacher_best) else teacher_final

    if verbose:
        print(f"\nTeacher test accuracy: {teacher_result['test_acc']:.4f}")
        print(f"Teacher checkpoint: {teacher_ckpt}")

    # ------------------------------------------------------------------
    # Phase 2: Student Gatekeeper
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# Phase 2: Student Training (seed={seed})")
        print(f"{'#' * 60}")

    student_dir = os.path.join(seed_dir, "student")
    os.makedirs(student_dir, exist_ok=True)

    student_result = run_student_training(
        cfg,
        seed,
        student_dir,
        teacher_ckpt_path=teacher_ckpt,
        verbose=verbose,
        set_seed_fn=set_seed,
        datasets_dir=datasets_dir,
        raw_image_datasets=raw_image_datasets,
    )

    # Find the best student checkpoint
    student_best = os.path.join(student_dir, f"student_best_seed_{seed}.pt")
    student_final = os.path.join(student_dir, f"student_final_seed_{seed}.pt")
    student_ckpt = student_best if os.path.exists(student_best) else student_final

    if verbose:
        print(f"\nStudent agreement: {student_result['agreement']:.4f}")
        print(f"Student checkpoint: {student_ckpt}")

    # ------------------------------------------------------------------
    # Phase 3: Final Classifier
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# Phase 3: Final Classifier Training (seed={seed})")
        print(f"{'#' * 60}")

    should_skip_final = bool(
        student_result.get("skipped", False)
    ) or not os.path.exists(student_ckpt)

    if should_skip_final:
        warnings.warn(
            "Skipping Final Classifier training because Student training was "
            "skipped or no Student checkpoint was produced.",
            UserWarning,
        )
        if verbose:
            print(
                "WARNING: Skipping Phase 3 (Final Classifier) because "
                "Student has no trainable data/checkpoint for this seed."
            )
        final_result = {
            "seed": seed,
            "architecture": "TS-MoE-FinalClassifier",
            "skipped": True,
            "skip_reason": "student_skipped_or_missing_checkpoint",
            "train_losses": [],
            "val_losses": [],
            "train_accs": [],
            "val_accs": [],
            "test_loss": None,
            "test_acc": None,
            "test_auc": None,
            "test_f1": None,
            "test_recall": None,
            "test_probs": [],
            "test_labels": [],
            "test_confusion_matrix": [],
            "num_classes": cfg.get("model", {}).get("num_classes", 2),
            "final_train_loss": None,
            "final_val_loss": None,
            "final_train_acc": None,
            "final_val_acc": None,
            "kernel_names": student_result.get("kernel_names", []),
            "num_kernels": student_result.get("num_kernels"),
            "comparison": {
                "final_classifier_acc": None,
                "teacher_oracle_acc": teacher_result.get("test_acc"),
                "baseline_acc": None,
                "speedup_factor": None,
                "routing_time_s": None,
                "inference_time_s": None,
                "pipeline_time_s": None,
            },
            "routing_analysis": {},
            "routing_time_s": None,
            "classifier_params": 0,
        }
    else:
        final_dir = os.path.join(seed_dir, "final_classifier")
        os.makedirs(final_dir, exist_ok=True)

        final_result = run_final_classifier_training(
            cfg,
            seed,
            final_dir,
            student_ckpt_path=student_ckpt,
            verbose=verbose,
            set_seed_fn=set_seed,
            datasets_dir=datasets_dir,
            raw_image_datasets=raw_image_datasets,
            teacher_test_acc=teacher_result["test_acc"],
            teacher_ckpt_path=teacher_ckpt,
        )

    pipeline_time = time.time() - t_pipeline_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "seed": seed,
        "pipeline_time_s": pipeline_time,
        "teacher_test_acc": teacher_result["test_acc"],
        "teacher_test_auc": teacher_result["test_auc"],
        "teacher_test_f1": teacher_result["test_f1"],
        "student_agreement": student_result["agreement"],
        "final_test_acc": final_result["test_acc"],
        "final_test_auc": final_result["test_auc"],
        "final_test_f1": final_result["test_f1"],
        "speedup_factor": final_result.get("comparison", {}).get("speedup_factor"),
    }

    if verbose:
        final_acc = summary["final_test_acc"]
        final_acc_str = "N/A" if final_acc is None else f"{final_acc:.4f}"
        speedup = summary["speedup_factor"]
        speedup_str = "N/A" if speedup is None else f"{speedup}×"
        print(f"\n{'=' * 60}")
        print(f"TS-MoE Pipeline Summary (seed={seed})")
        print(f"{'=' * 60}")
        print(f"Teacher accuracy:   {summary['teacher_test_acc']:.4f}")
        print(f"Student agreement:  {summary['student_agreement']:.4f}")
        print(f"Final accuracy:     {final_acc_str}")
        print(f"Speedup factor:     {speedup_str}")
        print(f"Total pipeline time: {pipeline_time:.1f}s")
        print(f"{'=' * 60}\n")

    # Save summary
    summary_path = os.path.join(seed_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "teacher": teacher_result,
        "student": student_result,
        "final": final_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a TS-MoE pipeline (Teacher → Student → Final Classifier)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "pneumonia_mnist_ts_moe.yml"),
        help="Path to YAML config file with model.architecture='TS-MoE'",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Validate architecture
    arch = cfg.get("model", {}).get("architecture", "original")
    if arch != "TS-MoE":
        print(
            f"Warning: model.architecture is '{arch}', expected 'TS-MoE'. "
            f"Proceeding anyway."
        )

    # Seeds
    seed_spec = cfg.get("misc", {}).get("seed", 0)
    seeds = seed_spec if isinstance(seed_spec, list) else [seed_spec]

    print("\n" + "=" * 60)
    print("TS-MoE Pipeline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Seeds: {seeds}")
    print(f"{'=' * 60}\n")

    # Output directory
    os.makedirs("outputs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"ts_moe_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_save = os.path.join(output_dir, "config.yml")
    with open(config_save, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to: {config_save}")
    print(f"Output directory: {output_dir}\n")

    # Run pipeline for each seed
    all_results = []
    for seed in seeds:
        result = run_ts_moe_pipeline(cfg, seed, output_dir, verbose=(len(seeds) == 1))
        all_results.append(result)

    # Aggregate across seeds
    summaries = [r["summary"] for r in all_results]
    aggregate = {}
    for key in [
        "teacher_test_acc",
        "student_agreement",
        "final_test_acc",
        "final_test_auc",
        "final_test_f1",
        "pipeline_time_s",
    ]:
        values = [s[key] for s in summaries if s.get(key) is not None]
        if values:
            aggregate[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    # Print aggregate
    print(f"\n{'=' * 60}")
    print(f"Aggregate Results ({len(seeds)} seed(s))")
    print(f"{'=' * 60}")
    for metric, stats in aggregate.items():
        print(f"  {metric:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"{'=' * 60}\n")

    # Save aggregate
    agg_path = os.path.join(output_dir, "aggregate_results.json")
    with open(agg_path, "w") as f:
        json.dump(
            {"seeds": seeds, "aggregate": aggregate, "summaries": summaries},
            f,
            indent=2,
        )
    print(f"Aggregate results saved to: {agg_path}")

    # Plot multi-seed ROC curves for the final classifier
    final_labels = [
        np.array(r["final"]["test_labels"])
        for r in all_results
        if r["final"].get("test_labels")
    ]
    final_probs = [
        np.array(r["final"]["test_probs"])
        for r in all_results
        if r["final"].get("test_probs")
    ]
    if final_labels and final_probs:
        num_classes = all_results[0]["final"].get("num_classes", 2)
        plot_multi_seed_roc_curves(
            final_labels,
            final_probs,
            os.path.join(output_dir, "final_roc_curve_multi_seed.png"),
            num_classes=num_classes,
        )
        print(f"Final classifier multi-seed ROC curve saved to: {output_dir}")

    # Plot multi-seed ROC curves for the teacher
    teacher_labels = [
        np.array(r["teacher"]["test_labels"])
        for r in all_results
        if r["teacher"].get("test_labels")
    ]
    teacher_probs = [
        np.array(r["teacher"]["test_probs"])
        for r in all_results
        if r["teacher"].get("test_probs")
    ]
    if teacher_labels and teacher_probs:
        num_classes = all_results[0]["teacher"].get("num_classes", 2)
        plot_multi_seed_roc_curves(
            teacher_labels,
            teacher_probs,
            os.path.join(output_dir, "teacher_roc_curve_multi_seed.png"),
            num_classes=num_classes,
        )
        print(f"Teacher multi-seed ROC curve saved to: {output_dir}")

    print(f"All outputs saved to: {output_dir}")
    print("\nTS-MoE pipeline complete!")


if __name__ == "__main__":
    main()
