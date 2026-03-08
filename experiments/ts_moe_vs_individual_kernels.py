"""
ts_moe_vs_individual_kernels.py

Comparison experiment: TS-MoE pipeline vs original DAQCNN with all kernels
vs DAQCNN trained on each individual kernel topology — all on TissueMNIST.

Models evaluated (20 seeds each):
    - TS-MoE Teacher          (all kernels, soft attention routing)
    - TS-MoE Final Classifier (sparse routed features)
    - DAQCNN All-Kernels      (all topologies concatenated, no routing)
    - DAQCNN kings            (single kernel)
    - DAQCNN horizontal       (single kernel)
    - DAQCNN cross            (single kernel)
    - DAQCNN ring             (single kernel)

Summary plot: macro ROC-AUC and accuracy for every model, mean ± std across
seeds, in a single side-by-side figure.

Usage (from project root):
    python experiments/ts_moe_vs_individual_kernels.py

    # Custom configs:
    python experiments/ts_moe_vs_individual_kernels.py \\
        --daqcnn_config  configs/tissue_mnist_multi_seed_3kern.yml \\
        --tsmoe_config   configs/tissue_mnist_ts_moe_3kern.yml \\
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \\
        --output_dir     outputs/ts_moe_vs_kernels_tissue
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.daqcnn_training import run_single_seed  # noqa: E402
from src.models.train_ts_moe import run_ts_moe_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DAQCNN_CONFIG = str(
    PROJECT_ROOT / "configs" / "tissue_mnist_multi_seed_3kern.yml"
)
DEFAULT_TSMOE_CONFIG = str(PROJECT_ROOT / "configs" / "tissue_mnist_ts_moe_3kern.yml")
DEFAULT_SEEDS = list(range(20))
DEFAULT_OUTPUT_DIR = str(
    PROJECT_ROOT / "outputs" / "ts_moe_vs_individual_kernels_tissue"
)

# Kernel names that exist in the tissue_mnist cached dataset.
# Must match kernel_topology_names in the DAQCNN config exactly.
INDIVIDUAL_KERNELS = ["kings", "horizontal", "cross", "ring"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_for_single_kernel(base_cfg: dict, kernel_name: str) -> dict:
    """Return a deep copy of base_cfg restricted to one kernel topology."""
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["kernel_topology_names"] = [kernel_name]
    return cfg


def _safe_float(val, fallback=float("nan")) -> float:
    """Return float(val) or fallback if val is None."""
    return float(val) if val is not None else fallback


def _aggregate(values: list) -> dict:
    """Compute mean / std / list for a list of floats, ignoring NaNs."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "values": values}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "values": values,
    }


# ---------------------------------------------------------------------------
# Per-seed runners
# ---------------------------------------------------------------------------


def run_daqcnn_seed(cfg: dict, seed: int, output_dir: str, verbose: bool) -> dict:
    """Run one DAQCNN seed and return {test_acc, test_auc}."""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    result = run_single_seed(
        cfg, seed, output_dir, verbose=verbose, set_seed_fn=set_seed
    )
    return {
        "test_acc": _safe_float(result.get("test_acc")),
        "test_auc": _safe_float(result.get("test_auc")),
    }


def run_tsmoe_seed(
    cfg: dict,
    seed: int,
    output_dir: str,
    verbose: bool,
) -> dict:
    """Run one full TS-MoE pipeline seed and return teacher + final metrics."""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    result = run_ts_moe_pipeline(cfg, seed, output_dir, verbose=verbose)

    teacher = result.get("teacher", {})
    final = result.get("final", {})

    return {
        "teacher_acc": _safe_float(teacher.get("test_acc")),
        "teacher_auc": _safe_float(teacher.get("test_auc")),
        "final_acc": _safe_float(final.get("test_acc")),
        "final_auc": _safe_float(final.get("test_auc")),
        "skipped": bool(final.get("skipped", False)),
    }


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------


def plot_comparison(
    results: dict,
    output_path: str,
    num_seeds: int,
) -> None:
    """
    Single figure with two side-by-side bar charts (Accuracy / ROC-AUC).

    Each bar represents one model variant; error bars show ± 1 std dev.
    Models are ordered as defined in PLOT_ORDER below.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed — skipping comparison plot.")
        return

    # ------------------------------------------------------------------
    # Define display order and cosmetics
    # ------------------------------------------------------------------
    PLOT_ORDER = [
        ("ts_moe_teacher", "TS-MoE\nTeacher", "#4C72B0"),
        ("ts_moe_final", "TS-MoE\nFinal Clf.", "#DD8452"),
        ("daqcnn_all", "DAQCNN\nAll Kernels", "#55A868"),
        ("daqcnn_kings", "DAQCNN\nKings", "#C44E52"),
        ("daqcnn_horizontal", "DAQCNN\nHorizontal", "#8172B2"),
        ("daqcnn_cross", "DAQCNN\nCross", "#937860"),
        ("daqcnn_ring", "DAQCNN\nRing", "#DA8BC3"),
    ]

    # Keep only models that actually have results
    present = [
        (key, label, color) for key, label, color in PLOT_ORDER if key in results
    ]
    n_models = len(present)

    if n_models == 0:
        warnings.warn("No results to plot.")
        return

    # ------------------------------------------------------------------
    # Extract values
    # ------------------------------------------------------------------
    labels = [label for _, label, _ in present]
    colors = [color for _, _, color in present]
    acc_means = [results[key]["acc"]["mean"] for key, _, _ in present]
    acc_stds = [results[key]["acc"]["std"] for key, _, _ in present]
    auc_means = [results[key]["auc"]["mean"] for key, _, _ in present]
    auc_stds = [results[key]["auc"]["std"] for key, _, _ in present]

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(max(12, n_models * 1.6), 6), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    x = np.arange(n_models)
    bar_w = 0.55

    for ax, means, stds, metric_name, y_label in [
        (axes[0], acc_means, acc_stds, "Accuracy", "Test Accuracy"),
        (axes[1], auc_means, auc_stds, "ROC-AUC", "Macro ROC-AUC (OvR)"),
    ]:
        bars = ax.bar(
            x,
            means,
            width=bar_w,
            color=colors,
            alpha=0.85,
            yerr=stds,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#333333"},
            zorder=3,
        )

        # Value labels above each bar
        for bar, mean, std in zip(bars, means, stds):
            if np.isnan(mean):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + std + 0.004,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#222222",
            )

        # Grid and styling
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(metric_name, fontsize=11, fontweight="bold", pad=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Dynamic y-axis: start just below the minimum mean
        valid = [m for m in means if not np.isnan(m)]
        if valid:
            y_min = max(0.0, min(valid) - 0.06)
            y_max = min(1.0, max(valid) + max(stds) + 0.06)
            ax.set_ylim(y_min, y_max)

    # Colour legend (models)
    legend_handles = [
        mpatches.Patch(color=color, label=label.replace("\n", " "))
        for (_, label, color) in present
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(n_models, 4),
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        f"TissueMNIST — TS-MoE vs Individual Kernels\n"
        f"({num_seeds} seeds, mean ± 1 std)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] Comparison figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TS-MoE Teacher, TS-MoE Final Classifier, DAQCNN All-Kernels, "
            "and DAQCNN per individual kernel on TissueMNIST across multiple seeds."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--daqcnn_config",
        type=str,
        default=DEFAULT_DAQCNN_CONFIG,
        help="Path to the DAQCNN YAML config (used for all-kernels and single-kernel runs).",
    )
    parser.add_argument(
        "--tsmoe_config",
        type=str,
        default=DEFAULT_TSMOE_CONFIG,
        help="Path to the TS-MoE YAML config.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="List of random seeds. Default: 0..19 (20 seeds).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for all outputs.",
    )
    parser.add_argument(
        "--skip_tsmoe",
        action="store_true",
        help="Skip the TS-MoE pipeline (useful when re-plotting existing results).",
    )
    parser.add_argument(
        "--skip_daqcnn",
        action="store_true",
        help="Skip all DAQCNN runs (useful when re-plotting existing results).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass verbose=True to all sub-runners.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    n_seeds = len(seeds)

    daqcnn_cfg = load_config(args.daqcnn_config)
    tsmoe_cfg = load_config(args.tsmoe_config)

    # Resolve which individual kernels to run from the daqcnn config.
    # Fall back to the module-level constant if not set in config.
    individual_kernels: list[str] = daqcnn_cfg.get("model", {}).get(
        "kernel_topology_names", INDIVIDUAL_KERNELS
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("TS-MoE vs Individual Kernels — TissueMNIST")
    print("=" * 70)
    print(f"  DAQCNN config   : {args.daqcnn_config}")
    print(f"  TS-MoE config   : {args.tsmoe_config}")
    print(f"  Seeds ({n_seeds})     : {seeds}")
    print(f"  Individual kernels: {individual_kernels}")
    print(f"  Output dir      : {output_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Storage: per-seed raw metrics
    # ------------------------------------------------------------------
    raw: dict[str, dict] = {
        "ts_moe_teacher": {"acc": [], "auc": []},
        "ts_moe_final": {"acc": [], "auc": []},
        "daqcnn_all": {"acc": [], "auc": []},
    }
    for k in individual_kernels:
        raw[f"daqcnn_{k}"] = {"acc": [], "auc": []}

    # ------------------------------------------------------------------
    # TS-MoE pipeline (Teacher + Student + Final Classifier)
    # ------------------------------------------------------------------
    if not args.skip_tsmoe:
        print(f"\n{'#' * 70}")
        print("# TS-MoE Pipeline")
        print(f"{'#' * 70}")

        tsmoe_root = output_dir / "ts_moe"
        tsmoe_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            print(f"\n  [TS-MoE] seed={seed} ...")
            seed_dir = str(tsmoe_root / f"seed_{seed}")
            try:
                metrics = run_tsmoe_seed(
                    tsmoe_cfg, seed, seed_dir, verbose=args.verbose
                )
            except Exception as exc:
                warnings.warn(f"TS-MoE seed={seed} failed: {exc}")
                metrics = {
                    "teacher_acc": float("nan"),
                    "teacher_auc": float("nan"),
                    "final_acc": float("nan"),
                    "final_auc": float("nan"),
                    "skipped": True,
                }

            raw["ts_moe_teacher"]["acc"].append(metrics["teacher_acc"])
            raw["ts_moe_teacher"]["auc"].append(metrics["teacher_auc"])

            if not metrics.get("skipped", False):
                raw["ts_moe_final"]["acc"].append(metrics["final_acc"])
                raw["ts_moe_final"]["auc"].append(metrics["final_auc"])
            else:
                raw["ts_moe_final"]["acc"].append(float("nan"))
                raw["ts_moe_final"]["auc"].append(float("nan"))

            print(
                f"  [TS-MoE] seed={seed} done — "
                f"teacher acc={metrics['teacher_acc']:.4f} auc={metrics['teacher_auc']:.4f} | "
                f"final acc={metrics['final_acc']:.4f} auc={metrics['final_auc']:.4f}"
            )

        # Checkpoint after all TS-MoE seeds
        _checkpoint(raw, output_dir, timestamp, label="after_tsmoe")

    else:
        print("\n[skip] TS-MoE runs skipped (--skip_tsmoe).")

    # ------------------------------------------------------------------
    # DAQCNN — All Kernels
    # ------------------------------------------------------------------
    if not args.skip_daqcnn:
        print(f"\n{'#' * 70}")
        print("# DAQCNN — All Kernels")
        print(f"{'#' * 70}")

        daqcnn_all_root = output_dir / "daqcnn_all"
        daqcnn_all_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            print(f"\n  [DAQCNN All] seed={seed} ...")
            seed_dir = str(daqcnn_all_root / f"seed_{seed}")
            try:
                metrics = run_daqcnn_seed(
                    daqcnn_cfg, seed, seed_dir, verbose=args.verbose
                )
            except Exception as exc:
                warnings.warn(f"DAQCNN All seed={seed} failed: {exc}")
                metrics = {"test_acc": float("nan"), "test_auc": float("nan")}

            raw["daqcnn_all"]["acc"].append(metrics["test_acc"])
            raw["daqcnn_all"]["auc"].append(metrics["test_auc"])
            print(
                f"  [DAQCNN All] seed={seed} done — "
                f"acc={metrics['test_acc']:.4f} auc={metrics['test_auc']:.4f}"
            )

        # ------------------------------------------------------------------
        # DAQCNN — Individual Kernels
        # ------------------------------------------------------------------
        for kernel_name in individual_kernels:
            print(f"\n{'#' * 70}")
            print(f"# DAQCNN — {kernel_name}")
            print(f"{'#' * 70}")

            kernel_root = output_dir / f"daqcnn_{kernel_name}"
            kernel_root.mkdir(parents=True, exist_ok=True)
            single_cfg = cfg_for_single_kernel(daqcnn_cfg, kernel_name)
            model_key = f"daqcnn_{kernel_name}"

            for seed in seeds:
                print(f"\n  [DAQCNN {kernel_name}] seed={seed} ...")
                seed_dir = str(kernel_root / f"seed_{seed}")
                try:
                    metrics = run_daqcnn_seed(
                        single_cfg, seed, seed_dir, verbose=args.verbose
                    )
                except Exception as exc:
                    warnings.warn(f"DAQCNN {kernel_name} seed={seed} failed: {exc}")
                    metrics = {"test_acc": float("nan"), "test_auc": float("nan")}

                raw[model_key]["acc"].append(metrics["test_acc"])
                raw[model_key]["auc"].append(metrics["test_auc"])
                print(
                    f"  [DAQCNN {kernel_name}] seed={seed} done — "
                    f"acc={metrics['test_acc']:.4f} auc={metrics['test_auc']:.4f}"
                )

            _checkpoint(raw, output_dir, timestamp, label=f"after_daqcnn_{kernel_name}")

    else:
        print("\n[skip] DAQCNN runs skipped (--skip_daqcnn).")

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    aggregated: dict[str, dict] = {}
    for model_key, data in raw.items():
        if len(data["acc"]) == 0:
            continue
        aggregated[model_key] = {
            "acc": _aggregate(data["acc"]),
            "auc": _aggregate(data["auc"]),
        }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Summary  (mean ± std across seeds)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<28}  {'Accuracy':>14}  {'ROC-AUC':>14}  {'N seeds':>8}")
    print("  " + "-" * 62)

    DISPLAY_ORDER = ["ts_moe_teacher", "ts_moe_final", "daqcnn_all"] + [
        f"daqcnn_{k}" for k in individual_kernels
    ]

    DISPLAY_NAMES = {
        "ts_moe_teacher": "TS-MoE Teacher",
        "ts_moe_final": "TS-MoE Final Classifier",
        "daqcnn_all": "DAQCNN All Kernels",
        **{f"daqcnn_{k}": f"DAQCNN {k}" for k in individual_kernels},
    }

    for model_key in DISPLAY_ORDER:
        if model_key not in aggregated:
            continue
        agg = aggregated[model_key]
        acc = agg["acc"]
        auc = agg["auc"]
        n_valid = len([v for v in acc["values"] if not np.isnan(v)])

        acc_str = (
            f"{acc['mean']:.4f} ± {acc['std']:.4f}"
            if not np.isnan(acc["mean"])
            else "N/A"
        )
        auc_str = (
            f"{auc['mean']:.4f} ± {auc['std']:.4f}"
            if not np.isnan(auc["mean"])
            else "N/A"
        )

        print(
            f"  {DISPLAY_NAMES.get(model_key, model_key):<28}  {acc_str:>14}  {auc_str:>14}  {n_valid:>8}"
        )

    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "seeds": seeds,
                "n_seeds": n_seeds,
                "individual_kernels": individual_kernels,
                "raw": raw,
                "aggregated": {
                    k: {
                        "acc": v["acc"],
                        "auc": v["auc"],
                    }
                    for k, v in aggregated.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\n[json] Results saved → {results_path}")

    # ------------------------------------------------------------------
    # Comparison plot
    # ------------------------------------------------------------------
    plot_path = str(output_dir / f"comparison_{timestamp}.png")
    plot_comparison(aggregated, plot_path, num_seeds=n_seeds)

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done.")


def _checkpoint(raw: dict, output_dir: Path, timestamp: str, label: str) -> None:
    """Save a partial results JSON so progress is not lost mid-run."""
    path = output_dir / f"checkpoint_{label}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  [checkpoint] Saved → {path}")


if __name__ == "__main__":
    main()
