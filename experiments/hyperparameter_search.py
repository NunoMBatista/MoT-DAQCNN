"""
Optuna-based hyperparameter search for all MoT-DAQCNN pipelines.

Supports three architectures:
    - "original"  : Original DAQCNN baseline
    - "TS-MoE"    : Teacher-Student Mixture-of-Experts
    - "gumbel"    : Gumbel-Softmax Mixture-of-Experts

Usage:
    python experiments/hyperparameter_search.py \
        --config configs/breast_mnist_multi_seed_3kern.yml \
        --search-config configs/hp_search/breast_mnist_original.yml \
        --n-trials 50

    # Resume a previous search (SQLite DB is auto-created):
    python experiments/hyperparameter_search.py \
        --config configs/breast_mnist_gumbel_3kern.yml \
        --search-config configs/hp_search/breast_mnist_gumbel.yml \
        --n-trials 100 \
        --resume

    # Compare best results across pipelines (after running all three):
    python experiments/hyperparameter_search.py --compare \
        outputs/hp_search_original_*/study.db \
        outputs/hp_search_gumbel_*/study.db \
        outputs/hp_search_tsmoe_*/study.db

Search Config Format:
    The search YAML defines which hyperparameters to tune and their ranges.
    See ``configs/hp_search/`` for examples.  Each entry maps a dotted config
    path (e.g. ``optim.lr``) to a distribution specification:

        optim.lr:
            type: loguniform
            low: 1e-5
            high: 1e-2

    Supported distribution types:
        - uniform(low, high)
        - loguniform(low, high)
        - int(low, high)             — uniform integer
        - int(low, high, step)       — stepped integer
        - categorical(choices)
        - fixed(value)               — not tuned, but overrides base config

Design:
    Each Optuna trial runs a *single seed* for speed.  The best top-K
    configurations found are then validated with full multi-seed runs
    using the existing ``robust_test_original_daqcnn.py`` script.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import torch
import yaml

# Optuna warns when categorical choices contain lists (e.g. ["chain"]).
# In this project we intentionally use list-valued choices for
# model.kernel_topology_names, so suppress only that specific warning.
warnings.filterwarnings(
    "ignore",
    message=(
        r"Choices for a categorical distribution should be a tuple.*"
        r"persistent storage.*"
    ),
    category=UserWarning,
)

# Optional W&B (kept optional so this script works without wandb installed)
try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def _load_dotenv_if_present(repo_root: str) -> None:
    """Load KEY=VALUE lines from ``<repo_root>/.env`` into os.environ.

    This loader is intentionally minimal and dependency-free.
    It supports blank lines, comments, optional ``export `` prefixes,
    and quoted/unquoted values. Existing environment variables are kept.
    """
    env_path = os.path.join(repo_root, ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        warnings.warn(f"Failed to read .env at {env_path}: {e}")
        return

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" not in s:
            continue

        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        # Do not override vars that are already set in the shell.
        if key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key like ``optim.lr``."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _deep_get(d: dict, dotted_key: str, default: Any = None) -> Any:
    """Get a value from a nested dict using a dotted key."""
    keys = dotted_key.split(".")
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


# ---------------------------------------------------------------------------
# Search-space sampling
# ---------------------------------------------------------------------------
def _sample_param(
    trial: optuna.Trial,
    name: str,
    spec: dict,
) -> Any:
    """Sample a single hyperparameter from an Optuna trial according to *spec*.

    Args:
        trial: Current Optuna trial.
        name: Parameter name (used as the Optuna param identifier).
        spec: Distribution specification dict with at least a ``type`` key.

    Returns:
        The sampled value.
    """
    dist_type = spec["type"].lower().strip()

    if dist_type in ("uniform", "float"):
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        if step is not None:
            return trial.suggest_float(name, low, high, step=float(step))
        return trial.suggest_float(name, low, high)

    elif dist_type in ("loguniform", "log_uniform", "logfloat", "log_float"):
        low = float(spec["low"])
        high = float(spec["high"])
        return trial.suggest_float(name, low, high, log=True)

    elif dist_type in ("int", "integer"):
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        if log:
            return trial.suggest_int(name, low, high, log=True)
        return trial.suggest_int(name, low, high, step=step)

    elif dist_type in ("categorical", "cat", "choice"):
        choices = spec["choices"]
        return trial.suggest_categorical(name, choices)

    elif dist_type == "fixed":
        # Not tuned — just inject a constant.  Useful for overriding the
        # base config without creating a new YAML.
        return spec["value"]

    else:
        raise ValueError(
            f"Unknown distribution type '{dist_type}' for parameter '{name}'. "
            f"Supported: uniform, loguniform, int, categorical, fixed."
        )


def build_trial_config(
    base_cfg: dict,
    search_space: dict,
    trial: optuna.Trial,
) -> dict:
    """Create a concrete config dict for this trial by sampling HPs.

    Args:
        base_cfg: The base YAML config (not mutated).
        search_space: The ``search_space`` section of the search config.
        trial: Current Optuna trial.

    Returns:
        A deep-copied config with sampled values injected.
    """
    cfg = copy.deepcopy(base_cfg)

    for dotted_key, spec in search_space.items():
        value = _sample_param(trial, dotted_key, spec)

        # Handle special yaml None / null conversions
        if value == "null" or value == "None":
            value = None

        _deep_set(cfg, dotted_key, value)

    return cfg


# ---------------------------------------------------------------------------
# Objective functions (one per pipeline)
# ---------------------------------------------------------------------------


def _run_original_trial(
    cfg: dict, seed: int, output_dir: str, trial: Any = None
) -> dict:
    """Run one trial of the original DAQCNN pipeline."""
    from src.models.daqcnn_training import run_single_seed

    result = run_single_seed(cfg, seed, output_dir, verbose=False, set_seed_fn=set_seed)
    return result


def _run_gumbel_trial(cfg: dict, seed: int, output_dir: str, trial: Any = None) -> dict:
    """Run one trial of the Gumbel-Softmax MoE pipeline."""
    from src.models.gumbel_moe_training import run_gumbel_moe

    result = run_gumbel_moe(cfg, seed, output_dir, verbose=False, set_seed_fn=set_seed)
    return result


def _run_ts_moe_trial(cfg: dict, seed: int, output_dir: str, trial: Any = None) -> dict:
    """Run one trial of the TS-MoE pipeline.

    Returns a flattened result dict with the final classifier metrics.
    """
    from src.models.train_ts_moe import run_ts_moe_pipeline

    ts_result = run_ts_moe_pipeline(cfg, seed, output_dir, verbose=False)

    # Flatten to standard form
    final = ts_result["final"]
    summary = ts_result["summary"]

    result = {
        "seed": seed,
        "train_losses": final.get("train_losses", []),
        "val_losses": final.get("val_losses", []),
        "train_accs": final.get("train_accs", []),
        "val_accs": final.get("val_accs", []),
        "test_loss": final.get("test_loss"),
        "test_acc": final.get("test_acc"),
        "test_auc": final.get("test_auc"),
        "test_f1": final.get("test_f1"),
        "test_recall": final.get("test_recall"),
        "teacher_test_acc": summary.get("teacher_test_acc"),
        "student_agreement": summary.get("student_agreement"),
        "pipeline_time_s": summary.get("pipeline_time_s"),
    }
    return result


# Map from architecture name to runner function
_RUNNERS = {
    "original": _run_original_trial,
    "TS-MoE": _run_ts_moe_trial,
    "ts-moe": _run_ts_moe_trial,
    "ts_moe": _run_ts_moe_trial,
    "gumbel": _run_gumbel_trial,
}


# ---------------------------------------------------------------------------
# Main objective
# ---------------------------------------------------------------------------


class HPSearchObjective:
    """Callable Optuna objective that wraps any pipeline.

    Attributes:
        base_cfg: The base YAML config (unmutated template).
        search_space: Dict mapping dotted config keys to distribution specs.
        architecture: One of "original", "TS-MoE", "gumbel".
        seed: Fixed seed used for every trial (single-seed exploration).
        output_root: Root output directory for all trials.
        metric: Which metric to optimise (e.g. "test_acc", "test_auc").
        direction: "maximize" or "minimize".
    """

    def __init__(
        self,
        base_cfg: dict,
        search_space: dict,
        architecture: str,
        seed: int,
        output_root: str,
        metric: str = "test_acc",
        trial_seeds: Optional[List[int]] = None,
    ):
        self.base_cfg = base_cfg
        self.search_space = search_space
        self.architecture = architecture
        self.seed = seed
        # If trial_seeds is given, each trial runs all of them and reports the mean.
        # Otherwise fall back to the single exploration seed for speed.
        self.trial_seeds = trial_seeds if trial_seeds is not None else [seed]
        self.output_root = output_root
        self.metric = metric

        runner = _RUNNERS.get(architecture)
        if runner is None:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Supported: {list(_RUNNERS.keys())}"
            )
        self._runner = runner

    def __call__(self, trial: optuna.Trial) -> float:
        """Run one trial and return the mean objective value across trial seeds."""
        t_start = time.time()

        # 1. Build config for this trial
        cfg = build_trial_config(self.base_cfg, self.search_space, trial)

        # 2. Create trial output directory
        trial_dir = os.path.join(self.output_root, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        # Save the trial config for reproducibility (seed-agnostic template)
        trial_cfg_path = os.path.join(trial_dir, "config.yml")
        with open(trial_cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # 3. Run the pipeline once per trial seed and collect results
        seed_results = []
        for s in self.trial_seeds:
            seed_dir = os.path.join(trial_dir, f"seed_{s}")
            os.makedirs(seed_dir, exist_ok=True)
            cfg_s = copy.deepcopy(cfg)
            cfg_s.setdefault("misc", {})["seed"] = s
            try:
                set_seed(s)
                result = self._runner(cfg_s, s, seed_dir, trial)
                seed_results.append(result)
            except Exception as e:
                error_path = os.path.join(seed_dir, "error.txt")
                with open(error_path, "w") as ef:
                    import traceback

                    ef.write(f"Trial {trial.number} seed {s} failed:\n")
                    ef.write(traceback.format_exc())
                warnings.warn(f"Trial {trial.number} seed {s} failed: {e}")

        if not seed_results:
            raise optuna.TrialPruned("All seeds failed for this trial")

        # Helper: collect non-None values for a metric key across seeds
        def _vals(key):
            return [r[key] for r in seed_results if r.get(key) is not None]

        # 4. Aggregate the objective metric (mean = what Optuna optimises)
        obj_values = _vals(self.metric)
        if not obj_values:
            warnings.warn(
                f"Trial {trial.number}: metric '{self.metric}' is None for all seeds."
            )
            raise optuna.TrialPruned(f"Metric '{self.metric}' is None")

        obj_value = float(np.mean(obj_values))
        obj_std = float(np.std(obj_values)) if len(obj_values) > 1 else 0.0

        # 5. Store aggregated metrics as user attributes
        for key in ("test_acc", "test_auc", "test_f1", "test_loss", "test_recall"):
            vals = _vals(key)
            if vals:
                trial.set_user_attr(key, float(np.mean(vals)))
                trial.set_user_attr(
                    f"{key}_std", float(np.std(vals)) if len(vals) > 1 else 0.0
                )

        trial.set_user_attr(f"{self.metric}_std", obj_std)
        trial.set_user_attr("n_seeds_completed", len(seed_results))
        trial.set_user_attr("trial_seeds", self.trial_seeds)
        trial.set_user_attr("duration_s", time.time() - t_start)

        # Store loss curves from the first seed for the learning-curve plot
        first = seed_results[0]
        if first.get("train_losses"):
            trial.set_user_attr("train_losses", first["train_losses"])
        if first.get("val_losses"):
            trial.set_user_attr("val_losses", first["val_losses"])

        # TS-MoE specific
        if self.architecture in ("TS-MoE", "ts-moe", "ts_moe"):
            ta_vals = _vals("teacher_test_acc")
            if ta_vals:
                trial.set_user_attr("teacher_test_acc", float(np.mean(ta_vals)))
            sa_vals = _vals("student_agreement")
            if sa_vals:
                trial.set_user_attr("student_agreement", float(np.mean(sa_vals)))

        # Save aggregate result summary
        agg_summary: dict = {
            "trial_number": trial.number,
            "seeds": self.trial_seeds,
            "n_seeds_completed": len(seed_results),
            f"{self.metric}_mean": obj_value,
            f"{self.metric}_std": obj_std,
        }
        for key in ("test_acc", "test_auc", "test_f1", "test_loss", "test_recall"):
            vals = _vals(key)
            if vals:
                agg_summary[f"{key}_mean"] = float(np.mean(vals))
                agg_summary[f"{key}_std"] = (
                    float(np.std(vals)) if len(vals) > 1 else 0.0
                )

        result_path = os.path.join(trial_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump(agg_summary, f, indent=2, default=str)

        elapsed = time.time() - t_start
        acc_vals = _vals("test_acc")
        auc_vals = _vals("test_auc")
        f1_vals = _vals("test_f1")
        acc_mean = float(np.mean(acc_vals)) if acc_vals else 0.0
        acc_std = float(np.std(acc_vals)) if len(acc_vals) > 1 else 0.0
        auc_mean = float(np.mean(auc_vals)) if auc_vals else 0.0
        f1_mean = float(np.mean(f1_vals)) if f1_vals else 0.0
        seeds_str = f"{len(seed_results)}/{len(self.trial_seeds)} seeds"
        print(
            f"  Trial {trial.number:4d} | "
            f"{self.metric}={obj_value:.4f}±{obj_std:.4f} | "
            f"acc={acc_mean:.4f}±{acc_std:.4f} "
            f"auc={auc_mean:.4f} "
            f"f1={f1_mean:.4f} | "
            f"{seeds_str} | {elapsed:.1f}s"
        )

        return float(obj_value)


# ---------------------------------------------------------------------------
# Multi-seed validation of top-K configs
# ---------------------------------------------------------------------------


def validate_top_k(
    study: optuna.Study,
    base_cfg: dict,
    search_space_keys: List[str],
    architecture: str,
    output_dir: str,
    *,
    top_k: int = 5,
    seeds: Optional[List[int]] = None,
) -> List[dict]:
    """Re-run the top-K configs with multiple seeds for robust validation.

    Args:
        study: Completed Optuna study.
        base_cfg: Base config (template).
        search_space_keys: List of dotted parameter keys that were tuned.
        architecture: Architecture name.
        output_dir: Output directory.
        top_k: Number of top trials to validate.
        seeds: Seeds for validation. Defaults to [0, 1, 2, 3, 4].

    Returns:
        List of dicts with aggregated validation metrics for each config.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    runner = _RUNNERS.get(architecture)
    if runner is None:
        raise ValueError(f"Unknown architecture: {architecture}")

    direction = study.direction
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if direction == optuna.study.StudyDirection.MAXIMIZE:
        sorted_trials = sorted(
            trials, key=lambda t: t.value or -float("inf"), reverse=True
        )
    else:
        sorted_trials = sorted(trials, key=lambda t: t.value or float("inf"))

    top_trials = sorted_trials[:top_k]
    validation_results = []

    print(f"\n{'=' * 60}")
    print(f"Validating Top-{min(top_k, len(top_trials))} Configurations")
    print(f"Seeds: {seeds}")
    print(f"{'=' * 60}\n")

    for rank, trial in enumerate(top_trials, 1):
        cfg = copy.deepcopy(base_cfg)
        for key in search_space_keys:
            if key in trial.params:
                _deep_set(cfg, key, trial.params[key])

        val_dir = os.path.join(output_dir, f"validation_rank{rank}_trial{trial.number}")
        os.makedirs(val_dir, exist_ok=True)

        # Save config
        with open(os.path.join(val_dir, "config.yml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        print(
            f"  Rank {rank} (trial {trial.number}, search {study.direction.name}={trial.value:.4f})"
        )
        print(f"  Params: {trial.params}")

        seed_results = []
        for seed in seeds:
            seed_dir = os.path.join(val_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            try:
                set_seed(seed)
                result = runner(cfg, seed, seed_dir, trial)
                seed_results.append(result)
                print(
                    f"    seed={seed}: acc={result.get('test_acc', 0):.4f} "
                    f"auc={result.get('test_auc', 0):.4f}"
                )
            except Exception as e:
                print(f"    seed={seed}: FAILED ({e})")

        if not seed_results:
            continue

        # Aggregate
        agg: Dict[str, Any] = {
            "rank": rank,
            "trial_number": trial.number,
            "search_value": trial.value,
            "params": trial.params,
        }
        for key in ("test_acc", "test_auc", "test_f1", "test_recall", "test_loss"):
            vals = [r[key] for r in seed_results if r.get(key) is not None]
            if vals:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }

        agg_path = os.path.join(val_dir, "validation_aggregate.json")
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2, default=str)

        validation_results.append(agg)

        mean_acc = agg.get("test_acc", {}).get("mean", 0)
        std_acc = agg.get("test_acc", {}).get("std", 0)
        print(f"  -> Validated: acc={mean_acc:.4f} +/- {std_acc:.4f}\n")

    return validation_results


# ---------------------------------------------------------------------------
# Pipeline comparison mode
# ---------------------------------------------------------------------------


def compare_studies(db_paths: List[str], output_dir: str):
    """Load multiple study databases and generate a comparison radar chart.

    Args:
        db_paths: Paths to Optuna SQLite study databases.
        output_dir: Where to save the comparison plots.
    """
    from src.utils.hp_search_plots import (
        plot_pipeline_comparison_radar,
    )

    os.makedirs(output_dir, exist_ok=True)

    pipeline_results: Dict[str, Dict[str, float]] = {}

    for db_path in db_paths:
        storage = f"sqlite:///{os.path.abspath(db_path)}"
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage)
            if not summaries:
                print(f"No studies found in {db_path}")
                continue
            study = optuna.load_study(
                study_name=summaries[0].study_name, storage=storage
            )
        except Exception as e:
            print(f"Failed to load {db_path}: {e}")
            continue

        name = study.study_name or os.path.basename(os.path.dirname(db_path))
        best = study.best_trial

        metrics: Dict[str, float] = {}
        for key in ("test_acc", "test_auc", "test_f1", "test_recall"):
            val = best.user_attrs.get(key)
            if val is not None:
                metrics[key] = float(val)

        if best.value is not None:
            metrics["objective"] = float(best.value)

        pipeline_results[name] = metrics
        print(f"Loaded {name}: {metrics}")

    if len(pipeline_results) < 2:
        print("Need at least 2 studies to compare.")
        return

    # Generate radar chart
    radar_path = os.path.join(output_dir, "pipeline_comparison_radar.png")
    plot_pipeline_comparison_radar(
        pipeline_results,
        radar_path,
        title="Pipeline Comparison — Best HP Configs",
    )
    print(f"Radar chart saved to: {radar_path}")

    # Save comparison table
    table_path = os.path.join(output_dir, "pipeline_comparison.json")
    with open(table_path, "w") as f:
        json.dump(pipeline_results, f, indent=2)
    print(f"Comparison table saved to: {table_path}")


# ---------------------------------------------------------------------------
# Generate best config YAML
# ---------------------------------------------------------------------------


def export_best_config(
    study: optuna.Study,
    base_cfg: dict,
    search_space: dict,
    output_path: str,
    *,
    validation_seeds: Optional[List[int]] = None,
) -> str:
    """Export the best trial's config as a ready-to-run YAML file.

    The exported config has the tuned hyperparameters injected into the
    base config, and uses the validation seeds (or original seeds) for
    multi-seed runs.

    Args:
        study: Completed Optuna study.
        base_cfg: Base config template.
        search_space: Search space definition.
        output_path: Where to write the YAML.
        validation_seeds: Seeds to use in the exported config.

    Returns:
        Path to the exported YAML.
    """
    best = study.best_trial
    cfg = copy.deepcopy(base_cfg)

    for key in search_space:
        if key in best.params:
            _deep_set(cfg, key, best.params[key])

    # Restore multi-seed for final training
    if validation_seeds:
        cfg.setdefault("misc", {})["seed"] = validation_seeds
    else:
        # Use the original seeds from the base config
        original_seeds = _deep_get(base_cfg, "misc.seed", [0, 1, 2])
        cfg.setdefault("misc", {})["seed"] = original_seeds

    with open(output_path, "w") as f:
        f.write("# Auto-generated by hyperparameter_search.py\n")
        f.write(f"# Best trial: {best.number} (value={best.value:.6f})\n")
        f.write(f"# Tuned params: {best.params}\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for MoT-DAQCNN pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Main search mode ---
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the base YAML config file.",
    )
    parser.add_argument(
        "--search-config",
        type=str,
        default=None,
        help="Path to the search space YAML.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for single-seed exploration trials (default: 42).",
    )
    parser.add_argument(
        "--trial-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "If provided, each trial is evaluated on ALL these seeds and the "
            "mean metric is used as the objective.  More robust but slower "
            "(runtime scales linearly with the number of seeds). "
            "Example: --trial-seeds 0 1 2  "
            "(default: single seed from --seed)"
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric to optimise (default: from search config or 'test_acc').",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        choices=["maximize", "minimize"],
        help="Optimisation direction (default: from search config or 'maximize').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous search from the SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: auto-generated under outputs/.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name. Default: auto-generated.",
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
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group (e.g., a study identifier).",
    )
    parser.add_argument(
        "--wandb-log-trials-table",
        action="store_true",
        help="If set, log a W&B Table with completed trials (params + objective + common attrs).",
    )

    # --- Validation ---
    parser.add_argument(
        "--validate-top-k",
        type=int,
        default=0,
        help="After search, validate the top K configs with multi-seed runs.",
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds for multi-seed validation (default: 0 1 2 3 4).",
    )

    # --- Comparison mode ---
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="DB_PATH",
        help="Compare studies from SQLite DBs and generate radar chart.",
    )

    # --- Sampler ---
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random", "cmaes"],
        help="Optuna sampler (default: tpe).",
    )

    # --- Misc ---
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots after search.",
    )

    args = parser.parse_args()

    # Load repo-root .env (if present) so WANDB_API_KEY/WANDB_ENTITY can be
    # picked up even when the shell did not explicitly source the file.
    _load_dotenv_if_present(PROJECT_ROOT)

    # === Comparison mode ===
    if args.compare:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cmp_dir = os.path.join("outputs", f"hp_comparison_{ts}")
        compare_studies(args.compare, cmp_dir)
        return

    # === Main search mode ===
    if args.config is None or args.search_config is None:
        parser.error("--config and --search-config are required for search mode.")

    base_cfg = load_yaml(args.config)
    search_cfg = load_yaml(args.search_config)

    # Parse search config
    search_space = search_cfg.get("search_space", {})
    if not search_space:
        parser.error("search_space section is empty or missing in search config.")

    # Search settings (can be in the search config or overridden via CLI)
    search_settings = search_cfg.get("settings", {})
    metric = args.metric or search_settings.get("metric", "test_acc")
    direction = args.direction or search_settings.get("direction", "maximize")
    n_trials = args.n_trials  # CLI always wins
    seed = args.seed

    # Architecture detection
    architecture = base_cfg.get("model", {}).get("architecture", "original")

    # Study name
    dataset_name = base_cfg.get("dataset", {}).get("name", "unknown")
    dataset_slug = dataset_name.replace("_", "")
    study_name = (
        args.study_name
        or search_settings.get("study_name")
        or f"hp_{architecture}_{dataset_slug}"
    )

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            "outputs", f"hp_search_{architecture}_{dataset_slug}_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # SQLite storage for persistence / resumption
    db_path = os.path.join(output_dir, "study.db")
    storage = f"sqlite:///{os.path.abspath(db_path)}"

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

            run_name = f"hp_search:{architecture}:{dataset_name}:{time.strftime('%Y%m%d_%H%M%S')}"
            wandb_group = args.wandb_group or study_name

            # Keep config JSON-serializable; store YAML dict under a single key.
            wandb_config = {
                "script": "experiments/hyperparameter_search.py",
                "base_config_path": args.config,
                "search_config_path": args.search_config,
                "output_dir": output_dir,
                "db_path": db_path,
                "study_name": study_name,
                "architecture": architecture,
                "dataset.name": dataset_name,
                "metric": metric,
                "direction": direction,
                "n_trials_requested": n_trials,
                "sampler": args.sampler,
                "resume": bool(args.resume),
                "trial_seeds": args.trial_seeds
                if args.trial_seeds is not None
                else [seed],
                "cfg.base": base_cfg,
                "cfg.search": search_cfg,
            }

            try:
                wandb_run = wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=run_name,
                    group=wandb_group,
                    tags=list(args.wandb_tags),
                    notes=args.wandb_notes,
                    config=wandb_config,
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] Failed to init W&B run: {e}\nContinuing without W&B.")
                wandb_run = None

    # Print header
    print("\n" + "=" * 70)
    print("  Hyperparameter Search")
    print("=" * 70)
    print(f"  Architecture:    {architecture}")
    print(f"  Base config:     {args.config}")
    print(f"  Search config:   {args.search_config}")
    print(f"  Dataset:         {dataset_name}")
    print(f"  Study name:      {study_name}")
    print(f"  Metric:          {metric} ({direction})")
    print(f"  Trials:          {n_trials}")
    trial_seeds = args.trial_seeds if args.trial_seeds is not None else [seed]
    if len(trial_seeds) == 1:
        print(f"  Seed:            {trial_seeds[0]}  (single-seed exploration)")
    else:
        print(f"  Trial seeds:     {trial_seeds}  (multi-seed — mean objective)")
    print(f"  Sampler:         {args.sampler}")
    print(f"  Output:          {output_dir}")
    print(f"  Storage:         {db_path}")
    print(f"  Resume:          {args.resume}")
    print(f"  Search space ({len(search_space)} params):")
    for key, spec in search_space.items():
        current_val = _deep_get(base_cfg, key)
        print(f"    {key:40s} {spec}  (base: {current_val})")
    print("=" * 70 + "\n")

    # Save search config and base config to output dir
    with open(os.path.join(output_dir, "base_config.yml"), "w") as f:
        yaml.dump(base_cfg, f, default_flow_style=False)
    with open(os.path.join(output_dir, "search_config.yml"), "w") as f:
        yaml.dump(search_cfg, f, default_flow_style=False)

    # --- Build sampler ---
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif args.sampler == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    # --- Create or load study ---
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=args.resume,
    )

    existing_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    if existing_trials > 0:
        print(f"Resuming study with {existing_trials} existing completed trials.\n")

    # --- Build objective ---
    trials_dir = os.path.join(output_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    objective = HPSearchObjective(
        base_cfg=base_cfg,
        search_space=search_space,
        architecture=architecture,
        seed=seed,
        output_root=trials_dir,
        metric=metric,
        trial_seeds=args.trial_seeds,
    )

    # --- Run optimisation ---
    t_search_start = time.time()

    # Real-time W&B trial logging via Optuna callback (one log per completed trial)
    def _wandb_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if wandb_run is None:
            return
        try:
            if trial.value is None:
                return
            payload: Dict[str, Any] = {
                "trial/number": trial.number,
                "trial/objective": trial.value,
                "trial/state": str(trial.state),
            }

            # Params
            for k, v in trial.params.items():
                payload[f"param/{k}"] = v

            # Common user_attrs you already set in the objective
            for a in (
                "test_acc",
                "test_auc",
                "test_f1",
                "test_recall",
                "test_loss",
                "duration_s",
                "n_seeds_completed",
            ):
                if a in trial.user_attrs:
                    payload[f"attr/{a}"] = trial.user_attrs.get(a)

            # Best-so-far objective
            if study.best_trial is not None and study.best_trial.value is not None:
                payload["study/best_objective_so_far"] = study.best_trial.value

            wandb_run.log(payload, step=trial.number)
        except Exception:
            # Never fail the search because of W&B logging
            return

    callbacks = [_wandb_trial_callback] if wandb_run is not None else None

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),  # don't crash on individual trial failures
        callbacks=callbacks,
    )

    search_time = time.time() - t_search_start

    # --- Results summary ---
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("\n" + "=" * 70)
    print("  Search Complete")
    print("=" * 70)
    print(f"  Total time:     {search_time:.1f}s ({search_time / 60:.1f} min)")
    print(f"  Completed:      {len(completed)} / {len(study.trials)} trials")
    print(f"  Failed/pruned:  {len(failed)}")

    if completed:
        best = study.best_trial
        print(f"\n  Best trial:     #{best.number}")
        print(f"  Best {metric}:  {best.value:.6f}")
        print("  Best params:")
        for k, v in best.params.items():
            if isinstance(v, float):
                print(f"    {k:40s} = {v:.6g}")
            else:
                print(f"    {k:40s} = {v}")

        # Extra metrics from user_attrs
        extra = {
            k: best.user_attrs.get(k)
            for k in ("test_acc", "test_auc", "test_f1", "test_recall", "test_loss")
            if best.user_attrs.get(k) is not None
        }
        if extra:
            print("  Full metrics:")
            for k, v in extra.items():
                print(f"    {k:40s} = {v:.6f}")

    print(f"{'=' * 70}\n")

    # --- Generate plots ---
    if not args.no_plots and completed:
        print("Generating visualizations...")
        from src.utils.hp_search_plots import generate_all_plots

        plots_dir = os.path.join(output_dir, "plots")
        generate_all_plots(
            study,
            plots_dir,
            study_name=f"{architecture} on {dataset_name}",
        )

    # --- Export best config ---
    if completed:
        best_cfg_path = os.path.join(output_dir, "best_config.yml")
        validation_seeds = args.validation_seeds or [0, 1, 2, 3, 4]
        export_best_config(
            study,
            base_cfg,
            search_space,
            best_cfg_path,
            validation_seeds=validation_seeds,
        )
        print(f"Best config exported to: {best_cfg_path}")
        print(
            f"  Run it with:  python experiments/robust_test_original_daqcnn.py "
            f"--config {best_cfg_path}\n"
        )

    # --- Validate top-K (optional) ---
    if args.validate_top_k > 0 and completed:
        val_dir = os.path.join(output_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)
        val_seeds = args.validation_seeds or [0, 1, 2, 3, 4]

        val_results = validate_top_k(
            study,
            base_cfg,
            list(search_space.keys()),
            architecture,
            val_dir,
            top_k=args.validate_top_k,
            seeds=val_seeds,
        )

        # Save validation summary
        val_summary_path = os.path.join(val_dir, "validation_summary.json")
        with open(val_summary_path, "w") as f:
            json.dump(val_results, f, indent=2, default=str)
        print(f"Validation summary saved to: {val_summary_path}")

    # --- Save final study summary ---
    best = study.best_trial if completed else None
    summary = {
        "study_name": study_name,
        "architecture": architecture,
        "dataset": dataset_name,
        "metric": metric,
        "direction": direction,
        "n_trials_requested": n_trials,
        "n_trials_completed": len(completed),
        "n_trials_failed": len(failed),
        "search_time_s": search_time,
        "best_trial": best.number if best is not None else None,
        "best_value": best.value if best is not None else None,
        "best_params": best.params if best is not None else None,
        "db_path": db_path,
    }
    summary_path = os.path.join(output_dir, "search_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll outputs saved to: {output_dir}")
    print("Search complete!")

    # Finalize W&B run (if enabled)
    if wandb_run is not None:
        try:
            # Headline results into summary
            wandb_run.summary["search/search_time_s"] = search_time
            wandb_run.summary["search/n_trials_completed"] = len(completed)
            wandb_run.summary["search/n_trials_failed_or_pruned"] = len(failed)
            if completed:
                wandb_run.summary["search/best_trial_number"] = int(
                    study.best_trial.number
                )
                wandb_run.summary["search/best_value"] = float(study.best_trial.value)

            # Optionally log a trials table for quick browsing
            if args.wandb_log_trials_table and completed:
                # Build columns from observed params
                param_names = set()
                for t in completed:
                    param_names.update(t.params.keys())
                param_names = sorted(param_names)

                columns = (
                    ["trial_number", "objective"]
                    + [f"param/{p}" for p in param_names]
                    + [
                        "attr/test_acc",
                        "attr/test_auc",
                        "attr/test_f1",
                        "attr/test_recall",
                        "attr/test_loss",
                        "attr/duration_s",
                    ]
                )
                table = wandb.Table(columns=columns)
                for t in completed:
                    row: List[Any] = [t.number, t.value]
                    for p in param_names:
                        row.append(t.params.get(p))
                    row.append(t.user_attrs.get("test_acc"))
                    row.append(t.user_attrs.get("test_auc"))
                    row.append(t.user_attrs.get("test_f1"))
                    row.append(t.user_attrs.get("test_recall"))
                    row.append(t.user_attrs.get("test_loss"))
                    row.append(t.user_attrs.get("duration_s"))
                    table.add_data(*row)
                wandb_run.log({"optuna/trials": table})

            # Save key files for convenience (does not force artifact upload)
            # Upload whenever they exist.
            key_files = [
                os.path.join(output_dir, "search_summary.json"),
                os.path.join(output_dir, "best_config.yml"),
                os.path.join(output_dir, "base_config.yml"),
                os.path.join(output_dir, "search_config.yml"),
                os.path.join(output_dir, "study.db"),
            ]

            # Validation outputs (if validation was enabled)
            val_dir = os.path.join(output_dir, "validation")
            key_files.append(os.path.join(val_dir, "validation_summary.json"))

            # Also upload per-rank validation aggregates if present
            if os.path.isdir(val_dir):
                try:
                    for root, _dirs, files in os.walk(val_dir):
                        for fname in files:
                            if fname.endswith(".json"):
                                key_files.append(os.path.join(root, fname))
                except Exception:
                    pass

            for p in key_files:
                try:
                    if os.path.exists(p):
                        wandb_run.save(p)
                except Exception:
                    pass
        finally:
            try:
                wandb_run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
