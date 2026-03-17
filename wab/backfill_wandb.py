#!/usr/bin/env python3
"""
Backfill existing experiment outputs into Weights & Biases (W&B).

This script scans an `outputs/` directory for:
  1) Robust evaluation runs produced by:
       experiments/robust_test_original_daqcnn.py
     Expected files (at minimum):
       - config.yml
       - aggregate_metrics.json
     Optional:
       - individual_results.json
       - *.png plots

  2) Optuna hyperparameter searches produced by:
       experiments/hyperparameter_search.py
     Expected files (at minimum):
       - search_summary.json
     Optional (highly recommended):
       - study.db
       - best_config.yml
       - plots/*.png
       - validation/validation_summary.json

It creates/updates W&B runs and logs:
  - configs (YAML/JSON) as W&B config
  - aggregate metrics into W&B summary
  - (optional) trial tables for Optuna (if `study.db` exists)
  - images/plots as W&B media
  - artifacts for key files (optional)

Requirements:
  pip install wandb pyyaml optuna pandas

Auth:
  - Prefer setting `WANDB_API_KEY` in the environment.
  - This script also supports a local `.env` file in the repo root
    (loaded automatically) containing:
        WANDB_API_KEY=...
    If neither is present and this runs in a non-interactive environment
    (no TTY), W&B will error with "api_key not configured (no-tty)".

Typical usage:
  python wab/backfill_wandb.py \
    --project MoT-DAQCNN \
    --entity nunombatista-university-of-coimbra \
    --outputs-dir outputs \
    --mode online

HPC / offline:
  python wab/backfill_wandb.py ... --mode offline
  # later, on a node with internet:
  wandb sync wandb/offline-run-*

Notes:
- To avoid duplicating runs, this script generates a deterministic run id from the
  output directory path. Re-running is idempotent: it will resume that run id.
- By default it does NOT upload model checkpoints; you can enable artifacts.

Python compatibility:
- This script is compatible with Python 3.6+ (it avoids `from __future__ import annotations`).
"""

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional deps (we'll fail gracefully with clear errors)
try:
    import yaml  # type: ignore
except Exception as e:
    yaml = None  # type: ignore

try:
    import wandb  # type: ignore
except Exception as e:
    wandb = None  # type: ignore


# Optional: load .env (repo root) without adding any dependency.
def _load_dotenv_if_present(repo_root: Path) -> None:
    """
    Load KEY=VALUE lines from `<repo_root>/.env` into os.environ if absent.

    This is intentionally minimal and does not try to replicate python-dotenv.
    Supported:
      - blank lines
      - comments starting with #
      - optional leading `export `
      - quoted or unquoted values
    """
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    try:
        raw = env_path.read_text(encoding="utf-8")
    except Exception as e:
        _warn(f"Failed to read .env at {env_path}: {e}")
        return

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
            v = v[1:-1]
        # Do not override explicitly provided environment vars
        if k not in os.environ:
            os.environ[k] = v


# Optuna + pandas are optional for hp-search backfill
try:
    import optuna  # type: ignore
except Exception:
    optuna = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# ----------------------------
# Utilities
# ----------------------------


def _die(msg: str, code: int = 2) -> None:
    print(f"[backfill_wandb] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _warn(msg: str) -> None:
    print(f"[backfill_wandb] WARNING: {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[backfill_wandb] {msg}")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        _die("pyyaml is not installed. Install with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {"_yaml": obj}


def _safe_flatten(
    d: Dict[str, Any], prefix: str = "", out: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Flatten nested dict into one level using dot keys.
    Good for W&B config/logging without losing too much structure.
    """
    if out is None:
        out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _safe_flatten(v, key, out)
        else:
            out[key] = v
    return out


def _hash_run_id(s: str) -> str:
    # W&B run ids: short is fine; use deterministic hash to avoid duplicates.
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:12]


def _infer_kind(dir_path: Path) -> str:
    """
    Determine whether this directory looks like:
      - robust run
      - hp_search run
      - unknown
    """
    if (dir_path / "search_summary.json").exists() or (dir_path / "study.db").exists():
        return "hp_search"
    if (dir_path / "aggregate_metrics.json").exists() and (
        dir_path / "config.yml"
    ).exists():
        return "robust_eval"
    return "unknown"


def _collect_pngs(dir_path: Path, max_images: int) -> List[Path]:
    pngs: List[Path] = []
    # Prefer "plots/" for hp search, but include top-level images too.
    for p in sorted(dir_path.rglob("*.png")):
        # Skip huge nested caches if any; be conservative.
        if "/.git/" in str(p).replace("\\", "/"):
            continue
        pngs.append(p)
        if len(pngs) >= max_images:
            break
    return pngs


def _as_wandb_key(s: str) -> str:
    """
    Make metric keys safe for W&B (avoid spaces and weird chars).
    """
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_.\-\/]+", "", s)
    return s


def _log_images(run, images: List[Path], root: Path) -> None:
    for img_path in images:
        try:
            rel = img_path.relative_to(root)
        except Exception:
            rel = img_path.name
        key = _as_wandb_key(f"plots/{str(rel)}")
        run.log({key: wandb.Image(str(img_path))})


def _fmt_bytes(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return str(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def _describe_local_file(p: Path, base_dir: Path) -> str:
    try:
        rel = str(p.relative_to(base_dir))
    except Exception:
        rel = str(p)
    try:
        st = p.stat()
        return f"{rel} ({_fmt_bytes(st.st_size)})"
    except Exception:
        return rel


def _maybe_add_artifact(
    run,
    name: str,
    artifact_type: str,
    files: List[Path],
    base_dir: Path,
) -> None:
    """
    Upload a set of files as a single W&B Artifact.

    Notes:
    - Files are added with paths relative to `base_dir` when possible.
    - This function is best-effort: individual file failures are warned but do not abort.
    - This function prints exactly what it will attempt to add.
    """
    # De-duplicate while preserving order
    dedup: List[Path] = []
    seen: set = set()
    for p in files:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)

    existing: List[Path] = [p for p in dedup if p.exists() and p.is_file()]
    missing: List[Path] = [p for p in dedup if not p.exists()]

    _info(f"Artifact {name} (type={artifact_type}) for {base_dir}:")
    _info(f"  candidate files: {len(dedup)}")
    _info(f"  existing files:  {len(existing)}")
    _info(f"  missing files:   {len(missing)}")

    if missing:
        # Print up to 50 missing to keep logs readable
        _info("  missing (first 50):")
        for p in missing[:50]:
            _info(f"    - {_describe_local_file(p, base_dir)}")

    if not existing:
        _warn(f"Artifact {name}: no existing files to add; skipping artifact upload.")
        return

    # Print exactly what will be added
    _info("  adding (all):")
    for p in existing:
        _info(f"    + {_describe_local_file(p, base_dir)}")

    art = wandb.Artifact(name=name, type=artifact_type)
    added = 0
    for p in existing:
        try:
            rel_name = str(p.relative_to(base_dir))
        except Exception:
            rel_name = p.name
        try:
            art.add_file(str(p), name=rel_name)
            added += 1
        except Exception as e:
            _warn(f"Failed to add artifact file {p}: {e}")

    if added > 0:
        run.log_artifact(art)
        _info(f"  uploaded artifact with {added} file(s).")
    else:
        _warn(f"Artifact {name}: no files were added successfully; nothing uploaded.")


# ----------------------------
# Robust eval backfill
# ----------------------------


def _backfill_robust_eval(
    run,
    out_dir: Path,
    *,
    max_images: int,
    enable_artifacts: bool,
) -> None:
    cfg = _read_yaml(out_dir / "config.yml")
    agg = _read_json(out_dir / "aggregate_metrics.json")

    # Put core metadata into config
    flat_cfg = _safe_flatten(cfg)
    run.config.update(flat_cfg, allow_val_change=True)

    # Also store lightweight provenance
    run.config.update(
        {
            "backfill.kind": "robust_eval",
            "backfill.output_dir": str(out_dir),
            "backfill.timestamp": int(time.time()),
        },
        allow_val_change=True,
    )

    # Metrics: aggregate_metrics.json has many entries like {mean,std,min,max,values}
    # We'll store mean/std into summary, and optionally a table of per-seed values.
    summary_updates: Dict[str, Any] = {}
    value_rows: List[Dict[str, Any]] = []

    for metric_name, stats in agg.items():
        if metric_name == "model_metadata":
            # Keep as config-ish info (summary is fine too, but config is more searchable)
            if isinstance(stats, dict):
                run.config.update(
                    {f"model_metadata.{k}": v for k, v in stats.items()},
                    allow_val_change=True,
                )
            continue

        if not isinstance(stats, dict):
            continue

        # mean/stats
        for k in ("mean", "std", "min", "max"):
            if k in stats:
                summary_updates[_as_wandb_key(f"{metric_name}/{k}")] = stats[k]

        # per-seed values
        vals = stats.get("values")
        if isinstance(vals, list):
            for i, v in enumerate(vals):
                value_rows.append(
                    {
                        "metric": metric_name,
                        "index": i,
                        "value": v,
                    }
                )

    # Update summary once
    for k, v in summary_updates.items():
        run.summary[k] = v

    # Optional individual results table (if available)
    ind_path = out_dir / "individual_results.json"
    if ind_path.exists():
        try:
            ind = _read_json(ind_path)
            if isinstance(ind, list):
                # Log a W&B Table with per-seed key metrics if possible
                cols = [
                    "seed",
                    "test_loss",
                    "test_acc",
                    "test_auc",
                    "test_f1",
                    "test_recall",
                    "final_train_loss",
                    "final_val_loss",
                    "final_train_acc",
                    "final_val_acc",
                    "architecture",
                ]
                table = wandb.Table(columns=cols)
                for r in ind:
                    if not isinstance(r, dict):
                        continue
                    table.add_data(
                        r.get("seed"),
                        r.get("test_loss"),
                        r.get("test_acc"),
                        r.get("test_auc"),
                        r.get("test_f1"),
                        r.get("test_recall"),
                        r.get("final_train_loss"),
                        r.get("final_val_loss"),
                        r.get("final_train_acc"),
                        r.get("final_val_acc"),
                        r.get("architecture"),
                    )
                run.log({"results/individual": table})
        except Exception as e:
            _warn(f"Failed reading {ind_path}: {e}")

    # If we don't have individual results, at least log the aggregate values table.
    if value_rows:
        try:
            table = wandb.Table(columns=["metric", "index", "value"])
            for row in value_rows:
                table.add_data(row["metric"], row["index"], row["value"])
            run.log({"results/aggregate_values": table})
        except Exception as e:
            _warn(f"Failed logging aggregate values table: {e}")

    # Log plots/images
    images = _collect_pngs(out_dir, max_images=max_images)
    if images:
        _log_images(run, images, out_dir)

    # Optional artifacts: configs + jsons + plots (NOT checkpoints)
    if enable_artifacts:
        files: List[Path] = []

        # Always include key result files if present
        for p in [
            out_dir / "config.yml",
            out_dir / "aggregate_metrics.json",
            out_dir / "individual_results.json",
        ]:
            if p.exists():
                files.append(p)

        # Include any JSONs under this run directory (future-proofing)
        # (bounded by max depth naturally via rglob)
        for p in sorted(out_dir.rglob("*.json")):
            if p.exists() and p not in files:
                files.append(p)

        # Add limited number of images
        files.extend(images[: max(0, min(max_images, 50))])

        art_name = f"robust_eval_{_hash_run_id(str(out_dir))}"
        _maybe_add_artifact(run, art_name, "backfill_outputs", files, out_dir)


# ----------------------------
# HP search backfill (Optuna)
# ----------------------------


def _optuna_trials_table_from_db(db_path: Path) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Returns (wandb.Table or None, stats_dict).
    Requires optuna. Uses pandas if available but does not strictly require it.
    """
    if optuna is None:
        return None, {"note": "optuna not installed; skipping study.db parsing"}

    storage = f"sqlite:///{db_path.resolve()}"
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if not summaries:
        return None, {"note": "no studies found in db"}

    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    stats = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "n_trials_failed": len(failed),
        "best_value": float(study.best_value) if completed else None,
        "best_trial_number": int(study.best_trial.number) if completed else None,
        "best_params": dict(study.best_trial.params) if completed else None,
        "best_user_attrs": dict(study.best_trial.user_attrs) if completed else None,
    }

    # Build a table with common columns:
    # - number, value, state, params..., user_attrs...
    # We'll keep it simple: include params and a handful of common attrs.
    # If pandas available, we can build a dataframe first; otherwise, manual.
    param_names: set[str] = set()
    attr_names: set[str] = set()
    for t in completed:
        param_names.update(t.params.keys())
        attr_names.update(t.user_attrs.keys())

    # Keep attr list bounded (user_attrs can be huge)
    common_attrs = [
        a
        for a in sorted(attr_names)
        if a in {"test_acc", "test_auc", "test_f1", "test_recall", "test_loss"}
    ]
    # You can add more by editing here.

    columns = (
        ["trial_number", "objective"]
        + [f"param/{p}" for p in sorted(param_names)]
        + [f"attr/{a}" for a in common_attrs]
    )
    table = wandb.Table(columns=columns)

    for t in completed:
        row: List[Any] = [t.number, t.value]
        for p in sorted(param_names):
            row.append(t.params.get(p))
        for a in common_attrs:
            row.append(t.user_attrs.get(a))
        table.add_data(*row)

    return table, stats


def _backfill_hp_search(
    run,
    out_dir: Path,
    *,
    max_images: int,
    enable_artifacts: bool,
) -> None:
    # search_summary.json is the minimum reliable metadata
    summary_path = out_dir / "search_summary.json"
    if not summary_path.exists():
        _die(f"hp_search directory missing search_summary.json: {out_dir}")

    search_summary = _read_json(summary_path)

    # Put search summary into config (flattened)
    if isinstance(search_summary, dict):
        run.config.update(
            _safe_flatten({"hp_search": search_summary}), allow_val_change=True
        )

    run.config.update(
        {
            "backfill.kind": "hp_search",
            "backfill.output_dir": str(out_dir),
            "backfill.timestamp": int(time.time()),
        },
        allow_val_change=True,
    )

    # Store top-level key results in summary for quick comparison
    for k in [
        "hp_search.best_value",
        "hp_search.n_trials_completed",
        "hp_search.n_trials_failed",
        "hp_search.search_time_s",
    ]:
        if k in run.config:
            run.summary[_as_wandb_key(k)] = run.config[k]

    # Log Optuna trials table if study.db exists
    db_path = out_dir / "study.db"
    if db_path.exists():
        try:
            table, stats = _optuna_trials_table_from_db(db_path)
            # stats into summary/config
            run.config.update(_safe_flatten({"optuna": stats}), allow_val_change=True)
            if table is not None:
                run.log({"optuna/trials": table})

            # headline stats in summary
            for sk in [
                "optuna.best_value",
                "optuna.n_trials_completed",
                "optuna.n_trials_total",
            ]:
                if sk in run.config:
                    run.summary[_as_wandb_key(sk)] = run.config[sk]
        except Exception as e:
            _warn(f"Failed to parse Optuna db {db_path}: {e}")
    else:
        _warn(f"No study.db found in {out_dir}; skipping Optuna trial backfill.")

    # Log validation results if they exist
    # - validation/validation_summary.json
    # - per-rank/per-trial aggregates under validation/**
    val_dir = out_dir / "validation"
    if val_dir.exists() and val_dir.is_dir():
        # Prefer the summary if present
        val_summary = val_dir / "validation_summary.json"
        if val_summary.exists():
            try:
                v = _read_json(val_summary)
                # Put in config for reference (can be large; keep in config as json-ish)
                run.config.update({"validation_summary": v}, allow_val_change=True)
            except Exception as e:
                _warn(f"Failed reading validation summary: {e}")

        # Also capture any other validation JSONs as a W&B table (lightweight)
        try:
            json_paths = sorted(p for p in val_dir.rglob("*.json") if p.is_file())
            if json_paths:
                table = wandb.Table(columns=["path", "json"])
                for p in json_paths:
                    try:
                        table.add_data(str(p.relative_to(out_dir)), _read_json(p))
                    except Exception:
                        # If a JSON can't be parsed, log as text path only
                        table.add_data(str(p.relative_to(out_dir)), None)
                run.log({"validation/json_files": table})
        except Exception as e:
            _warn(f"Failed logging validation JSON table: {e}")

    # Log plots/images
    images = _collect_pngs(out_dir, max_images=max_images)
    if images:
        _log_images(run, images, out_dir)

    # Optional artifacts: include DB + key configs + validation JSONs + plots
    if enable_artifacts:
        files: List[Path] = []

        # Key top-level files
        for p in [
            out_dir / "search_summary.json",
            out_dir / "base_config.yml",
            out_dir / "search_config.yml",
            out_dir / "best_config.yml",
            out_dir / "study.db",
        ]:
            if p.exists():
                files.append(p)

        # Validation JSONs (always include whenever they exist)
        val_dir = out_dir / "validation"
        if val_dir.exists() and val_dir.is_dir():
            for p in sorted(val_dir.rglob("*.json")):
                if p.exists() and p not in files:
                    files.append(p)

        # Any additional JSON summaries under this hp search folder (future-proofing)
        for p in sorted(out_dir.rglob("*.json")):
            if p.exists() and p not in files:
                files.append(p)

        # Attach limited number of plots/images
        files.extend(images[: max(0, min(max_images, 50))])

        art_name = f"hp_search_{_hash_run_id(str(out_dir))}"
        _maybe_add_artifact(run, art_name, "backfill_outputs", files, out_dir)


# ----------------------------
# Scan + dispatch
# ----------------------------


@dataclasses.dataclass(frozen=True)
class CandidateDir:
    path: Path
    kind: str


def _find_candidate_dirs(outputs_dir: Path) -> List[CandidateDir]:
    if not outputs_dir.exists():
        _die(f"outputs dir does not exist: {outputs_dir}")

    candidates: List[CandidateDir] = []
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        kind = _infer_kind(child)
        if kind == "unknown":
            continue
        candidates.append(CandidateDir(path=child, kind=kind))
    return candidates


def _make_run_name(kind: str, out_dir: Path) -> str:
    # Use directory name as run name for easy mapping
    return f"{kind}:{out_dir.name}"


def _group_from_kind(kind: str) -> str:
    return "hp_search" if kind == "hp_search" else "robust_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill existing outputs to W&B")
    parser.add_argument(
        "--entity", type=str, default=None, help="W&B entity (user or team)."
    )
    parser.add_argument(
        "--project", type=str, required=True, help="W&B project name (e.g. MoT-DAQCNN)."
    )
    parser.add_argument(
        "--outputs-dir", type=str, default="outputs", help="Path to outputs/ directory."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (online/offline/disabled).",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="all",
        choices=["all", "hp_search", "robust_eval"],
        help="Which kinds of outputs to backfill.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=30,
        help="Max number of PNG images to upload per run.",
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="If set, also upload key files as a W&B Artifact (can increase upload size).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and print what would be uploaded, but don't contact W&B.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only backfill a single output directory name (e.g. run_pneumoniamnist_20260101_120000).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=[],
        help="Optional tags to apply to all backfilled runs.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="Backfilled from existing outputs directory.",
        help="Run notes.",
    )
    args = parser.parse_args()

    if wandb is None:
        _die("wandb is not installed. Install with: pip install wandb")

    # Load `.env` from repo root (one directory above this script).
    # This addresses non-interactive environments where W&B can't prompt.
    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv_if_present(repo_root)

    outputs_dir = Path(args.outputs_dir).resolve()

    os.environ["WANDB_MODE"] = args.mode
    # Recommended on HPC to avoid hanging on git status, etc.
    os.environ.setdefault("WANDB_SILENT", "true")

    # Fail early with a clear message instead of W&B "no-tty" prompt failure.
    if args.mode != "disabled":
        if not os.environ.get("WANDB_API_KEY"):
            _die(
                "WANDB_API_KEY not set. Either:\n"
                "  1) `wandb login` (interactive), or\n"
                "  2) export WANDB_API_KEY=... in your shell, or\n"
                "  3) create a `.env` file at the repo root with `WANDB_API_KEY=...`"
            )

    candidates = _find_candidate_dirs(outputs_dir)
    if args.only:
        candidates = [c for c in candidates if c.path.name == args.only]
        if not candidates:
            _die(f"--only matched nothing under {outputs_dir}: {args.only}")

    if args.include != "all":
        candidates = [c for c in candidates if c.kind == args.include]

    if not candidates:
        _info(f"No candidates found under: {outputs_dir}")
        return

    _info(f"Found {len(candidates)} candidate run directories under: {outputs_dir}")
    for c in candidates:
        _info(f"  - {c.kind:11s} {c.path.name}")

    if args.dry_run or args.mode == "disabled":
        _info("Dry-run/disabled mode: not uploading to W&B.")
        return

    # Backfill each candidate directory as a deterministic run id
    for c in candidates:
        out_dir = c.path
        kind = c.kind

        run_id = _hash_run_id(str(out_dir))
        run_name = _make_run_name(kind, out_dir)
        group = _group_from_kind(kind)

        _info(f"Uploading {kind} from {out_dir} as run id={run_id} name={run_name}")

        # resume="allow" makes it idempotent across re-runs without clobbering
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            id=run_id,
            name=run_name,
            group=group,
            tags=list(args.tags),
            notes=args.notes,
            resume="allow",
            reinit=True,
            config={
                "backfill.source_dir": str(out_dir),
                "backfill.kind": kind,
            },
        )

        try:
            if kind == "robust_eval":
                _backfill_robust_eval(
                    run,
                    out_dir,
                    max_images=args.max_images,
                    enable_artifacts=args.artifacts,
                )
            elif kind == "hp_search":
                _backfill_hp_search(
                    run,
                    out_dir,
                    max_images=args.max_images,
                    enable_artifacts=args.artifacts,
                )
            else:
                _warn(f"Unknown kind {kind} for {out_dir}; skipping.")
        finally:
            run.finish()

    _info("Done.")


if __name__ == "__main__":
    main()
