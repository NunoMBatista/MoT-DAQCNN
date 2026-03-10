"""
Hyperparameter search visualization utilities.

Generates publication-quality plots from Optuna study results, including:
    - Optimization history (best objective over trials)
    - Parallel coordinate plots (HP combinations colored by objective)
    - Hyperparameter importance (fANOVA-based bar chart)
    - Contour / interaction plots (pairwise HP heatmaps)
    - Slice plots (marginal effect of each HP)
    - Pipeline comparison radar chart
    - Results summary table (CSV + HTML)

All functions accept an Optuna ``Study`` object and an output directory.
Static PNG images are produced via matplotlib/seaborn (always available),
with optional interactive Plotly HTML exports where beneficial.
"""

from __future__ import annotations

import csv
import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend for servers / CI

# ---------------------------------------------------------------------------
# Optuna imports (optional Plotly-based plots degrade gracefully)
# ---------------------------------------------------------------------------
try:
    import optuna
    from optuna.trial import FrozenTrial, TrialState
except ImportError as _e:
    raise ImportError(
        "optuna is required for hp_search_plots.  Install with:  pip install optuna"
    ) from _e

try:
    from optuna.visualization import (
        plot_contour as optuna_plot_contour,
    )
    from optuna.visualization import (
        plot_optimization_history as optuna_plot_history,
    )
    from optuna.visualization import (
        plot_parallel_coordinate as optuna_plot_parallel,
    )
    from optuna.visualization import (
        plot_param_importances as optuna_plot_importances,
    )
    from optuna.visualization import (
        plot_slice as optuna_plot_slice,
    )

    _HAS_OPTUNA_VIZ = True
except Exception:
    _HAS_OPTUNA_VIZ = False

try:
    import plotly
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# =========================================================================
# Helpers
# =========================================================================


def _completed_trials(study: optuna.Study) -> List[FrozenTrial]:
    """Return only COMPLETE trials from a study."""
    return [t for t in study.trials if t.state == TrialState.COMPLETE]


def _safe_save_plotly(fig, path: str) -> bool:
    """Try to save a Plotly figure as static PNG + interactive HTML.

    Returns True if at least one format succeeded.
    """
    saved = False
    # Static PNG via kaleido
    try:
        fig.write_image(path)
        saved = True
    except Exception:
        pass
    # Interactive HTML
    try:
        html_path = os.path.splitext(path)[0] + ".html"
        fig.write_html(html_path)
        saved = True
    except Exception:
        pass
    return saved


def _param_names_from_study(study: optuna.Study) -> List[str]:
    """Collect all parameter names seen across completed trials."""
    names: set = set()
    for trial in _completed_trials(study):
        names.update(trial.params.keys())
    return sorted(names)


# =========================================================================
# 1. Optimization History
# =========================================================================


def plot_optimization_history(
    study: optuna.Study,
    save_path: str,
    *,
    title: str = "Optimization History",
) -> str:
    """Plot the best objective value achieved after each trial.

    Saves both a matplotlib PNG and (if Plotly is available) an interactive
    HTML.

    Args:
        study: Completed Optuna study.
        save_path: Output PNG path.
        title: Plot title.

    Returns:
        The path to the saved PNG.
    """
    trials = _completed_trials(study)
    if not trials:
        warnings.warn("No completed trials — skipping optimization history plot.")
        return save_path

    # -- Plotly (interactive) --
    if _HAS_OPTUNA_VIZ:
        try:
            fig = optuna_plot_history(study)
            fig.update_layout(title=title)
            _safe_save_plotly(fig, save_path)
        except Exception:
            pass

    # -- Matplotlib (always) --
    trial_numbers = [t.number for t in trials]
    values = [float(t.value) for t in trials if t.value is not None]
    trial_numbers = trial_numbers[: len(values)]  # align lengths

    direction = study.direction
    if direction == optuna.study.StudyDirection.MAXIMIZE:
        best_so_far = np.maximum.accumulate(values)
    else:
        best_so_far = np.minimum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        trial_numbers,
        values,
        alpha=0.45,
        s=30,
        label="Trial value",
        color="steelblue",
        edgecolors="navy",
        linewidths=0.3,
    )
    ax.plot(
        trial_numbers, best_so_far, color="crimson", linewidth=2, label="Best so far"
    )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Objective value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


# =========================================================================
# 2. Parallel Coordinate Plot
# =========================================================================


def plot_parallel_coordinates(
    study: optuna.Study,
    save_path: str,
    *,
    title: str = "Parallel Coordinate Plot",
    params: Optional[List[str]] = None,
) -> str:
    """Plot all HP axes colored by objective value.

    Args:
        study: Completed Optuna study.
        save_path: Output PNG path.
        title: Plot title.
        params: Subset of parameter names to show (None = all).

    Returns:
        Path to saved PNG.
    """
    trials = _completed_trials(study)
    if not trials:
        return save_path

    # Plotly interactive version (best for parallel coordinates)
    if _HAS_OPTUNA_VIZ:
        try:
            fig = optuna_plot_parallel(study, params=params)
            fig.update_layout(title=title)
            _safe_save_plotly(fig, save_path)
        except Exception:
            pass

    # Matplotlib fallback — simplified version
    all_params = params or _param_names_from_study(study)
    if not all_params:
        return save_path

    # Build data matrix
    data_rows: list = []
    obj_values: list = []
    for t in trials:
        row = []
        for p in all_params:
            v = t.params.get(p, np.nan)
            # Convert booleans / strings to numeric
            if isinstance(v, bool):
                v = int(v)
            elif isinstance(v, str):
                v = hash(v) % 1000  # rough numeric encoding
            row.append(v)
        data_rows.append(row)
        obj_values.append(t.value)

    data = np.array(data_rows, dtype=float)
    obj_values_arr = np.array(obj_values, dtype=float)

    # Normalise each column to [0, 1] for display
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    data_norm = (data - col_min) / col_range

    fig, ax = plt.subplots(figsize=(max(10, len(all_params) * 1.5), 6))
    cmap = matplotlib.colormaps["RdYlGn"]  # red = bad, green = good
    norm = mcolors.Normalize(
        vmin=float(np.nanmin(obj_values_arr)), vmax=float(np.nanmax(obj_values_arr))
    )

    x_ticks = np.arange(len(all_params))
    for i in range(len(trials)):
        color = cmap(norm(obj_values_arr[i]))
        ax.plot(x_ticks, data_norm[i], color=color, alpha=0.5, linewidth=1.2)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(all_params, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalised value")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Objective value")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


# =========================================================================
# 3. Hyperparameter Importances
# =========================================================================


def plot_hp_importances(
    study: optuna.Study,
    save_path: str,
    *,
    title: str = "Hyperparameter Importances",
) -> str:
    """Bar chart of fANOVA-based importance scores.

    Falls back to a simpler correlation-based importance if fANOVA fails
    (e.g. too few trials).

    Args:
        study: Completed Optuna study.
        save_path: Output PNG path.
        title: Plot title.

    Returns:
        Path to saved PNG.
    """
    trials = _completed_trials(study)
    if len(trials) < 4:
        warnings.warn(
            f"Only {len(trials)} completed trials — need ≥ 4 for importance analysis."
        )
        return save_path

    importances: Dict[str, float] = {}

    # Try Optuna's built-in fANOVA importance
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        # Fallback: Spearman rank correlation with objective
        all_params = _param_names_from_study(study)
        obj = np.array([t.value for t in trials])
        for p in all_params:
            vals = []
            for t in trials:
                v = t.params.get(p, np.nan)
                if isinstance(v, bool):
                    v = float(v)
                elif isinstance(v, str):
                    v = float(hash(v) % 10000)
                vals.append(float(v) if v is not None else np.nan)
            arr = np.array(vals)
            valid = ~np.isnan(arr)
            if valid.sum() > 2:
                try:
                    from scipy.stats import spearmanr

                    corr, _ = spearmanr(arr[valid], obj[valid])
                    importances[p] = float(abs(corr)) if not np.isnan(corr) else 0.0
                except ImportError:
                    importances[p] = 0.0
            else:
                importances[p] = 0.0

    if not importances:
        return save_path

    # Plotly interactive
    if _HAS_OPTUNA_VIZ:
        try:
            fig = optuna_plot_importances(study)
            fig.update_layout(title=title)
            _safe_save_plotly(fig, save_path)
        except Exception:
            pass

    # Matplotlib
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    scores = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    bars = ax.barh(
        range(len(names)), scores, color="steelblue", edgecolor="navy", linewidth=0.5
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


# =========================================================================
# 4. Contour Plots (pairwise interactions)
# =========================================================================


def plot_contour_plots(
    study: optuna.Study,
    save_path: str,
    *,
    params: Optional[List[str]] = None,
    title: str = "HP Interaction Contours",
    max_pairs: int = 10,
) -> str:
    """2-D contour heatmaps for the most important HP pairs.

    Args:
        study: Completed Optuna study.
        save_path: Output PNG path.
        params: Subset of parameter names. If None, picks the top-importance
            parameters (up to 5).
        title: Plot suptitle.
        max_pairs: Maximum number of pairs to plot.

    Returns:
        Path to saved PNG.
    """
    trials = _completed_trials(study)
    if len(trials) < 4:
        return save_path

    if params is None:
        try:
            imp = optuna.importance.get_param_importances(study)
            params = list(imp.keys())[:5]
        except Exception:
            params = _param_names_from_study(study)[:5]

    if len(params) < 2:
        return save_path

    # Plotly interactive (all-in-one)
    if _HAS_OPTUNA_VIZ:
        try:
            fig = optuna_plot_contour(study, params=params)
            fig.update_layout(title=title)
            _safe_save_plotly(fig, save_path)
        except Exception:
            pass

    # Matplotlib pairwise scatter with color
    from itertools import combinations

    pairs = list(combinations(params, 2))[:max_pairs]
    if not pairs:
        return save_path

    ncols = min(3, len(pairs))
    nrows = math.ceil(len(pairs) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )

    cmap = matplotlib.colormaps["viridis"]

    for idx, (p1, p2) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        x_vals, y_vals, c_vals = [], [], []
        for t in trials:
            v1 = t.params.get(p1)
            v2 = t.params.get(p2)
            if v1 is None or v2 is None:
                continue
            if isinstance(v1, bool):
                v1 = int(v1)
            if isinstance(v2, bool):
                v2 = int(v2)
            if isinstance(v1, str) or isinstance(v2, str):
                continue  # skip string params in scatter
            x_vals.append(float(v1))
            y_vals.append(float(v2))
            c_vals.append(t.value)

        if len(x_vals) < 3:
            ax.set_visible(False)
            continue

        sc = ax.scatter(
            x_vals,
            y_vals,
            c=c_vals,
            cmap=cmap,
            s=40,
            alpha=0.7,
            edgecolors="gray",
            linewidths=0.3,
        )
        ax.set_xlabel(p1, fontsize=9)
        ax.set_ylabel(p2, fontsize=9)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =========================================================================
# 5. Slice Plots (marginal effect of each HP)
# =========================================================================


def plot_slice_plots(
    study: optuna.Study,
    save_path: str,
    *,
    params: Optional[List[str]] = None,
    title: str = "Hyperparameter Slice Plots",
) -> str:
    """One subplot per HP: objective value vs. that HP's value.

    Args:
        study: Completed Optuna study.
        save_path: Output PNG path.
        params: Subset of parameter names. None = all.
        title: Plot suptitle.

    Returns:
        Path to saved PNG.
    """
    trials = _completed_trials(study)
    if not trials:
        return save_path

    if params is None:
        params = _param_names_from_study(study)
    if not params:
        return save_path

    # Plotly interactive
    if _HAS_OPTUNA_VIZ:
        try:
            fig = optuna_plot_slice(study, params=params)
            fig.update_layout(title=title)
            _safe_save_plotly(fig, save_path)
        except Exception:
            pass

    # Matplotlib
    ncols = min(4, len(params))
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, p in enumerate(params):
        ax = axes[idx // ncols][idx % ncols]
        x_vals, y_vals = [], []
        is_categorical = False
        for t in trials:
            v = t.params.get(p)
            if v is None:
                continue
            if isinstance(v, bool):
                v = int(v)
            elif isinstance(v, str):
                is_categorical = True
            x_vals.append(v)
            y_vals.append(t.value)

        if not x_vals:
            ax.set_visible(False)
            continue

        if is_categorical:
            # Box plot for categorical
            categories = sorted(set(x_vals), key=str)
            cat_data = {c: [] for c in categories}
            for x, y in zip(x_vals, y_vals):
                cat_data[x].append(y)
            ax.boxplot(
                [cat_data[c] for c in categories], labels=[str(c) for c in categories]
            )
        else:
            ax.scatter(
                x_vals,
                y_vals,
                alpha=0.5,
                s=25,
                color="steelblue",
                edgecolors="navy",
                linewidths=0.3,
            )
            # Trend line
            try:
                x_arr = np.array(x_vals, dtype=float)
                y_arr = np.array(y_vals, dtype=float)
                valid = np.isfinite(x_arr) & np.isfinite(y_arr)
                if valid.sum() > 2:
                    z = np.polyfit(x_arr[valid], y_arr[valid], 1)
                    p_line = np.poly1d(z)
                    x_sorted = np.sort(x_arr[valid])
                    ax.plot(x_sorted, p_line(x_sorted), "r--", alpha=0.5, linewidth=1.5)
            except Exception:
                pass

        ax.set_xlabel(p, fontsize=9)
        ax.set_ylabel("Objective" if idx % ncols == 0 else "")
        ax.grid(True, alpha=0.3)

    for idx in range(len(params), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =========================================================================
# 6. Top-K Trials Summary Table
# =========================================================================


def save_results_table(
    study: optuna.Study,
    output_dir: str,
    *,
    top_k: int = 20,
) -> Tuple[str, str]:
    """Save a sortable summary of all trials as CSV and HTML.

    Args:
        study: Completed Optuna study.
        output_dir: Directory to write files into.
        top_k: How many top trials to highlight in HTML.

    Returns:
        Tuple of (csv_path, html_path).
    """
    trials = _completed_trials(study)
    if not trials:
        return "", ""

    all_params = _param_names_from_study(study)
    direction = study.direction

    # Sort by objective
    if direction == optuna.study.StudyDirection.MAXIMIZE:
        sorted_trials = sorted(
            trials, key=lambda t: t.value or -float("inf"), reverse=True
        )
    else:
        sorted_trials = sorted(trials, key=lambda t: t.value or float("inf"))

    # --- CSV ---
    csv_path = os.path.join(output_dir, "hp_search_results.csv")
    fieldnames = ["rank", "trial", "value"] + all_params + ["duration_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, t in enumerate(sorted_trials, 1):
            row: Dict[str, Any] = {
                "rank": rank,
                "trial": t.number,
                "value": f"{t.value:.6f}" if t.value is not None else "",
            }
            for p in all_params:
                v = t.params.get(p)
                if isinstance(v, float):
                    row[p] = f"{v:.6g}"
                else:
                    row[p] = v
            dur = (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_start and t.datetime_complete
                else ""
            )
            row["duration_s"] = f"{dur:.1f}" if isinstance(dur, float) else ""
            writer.writerow(row)

    # --- HTML ---
    html_path = os.path.join(output_dir, "hp_search_results.html")
    with open(html_path, "w") as f:
        f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        f.write("<title>HP Search Results</title>\n")
        f.write("<style>\n")
        f.write("body{font-family:system-ui,sans-serif;margin:20px;}\n")
        f.write("table{border-collapse:collapse;width:100%;font-size:13px;}\n")
        f.write("th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;}\n")
        f.write(
            "th{background:#2c3e50;color:white;position:sticky;top:0;cursor:pointer;}\n"
        )
        f.write("tr:nth-child(even){background:#f9f9f9;}\n")
        f.write(f"tr:nth-child(-n+{top_k + 1}){{background:#e8f5e9;}}\n")
        f.write("tr:hover{background:#fff3e0;}\n")
        f.write(".best{background:#c8e6c9 !important;font-weight:bold;}\n")
        f.write("h1{color:#2c3e50;}\n")
        f.write("</style>\n</head><body>\n")
        f.write(f"<h1>Hyperparameter Search Results — {study.study_name}</h1>\n")
        f.write(
            f"<p>{len(trials)} completed trials | "
            f"Direction: {study.direction.name} | "
            f"Best value: {study.best_value:.6f} (trial {study.best_trial.number})</p>\n"
        )

        f.write("<table id='results'>\n<thead><tr>")
        for col in fieldnames:
            f.write(f"<th>{col}</th>")
        f.write("</tr></thead>\n<tbody>\n")

        for rank, t in enumerate(sorted_trials, 1):
            cls = " class='best'" if rank == 1 else ""
            f.write(f"<tr{cls}>")
            f.write(f"<td>{rank}</td><td>{t.number}</td>")
            f.write(f"<td>{t.value:.6f}</td>" if t.value is not None else "<td></td>")
            for p in all_params:
                v = t.params.get(p)
                if isinstance(v, float):
                    f.write(f"<td>{v:.6g}</td>")
                else:
                    f.write(f"<td>{v}</td>")
            dur = (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_start and t.datetime_complete
                else ""
            )
            f.write(f"<td>{dur:.1f}</td>" if isinstance(dur, float) else "<td></td>")
            f.write("</tr>\n")

        f.write("</tbody></table>\n")

        # Simple JS for column sorting
        f.write("""
<script>
document.querySelectorAll('#results th').forEach((th, idx) => {
  th.addEventListener('click', () => {
    const tbody = document.querySelector('#results tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = th.dataset.asc !== '1';
    th.dataset.asc = asc ? '1' : '0';
    rows.sort((a, b) => {
      const av = a.children[idx].textContent.trim();
      const bv = b.children[idx].textContent.trim();
      const an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
</script>
""")
        f.write("</body></html>")

    return csv_path, html_path


# =========================================================================
# 7. Radar Chart — Pipeline Comparison
# =========================================================================


def plot_pipeline_comparison_radar(
    pipeline_results: Dict[str, Dict[str, float]],
    save_path: str,
    *,
    metrics: Optional[List[str]] = None,
    title: str = "Pipeline Comparison",
) -> str:
    """Radar chart comparing best results across different pipelines.

    Args:
        pipeline_results: Dict mapping pipeline name to a dict of
            metric_name -> value. Example::

                {
                    "Original DAQCNN": {"test_acc": 0.85, "test_auc": 0.91, ...},
                    "TS-MoE": {"test_acc": 0.88, "test_auc": 0.94, ...},
                    "Gumbel MoE": {"test_acc": 0.86, "test_auc": 0.92, ...},
                }

        save_path: Output PNG path.
        metrics: Which metrics to plot. None = auto-detect from data.
        title: Plot title.

    Returns:
        Path to saved PNG.
    """
    if not pipeline_results:
        return save_path

    # Auto-detect metrics from first pipeline
    if metrics is None:
        all_metrics: set = set()
        for m in pipeline_results.values():
            all_metrics.update(m.keys())
        metrics = sorted(all_metrics)

    if not metrics:
        return save_path

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = matplotlib.colormaps["Set2"](np.linspace(0, 1, len(pipeline_results)))

    for (name, results), color in zip(pipeline_results.items(), colors):
        values = [results.get(m, 0.0) for m in metrics]
        values += values[:1]
        ax.plot(
            angles, values, "o-", linewidth=2, label=name, color=color, markersize=6
        )
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title(title, pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =========================================================================
# 8. Top-K Configs Bar Comparison
# =========================================================================


def plot_top_k_comparison(
    study: optuna.Study,
    save_path: str,
    *,
    top_k: int = 10,
    title: str = "Top Configurations",
) -> str:
    """Horizontal bar chart of the top-K trial objective values.

    Args:
        study: Completed Optuna study.
        save_path: Output PNG path.
        top_k: Number of top trials to show.
        title: Plot title.

    Returns:
        Path to saved PNG.
    """
    trials = _completed_trials(study)
    if not trials:
        return save_path

    direction = study.direction
    if direction == optuna.study.StudyDirection.MAXIMIZE:
        sorted_trials = sorted(
            trials, key=lambda t: t.value or -float("inf"), reverse=True
        )
    else:
        sorted_trials = sorted(trials, key=lambda t: t.value or float("inf"))

    top = sorted_trials[:top_k]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.5)))
    labels = [f"Trial {t.number}" for t in top]

    colors = matplotlib.colormaps["RdYlGn"](np.linspace(0.9, 0.3, len(top)))
    values_f = [float(t.value) for t in top if t.value is not None]
    bars = ax.barh(
        range(len(values_f)),
        values_f,
        color=colors[: len(values_f)],
        edgecolor="gray",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(values_f)))
    ax.set_yticklabels(labels[: len(values_f)], fontsize=9)
    ax.set_xlabel("Objective value")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, values_f):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {val:.4f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


# =========================================================================
# 9. Learning Curve Comparison (from user_attrs)
# =========================================================================


def plot_best_trial_learning_curves(
    study: optuna.Study,
    save_path: str,
    *,
    top_k: int = 5,
    title: str = "Learning Curves — Top Trials",
) -> str:
    """Plot training/validation loss curves for the top-K trials.

    Requires that the objective function stored ``train_losses`` and
    ``val_losses`` lists as user_attrs on each trial.

    Args:
        study: Completed Optuna study.
        save_path: Output PNG path.
        top_k: Number of top trials to show.
        title: Plot title.

    Returns:
        Path to saved PNG.
    """
    trials = _completed_trials(study)
    if not trials:
        return save_path

    direction = study.direction
    if direction == optuna.study.StudyDirection.MAXIMIZE:
        sorted_trials = sorted(
            trials, key=lambda t: t.value or -float("inf"), reverse=True
        )
    else:
        sorted_trials = sorted(trials, key=lambda t: t.value or float("inf"))

    top = [t for t in sorted_trials[:top_k] if "val_losses" in t.user_attrs]

    if not top:
        return save_path

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, len(top)))

    for t, color in zip(top, colors):
        label = f"T{t.number} ({t.value:.4f})"
        train = t.user_attrs.get("train_losses", [])
        val = t.user_attrs.get("val_losses", [])
        if train:
            axes[0].plot(train, label=label, color=color, alpha=0.8)
        if val:
            axes[1].plot(val, label=label, color=color, alpha=0.8)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


# =========================================================================
# 10. Master plotting function
# =========================================================================


def generate_all_plots(
    study: optuna.Study,
    output_dir: str,
    *,
    study_name: Optional[str] = None,
    pipeline_comparison: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, str]:
    """Generate all hyperparameter search visualizations at once.

    This is the main entry point — call this after the study finishes.

    Args:
        study: Completed Optuna study.
        output_dir: Directory to save all outputs.
        study_name: Human-readable name for titles. Defaults to study.study_name.
        pipeline_comparison: Optional dict for the radar comparison chart.
            Maps pipeline names to metric dicts.

    Returns:
        Dict mapping plot name to file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    name = study_name or study.study_name or "HP Search"
    paths: Dict[str, str] = {}

    # 1. Optimization history
    p = os.path.join(output_dir, "optimization_history.png")
    plot_optimization_history(study, p, title=f"{name} — Optimization History")
    paths["optimization_history"] = p

    # 2. Parallel coordinates
    p = os.path.join(output_dir, "parallel_coordinates.png")
    plot_parallel_coordinates(study, p, title=f"{name} — Parallel Coordinates")
    paths["parallel_coordinates"] = p

    # 3. HP importances
    p = os.path.join(output_dir, "hp_importances.png")
    plot_hp_importances(study, p, title=f"{name} — HP Importances")
    paths["hp_importances"] = p

    # 4. Contour plots
    p = os.path.join(output_dir, "contour_plots.png")
    plot_contour_plots(study, p, title=f"{name} — HP Interactions")
    paths["contour_plots"] = p

    # 5. Slice plots
    p = os.path.join(output_dir, "slice_plots.png")
    plot_slice_plots(study, p, title=f"{name} — Slice Plots")
    paths["slice_plots"] = p

    # 6. Top-K comparison
    p = os.path.join(output_dir, "top_k_comparison.png")
    plot_top_k_comparison(study, p, title=f"{name} — Top Configurations")
    paths["top_k_comparison"] = p

    # 7. Learning curves of best trials
    p = os.path.join(output_dir, "best_learning_curves.png")
    plot_best_trial_learning_curves(
        study, p, title=f"{name} — Best Trial Learning Curves"
    )
    paths["best_learning_curves"] = p

    # 8. Results table (CSV + HTML)
    csv_path, html_path = save_results_table(study, output_dir)
    paths["results_csv"] = csv_path
    paths["results_html"] = html_path

    # 9. Pipeline comparison radar (if provided)
    if pipeline_comparison:
        p = os.path.join(output_dir, "pipeline_comparison_radar.png")
        plot_pipeline_comparison_radar(
            pipeline_comparison, p, title=f"{name} — Pipeline Comparison"
        )
        paths["pipeline_comparison_radar"] = p

    # 10. Save best trial config as YAML for easy re-use
    try:
        best = study.best_trial
        best_config_path = os.path.join(output_dir, "best_trial_config.json")
        with open(best_config_path, "w") as f:
            json.dump(
                {
                    "trial_number": best.number,
                    "value": best.value,
                    "params": best.params,
                    "user_attrs": {
                        k: v
                        for k, v in best.user_attrs.items()
                        if k not in ("train_losses", "val_losses")
                    },
                },
                f,
                indent=2,
                default=str,
            )
        paths["best_trial_config"] = best_config_path
    except Exception:
        pass

    print(f"\n{'=' * 60}")
    print(f"HP Search Plots saved to: {output_dir}")
    print(f"{'=' * 60}")
    for plot_name, plot_path in paths.items():
        if plot_path:
            print(f"  {plot_name:30s} -> {os.path.basename(plot_path)}")
    print(f"{'=' * 60}\n")

    return paths
