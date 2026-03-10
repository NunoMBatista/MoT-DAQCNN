# Hyperparameter Search System

> **Module:** `experiments/hyperparameter_search.py`
> **Plots:** `src/utils/hp_search_plots.py`
> **Configs:** `configs/hp_search/`

## Overview

The hyperparameter search system uses [Optuna](https://optuna.org/) to efficiently explore the hyperparameter space of all three MoT-DAQCNN pipelines:

| Pipeline | Architecture flag | Base config example |
|---|---|---|
| Original DAQCNN | `original` | `breast_mnist_multi_seed_3kern.yml` |
| Teacher-Student MoE | `TS-MoE` | `breast_mnist_ts_moe_3kern.yml` |
| Gumbel-Softmax MoE | `gumbel` | `breast_mnist_gumbel_3kern.yml` |

The search runs each trial with a **single seed** for speed (fast exploration), then the best configurations can be validated with full multi-seed runs using the existing `robust_test_original_daqcnn.py` script.

### Why Optuna?

- **TPE sampler** (Tree-Parzen Estimator): Bayesian-style search that focuses on promising regions of the HP space, much more efficient than grid or random search.
- **Persistent storage**: SQLite backend lets you resume interrupted searches, inspect results mid-run, and compare across studies.
- **Built-in pruning**: Failed trials are handled gracefully without crashing the entire search.
- **Lightweight**: No heavy distributed computing dependencies (unlike Ray Tune).

---

## Quick Start

### 1. Run a search

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist_gumbel_3kern.yml \
    --search-config configs/hp_search/breast_mnist_gumbel.yml \
    --n-trials 60
```

### 2. Resume an interrupted search

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist_gumbel_3kern.yml \
    --search-config configs/hp_search/breast_mnist_gumbel.yml \
    --n-trials 100 \
    --resume \
    --output-dir outputs/hp_search_gumbel_breastmnist_20250101_120000
```

### 3. Run a search with automatic multi-seed validation of top configs

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist_ts_moe_3kern.yml \
    --search-config configs/hp_search/breast_mnist_ts_moe.yml \
    --n-trials 80 \
    --validate-top-k 5 \
    --validation-seeds 0 1 2 3 4
```

### 4. Compare best results across pipelines

```bash
python experiments/hyperparameter_search.py --compare \
    outputs/hp_search_original_breastmnist_*/study.db \
    outputs/hp_search_gumbel_breastmnist_*/study.db \
    outputs/hp_search_TS-MoE_breastmnist_*/study.db
```

### 5. Run the best config with full multi-seed evaluation

After a search, the best config is auto-exported:

```bash
python experiments/robust_test_original_daqcnn.py \
    --config outputs/hp_search_gumbel_breastmnist_<timestamp>/best_config.yml
```

---

## Search Config Format

Search configs are YAML files with two sections: `settings` and `search_space`.

### Settings

```yaml
settings:
  metric: test_acc       # Which metric to optimise
  direction: maximize    # "maximize" or "minimize"
  study_name: null       # Auto-generated if null
```

### Search Space

Each entry maps a **dotted config path** (e.g., `optim.lr`) to a **distribution specification**:

```yaml
search_space:
  optim.lr:
    type: loguniform
    low: 1.0e-5
    high: 5.0e-3

  model.dropout:
    type: uniform
    low: 0.2
    high: 0.7

  optim.patience:
    type: int
    low: 5
    high: 25
    step: 5

  model.activation:
    type: categorical
    choices: ["relu", "gelu"]

  gumbel.router_hidden_dim:
    type: fixed
    value: 64
```

### Supported Distribution Types

| Type | Parameters | Description | Example |
|------|-----------|-------------|---------|
| `uniform` | `low`, `high`, optional `step` | Uniform float in [low, high] | LR schedule eta_min |
| `loguniform` | `low`, `high` | Log-uniform float (for values spanning orders of magnitude) | Learning rate, weight decay |
| `int` | `low`, `high`, optional `step`, optional `log` | Uniform integer | Patience, epochs, hidden dims |
| `categorical` | `choices` (list) | One value from a discrete set | Activation, boolean flags, batch size |
| `fixed` | `value` | Not tuned — injects a constant | Override base config without a new YAML |

### Dotted Path Convention

The dotted path corresponds to nested YAML keys. For example:

| Dotted path | YAML location |
|---|---|
| `optim.lr` | `optim: { lr: ... }` |
| `model.dropout` | `model: { dropout: ... }` |
| `gumbel.sparsity_weight` | `gumbel: { sparsity_weight: ... }` |
| `ts_moe.lambda_max` | `ts_moe: { lambda_max: ... }` |
| `dataset.batch_size` | `dataset: { batch_size: ... }` |

---

## Provided Search Configs

### `configs/hp_search/breast_mnist_original.yml`

Tunes the **Original DAQCNN** on BreastMNIST. Covers:

| Parameter | Type | Range | Why |
|---|---|---|---|
| `optim.lr` | loguniform | [1e-5, 5e-3] | Most impactful HP universally |
| `optim.weight_decay` | loguniform | [1e-8, 1e-3] | L2 regularisation strength |
| `optim.grad_clip` | uniform | [0.5, 5.0] | Gradient stability |
| `optim.patience` | int | [5, 25] step 5 | Early stopping sensitivity |
| `optim.use_scheduler` | categorical | [true, false] | Cosine annealing toggle |
| `model.dropout` | uniform | [0.2, 0.7] | Head regularisation |
| `model.activation` | categorical | [relu, gelu] | Nonlinearity choice |
| `dataset.batch_size` | categorical | [16, 32, 64] | Gradient noise level |

**8 parameters, ~60 trials recommended.**

### `configs/hp_search/breast_mnist_gumbel.yml`

Tunes the **Gumbel-Softmax MoE** on BreastMNIST. Adds Gumbel-specific parameters on top of the standard optimizer/model HPs:

| Parameter | Type | Range | Why |
|---|---|---|---|
| `gumbel.sparsity_weight` | loguniform | [1e-4, 0.5] | Budget loss λ — too high collapses routing |
| `gumbel.tau_start` | uniform | [0.5, 5.0] | Initial temperature — exploration breadth |
| `gumbel.tau_end` | loguniform | [0.01, 0.5] | Final temperature — discreteness |
| `gumbel.tau_anneal_epochs` | int | [30, 100] step 10 | Annealing speed |
| `gumbel.hard_after_epoch` | int | [40, 120] step 10 | When to switch to STE |
| *(plus all optimizer/model HPs)* | | | |

**13 parameters, ~80 trials recommended.**

### `configs/hp_search/breast_mnist_ts_moe.yml`

Tunes the **Teacher-Student MoE** on BreastMNIST. This is the most complex pipeline with three phases, so it has the most tunable parameters:

**Phase 1 — Teacher:**

| Parameter | Type | Range | Why |
|---|---|---|---|
| `ts_moe.lambda_max` | loguniform | [0.01, 2.0] | Entropy penalty strength |
| `ts_moe.lambda_warmup_epochs` | int | [10, 80] step 10 | Warmup schedule length |
| `ts_moe.lambda_entropy_start` | uniform | [0.0, 0.1] | Initial penalty (usually 0) |
| `ts_moe.attention_hidden_dim` | categorical | [16, 32, 64] | SE gate capacity |
| `ts_moe.attention_gate_zero_init` | categorical | [true, false] | Uniform initial routing |

**Phase 2 — Student:**

| Parameter | Type | Range | Why |
|---|---|---|---|
| `ts_moe.student_hidden_dims` | categorical | [[32,16], [64,32], [128,64]] | MLP capacity |
| `ts_moe.student_lr` | loguniform | [1e-4, 1e-2] | Student learning rate |
| `ts_moe.student_epochs` | int | [15, 60] step 5 | Training budget |
| `ts_moe.student_weighted_ce` | categorical | [true, false] | Handle label imbalance |
| `ts_moe.student_balanced_sampler` | categorical | [true, false] | Aggressive balancing |
| `ts_moe.confidence_threshold` | uniform | [0.0, 0.6] | Filter ambiguous labels |
| `ts_moe.student_kd_alpha` | uniform | [0.0, 1.0] | Hard vs soft distillation |
| `ts_moe.student_kd_temperature` | uniform | [1.0, 8.0] | Soft-target temperature |
| `ts_moe.student_features_stats` | categorical | [true, false] | Extra feature: mean/std/min/max |
| `ts_moe.student_features_range_energy` | categorical | [true, false] | Extra feature: range/L2/L1 |
| `ts_moe.student_features_gradients` | categorical | [true, false] | Extra feature: gradients |

**Phase 3 — Final Classifier:**

| Parameter | Type | Range | Why |
|---|---|---|---|
| `ts_moe.final_lr` | loguniform | [1e-5, 5e-3] | Classifier learning rate |
| `ts_moe.final_epochs` | int | [50, 150] step 10 | Training budget |
| `ts_moe.use_mask_channel` | categorical | [true, false] | Extra routing mask input |

**27 parameters total, ~80+ trials recommended.**

---

## Output Structure

After a search, the output directory contains:

```
outputs/hp_search_gumbel_breastmnist_20250101_120000/
├── base_config.yml              # Copy of the base config used
├── search_config.yml            # Copy of the search space config
├── search_summary.json          # Study metadata and best trial info
├── best_config.yml              # Ready-to-run YAML with best HPs injected
├── study.db                     # SQLite database (for resumption / inspection)
│
├── trials/                      # Per-trial outputs
│   ├── trial_0000/
│   │   ├── config.yml           # This trial's full config
│   │   ├── result.json          # Metrics summary
│   │   ├── loss_curve_seed_42.png
│   │   ├── roc_curve_seed_42.png
│   │   └── ...                  # All standard per-seed outputs
│   ├── trial_0001/
│   └── ...
│
├── plots/                       # Search-level visualisations
│   ├── optimization_history.png # Best objective over trials
│   ├── optimization_history.html
│   ├── parallel_coordinates.png # All HPs colored by objective
│   ├── parallel_coordinates.html
│   ├── hp_importances.png       # fANOVA importance bar chart
│   ├── hp_importances.html
│   ├── contour_plots.png        # Pairwise HP interaction heatmaps
│   ├── contour_plots.html
│   ├── slice_plots.png          # Marginal effect of each HP
│   ├── slice_plots.html
│   ├── top_k_comparison.png     # Bar chart of top-10 trial values
│   ├── best_learning_curves.png # Train/val loss of top-5 trials
│   ├── hp_search_results.csv    # Full sortable results table
│   ├── hp_search_results.html   # Interactive HTML table
│   └── best_trial_config.json   # Best trial params + metadata
│
└── validation/                  # (if --validate-top-k was used)
    ├── validation_rank1_trial42/
    │   ├── config.yml
    │   ├── seed_0/ ...
    │   ├── seed_1/ ...
    │   └── validation_aggregate.json
    └── validation_summary.json
```

---

## Visualisations Explained

### 1. Optimization History (`optimization_history.png`)

Shows the objective value (e.g., test accuracy) for each completed trial as a scatter plot, with a red line tracking the best value found so far.

**What to look for:**
- **Convergence**: If the red line flattens, the search has likely found a near-optimal region. You can stop.
- **No convergence**: If it's still climbing/descending at the end, run more trials.
- **Scattered points**: High variance suggests the HP space has many local optima or that some HPs are very sensitive.

### 2. Parallel Coordinate Plot (`parallel_coordinates.png`)

Each vertical axis represents one hyperparameter. Each trial is drawn as a polyline connecting its HP values, colored by objective value (green = good, red = bad).

**What to look for:**
- **Clustered green lines**: A narrow band of good values for a given HP suggests a clear optimal range.
- **Uniform color distribution**: That HP probably doesn't matter much.
- **Crossing patterns**: Interactions between HPs (e.g., high LR works only with high weight decay).

### 3. Hyperparameter Importances (`hp_importances.png`)

Bar chart ranking HPs by their contribution to objective variance (fANOVA analysis). Falls back to Spearman correlation if fANOVA fails with too few trials.

**What to look for:**
- **Dominant parameters**: Focus your future tuning on the top 2–3 parameters.
- **Negligible parameters**: Fix these at their default values and remove from the search to reduce dimensionality.

### 4. Contour Plots (`contour_plots.png`)

Pairwise scatter plots for the most important HP pairs, colored by objective value. Reveals interactions between HPs.

**What to look for:**
- **Diagonal colour patterns**: The two HPs interact — good values of one depend on the other.
- **Horizontal/vertical bands**: Only one of the two HPs matters in that pair.
- **Hot spots**: Small regions where both HPs must be in a specific range.

### 5. Slice Plots (`slice_plots.png`)

One subplot per HP showing objective value vs. that HP's value. Includes a linear trend line.

**What to look for:**
- **Strong trend**: That HP has a clear effect — follow the trend.
- **U-shape or inverted U**: The HP has an optimal range; extremes are bad.
- **Flat cloud**: That HP has little effect — consider fixing it.

### 6. Top-K Comparison (`top_k_comparison.png`)

Horizontal bar chart showing the objective values of the top 10 trial configurations.

**What to look for:**
- **Tight cluster at the top**: Multiple good configs exist — the pipeline is robust.
- **Big gap between #1 and #2**: The best config is special — validate carefully with multi-seed runs.

### 7. Best Trial Learning Curves (`best_learning_curves.png`)

Training and validation loss curves for the top 5 trials, overlaid.

**What to look for:**
- **Overfitting**: Training loss decreasing while validation loss increases.
- **Underfitting**: Both losses plateau high — need more capacity or training.
- **Training instability**: Spiky loss curves suggest LR or gradient issues.

### 8. Results Table (`hp_search_results.html`)

Interactive HTML table with all trials, sorted by objective value. Click column headers to sort. Top trials are highlighted in green. Also exported as CSV for programmatic analysis.

### 9. Pipeline Comparison Radar (`pipeline_comparison_radar.png`)

Only generated in `--compare` mode. Radar chart comparing the best results from different pipelines across multiple metrics (accuracy, AUC, F1, recall).

---

## CLI Reference

```
python experiments/hyperparameter_search.py [OPTIONS]
```

### Search Mode (default)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | *required* | Path to the base YAML config |
| `--search-config` | str | *required* | Path to the search space YAML |
| `--n-trials` | int | 50 | Number of Optuna trials |
| `--seed` | int | 42 | Fixed seed for single-seed exploration |
| `--metric` | str | from search config | Metric to optimise |
| `--direction` | str | from search config | `maximize` or `minimize` |
| `--resume` | flag | false | Resume from existing SQLite DB |
| `--output-dir` | str | auto | Output directory path |
| `--study-name` | str | auto | Optuna study name |
| `--sampler` | str | `tpe` | Sampler: `tpe`, `random`, or `cmaes` |
| `--no-plots` | flag | false | Skip plot generation |

### Validation Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--validate-top-k` | int | 0 | Validate top K configs with multi-seed runs |
| `--validation-seeds` | int list | 0 1 2 3 4 | Seeds for validation runs |

### Comparison Mode

| Flag | Type | Description |
|---|---|---|
| `--compare` | str list | Paths to `study.db` files from different searches |

---

## Practical Guide

### How many trials do I need?

As a rule of thumb:

| Search space dimensions | Recommended trials |
|---|---|
| 5–8 parameters | 40–60 |
| 9–15 parameters | 60–100 |
| 15–30 parameters | 100–200 |

The TPE sampler is most effective after ~20 trials (it needs initial random exploration before it starts being "smart"). If you have a budget of N trials, Optuna will use the first ~10 as random exploration and then switch to informed sampling.

### Which sampler should I use?

| Sampler | Best for | Notes |
|---|---|---|
| `tpe` (default) | Most cases | Bayesian-style, handles categorical + continuous HPs well |
| `random` | Baseline comparison, or when you want unbiased coverage | Simple but inefficient |
| `cmaes` | All-continuous search spaces | Very efficient for continuous HPs, but cannot handle categoricals |

For TS-MoE with many boolean/categorical flags, **stick with TPE**.

### How do I choose what to tune?

**Always tune:**
- Learning rate (`optim.lr`) — by far the most impactful HP in almost every deep learning setup
- Dropout (`model.dropout`) — critical for small datasets like BreastMNIST

**Tune if using that pipeline:**
- Gumbel: `gumbel.sparsity_weight`, `gumbel.tau_start`, `gumbel.tau_end`
- TS-MoE: `ts_moe.lambda_max`, `ts_moe.lambda_warmup_epochs`, `ts_moe.student_kd_alpha`

**Usually safe to fix:**
- `optim.epochs` — use a high value with early stopping
- `dataset.num_workers` — has no effect on model quality
- Quantum parameters — require cache regeneration (search in a separate outer loop)

### Recommended workflow

1. **Start broad**: Run 50–80 trials with the provided search configs.
2. **Inspect importance plots**: Identify which HPs matter most.
3. **Narrow the search**: Create a new search config with tighter ranges around the promising region, and fix unimportant HPs.
4. **Run targeted search**: 30–50 more trials in the narrowed space.
5. **Validate**: Run `--validate-top-k 5` to get multi-seed statistics.
6. **Deploy**: Use the exported `best_config.yml` for your final experiments.

### Interpreting failed / pruned trials

If many trials fail, check the `error.txt` files in the individual trial directories. Common causes:

- **`RuntimeError: GumbelMoE requires a pre-computed quantum dataset cache`**: You need to run `create_quantum_dataset.py` first with matching kernel settings.
- **CUDA out of memory**: Try reducing batch size (set a `fixed` value in search config) or reducing the number of kernels.
- **NaN loss**: Usually caused by learning rate too high combined with low gradient clipping. The search handles this gracefully — it marks the trial as pruned and moves on.

### Tuning quantum parameters

The quantum parameters (`kernel_size`, `stride`, `evolution_time`, `kernel_topology_names`) are **not included** in the search configs because changing them requires regenerating the quantum dataset cache. To tune them:

1. Pick 3–5 candidate quantum configurations manually.
2. For each, generate a cache with `create_quantum_dataset.py`.
3. Run a full HP search for each cache.
4. Compare across caches using `--compare` mode.

This "outer loop" approach keeps the inner search efficient while still exploring the quantum design space.

---

## Extending the Search

### Adding a new parameter to the search

1. Identify the dotted path in the base YAML config (e.g., `ts_moe.student_batch_size`).
2. Add an entry to the search space section of your search config:
   ```yaml
   ts_moe.student_batch_size:
     type: int
     low: 64
     high: 512
     step: 64
   ```
3. Re-run the search. That's it — no code changes needed.

### Creating a search config for a new dataset

1. Copy one of the existing search configs (e.g., `breast_mnist_original.yml`).
2. Adjust ranges based on dataset characteristics:
   - **Larger datasets** (e.g., TissueMNIST): wider patience range, possibly higher batch sizes.
   - **Multi-class datasets**: wider dropout range, possibly higher weight decay.
   - **Imbalanced datasets**: make sure to search `student_weighted_ce` and `student_balanced_sampler`.
3. Point `--config` at the appropriate base config for that dataset.

### Accessing the Optuna study programmatically

```python
import optuna

storage = "sqlite:///outputs/hp_search_gumbel_breastmnist_*/study.db"
summaries = optuna.study.get_all_study_summaries(storage=storage)
study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)

# Best trial
print(study.best_trial.params)
print(study.best_trial.value)

# All completed trials
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(trial.number, trial.value, trial.params)

# Importance analysis
importances = optuna.importance.get_param_importances(study)
print(importances)
```

---

## Dependencies

The hyperparameter search system requires:

| Package | Purpose | Install |
|---|---|---|
| `optuna` | Core HP search engine | `pip install optuna` |
| `plotly` | Interactive HTML plots (optional, degrades gracefully) | `pip install plotly` |
| `kaleido` | Static PNG export from Plotly (optional) | `pip install kaleido` |
| `matplotlib` | Always-available static plots | (already in project) |
| `seaborn` | Enhanced matplotlib aesthetics | (already in project) |

All dependencies except `optuna` are optional — the system degrades gracefully if they're missing, falling back to matplotlib-only plots.