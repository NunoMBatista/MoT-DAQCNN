# Head-Capacity Sweep — Design and Implementation Plan

## Goal

Trace how the quantum-vs-classical AUC gap evolves as a function of the
trainable classifier's capacity. The motivating observation: at the
linear-probe level the gap is +3–6 pp AUC in favour of quantum kernels;
at the full CNN head the gap collapses to ≤1 pp. The hypothesis: the
gap shrinks monotonically as head capacity grows, and quantum features
are most useful when the downstream classifier is constrained.

This becomes the keystone experiment for the IEEE QAI 2026 paper — the
single contribution that distinguishes us from Simen et al. and gives
the paper a mechanistic claim with practical implications.

---

## Design

### Heads (5 capacity levels)

All heads end in `Linear(num_classes)`. All share the same training
loop (Adam + early stopping + val-based selection + optional cosine LR
schedule). The dropout and weight_decay HPs are searched per cell.

| # | Head | Architecture | Params (1k feats) | Params (4k feats) |
|---|---|---|---|---|
| 1 | **Linear** | `Flatten → LazyLinear(K)` | ~7k | ~29k |
| 2 | **MLP-1** | `Flatten → LazyLinear(32) → Act → Drop → Linear(32, K)` | ~117k | ~467k |
| 3 | **MLP-2** | `Flatten → LazyLinear(64) → Act → Drop → Linear(64, 32) → Act → Drop → Linear(32, K)` | ~235k | ~935k |
| 4 | **CNN-small** | existing `build_classification_head` with `hidden_channels=16` | ~6k | ~6k |
| 5 | **CNN-large** | existing `build_classification_head` with `hidden_channels=64` (current default) | ~100k | ~100k |

Note: the MLP heads can end up with *more* parameters than the CNN
heads because their linear layer scales with the flattened feature
dimension, while the CNN's conv params are independent of spatial
size. Report param counts alongside architectural labels in the figure
— x-axis ordering follows architectural expressivity, not parameter
count, but parameter counts are annotated for transparency.

### Feature sources (8 total)

**Floor reference (784 dim):**
- Raw pixels — flattened 28×28 input. Different dimensionality from the
  matched-dim pairs, but acts as the no-feature-engineering floor. The
  figure shows this as a separate reference line; the matched-dim
  comparisons live in 1k and 4k scales below.

**1-kernel scale (~3645 dim):**
- Digital-Z 1k (ring) — best from 1a
- Digital-ZZ 1k (cross) — best from 1b
- Analog-ZZ 1k (star) — best from 1c
- Classical Random ×45 filters (seed-controlled)

**4-kernel scale (~14580 dim):**
- Digital-ZZ 4k (g+c+h+ring) — best from 1h
- Analog-ZZ 4k (g+c+h+kings) — best from 1i
- Classical Random ×180 filters (seed-controlled)

Total: **40 cells** (5 heads × 8 sources).

### HP search per cell

Small Optuna study per cell, **30 trials**, objective = `val_acc`.
Search space:

```yaml
optim.lr:           loguniform [1e-5, 5e-3]
optim.weight_decay: loguniform [1e-8, 1e-3]
model.dropout:      uniform    [0.0, 0.7]
```

Everything else fixed: Adam optimizer, cosine LR scheduler on/off
(fixed to `false` for the linear head, `true` for others — small
linear heads don't benefit from LR scheduling on this dataset),
epochs=100, patience=30, grad_clip=1.0, batch_size=32, activation=gelu.

Topology and quantum HPs are NOT searched — the feature source is
fixed per cell, and the features themselves come from disk.

### Validation

Best trial per cell → train on 10 seeds (0..9) → report
`mean ± std` for test_acc, test_auc, test_f1.

### Total compute estimate

- 40 cells × 30 Optuna trials × ~30 s per training (cached features) = ~10 hr
- 40 cells × 10 validation seeds × ~45 s = ~5 hr
- **Total ~15 hr**, can run overnight. CNN-large and MLP-2 cells will
  be at the longer end. Raw-pixel cells will be faster (smaller dim).

---

## Output format

```
outputs/paper_results/capacity_sweep/
├── breast_mnist/
│   ├── cells/
│   │   ├── {head}_{feature_source}/
│   │   │   ├── search_config.yml
│   │   │   ├── search_summary.json
│   │   │   ├── study.db
│   │   │   └── validation/
│   │   │       └── seed_{0..9}/
│   │   │           └── metrics.json
│   ├── summary.csv          # one row per cell with mean/std test metrics
│   └── per_seed.csv         # one row per (cell, seed)
└── plots/
    └── capacity_sweep_auc.pdf
```

`summary.csv` columns:
```
head, feature_source, dim, param_count,
test_acc_mean, test_acc_std,
test_auc_mean, test_auc_std,
test_f1_mean,  test_f1_std,
best_lr, best_dropout, best_wd
```

---

## Implementation

### New files

1. **`src/utils/training_utils.py`** — add three helpers:
   ```python
   def build_linear_head(num_classes): ...        # Flatten + LazyLinear
   def build_mlp1_head(num_classes, hidden=32, dropout=0.5, activation="gelu"): ...
   def build_mlp2_head(num_classes, h1=64, h2=32, dropout=0.5, activation="gelu"): ...
   ```
   Existing `build_classification_head` already supports `hidden_channels`,
   so CNN-small and CNN-large are just two calls with different values.

2. **`experiments/head_capacity_sweep.py`** — new experiment script.

3. **`experiments/plot_capacity_sweep.py`** — figure generator.

### `experiments/head_capacity_sweep.py` — pseudocode

```python
# Args
#   --dataset breast_mnist
#   --heads linear mlp1 mlp2 cnn_small cnn_large
#   --feature-sources <list of named configs>
#   --n-trials 30
#   --validation-seeds 0..9
#   --output-dir outputs/paper_results/capacity_sweep/

FEATURE_SOURCES = {
    "digital_z_1k":   {"config": "configs/breast_mnist/digital_z_best.yml"},
    "digital_zz_1k":  {"config": "configs/breast_mnist/digital_zz_best.yml"},
    "analog_zz_1k":   {"config": "configs/breast_mnist/analog_zz_best.yml"},
    "random_45":      {"random_filters": 45,  "ref_config": ".../1_kernel_3x3.yml"},
    "digital_zz_4k":  {"config": "configs/breast_mnist/digital_zz_4k_best.yml"},
    "analog_zz_4k":   {"config": "configs/breast_mnist/analog_zz_4k_best.yml"},
    "random_180":     {"random_filters": 180, "ref_config": ".../1_kernel_3x3.yml"},
}

HEADS = {
    "linear":     lambda nc: build_linear_head(nc),
    "mlp1":       lambda nc: build_mlp1_head(nc, hidden=32),
    "mlp2":       lambda nc: build_mlp2_head(nc, h1=64, h2=32),
    "cnn_small":  lambda nc: build_classification_head(in_ch, nc, hidden_channels=16),
    "cnn_large":  lambda nc: build_classification_head(in_ch, nc, hidden_channels=64),
}

for source_name in feature_sources:
    feats_train, y_train, feats_val, y_val, feats_test, y_test = load_features(source_name)
    for head_name in heads:
        cell_dir = output_dir / f"{head_name}_{source_name}"
        study = optuna.create_study(study_name=f"{head_name}_{source_name}",
                                     direction="maximize",
                                     storage=f"sqlite:///{cell_dir}/study.db")

        def objective(trial):
            lr = trial.suggest_loguniform("lr", 1e-5, 5e-3)
            wd = trial.suggest_loguniform("wd", 1e-8, 1e-3)
            do = trial.suggest_uniform("dropout", 0.0, 0.7)
            head = build_head(head_name, dropout=do)
            return train_and_eval(head, feats_train, y_train, feats_val, y_val,
                                   lr=lr, wd=wd, seed=0)  # single seed for search

        study.optimize(objective, n_trials=30)

        # Validate top-1 over 10 seeds
        best = study.best_params
        for seed in range(10):
            head = build_head(head_name, dropout=best["dropout"])
            metrics = train_and_eval(head, feats_train, y_train, feats_val, y_val,
                                      feats_test, y_test,
                                      lr=best["lr"], wd=best["wd"], seed=seed)
            save_seed_metrics(cell_dir / f"seed_{seed}.json", metrics)

        aggregate_to_summary_csv(cell_dir, summary_csv)
```

### Feature loading

For quantum sources: call `find_cached_quantum_dataset(cfg)` and
`load_cached_quantum_dataset(...)`. The cache returns DataLoaders;
collect into tensors once and reuse across all 30 trials × 10 seeds —
no need to re-read from disk per training.

For random-filter sources: load raw images via `get_dataloaders`,
apply a fixed Gaussian random conv (seeded), collect into tensors.
**Important:** the random filter seed must be fixed (e.g., 42) so the
filter bank is identical across all 5 heads — otherwise we'd be
varying both the head and the feature substrate.

### Training loop

A stripped-down version of `run_single_seed`:
- Forward: just `head(features)` since features are pre-extracted.
- Loss: cross-entropy.
- Early stopping on val loss, patience=30.
- Track best val_acc and corresponding model state.
- Return test metrics from the best-val checkpoint.

No need for `DAQCNN` or `ClassicalBaselineCNN` wrapper classes — the
head IS the model in this experiment.

### Configs needed

These five "best config" YAMLs must be produced from the HP search
results before running the sweep:

- `configs/breast_mnist/digital_z_best.yml` (from 1a, topology=ring)
- `configs/breast_mnist/digital_zz_best.yml` (from 1b, topology=cross)
- `configs/breast_mnist/analog_zz_best.yml` (from 1c, topology=star)
- `configs/breast_mnist/digital_zz_4k_best.yml` (from 1h, topology=g+c+h+ring)
- `configs/breast_mnist/analog_zz_4k_best.yml` (from 1i, topology=g+c+h+kings)

Each is a copy of the corresponding 1-kern or 4-kern base config with
the winning topology, encoding mode, and `include_correlators` set
correctly. The optim HPs in these files don't matter for the sweep
(we search per cell) but should be present for reproducibility.

Caches must exist for these topologies. Already present for the ones
that came out of the searches; verify before running.

---

## Risks and mitigations

**Risk 1 — Hypothesis fails: gap is flat or non-monotonic.**
This is the empirical question. If the gap doesn't shrink monotonically,
the paper's framing has to change again. But even a flat-gap result is
informative ("quantum advantage is head-capacity invariant in this
regime"). Run the sweep, see what happens.

**Risk 2 — Linear head undertrains.**
A linear head on 14580 features can be sensitive to regularization. If
the linear cells perform worse than the SVM probe, that's a sign the
training loop isn't fully optimising. Mitigation: include a sanity check
that the linear head matches the SVM probe AUC within ~1 pp on the same
features. If not, increase patience or extend epochs.

**Risk 3 — Optuna 30 trials too few.**
The HP search landscape for a head-only model is shallow (3 HPs).
30 trials should converge. If not, bump to 50.

**Risk 4 — Random-filter feature variance contaminates the
classical baseline.**
The random filters are seeded once, so the feature substrate is fixed,
but the *quality* of those random filters depends on the seed. To avoid
cherry-picking, run with 3 different filter-bank seeds and report the
average — adds modest compute (3× the classical cells = ~6 hr extra).
Decide whether this is worth the rigor cost.

**Risk 5 — Cells are not independent.**
Optuna study DBs are per-cell, training is per-cell, but the cached
features are shared across cells at the same scale. This is fine as
long as we don't accidentally reuse a head state between cells.

---

## Order of operations

1. Generate the five "best" config YAMLs from the HP search results.
2. Verify caches exist for the winning topologies.
3. Add `build_linear_head`, `build_mlp1_head`, `build_mlp2_head` to
   `training_utils.py`.
4. Write `experiments/head_capacity_sweep.py`.
5. Smoke test: run 1 cell (e.g., linear + digital_zz_1k) with n_trials=3,
   validation_seeds=[0], confirm output format and training behaviour.
6. Run all 35 cells (overnight).
7. Aggregate → CSV → figure.

## Open items deferred

- Whether to also add a "trained classical filters" feature source —
  would test "trained kernels vs random kernels at low head capacity".
  Tangential to the central claim but mildly interesting. Skip for v1.
- Whether to do a small filter-bank seed average for the random
  baselines — see Risk 4. Decide before running.
