# Weights & Biases (W&B) in MoT-DAQCNN

This repository supports two workflows:

1) **Direct / real-time logging (recommended going forward)**  
   Experiments log to W&B while they run (useful on multi-device/HPC runs).

2) **Backfill logging (useful for existing runs)**  
   Run experiments normally (they write to `outputs/`), then upload results afterward.

W&B project: https://wandb.ai/nunombatista-university-of-coimbra/MoT-DAQCNN/

---

## 1) Install + authenticate

Install dependencies in your environment:

```bash
pip install wandb pyyaml optuna pandas
```

### Authentication options (pick one)

**A) API key via `.env` (recommended for HPC/no-tty jobs)**

Create a file at the repo root:

- `MoT-DAQCNN/.env`

with:

```bash
WANDB_API_KEY=YOUR_KEY_HERE
```

**B) Interactive login (works on a machine with a TTY)**

```bash
wandb login
```

---

## 2) Direct (real-time) logging from experiment scripts

Both scripts below support W&B logging behind a toggle, so they still run fine without W&B.

### Robust evaluation runner (multi-seed)

Script:
- `experiments/robust_test_original_daqcnn.py`

Enable W&B:

```bash
python experiments/robust_test_original_daqcnn.py \
  --config configs/pneumonia_mnist.yml \
  --wandb \
  --wandb-entity nunombatista-university-of-coimbra \
  --wandb-project MoT-DAQCNN \
  --wandb-mode online
```

HPC-friendly offline mode:

```bash
python experiments/robust_test_original_daqcnn.py \
  --config configs/pneumonia_mnist.yml \
  --wandb \
  --wandb-entity nunombatista-university-of-coimbra \
  --wandb-project MoT-DAQCNN \
  --wandb-mode offline

# later:
wandb sync wandb/offline-run-*
```

What gets logged in real-time:
- per-seed summary metrics (accuracy/AUC/F1/recall/loss, etc.)
- per-epoch loss curves when available
- aggregate metrics + key output files are attached at the end of the run

### Optuna hyperparameter search

Script:
- `experiments/hyperparameter_search.py`

Enable W&B:

```bash
python experiments/hyperparameter_search.py \
  --config configs/breast_mnist_multi_seed_3kern.yml \
  --search-config configs/hp_search/breast_mnist_original.yml \
  --n-trials 50 \
  --wandb \
  --wandb-entity nunombatista-university-of-coimbra \
  --wandb-project MoT-DAQCNN \
  --wandb-mode online
```

Optional: log a “completed trials” table at the end:

```bash
python experiments/hyperparameter_search.py ... --wandb --wandb-log-trials-table
```

What gets logged in real-time:
- one W&B log per completed trial (objective + params + common user attrs)
- “best objective so far” over trials
- final summary + key files are attached at the end

---

## 3) Backfill existing results from `outputs/` to W&B

Script:

- `wab/backfill_wandb.py`

It scans `outputs/` for known output formats and creates one W&B run per output directory.

### Upload everything it recognizes

```bash
python wab/backfill_wandb.py \
  --entity nunombatista-university-of-coimbra \
  --project MoT-DAQCNN \
  --outputs-dir outputs \
  --mode online
```

### HPC-friendly offline mode (+ later sync)

```bash
python wab/backfill_wandb.py \
  --entity nunombatista-university-of-coimbra \
  --project MoT-DAQCNN \
  --outputs-dir outputs \
  --mode offline

# later (on a machine with internet):
wandb sync wandb/offline-run-*
```

### Only backfill a single outputs folder

```bash
python wab/backfill_wandb.py \
  --entity nunombatista-university-of-coimbra \
  --project MoT-DAQCNN \
  --only run_pneumoniamnist_20260101_120000
```

### Only backfill one type

```bash
# only robust evaluation runs
python wab/backfill_wandb.py ... --include robust_eval

# only hyperparameter searches
python wab/backfill_wandb.py ... --include hp_search
```

### Dry run (no upload)

```bash
python wab/backfill_wandb.py ... --dry-run
```

### Optional: upload key files as W&B Artifacts

```bash
python wab/backfill_wandb.py ... --artifacts
```

> Artifacts can increase upload size. By default, the script focuses on configs, metrics, and plots (and avoids uploading heavy checkpoints).

---

## 4) Backfill vs direct logging: are they equivalent?

**Not fully equivalent.**

Backfilling *is equivalent* for:
- final aggregate metrics (e.g., mean/std across seeds),
- saved plots/images,
- configs and summaries,
- Optuna trial table (if `study.db` exists and is accessible).

Backfilling is *not equivalent* for:
- real-time monitoring (loss curves updating live),
- system metrics/time-series (GPU utilization, step time),
- live alerts, early aborts, comparing runs mid-flight,
- exact per-step logs if those weren’t saved to disk.

### Practical recommendation
- Use **direct logging** for new experiments (you’ll want the live dashboard).
- Keep **backfill** around for legacy runs and for uploading extra files/plots from `outputs/`.