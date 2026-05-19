# Experiments TODO

Status as of 2026-05-19. The 11 BreastMNIST HP searches (1a–1k) are
**complete** under `outputs/paper_results/hp_search/`. Linear probing
(`outputs/paper_results/linear_probing/breast_mnist.csv`) was run with
old topology picks and needs re-running with the corrected best configs.

The remaining work is reordered to reflect the new paper framing: the
quantum-vs-classical gap visible at the linear probe level (+3–6 pp AUC)
collapses to ≤1 pp at the full CNN head. The central new experiment is a
**head-capacity sweep** that traces how the gap evolves with classifier
expressivity.

---

## Priority 0 — Head-capacity sweep (NEW, highest impact)

Train the **same** frozen features with progressively more expressive
trainable heads and report the quantum-vs-classical gap at each capacity
level. This is the keystone experiment for the paper.

**Heads to compare:**

1. **Linear** — `LazyLinear(num_classes)` only. Apples-to-apples with the SVM
   probe but inside the same training loop (Adam, cosine LR, early stopping,
   val-based selection).
2. **MLP-1** — `LazyLinear(H) → Act → Dropout → Linear(num_classes)`.
3. **CNN head (current)** — the existing `build_classification_head`.

**Feature sources (use the best configs from Section 1):**

- Digital-ZZ 1-kern (cross topology) — best 1b
- Analog-ZZ 1-kern (star topology) — best 1c
- Digital-ZZ 4-kern (grid+chain+horizontal+ring) — best 1h
- Analog-ZZ 4-kern (grid+chain+horizontal+kings) — best 1i
- Classical Random ×45 filters (matches dim of 1-kern quantum)
- Classical Random ×180 filters (matches dim of 4-kern quantum)

**Protocol:** 10 seeds per (head, feature) cell. Light HP sweep over lr +
dropout + weight_decay only — the head is small enough that grid search
or ~30 Optuna trials is enough. Topology and Hamiltonian parameters are
fixed (frozen quantum features).

Commands to be added once the experiment script exists. See
`docs/plans/capacity_sweep.md` for the design and implementation plan.

---

## Priority 1 — Re-run linear probing with corrected best configs

Existing `linear_probing/breast_mnist.csv` used `kings` for the 1-kern
quantum entries; the HP searches now pick `cross` (Digital-ZZ) and
`star` (Analog-ZZ). Re-run with the new topology picks so the probe
and CNN comparisons reference the same models. Also probe **all 8
topologies** independently — if the best probing topology differs from
the best CNN topology, that's another piece of evidence that head
capacity matters.

```bash
python experiments/feature_probing_ablation.py \
    --classical-1kern configs/breast_mnist/classical_baseline_base_1kern.yml \
    --quantum-1kern configs/breast_mnist/digital_zz_best.yml \
    --classical-4kern configs/breast_mnist/classical_baseline_base.yml \
    --quantum-4kern configs/breast_mnist/original/4_kernels_3x3_digital_zz.yml \
    --seeds 0 1 2 3 4
```

The `--quantum-*` configs must be updated to the new best topologies
before running. Cheap to re-run (~30 min) — the heavy work is generating
the quantum feature caches, which are already on disk.

---

## Completed — BreastMNIST HP searches (10 unique runs)

All searches use `val_acc` as the objective, 200 trials (100 for raw
pixels), validated on 10 seeds. Results in
`outputs/paper_results/hp_search/`.

| # | Model | val_acc | test_acc | test_auc | Best topology |
|---|---|---|---|---|---|
| 1a | Digital-Z 1k | 0.962 | 84.8 ± 1.8 | 86.8 ± 0.7 | ring |
| 1b | Digital+ZZ 1k | 0.949 | 85.5 ± 1.5 | 88.3 ± 1.2 | cross |
| 1c | Analog+ZZ 1k | 0.949 | 84.4 ± 0.9 | 88.9 ± 1.0 | star |
| 1d | Classical Random 1k | 0.936 | 86.2 ± 0.7 | 89.5 ± 1.0 | — |
| 1e | Classical Trainable 1k | 0.949 | 86.2 ± 1.3 | 88.9 ± 1.5 | — |
| 1f | Vanilla CNN | 0.936 | 82.8 ± 1.3 | 86.0 ± 1.4 | — |
| 1g | Raw pixels | 0.885 | 82.4 ± 1.0 | 84.3 ± 0.4 | — |
| 1h | Digital+ZZ 4k | 0.949 | 85.2 ± 1.6 | 90.1 ± 0.5 | g+c+h+ring |
| 1i | Analog+ZZ 4k | 0.949 | 84.7 ± 1.8 | 89.3 ± 1.0 | g+c+h+kings |
| 1j | Classical Random 4k | 0.923 | 85.8 ± 1.4 | 89.3 ± 0.9 | — |
| 1k | Classical Trainable 4k | 0.936 | 83.5 ± 2.9 | 86.3 ± 1.6 | — |

The original commands for all 11 searches are preserved in git history if
re-runs are needed.

---

## Deferred — PneumoniaMNIST (parked)

Will revisit only after the capacity sweep is complete and the paper
draft is solid. Goal would be a robustness check: confirm the
qualitative pattern (quantum > classical at probe, ≈ at CNN) replicates
on a second medical dataset. If yes, one extra row per table. If no,
the paper scope tightens to BreastMNIST.

Requires:
- Base configs (copy from `configs/breast_mnist/original/`, set
  `dataset.name: pneumonia_mnist`).
- Quantum feature caches for digital_zz and analog_zz.
- Best 4–6 model variants (not the full 11), only the ones that matter
  for the capacity-sweep claim.

---

## Deferred — TissueMNIST, CKA similarity, noise model

Not on the critical path for the IEEE QAI 2026 submission. Revisit only
if the paper is finished early. CKA in particular is tangential to the
new framing (topology complementarity is interesting but not central to
the capacity-as-mechanism claim).
