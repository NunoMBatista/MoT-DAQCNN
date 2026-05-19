# Post-Sweep Workflow

What to do once `outputs/paper_results/capacity_sweep/breast_mnist/summary.csv`
exists. Order matters: sanity-check first, then plot, then write.

---

## Step 1. Sanity-check the sweep results (~15 min)

Before plotting or writing, eyeball the CSV to make sure nothing is
broken. Specifically:

- Linear-head AUC on `digital_zz_1k` should be close to the SVM probe
  AUC at the same topology (cross, ~0.821). If it's wildly off (more
  than ~2 AUC below), the training loop is undertraining the linear
  head and we need to investigate before trusting anything else.
- Within each head, AUC should be roughly monotonic across feature
  sources (random_45 ≤ digital_zz_1k for the linear head at least).
  Wild outliers suggest a bug.
- The CNN-large head AUCs on quantum sources should be close to the
  end-to-end HP-search results from Section 1 of `experiments_todo.md`
  (Digital-ZZ 4k: 0.901, Analog-ZZ 4k: 0.893). If off by more than a
  few AUC, something's wrong.

If anything looks off, dig into the relevant `cells/{head}__{source}/`
directory and inspect the validation seed JSONs and study.db.

---

## Step 2. Plot (~5 min)

```bash
/home/nuno/python_envs/ML/bin/python experiments/plot_capacity_sweep.py
```

Writes two figures to
`outputs/paper_results/capacity_sweep/breast_mnist/plots/`:

- `capacity_sweep.pdf`: two-panel plot of AUC vs head capacity, one
  panel per scale (1-kernel and 4-kernel). One line per feature
  source. Std band over 10 seeds.
- `capacity_sweep_gap.pdf`: side-by-side line plots of the
  quantum-minus-classical AUC gap at each head capacity. **This is
  the keystone figure for the paper.** If the gap shrinks
  monotonically (or near-monotonically) toward zero as we move from
  linear to cnn_large, the central hypothesis is supported.

Inspect both visually before committing to interpretation.

---

## Step 3. Decide what the data says (~30 min thinking)

There are three plausible outcomes from the gap-vs-capacity plot. Each
implies a different paper framing:

**(a) Monotonic shrinkage (hypothesis confirmed):** gap is largest at
linear head, smallest at cnn_large. Frame paper around "head capacity
equalises features." Strongest paper version.

**(b) Plateau or saturation:** gap is roughly constant across head
capacities. Means the head doesn't "rescue" weak features by training
harder. Different framing: "quantum features are intrinsically more
separable; this advantage propagates through head capacity uniformly".

**(c) Non-monotonic / flat / noisy:** the gap is essentially zero at
every head capacity. Hard interpretation. Probably means the
linear-Adam head is already extracting almost everything, or that the
classical-random baseline has reached parity. Weakest outcome.

We pick the paper's framing based on which we see.

---

## Step 4. Update the paper Results section (~1 day of writing)

The Results section is currently a placeholder. With the sweep CSV
and the topology-probing CSV both in hand we now have everything
needed to write it. Structure:

- **V.A Linear feature probing across topologies.** Use the heatmap
  figure (`outputs/paper_results/linear_probing/plots/topology_sweep_heatmap.pdf`).
  Honest matched-topology encoding comparison: mean encoding effect is
  small and topology-dependent. ZZ effect is real and consistent.
- **V.B End-to-end classification (10-seed).** Table of the 11 model
  variants with test_acc, test_auc, std. Numbers in
  `outputs/paper_results/hp_search/*/validation/validation_summary.json`.
- **V.C Head-capacity sweep.** This is the headline. Use the
  capacity-sweep plot. Tell the story of gap vs capacity.
- **V.D Discussion.** Use the draft paragraph already in the
  Conclusion section about analog encoding scope (mark currently as
  `[Draft, still to be determined]`).

While writing, also update the Introduction's three-pronged
contribution list to add the head-capacity sweep as a fourth item.

---

## Step 5. Decide on PneumoniaMNIST (~1 hour of thinking, ~2 days of compute)

By the time we finish Section V, we'll have a clear sense of whether
the paper's claims are robust. Decide then whether to also run
PneumoniaMNIST as a robustness check.

If yes: launch a subset of HP searches (5-6 model variants) on
Pneumonia. ~24-36 hours of compute. While that runs, write the rest
of the paper. Add a "PneumoniaMNIST replication" subsection at the
end of Section V when results land.

If no: be explicit in the paper that scope is BreastMNIST and that
multi-dataset replication is left to future work.

My current recommendation: yes, run a subset. Reviewer-defensibility
matters and we can pipeline it during paper writing.

---

## Step 6. Finalize the title (~10 min)

The current title commits to "Analog-Encoded Rydberg Quantum Kernels"
as the headline. The data doesn't support analog being the headline.
Candidates that match the actual findings:

1. **"Quantum Kernel Advantage and Classifier Capacity: A DAQCNN
   Study on BreastMNIST"** — pitches the head-capacity finding as
   the contribution.
2. **"Rydberg Quantum Kernels for Medical Image Classification: A
   Controlled Ablation"** — neutral, dignified, doesn't oversell.
3. **"When Do Quantum Features Help? A Capacity-Sensitivity Analysis
   of DAQCNN"** — same headline as (1), more provocative.

Pick after seeing the sweep results. If hypothesis (a) holds, (1) or
(3). If outcomes (b)/(c), (2) is safest.

---

## Step 7. Abstract and Introduction polish (~half day)

Write abstract last. With the data in hand, the abstract is straightforward:
state the methodology, the three findings (feature-level quantum
advantage, ZZ ablation, head-capacity story), and the scope limitation
(image classification doesn't tap analog's extra function class).

Update the Intro's three-pronged contribution list to four-pronged.

---

## Skip list (do NOT do)

- **CKA**: deferred. Mentioned in `experiments_todo.md` as parked.
- **TissueMNIST / harder datasets**: out of scope for this submission.
- **Trainable Hamiltonian parameters / encoder rescaling**: cool
  research direction but adds risk and complexity at deadline pressure.
  Future work.
- **T-evolution-time sweep with entanglement metric**: appealing but
  only if everything else lands cleanly and we have a spare day. Not
  load-bearing.

---

## My opinion on what's actually next

1. Run the steps above in order.
2. Don't add more experiments unless something in the sweep results
   genuinely changes the framing.
3. The biggest risk to the deadline is *paper writing time*, not
   *compute time*. Once the sweep finishes, prioritise writing over
   running more experiments.
