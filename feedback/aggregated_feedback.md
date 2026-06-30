# Aggregated Feedback — DAQCNN Paper

Consolidated from Gabriel (handwritten, red pen, mostly Portuguese) and Oscar (PDF
comments). Page numbers refer to the reviewed PDF. Items both reviewers raised are
flagged **[Both]**.

## 1. Title and scope

- **Oscar (p1):** The paper applies only to neutral-atom systems; the title should
  say so. Consider narrowing the title to reflect the neutral-atom (Rydberg) scope
  rather than "quantum" in general.

## 2. Section structure and "no subsection without a preamble"

Gabriel's recurring structural rule: never open a subsection directly with a
subsubsection or with content — add a short preamble paragraph first.

- **Gabriel (p1):** "Avoid starting a subsection without a preamble." Pointed at
  §II Background opening straight into *A. Rydberg Hamiltonian*. Add an intro
  sentence to §II before subsection A.
- **Gabriel (p4):** Same issue at §IV Experimental Setup — add a preamble before
  *A. Datasets*.
- **Gabriel (p2):** The *C. MedMNIST* dataset description currently sits in the
  Background; move it into the Experimental Setup (§IV). It is circled/struck and
  marked "say this in the experimental setup (section IV)."

## 3. The a)/b)/c)/d) enumerated paragraphs

- **Oscar (p3):** A block "should be divided into a), b), c), and d)" — wants
  clearer enumeration of the items there. **[Both]** (Gabriel addresses the same
  paragraphs from the formatting side.)
- **Gabriel (p5, p7):** Don't use LaTeX `\paragraph` for the a)/b)/c)/d) items
  nested inside a subsubsection — use `\subsubsection` instead (repeated on both
  the Results §V and Discussion paragraphs).

## 4. Figure and table placement / referencing

- **Gabriel (p2, p3, p8):** **Figure 1** (and the figures generally) should sit on
  the page where first cited, not later. Fig 1 is cited on the page before it
  appears — move it up to the top of that earlier page. Same note repeated for
  Fig 3 on p8 ("this figure has to go to the previous page").
- **Gabriel (p6):** **Table III is never cited/referenced in the text.** Either
  cite it or, given space limits, move it to an external pointer (the paper's
  GitHub) and reference that. **[RESOLVED — removed from `paper_extended.tex`;
  its verified TissueMNIST probing numbers folded into prose; paper now 9 pages.]**
- **Gabriel (p3):** In the flattening notation `(N, M f_k, H_out, W_out)`, use
  explicit `×` separators (`M × f_k × H_out × W_out`) instead of commas so it
  reads as a product, not a tuple.
- **Gabriel (p3):** In the two-baseline list ("Raw pixels... Random filters..."),
  fix the punctuation — insert "and", drop the colons used as separators.

## 5. Length / conciseness (cut and compress)

- **Gabriel (p4, §E Classical Head):** Repeated material here — reduce the text.
- **Gabriel (p4, §F Head-Capacity Sweep):** "Can you merge/fix these two
  paragraphs?"
- **Gabriel (p4):** The paragraph re-stating that the end-to-end comparison also
  evaluates poly-2/RFF — "isn't this already clear in Table IV?" Consider cutting.
- **Gabriel (p5, §V-A Linear Feature Probing):** Reduce/eliminate the unnecessary
  text in the long "representational measure / not the operating point" paragraph.
- **Gabriel (p7, big margin note):** Can't all the ablation-study results (the
  dense a)–d) prose) be summarized in a single table or figure — e.g. a
  Pareto-optimal curve? Then cut the surrounding text, keeping the *Interpretation*
  paragraph. ("doable?")
- **Gabriel (p8, big note):** **The Conclusion is too long.** Compress it to a
  single paragraph. Most of the current content belongs in a *Discussion*
  subsection (maybe §V-D) or a new standalone Discussion section — a paper of this
  scope justifies a dedicated Discussion section.

## 6. Reviewer-friendliness — point reviewers to the evidence

- **Gabriel (p5, big note):** "Help the reviewers — each one is reviewing ~10
  papers and will be impatient." Where the text makes a claim shown in Fig 3, say
  explicitly where in Fig 3 it appears (panel/cell). Add concrete pointers from
  claims to the exact figure location.

## 7. Citations (add more, on purpose)

- **Gabriel (p9, big note):** Papers get rejected (even unfairly) for having few
  citations. Add:
  - Our own **IEEE QAI 2025 paper** (to match the target conference).
  - Gabriel's **polyp-detection / YOLO paper** to give context when discussing
    classical AI on medical-imaging datasets.

## 8. Acronyms and clarity (define on first use)

- **Gabriel (p1, abstract):** Underlined **"leave separability headroom"** and wrote
  "?? what does it mean? — rephrase!" The phrase is unclear; reword it in plainer
  language.
- **Oscar (p3):** "sigla" on **AUROC** — define the acronym.
- **Oscar (p4):** "sigla" on **HP** ("selected by HP search") — spell out
  "hyperparameter."
- **Oscar (p2):** "So the training happens classically and the weights are passed
  to the quantum kernel? Unclear." The frozen-quantum / trained-classical split
  needs to be stated more clearly — clarify that nothing is trained in the quantum
  layer.
- **Oscar (p3):** On the patch-extraction description: "the idea is to apply
  [Hamiltonian evolution] to extract features from the patches and hand them to a
  classical computer?" — confirm/clarify the pipeline intent in plain language.

## 9. Implementation / hardware details

- **Oscar (p5):** **[Both]** "Which CPU and GPU? More details needed." and "Did you
  use GPUs? Unclear." The §IV-D Implementation sentence ("single workstation CPU...
  without dedicated GPU") is ambiguous — specify the exact hardware and whether GPUs
  were used at all. Gabriel marks the same Implementation text (suggests it may
  belong in the Experimental Setup).

## 10. Empty section

- **Oscar (p8):** **The Acknowledgment section is empty** ("Watch this, it's
  empty") — fill it in or remove it before submission.

## 11. Gabriel's proposed new Discussion content (pp9–10)

Gabriel drafted concrete prose to seed a new Discussion/future-work subsection,
framed around *when* quantum kernels help and why graph-structured data is a more
natural fit. Paraphrased:

> Quantum kernels are not universally useful. Their usefulness depends on the
> alignment between the quantum feature space and the structure of the underlying
> data. The TissueMNIST results support this: on the hardest dataset the quantum
> kernel outperforms tuned classical nonlinearity at low classifier capacity. The
> benefit appears only when the data contains structure the kernel can expose and
> the classifier is too weak to learn that structure on its own.
>
> **Why graphs may be more promising.** Prior work shows local detuning improves
> expressiveness on graph data, and from a physics standpoint Rydberg systems
> naturally define interaction graphs, distance-dependent couplings, and many-body
> correlations — all graph-native concepts. Physics is naturally graph-oriented, so
> the most promising domains for neutral-atom quantum kernels are probably
> (1) graph learning, (2) physical systems, (3) scientific relational data. These
> are much closer to the native physics of Rydberg interactions than 28×28
> grayscale images. Data that maps naturally onto graphs suits neutral-atom
> interactions better and is the subject of further study.

This is offered as text to fold into the compressed Conclusion / new Discussion
section (ties back to items 5 and 2).

---

### Quick action checklist

- [X] Narrow title to neutral-atom scope (Oscar) — inserted "Rydberg-Atom":
  "Reproducing the Apparent Quantum-Kernel Advantage of Rydberg-Atom DAQCNN
  using Classical Nonlinearity"
- [X] Add preambles to §II and §IV before their first subsection (Gabriel)
- [X] Move MedMNIST dataset description from Background to §IV (Gabriel) — §IV.A
  already covered it in full, so the redundant §II.C subsection was deleted
- [X] Convert a)/b)/c)/d) `\paragraph` items to `\subsubsection` (Gabriel) — done
  in the Results section; the Conclusion items are deferred to the Wave 3 rework
- [X] Move Fig 1 and Fig 3 to the page of first citation (Gabriel) — Fig 1 DONE
  (page 2, with its citation). Fig 3: after the Wave 3 repagination it still renders
  on page 7 while its §V-C discussion is on page 6, because page-6's single
  full-width top slot is owned by the end-to-end table. Forcing Fig 3 onto page 6
  would displace that table off its own discussion. Accepted as the structural
  optimum (figure is adjacent, immediately after its discussion)
- [X] ~~Cite Table III in text, or externalize it~~ — RESOLVED: Table III removed,
  numbers folded into prose (Gabriel)
- [X] Fix `×` notation and baseline-list punctuation on p3 (Gabriel)
- [X] Trim §E (cut redundant HP enumeration → pointer to Table II) and cut the
  §IV.B Models end-to-end restatement ("clear in the table") (Gabriel). §F left
  as-is: already consolidated; the §E trim removes the cross-section HP redundancy
- [~] PARKED: Pareto figure prototyped (`experiments/plot_capacity_pareto.py`,
  `docs/paper/figures/capacity_pareto.{png,pdf}`) but NOT included. It contradicts
  the thesis (quantum sits on/atop the frontier on Breast+Pneumonia because the
  frontier takes global maxima, not matched-dim comparisons), breaks on TissueMNIST
  (rebuilt CSV has param_count=0), and shows non-operating-point AUCs. Artifacts
  kept for later; user will decide.
- [X] Compress Conclusion to one paragraph; create a Discussion section (Gabriel)
  — new §VI Discussion (4 subsections incl. graph-suitability), §VII Conclusion =
  one paragraph; the three former conclusion `\paragraph`s became `\subsection`s
- [X] Add explicit Fig-3 location pointers to claims (Gabriel) — added e.g.
  "Fig. 3, bottom row, linear head" to the capacity claims in Discussion
- [X] Add IEEE QAI 2025 citation (Gabriel) — added `batista2025medical` (the
  cold-atom reservoir-computing medical-imaging QAI'25 paper; co-authored by Oscar
  and Gabriel, so it also serves the collaborator interest) and cited it in the
  Introduction as shared frozen-dynamics related work. YOLO polyp paper DROPPED as
  off-topic (user agreed)
- [X] Strengthen related-work citations (user request) — added `lau2024modular`
  (QRC for MNIST/Fashion-MNIST/CIFAR with ZZ-coupled reservoirs; verified via
  Semantic Scholar + full text). Verified `kornjaca2024largescale` DOES do image
  classification (MNIST + Plant Village, confirmed via ar5iv full text, not the
  abstract) and added it to the Introduction group; its Discussion synthetic-data
  claim re-checked and accurate. Still optional/not added: Das 2025 (Rydberg QRC
  denoising on Aquila), Zhang 2025 (QRC image clf claiming a quantum-kernel edge)
- [X] Trim the long §V-A "representational measure / not the operating point"
  paragraph (Gabriel p5) — tightened, kept the test-selection-bias point
- [X] Rephrase "leave separability headroom" in the abstract (Gabriel)
- [X] Define AUROC and HP on first use (Oscar)
- [X] Clarify frozen-quantum / trained-classical split + pipeline intent (Oscar)
  — added an explicit "no trainable weights enter the quantum layer" sentence and
  a §III.B preamble describing the extract-then-hand-to-classical flow
- [X] Specify CPU/GPU hardware in Implementation (Both) — corrected the false "no
  GPU" claim. Now: Breast/Pneumonia caches on AMD Threadripper 3960X, heads trained
  on an NVIDIA RTX 3090; TissueMNIST on HPC-cluster Xeon Gold 6448Y CPU nodes
  (128 threads, 480 GB), CPU-only there
- [X] (User request) Shortened the Table III caption; moved the seed counts, tuning
  protocol, and per-dataset winning topologies into a §V-B preamble (which also
  fixes the missing preamble before that table)
- [X] Fill or remove the empty Acknowledgment (Oscar) — commented out for the
  anonymous submission, with a camera-ready reminder
- [X] Fold in Gabriel's graph-suitability Discussion prose (Gabriel) — now §VI-C
  "When are quantum kernels promising?", hedged as speculation per the writing rules
