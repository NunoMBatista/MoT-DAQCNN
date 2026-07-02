"""PROTOTYPE: capacity-vs-AUC Pareto view of the head-capacity sweep.

One panel per dataset. x = trainable parameter count (head capacity, log scale),
y = test AUC. Every (head, source) cell is a point, coloured by camp (quantum /
classical-nonlinear / random / raw). The black stepped line is the Pareto
frontier over ALL points (best AUC achievable at or below each capacity budget);
frontier points are ringed and coloured by their camp, so you can read off which
camp owns the frontier at each capacity.

Story it should make visible: the frontier is almost entirely classical; quantum
touches it only at the TissueMNIST low-capacity corner.

Run from repo root:  python experiments/plot_capacity_pareto.py
Writes docs/paper/figures/capacity_pareto.{pdf,png}.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CS = os.path.join(ROOT, "outputs/paper_results/capacity_sweep_std")

DATASETS = [
    ("BreastMNIST", f"{CS}/breast_mnist/summary.csv"),
    ("PneumoniaMNIST", f"{CS}/pneumonia_mnist/summary.csv"),
    ("TissueMNIST", f"{CS}/tissue_mnist/summary_rebuilt.csv"),
]

# source -> (camp, colour, marker)
QUANTUM = ["digital_z_1k", "digital_zz_1k", "digital_zz_4k",
           "analog_z_1k", "analog_zz_1k", "analog_zz_4k"]
NONLIN = ["poly2_45", "rff_45", "rff_180"]
RANDOM = ["random_9", "random_45", "random_180"]
RAW = ["raw"]

CAMP = {}
for s in QUANTUM: CAMP[s] = ("Quantum", "#08306b", "o")
for s in NONLIN:  CAMP[s] = ("Classical nonlinear", "#e67e22", "s")
for s in RANDOM:  CAMP[s] = ("Random projection", "#7f8c8d", "^")
for s in RAW:     CAMP[s] = ("Raw pixels", "#bdc3c7", "x")


def load(path):
    """Return list of (param_count, auc, source) for sources we plot."""
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            s = r["source"]
            if s not in CAMP:
                continue
            pts.append((int(r["param_count"]), float(r["test_auc_mean"]), s))
    return pts


def pareto(pts):
    """Indices of non-dominated points: maximise AUC, minimise params."""
    keep = []
    for i, (pi, ai, _) in enumerate(pts):
        dominated = any(
            (pj <= pi and aj >= ai and (pj < pi or aj > ai))
            for j, (pj, aj, _) in enumerate(pts) if j != i
        )
        if not dominated:
            keep.append(i)
    return keep


fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

for ax, (name, path) in zip(axes, DATASETS):
    pts = load(path)
    front = set(pareto(pts))

    # scatter all points, coloured by camp; frontier points get a black ring
    for i, (p, a, s) in enumerate(pts):
        _, col, mk = CAMP[s]
        on = i in front
        if mk == "x":  # unfilled marker: no edgecolor
            ax.scatter(p, a, c=col, marker=mk, s=26, zorder=2, alpha=0.7)
            continue
        ax.scatter(p, a, c=col, marker=mk, s=70 if on else 24,
                   edgecolors="black" if on else "none",
                   linewidths=1.4 if on else 0, zorder=4 if on else 2,
                   alpha=1.0 if on else 0.5)

    # stepped Pareto frontier line
    fp = sorted([pts[i] for i in front], key=lambda t: t[0])
    if fp:
        xs, ys = [fp[0][0]], [fp[0][1]]
        for p, a, _ in fp[1:]:
            xs += [p, p]; ys += [ys[-1], a]
        ax.plot(xs, ys, color="black", lw=1.0, ls="--", zorder=1, alpha=0.7)

    ax.set_xscale("log")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("trainable parameters (head)")
    ax.grid(True, which="both", ls=":", alpha=0.3)

axes[0].set_ylabel("test AUC")

legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#08306b",
           markersize=7, label="Quantum"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#e67e22",
           markersize=7, label="Classical nonlinear"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#7f8c8d",
           markersize=7, label="Random projection"),
    Line2D([0], [0], marker="x", color="#bdc3c7", lw=0, markersize=7,
           label="Raw pixels"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor="black", markersize=8, label="on Pareto frontier"),
    Line2D([0], [0], color="black", ls="--", lw=1, label="Pareto frontier"),
]
fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.04))

fig.tight_layout(rect=[0, 0.04, 1, 1])
out = os.path.join(ROOT, "docs/paper/figures/capacity_pareto")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print("wrote", out + ".png")
