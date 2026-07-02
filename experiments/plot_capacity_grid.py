"""Combined head-capacity figure for the paper: all datasets in one grid.

Rows are datasets, columns are the four feature families. Every source curve
from the per-dataset facets is kept (+/-1 SD bands, raw anchor); the compression
is one figure with shared per-row y-axes and a single legend. At each head a
gold star marks the best source across all families, labelled by the winning
camp (quantum vs classical) and its margin over the other camp.

Run from the repo root:
    python experiments/plot_capacity_grid.py
Writes docs/paper/figures/capacity_sweep_grid.{pdf,png}.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HLAB = ["Linear", "MLP-1", "MLP-2", "CNN-16", "CNN-64"]

# scale key -> (linestyle, marker)
SCALE_STYLE = {
    "z":  ((0, (1, 1.4)), "v"),
    "1k": ("solid",       "o"),
    "4k": ((0, (5, 2)),   "s"),
}

# column families: (title, color, [(source, scale)])
FAMILIES = [
    ("Quantum (digital)", "#08306b",
     [("digital_z_1k", "z"), ("digital_zz_1k", "1k"), ("digital_zz_4k", "4k")]),
    ("Quantum (analog)", "#c0392b",
     [("analog_z_1k", "z"), ("analog_zz_1k", "1k"), ("analog_zz_4k", "4k")]),
    ("Random projection", "#bcbd22",
     [("random_9", "z"), ("random_45", "1k"), ("random_180", "4k")]),
    ("Classical nonlinear", "#e67e22",
     [("poly2_45", "z"), ("rff_45", "1k"), ("rff_180", "4k")]),
]

CS = "outputs/paper_results/capacity_sweep_std"
DATASETS = [
    ("BreastMNIST", f"{CS}/breast_mnist/summary.csv"),
    ("PneumoniaMNIST", f"{CS}/pneumonia_mnist/summary.csv"),
    ("TissueMNIST", f"{CS}/tissue_mnist/summary_rebuilt.csv"),  # preliminary, digital-only
]

# camps for the cross-group margin: columns 0,1 = quantum; 2,3 = classical
SRC_COL = {}
for ci, (title, color, lines) in enumerate(FAMILIES):
    for src, scale in lines:
        SRC_COL[src] = ci
QUANTUM_SRCS = [s for s, c in SRC_COL.items() if c in (0, 1)]
CLASSICAL_SRCS = [s for s, c in SRC_COL.items() if c in (2, 3)]


def load(path):
    d = {}
    for r in csv.DictReader(open(os.path.join(ROOT, path))):
        try:
            d[(r["head"], r["source"])] = (float(r["test_auc_mean"]),
                                           float(r["test_auc_std"]))
        except (ValueError, KeyError):
            pass
    return d


def series(d, src):
    m = np.array([d.get((h, src), (np.nan, np.nan))[0] for h in HEADS])
    s = np.array([d.get((h, src), (np.nan, np.nan))[1] for h in HEADS])
    return m, s


data = [(name, load(path)) for name, path in DATASETS]
x = np.arange(len(HEADS))

fig, axes = plt.subplots(3, 4, figsize=(7.1, 3.9), sharex=True)

for ri, (dname, d) in enumerate(data):
    raw_m, _ = series(d, "raw")
    allv = [m for (m, s) in d.values() if not np.isnan(m)]
    ylo, yhi = min(allv) - 0.020, max(allv) + 0.042  # headroom for star labels

    for ci, (title, color, lines) in enumerate(FAMILIES):
        ax = axes[ri, ci]
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.25, lw=0.5)
        ax.plot(x, raw_m, color="#999999", ls=(0, (1, 1)), lw=1.0, alpha=0.7,
                marker="x", markersize=3, zorder=1)
        any_data = False
        for src, scale in lines:
            ls, mk = SCALE_STYLE[scale]
            m, s = series(d, src)
            if np.isnan(m).all():
                continue
            any_data = True
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, lw=0, zorder=2)
            ax.plot(x, m, color=color, ls=ls, lw=1.6, marker=mk, markersize=3.6, zorder=3)
        if not any_data:
            ax.text(0.5, 0.5, "not run\n(digital-only)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="#999999", style="italic")
        if ri == 0:
            ax.set_title(title, fontsize=8.5, color=color, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{dname}\nTest AUC", fontsize=8)
        ax.tick_params(labelsize=6.5)

    # winner star + cross-camp margin (best quantum vs best classical) per head
    row_anno = {}
    for hi in range(len(HEADS)):
        def camp_best(srcs):
            cand = [(series(d, s)[0][hi], s) for s in srcs
                    if not np.isnan(series(d, s)[0][hi])]
            return max(cand) if cand else None
        bq, bc = camp_best(QUANTUM_SRCS), camp_best(CLASSICAL_SRCS)
        if bq is None or bc is None:
            continue
        quantum_wins = bq[0] >= bc[0]
        best_v, best_src = bq if quantum_wins else bc
        margin = abs(bq[0] - bc[0])
        tag = "Q" if quantum_wins else "C"
        ci = SRC_COL[best_src]
        axes[ri, ci].plot(hi, best_v, marker="*", ms=8.5, mfc="gold", mec="black",
                          mew=0.6, ls="none", zorder=6)
        row_anno.setdefault(ci, []).append((hi, best_v, tag + f"{margin:.4f}"[1:]))

    # place labels: edge-aware alignment, data-aware height (clear of every
    # curve and star within the label's horizontal extent), then resolve
    # label-label collisions explicitly
    last = len(HEADS) - 1
    TH = 0.011          # label text height in data units
    for ci, items in row_anno.items():
        col_srcs = [s for s, _ in FAMILIES[ci][2]] + ["raw"]
        placed = []     # (x0, x1, y0, y1) bands of labels already set
        for hi, bv, text in sorted(items):
            ha = "left" if hi == 0 else "right" if hi == last else "center"
            xpos = hi + (0.08 if hi == 0 else -0.08 if hi == last else 0)
            # horizontal extent: `reach` is how far the label overhangs toward a
            # curve (for vertical clearance); the collision band [x0,x1] is the
            # text footprint (~0.62 head-units half-width) used to detect
            # label-label overlap, slightly wider so adjacent labels separate
            if hi == 0:
                x0, x1, reach = 0.0, 0.9, 1.0
            elif hi == last:
                x0, x1, reach = last - 0.9, float(last), 1.0
            else:
                x0, x1, reach = hi - 0.62, hi + 0.62, 0.5
            # everything under the label: own head, the reachable part of each
            # neighbouring segment, and any neighbouring star (+ marker radius)
            local = [bv]
            for s in col_srcs:
                m, _ = series(d, s)
                if np.isnan(m[hi]):
                    continue
                local.append(m[hi])
                for nb in (hi - 1, hi + 1):
                    if 0 <= nb <= last and not np.isnan(m[nb]):
                        local.append(m[hi] + reach * (m[nb] - m[hi]))
            for ohi, obv, _ in items:
                if ohi != hi and abs(ohi - hi) <= 1:
                    local.append(obv + 0.010)

            def collide(y0, y1):
                return any(x0 < px1 and px0 < x1 and y0 < py1 + 0.002
                           and y1 > py0 - 0.002 for px0, px1, py0, py1 in placed)

            if (bv - ylo) / (yhi - ylo) > 0.78:
                ytxt, va = min(local) - 0.010, "top"
                while collide(ytxt - TH, ytxt):
                    ytxt -= 0.004
                band = (ytxt - TH, ytxt)
            else:
                ytxt, va = max(local) + 0.010, "bottom"
                while collide(ytxt, ytxt + TH):
                    ytxt += 0.004
                if ytxt > yhi - TH:  # would poke out the top: flip below
                    ytxt, va = min(local) - 0.010, "top"
                    while collide(ytxt - TH, ytxt):
                        ytxt -= 0.004
                    band = (ytxt - TH, ytxt)
                else:
                    band = (ytxt, ytxt + TH)
            placed.append((x0, x1, band[0], band[1]))
            axes[ri, ci].text(xpos, ytxt, text, ha=ha, va=va, fontsize=5.2,
                              color="black", zorder=7)

for ax in axes[2]:
    ax.set_xticks(x)
    ax.set_xticklabels(HLAB, fontsize=7, rotation=90, ha="center", va="top")

handles = [
    Line2D([0], [0], color="#444", ls=SCALE_STYLE["z"][0], marker="v", ms=4,
           label=r"smallest ($\langle Z\rangle$ / $\times$9 / poly-2)"),
    Line2D([0], [0], color="#444", ls="solid", marker="o", ms=4, label="single-kernel, 45-dim"),
    Line2D([0], [0], color="#444", ls=(0, (5, 2)), marker="s", ms=4, label="four-kernel, 180-dim"),
    Line2D([0], [0], color="#999", ls=(0, (1, 1)), marker="x", ms=4, label="raw pixels"),
    Line2D([0], [0], color="black", marker="*", mfc="gold", mec="black", ms=8,
           ls="none", label="best at capacity; Q.0136 = quantum leads classical by 0.0136"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
           frameon=False, handlelength=2.6, bbox_to_anchor=(0.5, -0.03))

fig.tight_layout(rect=[0, 0.05, 1, 1.0])
out = os.path.join(ROOT, "docs/paper/figures/capacity_sweep_grid")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", bbox_inches="tight", dpi=175)
print(f"wrote {out}.pdf / .png")
