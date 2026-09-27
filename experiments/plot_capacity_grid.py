"""Combined head-capacity figure for the paper: all datasets in one grid.

Rows are datasets, columns are the four feature families. Every source curve
from the per-dataset facets is kept (+/-1 SD bands, raw anchor); the compression
is one figure with shared per-row y-axes and a single legend. At each head a
gold star marks the validation-selected leader across all families; the camp
is the star's column, and the margins are stated in the text (A11: the
5.2 pt margin labels were dropped from the paper figure; --star-labels
restores them).

Run from the repo root:
    python experiments/plot_capacity_grid.py
Writes docs/paper/figures/capacity_sweep_grid.{pdf,png}.
"""
import os
import numpy as np
import val_selected_margins as val_sel
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts in the PDF, not Type 3
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse

# A11 (camera-ready): density variants for the user to compare; the defaults
# reproduce the paper figure exactly
ap = argparse.ArgumentParser()
ap.add_argument("--no-bands", action="store_true", help="drop the +/-1 SD bands")
ap.add_argument("--star-labels", action="store_true",
                help="add the Q/C margin labels next to the winner stars")
ap.add_argument("--out", default=None,
                help="output stem (default docs/paper/figures/capacity_sweep_grid)")
ARGS = ap.parse_args()

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

# Curves and camp selection both read the per-seed JSONs. They reproduce the
# summary CSVs exactly (checked on all 195 cells) and additionally carry the
# per-seed validation accuracy that the camp selection needs.
DATASETS = val_sel.DATASETS

# camps for the cross-group margin: columns 0,1 = quantum; 2,3 = classical
SRC_COL = {}
for ci, (title, color, lines) in enumerate(FAMILIES):
    for src, scale in lines:
        SRC_COL[src] = ci
QUANTUM_SRCS = [s for s, c in SRC_COL.items() if c in (0, 1)]
CLASSICAL_SRCS = [s for s, c in SRC_COL.items() if c in (2, 3)]


def series(cells, src):
    """Per-head mean and SD of test AUC over the 10 validation seeds."""
    means, sds = [], []
    for head in HEADS:
        cell = cells.get((head, src))
        if cell is None:
            means.append(np.nan)
            sds.append(np.nan)
            continue
        vals = np.array(cell["test_aucs"])
        means.append(vals.mean())
        sds.append(vals.std())
    return np.array(means), np.array(sds)


# The camps must be exactly the ones the selection module ranks, or the figure
# and the reported margins would answer different questions.
assert sorted(QUANTUM_SRCS) == sorted(val_sel.QUANTUM)
assert sorted(CLASSICAL_SRCS) == sorted(val_sel.CLASSICAL)

data = [(name, val_sel.load_cells(sweep_dir)) for name, sweep_dir in DATASETS]
x = np.arange(len(HEADS))

fig, axes = plt.subplots(3, 4, figsize=(7.1, 3.9), sharex=True)

for ri, (dname, d) in enumerate(data):
    raw_m, _ = series(d, "raw")
    allv = [float(np.mean(cell["test_aucs"])) for cell in d.values()]
    # headroom for the star labels (less when they are off)
    ylo, yhi = min(allv) - 0.020, max(allv) + (0.042 if ARGS.star_labels else 0.020)

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
            if not ARGS.no_bands:
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
    for hi, head in enumerate(HEADS):
        # Each camp's representative is chosen on VALIDATION accuracy (reviewer
        # 4 #3); the value plotted is that fixed choice's test AUC, so no test
        # number takes part in the selection.
        def camp_rep(srcs):
            src = val_sel.pick(d, head, srcs, "val_acc_mean")
            return (series(d, src)[0][hi], src) if src is not None else None
        bq, bc = camp_rep(QUANTUM_SRCS), camp_rep(CLASSICAL_SRCS)
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
    for ci, items in (row_anno.items() if ARGS.star_labels else []):
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
           label=r"$\langle Z\rangle$ / $\times$9 / poly-2"),
    Line2D([0], [0], color="#444", ls="solid", marker="o", ms=4, label="single-kernel, 45-dim"),
    Line2D([0], [0], color="#444", ls=(0, (5, 2)), marker="s", ms=4, label="four-kernel, 180-dim"),
    Line2D([0], [0], color="#999", ls=(0, (1, 1)), marker="x", ms=4, label="raw pixels"),
    Line2D([0], [0], color="black", marker="*", mfc="gold", mec="black", ms=8,
           ls="none", label=("val-selected leader; Q.0136 = quantum leads classical by 0.0136"
                            if ARGS.star_labels else "val-selected leader")),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
           frameon=False, handlelength=2.6, bbox_to_anchor=(0.5, -0.03))

fig.tight_layout(rect=[0, 0.05, 1, 1.0])
out = ARGS.out or os.path.join(ROOT, "docs/paper/figures/capacity_sweep_grid")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", bbox_inches="tight", dpi=175)
print(f"wrote {out}.pdf / .png")
