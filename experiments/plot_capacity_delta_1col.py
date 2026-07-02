"""Single-column variant of the quantum-minus-classical delta heatmap.
Collapses the three per-dataset panels into ONE combined heatmap (15 rows =
3 datasets x 5 quantum sources, 5 head columns) so it fits \\columnwidth.
Writes docs/paper/figures/capacity_delta_1col.pdf."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE = "outputs/paper_results/capacity_sweep_std"
DATASETS = [("breast_mnist", "summary.csv", "Breast"),
            ("pneumonia_mnist", "summary.csv", "Pneumonia"),
            ("tissue_mnist", "summary_rebuilt.csv", "Tissue")]
HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HEAD_LABELS = ["Linear", "MLP-1", "MLP-2", "CNN-16", "CNN-64"]
QUANT = ["digital_zz_1k", "digital_zz_4k", "analog_zz_1k", "analog_zz_4k", "digital_z_1k"]
QLABELS = ["Dig-ZZ 1k", "Dig-ZZ 4k", "Ana-ZZ 1k", "Ana-ZZ 4k", "Dig-Z 1k"]
CLASS = ["random_9", "random_45", "random_180", "poly2_45", "rff_45", "rff_180"]

# Build the stacked (15 x 5) matrix and row labels.
M = np.full((len(DATASETS) * len(QUANT), len(HEADS)), np.nan)
row_labels = []
for di, (ds, fn, _) in enumerate(DATASETS):
    df = pd.read_csv(f"{BASE}/{ds}/{fn}").set_index(["head", "source"])["test_auc_mean"]
    for ri, q in enumerate(QUANT):
        row_labels.append(QLABELS[ri])
        for ci, h in enumerate(HEADS):
            best_c = max((df.get((h, c), np.nan) for c in CLASS
                          if (h, c) in df.index), default=np.nan)
            if (h, q) in df.index:
                M[di * len(QUANT) + ri, ci] = df.loc[(h, q)] - best_c

fig, ax = plt.subplots(figsize=(3.4, 2.75))
norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05)
cmap = plt.get_cmap("RdBu_r")
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

for r in range(M.shape[0]):
    for c in range(M.shape[1]):
        if not np.isnan(M[r, c]):
            # White text on dark cells, black on light ones (perceived luminance).
            rr, gg, bb, _ = cmap(norm(M[r, c]))
            lum = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
            ax.text(c, r, f"{M[r, c]:+.3f}", ha="center", va="center",
                    fontsize=5.0, color="white" if lum < 0.5 else "black")

# Separators between the three dataset blocks + dataset labels on the left.
for b in (1, 2):
    ax.axhline(b * len(QUANT) - 0.5, color="black", lw=1.1)
for di, (_, _, name) in enumerate(DATASETS):
    ax.text(-2.05, di * len(QUANT) + (len(QUANT) - 1) / 2, name, rotation=90,
            ha="center", va="center", fontsize=7.5, fontweight="bold")

ax.set_xticks(range(len(HEADS)))
ax.set_xticklabels(HEAD_LABELS, fontsize=6.5, rotation=30, ha="right")
ax.set_yticks(range(M.shape[0]))
ax.set_yticklabels(row_labels, fontsize=6.0)
ax.tick_params(length=0)

# Thin vertical colorbar on the right: adds width, not height, so it does not
# re-introduce the wasted bottom strip a horizontal bar would.
cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.04, pad=0.03)
cb.set_label("Quantum $-$ best-classical AUC", fontsize=6.0, rotation=270,
             labelpad=9)
cb.ax.tick_params(labelsize=5.5)

out = "docs/paper/figures/capacity_delta_1col.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print("wrote", out)
