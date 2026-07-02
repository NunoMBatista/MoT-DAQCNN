"""Capacity quantum-minus-classical delta heatmap. For each dataset and head,
the cell is (quantum source AUC) - (best classical source AUC at that head),
so red = quantum ahead, blue = classical ahead. Visualises capacity
conditioning: the quantum lead at low-capacity heads shrinks/reverses at the
CNN heads. Writes docs/paper/figures/capacity_delta_ext.pdf."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE = "outputs/paper_results/capacity_sweep_std"
DATASETS = [("breast_mnist", "summary.csv", "BreastMNIST"),
            ("pneumonia_mnist", "summary.csv", "PneumoniaMNIST"),
            ("tissue_mnist", "summary_rebuilt.csv", "TissueMNIST")]
HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HEAD_LABELS = ["Linear", "MLP-1", "MLP-2", "CNN-16", "CNN-64"]
QUANT = ["digital_zz_1k", "digital_zz_4k", "analog_zz_1k", "analog_zz_4k", "digital_z_1k"]
QLABELS = ["Dig-ZZ 1k", "Dig-ZZ 4k", "Ana-ZZ 1k", "Ana-ZZ 4k", "Dig-Z 1k"]
CLASS = ["random_9", "random_45", "random_180", "poly2_45", "rff_45", "rff_180"]

fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.3))
norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05)

for ax, (ds, fn, title) in zip(axes, DATASETS):
    df = pd.read_csv(f"{BASE}/{ds}/{fn}").set_index(["head", "source"])["test_auc_mean"]
    M = np.full((len(QUANT), len(HEADS)), np.nan)
    for ci, h in enumerate(HEADS):
        best_c = max((df.get((h, c), np.nan) for c in CLASS
                      if (h, c) in df.index), default=np.nan)
        for ri, q in enumerate(QUANT):
            if (h, q) in df.index:
                M[ri, ci] = df.loc[(h, q)] - best_c
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    for ri in range(len(QUANT)):
        for ci in range(len(HEADS)):
            if not np.isnan(M[ri, ci]):
                ax.text(ci, ri, f"{M[ri, ci]:+.3f}", ha="center", va="center",
                        fontsize=5.5, color="black")
    ax.set_xticks(range(len(HEADS)))
    ax.set_xticklabels(HEAD_LABELS, fontsize=6.5, rotation=30, ha="right")
    ax.set_yticks(range(len(QUANT)))
    ax.set_yticklabels(QLABELS if ds == "breast_mnist" else [""] * len(QUANT), fontsize=6.5)
    ax.set_title(title, fontsize=8.5)

cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
cb.set_label("Quantum $-$ best-classical AUC", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
out = "docs/paper/figures/capacity_delta_ext.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
