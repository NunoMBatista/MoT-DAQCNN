"""Delta heatmap: source x head, coloured by AUC minus the best classical source
at that head. Diverging colormap centered at 0; red = beats the best classical
baseline, blue = below it. Compact view of all sources at once -- a near-uniform
non-red field is the visual statement that nothing beats classical.

Usage: python experiments/plot_capacity_heatmap.py --csv <summary.csv> --out <png>
"""
import csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HLAB = ["Linear", "MLP-1", "MLP-2", "CNN-16", "CNN-64"]

# row order, grouped by family
ROWS = [
    ("Digital-Z", "digital_z_1k"), ("Digital-ZZ 1k", "digital_zz_1k"),
    ("Digital-ZZ 4k", "digital_zz_4k"),
    ("Analog-Z", "analog_z_1k"), ("Analog-ZZ 1k", "analog_zz_1k"),
    ("Analog-ZZ 4k", "analog_zz_4k"),
    ("Random x9", "random_9"), ("Random x45", "random_45"),
    ("Random x180", "random_180"),
    ("poly-2", "poly2_45"), ("RFF 45", "rff_45"), ("RFF 180", "rff_180"),
]
# the reference at each head = best of these classical sources
CLASSICAL = ["random_9", "random_45", "random_180", "poly2_45", "rff_45", "rff_180"]
GROUP_SEP = [3, 6, 9]  # draw a divider after these row indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    d = {}
    for r in csv.DictReader(open(args.csv)):
        d[(r["head"], r["source"])] = float(r["test_auc_mean"])

    ref = {h: max(d[(h, s)] for s in CLASSICAL if (h, s) in d) for h in HEADS}
    M = np.full((len(ROWS), len(HEADS)), np.nan)
    for i, (_, src) in enumerate(ROWS):
        for j, h in enumerate(HEADS):
            if (h, src) in d:
                M[i, j] = d[(h, src)] - ref[h]

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    vmax = 0.05
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(HEADS))); ax.set_xticklabels(HLAB)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS])
    for i in range(len(ROWS)):
        for j in range(len(HEADS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                        fontsize=7, color="black")
    for g in GROUP_SEP:
        ax.axhline(g - 0.5, color="white", lw=2)
    ax.set_title(args.title or "AUC minus best classical source")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta$AUC vs. best classical (red = quantum better)")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=160)
    fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
