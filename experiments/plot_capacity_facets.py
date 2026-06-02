"""Small-multiples capacity-sweep figure: one mini-panel per source family, so
all ~12 sources are visible without a 12-line tangle. Shared y-axis; raw pixels
drawn faintly in every panel as a common cross-panel anchor. Within a panel,
line style encodes scale (dotted = Z-only/9-dim, solid = single-kernel/45-dim,
dashed = four-kernel/180-dim).

Usage: python experiments/plot_capacity_facets.py --csv <summary.csv> --out <png>
"""
import csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HLAB = ["Linear", "MLP-1", "MLP-2", "CNN-16", "CNN-64"]

# (source, linestyle, label) per family panel; color is per-family
PANELS = [
    ("Quantum: digital", "#08306b", [
        ("digital_z_1k", ":", r"Digital-Z"),
        ("digital_zz_1k", "-", r"Digital-ZZ 1k"),
        ("digital_zz_4k", "--", r"Digital-ZZ 4k")]),
    ("Quantum: analog", "#c0392b", [
        ("analog_z_1k", ":", r"Analog-Z"),
        ("analog_zz_1k", "-", r"Analog-ZZ 1k"),
        ("analog_zz_4k", "--", r"Analog-ZZ 4k")]),
    ("Random projection", "#bcbd22", [
        ("random_9", ":", r"Random$\times$9"),
        ("random_45", "-", r"Random$\times$45"),
        ("random_180", "--", r"Random$\times$180")]),
    ("Classical nonlinear", "#e67e22", [
        ("poly2_45", ":", r"poly-2 (45)"),
        ("rff_45", "-", r"RFF 45"),
        ("rff_180", "--", r"RFF 180")]),
]


def load(csv_path):
    d = {}
    for r in csv.DictReader(open(csv_path)):
        d[(r["head"], r["source"])] = float(r["test_auc_mean"])
    return d


def series(d, src):
    return np.array([d.get((h, src), np.nan) for h in HEADS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    d = load(args.csv)
    raw = series(d, "raw")

    # shared y-range from all plotted sources
    allv = [v for k, v in d.items() if not np.isnan(v)]
    ylo, yhi = min(allv) - 0.005, max(allv) + 0.005
    x = np.arange(len(HEADS))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 4.6), sharex=True, sharey=True)
    for ax, (title, color, lines) in zip(axes.ravel(), PANELS):
        # common anchor: raw pixels, faint gray
        ax.plot(x, raw, color="#999999", ls=(0, (1, 1)), lw=1.2, alpha=0.7,
                marker="x", markersize=4, label="Raw pixels", zorder=1)
        for src, style, lab in lines:
            ax.plot(x, series(d, src), color=color, ls=style, lw=1.9,
                    marker="o", markersize=4, label=lab, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    for ax in axes[1]:
        ax.set_xticks(x); ax.set_xticklabels(HLAB, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test AUC")
    if args.title:
        fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=160)
    fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
