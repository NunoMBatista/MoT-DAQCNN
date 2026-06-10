"""Small-multiples capacity-sweep figure: one mini-panel per source family, so
all ~12 sources are visible without a 12-line tangle. Shared y-axis; raw pixels
drawn faintly in every panel as a common cross-panel anchor. Within a panel,
scale is encoded by both line style and marker (dotted+triangle = smallest,
solid+circle = single-kernel/45-dim, long-dash+square = four-kernel/180-dim).

Usage: python experiments/plot_capacity_facets.py --csv <summary.csv> --out <png>
"""
import csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HLAB = ["Linear", "MLP-1", "MLP-2", "CNN-16", "CNN-64"]

# scale key -> (linestyle, marker). Each scale gets BOTH a distinct dash pattern
# and a distinct marker shape, so the three same-colour lines in a panel stay
# unambiguous even in a short legend handle (solid vs dashed alone reads poorly).
SCALE_STYLE = {
    "z":  ((0, (1, 1.4)), "v"),   # smallest scale  : dotted     + down-triangle
    "1k": ("solid",       "o"),   # single-kernel/45: solid      + circle
    "4k": ((0, (5, 2)),   "s"),   # four-kernel/180 : long-dashed + square
}

# (source, scale_key, label) per family panel; color is per-family
PANELS = [
    ("Quantum: digital", "#08306b", [
        ("digital_z_1k",  "z",  r"Digital-Z"),
        ("digital_zz_1k", "1k", r"Digital-ZZ 1k"),
        ("digital_zz_4k", "4k", r"Digital-ZZ 4k")]),
    ("Quantum: analog", "#c0392b", [
        ("analog_z_1k",  "z",  r"Analog-Z"),
        ("analog_zz_1k", "1k", r"Analog-ZZ 1k"),
        ("analog_zz_4k", "4k", r"Analog-ZZ 4k")]),
    ("Random projection", "#bcbd22", [
        ("random_9",   "z",  r"Random$\times$9"),
        ("random_45",  "1k", r"Random$\times$45"),
        ("random_180", "4k", r"Random$\times$180")]),
    ("Classical nonlinear", "#e67e22", [
        ("poly2_45", "z",  r"poly-2 (45)"),
        ("rff_45",   "1k", r"RFF 45"),
        ("rff_180",  "4k", r"RFF 180")]),
]


def load(csv_path):
    d = {}
    for r in csv.DictReader(open(csv_path)):
        d[(r["head"], r["source"])] = (float(r["test_auc_mean"]),
                                       float(r["test_auc_std"]))
    return d


def series(d, src):
    """Return (means, stds) over the head ladder for a source."""
    m = np.array([d.get((h, src), (np.nan, np.nan))[0] for h in HEADS])
    s = np.array([d.get((h, src), (np.nan, np.nan))[1] for h in HEADS])
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    d = load(args.csv)
    raw_m, _ = series(d, "raw")

    # shared y-range from all plotted means (+/- a hair of headroom for the bands)
    allv = [m for (m, s) in d.values() if not np.isnan(m)]
    ylo, yhi = min(allv) - 0.02, max(allv) + 0.015
    x = np.arange(len(HEADS))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 4.6), sharex=True, sharey=True)
    for ax, (title, color, lines) in zip(axes.ravel(), PANELS):
        # common anchor: raw pixels, faint gray (no band, kept as a clean reference)
        ax.plot(x, raw_m, color="#999999", ls=(0, (1, 1)), lw=1.2, alpha=0.7,
                marker="x", markersize=4, label="Raw pixels", zorder=1)
        for src, scale, lab in lines:
            ls, mk = SCALE_STYLE[scale]
            m, s = series(d, src)
            # +/-1 SD band over the 10 evaluation seeds
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.12,
                            lw=0, zorder=2)
            ax.plot(x, m, color=color, ls=ls, lw=1.9,
                    marker=mk, markersize=4.5, label=lab, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.3)
        # longer handle so the dash patterns are visible, not just the marker
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9,
                  handlelength=3.0, handletextpad=0.5)
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
