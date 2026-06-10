"""Plot the linear probing topology sweep results.

Reads outputs/paper_results/linear_probing/breast_mnist_topology_sweep.csv
and produces two figures:
  (1) Two heatmaps (1-kernel and 4-kernel scales) of AUC across
      topology x encoding cells. Cells are annotated with their AUC.
  (2) A grouped bar chart of the 1-kernel data, with topologies sorted
      by the mean AUC across encodings, plus reference lines for the
      raw-pixel and matched-dimension random-filter baselines.

Outputs land in outputs/paper_results/linear_probing/plots/.
"""

import argparse
import csv
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

# Baseline AUCs from outputs/paper_results/linear_probing/breast_mnist.csv
# (raw pixels + random filters at matched dim, single source of truth)
BASELINE_RAW = 0.7888
# Seed-averaged random-projection baselines (mean over projection seeds, not a
# single draw). The old single-seed random_45 (0.7659) was a 5th-percentile low
# draw; reporting the seed mean avoids flattering the quantum kernels.
BASELINE_RANDOM_45 = 0.7843      # matches 1-kern Z+ZZ dim (3645)
BASELINE_RANDOM_180 = 0.7654     # matches 4-kern Z+ZZ dim (14580)


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["dim"] = int(r["dim"])
            r["acc"] = float(r["acc"])
            r["auc"] = float(r["auc"])
            rows.append(r)
    return rows


def build_matrix(rows, scale, topology_order, encoding_order):
    """Return a (n_encoding x n_topology) matrix of AUC values."""
    M = np.full((len(encoding_order), len(topology_order)), np.nan)
    for r in rows:
        if r["scale"] != scale:
            continue
        if r["encoding"] not in encoding_order:   # skip encodings we don't plot (e.g. Analog-Z)
            continue
        i = encoding_order.index(r["encoding"])
        if r["topology_set"] in topology_order:
            j = topology_order.index(r["topology_set"])
            M[i, j] = r["auc"]
    return M


def plot_heatmap(M, topology_order, encoding_order, title, baseline_label,
                 baseline_auc, ax):
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.74, vmax=0.86)
    ax.set_xticks(range(len(topology_order)))
    ax.set_xticklabels(topology_order, rotation=30, ha="right")
    ax.set_yticks(range(len(encoding_order)))
    ax.set_yticklabels(encoding_order)
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v < 0.80 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=txt_color, fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("AUC")
    cb.ax.axhline(baseline_auc, color="red", linewidth=1.2)
    cb.ax.text(1.3, baseline_auc, f"  {baseline_label}", color="red",
               va="center", transform=cb.ax.get_yaxis_transform(), fontsize=8)


def plot_bars_1k(rows, out_path):
    """Grouped bar chart at 1-kernel scale, sorted by mean AUC across encodings."""
    enc_order = ["Digital-Z", "Digital-ZZ", "Analog-ZZ"]
    topos = OrderedDict()
    for r in rows:
        if r["scale"] != "1k":
            continue
        topos.setdefault(r["topology_set"], {})[r["encoding"]] = r["auc"]

    topo_means = {t: np.mean(list(d.values())) for t, d in topos.items()}
    topo_order = sorted(topos.keys(), key=lambda t: -topo_means[t])

    n_t = len(topo_order)
    width = 0.25
    x = np.arange(n_t)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"Digital-Z": "#7f8c8d", "Digital-ZZ": "#2980b9", "Analog-ZZ": "#c0392b"}
    for k, enc in enumerate(enc_order):
        vals = [topos[t].get(enc, np.nan) for t in topo_order]
        bars = ax.bar(x + (k - 1) * width, vals, width,
                      label=enc, color=colors[enc])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                        rotation=90)

    ax.axhline(BASELINE_RANDOM_45, color="gray", linestyle="--", linewidth=1)
    ax.text(n_t - 0.4, BASELINE_RANDOM_45 + 0.001,
            f"random×45  ({BASELINE_RANDOM_45:.3f})",
            color="gray", fontsize=8, ha="right", va="bottom")
    ax.axhline(BASELINE_RAW, color="black", linestyle=":", linewidth=1)
    ax.text(n_t - 0.4, BASELINE_RAW + 0.001,
            f"raw pixels  ({BASELINE_RAW:.3f})",
            color="black", fontsize=8, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(topo_order, rotation=30, ha="right")
    ax.set_ylabel("Linear-probe AUC (BreastMNIST test)")
    ax.set_title("1-kernel probing: AUC by topology and encoding "
                 "(topologies ordered by mean AUC)")
    ax.set_ylim(0.74, 0.87)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/paper_results/linear_probing/breast_mnist_topology_sweep.csv")
    ap.add_argument("--out-dir", default="outputs/paper_results/linear_probing/plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.csv)

    # === Heatmaps (one fig with two panels) ===
    enc_order = ["Digital-Z", "Digital-ZZ", "Analog-ZZ"]
    topo_order_1k = ["kings", "horizontal", "vertical", "cross",
                     "ring", "chain", "star", "grid"]
    topo_order_4k = [
        "grid+chain+horizontal+kings",
        "grid+chain+horizontal+vertical",
        "grid+chain+horizontal+cross",
        "grid+chain+horizontal+ring",
        "grid+chain+horizontal+star",
    ]

    M_1k = build_matrix(rows, "1k", topo_order_1k, enc_order)
    M_4k = build_matrix(rows, "4k", topo_order_4k, enc_order)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2),
                             gridspec_kw={"height_ratios": [1, 0.7]})
    plot_heatmap(M_1k, topo_order_1k, enc_order,
                 "1-kernel probing AUC (BreastMNIST)",
                 "random×45", BASELINE_RANDOM_45, axes[0])
    short_4k = [s.replace("grid+chain+horizontal+", "g+c+h+") for s in topo_order_4k]
    plot_heatmap(M_4k, short_4k, enc_order,
                 "4-kernel probing AUC (BreastMNIST)",
                 "random×180", BASELINE_RANDOM_180, axes[1])
    fig.tight_layout()
    heatmap_pdf = os.path.join(args.out_dir, "topology_sweep_heatmap.pdf")
    fig.savefig(heatmap_pdf, bbox_inches="tight")
    fig.savefig(heatmap_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)

    # === Grouped bar chart for 1-kernel ===
    bars_pdf = os.path.join(args.out_dir, "topology_sweep_1k_bars.pdf")
    plot_bars_1k(rows, bars_pdf)

    print(f"Wrote {heatmap_pdf}")
    print(f"Wrote {heatmap_pdf.replace('.pdf','.png')}")
    print(f"Wrote {bars_pdf}")
    print(f"Wrote {bars_pdf.replace('.pdf','.png')}")


if __name__ == "__main__":
    main()
