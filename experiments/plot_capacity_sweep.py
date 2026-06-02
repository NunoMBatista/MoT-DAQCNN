"""Plot the head-capacity sweep results.

Reads outputs/paper_results/capacity_sweep/breast_mnist/summary.csv and
produces a multi-panel figure showing how test AUC evolves with head
capacity for each feature source.

Layout:
  - Top panel: 1-kernel-scale sources (raw, random_45, digital_z_1k,
    digital_zz_1k, analog_zz_1k). All matched at ~3645 dim except raw
    (784) and digital_z (729) which are flagged as lower-dim references.
  - Bottom panel: 4-kernel-scale sources (raw, random_180,
    digital_zz_4k, analog_zz_4k). All matched at 14580 dim except raw.

Each line shows mean test AUC with std-error band over 10 seeds.
Heads are ordered by architectural expressivity: linear -> mlp1 ->
mlp2 -> cnn_small -> cnn_large.

The keystone observation we expect to see: the quantum-classical gap
shrinks (or fluctuates) as head capacity grows.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


HEAD_ORDER = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HEAD_LABELS = {
    "linear":    "Linear",
    "mlp1":      "MLP-1",
    "mlp2":      "MLP-2",
    "cnn_small": "CNN-16",
    "cnn_large": "CNN-64",
}

# Source styling
SOURCES_1K = ["raw", "random_45", "digital_z_1k", "digital_zz_1k", "analog_zz_1k"]
SOURCES_4K = ["raw", "random_180", "digital_zz_4k", "analog_zz_4k"]

SRC_LABEL = {
    "raw":           "Raw pixels (784)",
    "random_45":     "Random x45 (3645)",
    "random_180":    "Random x180 (14580)",
    "digital_z_1k":  "Digital-Z 1k (729)",
    "digital_zz_1k": "Digital-ZZ 1k (3645)",
    "analog_zz_1k":  "Analog-ZZ 1k (3645)",
    "digital_zz_4k": "Digital-ZZ 4k (14580)",
    "analog_zz_4k":  "Analog-ZZ 4k (14580)",
}

SRC_COLOR = {
    "raw":           "#7f7f7f",   # gray
    "random_45":     "#bcbd22",   # olive
    "random_180":    "#bcbd22",   # olive
    "digital_z_1k":  "#1f77b4",   # blue (light)
    "digital_zz_1k": "#08306b",   # navy
    "analog_zz_1k":  "#c0392b",   # red
    "digital_zz_4k": "#08306b",   # navy
    "analog_zz_4k":  "#c0392b",   # red
}

SRC_STYLE = {
    "raw":           ":",
    "random_45":     "--",
    "random_180":    "--",
    "digital_z_1k":  "-.",
    "digital_zz_1k": "-",
    "analog_zz_1k":  "-",
    "digital_zz_4k": "-",
    "analog_zz_4k":  "-",
}

SRC_MARKER = {
    "raw":           "x",
    "random_45":     "s",
    "random_180":    "s",
    "digital_z_1k":  "v",
    "digital_zz_1k": "o",
    "analog_zz_1k":  "o",
    "digital_zz_4k": "o",
    "analog_zz_4k":  "o",
}


def load_summary(csv_path):
    """Returns {(head, source): {auc_mean, auc_std, acc_mean, acc_std, params}}."""
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["head"], row["source"])
            out[key] = {
                "auc_mean": float(row["test_auc_mean"]),
                "auc_std":  float(row["test_auc_std"]),
                "acc_mean": float(row["test_acc_mean"]),
                "acc_std":  float(row["test_acc_std"]),
                "param_count": int(row.get("param_count", 0)),
            }
    return out


def plot_panel(ax, data, sources, title):
    x = np.arange(len(HEAD_ORDER))
    for src in sources:
        means, stds = [], []
        for head in HEAD_ORDER:
            entry = data.get((head, src))
            if entry is None:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(entry["auc_mean"])
                stds.append(entry["auc_std"])
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(x, means,
                color=SRC_COLOR[src], linestyle=SRC_STYLE[src],
                marker=SRC_MARKER[src], markersize=6,
                linewidth=1.7, label=SRC_LABEL[src])
        ax.fill_between(x, means - stds, means + stds,
                        color=SRC_COLOR[src], alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels([HEAD_LABELS[h] for h in HEAD_ORDER])
    ax.set_ylabel("Test AUC")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


# Merged single-panel: both scales on one axis. Color encodes the source family,
# linestyle encodes the scale (solid = single kernel / 45-dim, dashed = four
# kernels / 180-dim, dotted = raw pixels). Two-section legend keeps it readable.
MERGED_SOURCES = ["raw", "random_45", "random_180", "rff_45", "rff_180",
                  "digital_zz_1k", "digital_zz_4k"]

SCALE_STYLE = {  # linestyle by scale, independent of family colour
    "raw": ":", "random_45": "-", "random_180": "--",
    "rff_45": "-", "rff_180": "--",
    "digital_zz_1k": "-", "digital_zz_4k": "--",
}

# extra colours/markers for the nonlinear-control family
SRC_COLOR["rff_45"] = "#e67e22"   # orange
SRC_COLOR["rff_180"] = "#e67e22"
SRC_MARKER["rff_45"] = "^"
SRC_MARKER["rff_180"] = "^"

FAMILY_LEGEND = [("Raw pixels", "#7f7f7f"), ("Random proj.", "#bcbd22"),
                 ("RFF (nonlinear)", "#e67e22"), ("Digital-ZZ (best quantum)", "#08306b")]


def plot_merged(ax, data, title):
    from matplotlib.lines import Line2D
    x = np.arange(len(HEAD_ORDER))
    for src in MERGED_SOURCES:
        means = np.array([(data.get((h, src)) or {}).get("auc_mean", np.nan)
                          for h in HEAD_ORDER])
        stds = np.array([(data.get((h, src)) or {}).get("auc_std", np.nan)
                         for h in HEAD_ORDER])
        ax.plot(x, means, color=SRC_COLOR[src], linestyle=SCALE_STYLE[src],
                marker=SRC_MARKER[src], markersize=5, linewidth=1.8, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([HEAD_LABELS[h] for h in HEAD_ORDER])
    ax.set_ylabel("Test AUC")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    # two-section legend: family (colour) + scale (linestyle)
    fam_handles = [Line2D([0], [0], color=c, lw=2, marker="o", label=n)
                   for n, c in FAMILY_LEGEND]
    scale_handles = [
        Line2D([0], [0], color="black", lw=2, ls="-", label="Single kernel (45-dim)"),
        Line2D([0], [0], color="black", lw=2, ls="--", label="Four kernels (180-dim)"),
        Line2D([0], [0], color="#7f7f7f", lw=2, ls=":", label="Raw pixels (784-dim)"),
    ]
    leg1 = ax.legend(handles=fam_handles, loc="upper left", fontsize=8,
                     title="Source", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=scale_handles, loc="lower right", fontsize=8,
              title="Scale", framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="outputs/paper_results/capacity_sweep/breast_mnist/summary.csv")
    ap.add_argument("--out-dir",
                    default="outputs/paper_results/capacity_sweep/breast_mnist/plots")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Sweep results not found at {args.csv}. "
                                "Has the sweep finished?")

    data = load_summary(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    # Dataset label from the csv path (e.g. .../pneumonia_mnist/summary.csv)
    ds_label = ("PneumoniaMNIST" if "pneumonia" in args.csv.lower()
                else "BreastMNIST" if "breast" in args.csv.lower()
                else os.path.basename(os.path.dirname(os.path.dirname(args.csv))))

    # === Merged single-panel figure: both scales on one axis ===
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    plot_merged(ax, data, f"AUC vs. head capacity ({ds_label})")
    fig.tight_layout()
    out_pdf = os.path.join(args.out_dir, "capacity_sweep.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)

    # === Companion: quantum-vs-classical gap (per matched-dim pair) ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(HEAD_ORDER))

    # 1-kern gap: quantum sources minus random_45
    rng_key = "random_45"
    for src in ["digital_zz_1k", "analog_zz_1k"]:
        gaps = []
        for head in HEAD_ORDER:
            q = data.get((head, src))
            r = data.get((head, rng_key))
            if q is None or r is None:
                gaps.append(np.nan)
            else:
                gaps.append(q["auc_mean"] - r["auc_mean"])
        axes[0].plot(x, gaps,
                     color=SRC_COLOR[src], marker=SRC_MARKER[src], linewidth=1.7,
                     label=f"{SRC_LABEL[src].split()[0]}-{SRC_LABEL[src].split()[1]} - {SRC_LABEL[rng_key].split()[0]}")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([HEAD_LABELS[h] for h in HEAD_ORDER])
    axes[0].set_ylabel(r"$\Delta$ AUC (quantum minus random)")
    axes[0].set_title("1-kernel: quantum-vs-classical-random AUC gap")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    # 4-kern gap
    rng_key = "random_180"
    for src in ["digital_zz_4k", "analog_zz_4k"]:
        gaps = []
        for head in HEAD_ORDER:
            q = data.get((head, src))
            r = data.get((head, rng_key))
            if q is None or r is None:
                gaps.append(np.nan)
            else:
                gaps.append(q["auc_mean"] - r["auc_mean"])
        axes[1].plot(x, gaps,
                     color=SRC_COLOR[src], marker=SRC_MARKER[src], linewidth=1.7,
                     label=f"{SRC_LABEL[src].split()[0]}-{SRC_LABEL[src].split()[1]} - {SRC_LABEL[rng_key].split()[0]}")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([HEAD_LABELS[h] for h in HEAD_ORDER])
    axes[1].set_ylabel(r"$\Delta$ AUC (quantum minus random)")
    axes[1].set_title("4-kernel: quantum-vs-classical-random AUC gap")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    gap_pdf = os.path.join(args.out_dir, "capacity_sweep_gap.pdf")
    fig.savefig(gap_pdf, bbox_inches="tight")
    fig.savefig(gap_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_pdf.replace('.pdf', '.png')}")
    print(f"Wrote {gap_pdf}")
    print(f"Wrote {gap_pdf.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
