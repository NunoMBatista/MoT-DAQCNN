"""Atom-array topology figure: the interaction graph each 3x3 geometry induces
over the 9 patch pixels. Nodes sit at the canonical 3x3 pixel positions; edges
are weighted by the real coupling V_ij = 1/r_ij^6 computed from each topology's
actual atom coordinates. Writes docs/paper/figures/atom_topologies_ext.pdf."""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.physics.kernel_topologies import get_3x3_kernel_set

ORDER = ["kings", "horizontal", "vertical", "cross",
         "ring", "chain", "star", "grid"]

# canonical pixel positions: pixel i at (col, -row) so row 0 is on top
NODE = np.array([[i % 3, -(i // 3)] for i in range(9)], dtype=float)

ks = get_3x3_kernel_set()
fig, axes = plt.subplots(2, 4, figsize=(8.0, 3.2))

for ax, name in zip(axes.flat, ORDER):
    coords = ks[name]
    # coupling for every pair from the real geometry
    pairs = []
    for i in range(9):
        for j in range(i + 1, 9):
            r = np.linalg.norm(coords[i] - coords[j])
            v = 1.0 / r**6 if r > 0 else 0.0
            if v > 0.02:
                pairs.append((i, j, v))
    vmax = max((v for *_, v in pairs), default=1.0)
    for i, j, v in pairs:
        w = 0.4 + 3.2 * (v / vmax)            # line width by strength
        a = 0.25 + 0.65 * (v / vmax)          # alpha by strength
        ax.plot([NODE[i, 0], NODE[j, 0]], [NODE[i, 1], NODE[j, 1]],
                "-", color="0.15", lw=w, alpha=a, zorder=1)
    ax.scatter(NODE[:, 0], NODE[:, 1], s=55, color="white",
               edgecolors="black", linewidths=1.0, zorder=2)
    ax.set_title(name, fontsize=9.5)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-2.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

fig.tight_layout()
out = "docs/paper/figures/atom_topologies_ext.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
