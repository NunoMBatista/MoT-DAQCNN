"""Visualize atom positions and interaction edges for a kernel topology.

This script plots the coordinates produced by ``create_kings_graph_geometry``
and connects every pair of atoms with a nonzero interaction strength
J_ij = scaling_factor / r_ij^6. Use it to verify kernels/topologies quickly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make local src importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.physics.kernel_topologies as topologies


def compute_edges(physical_coords: np.ndarray, scaling: float) -> list[tuple[int, int, float]]:
    """Return list of (i, j, J_ij) for all pairs with nonzero coupling using physical coords."""
    n = len(physical_coords)
    edges: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r_ij = np.linalg.norm(physical_coords[i] - physical_coords[j])
            if r_ij <= 1e-6:
                continue
            J_ij = scaling / (r_ij**6)
            if J_ij != 0.0:
                edges.append((i, j, J_ij)) # type: ignore
    return edges


def canonical_layout(n_qubits: int) -> np.ndarray:
    """Return a compact display layout on an N x N grid for plotting only."""
    grid_size = int(round(np.sqrt(n_qubits)))
    coords = []
    for r in range(grid_size):
        for c in range(grid_size):
            coords.append([r, c])
    return np.array(coords[:n_qubits], dtype=float)


def plot_topology(display_coords: np.ndarray, edges: list[tuple[int, int, float]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(display_coords[:, 0], display_coords[:, 1], c="tab:blue", s=120, zorder=3)
    for idx, (x, y) in enumerate(display_coords):
        ax.text(x, y + 0.08, str(idx), ha="center", va="bottom", fontsize=10)

    if edges:
        max_J = max(w for *_, w in edges)
        for i, j, J in edges:
            x0, y0 = display_coords[i]
            x1, y1 = display_coords[j]
            width = 1.5 * (J / max_J) if max_J > 0 else 1.0
            ax.plot([x0, x1], [y0, y1], color="tab:orange", linewidth=width, alpha=0.8)
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(xm, ym, f"{J:.2e}", fontsize=8, color="tab:orange", ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (display)")
    ax.set_ylabel("y (display)")
    grid_size = int(round(np.sqrt(len(display_coords))))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    kernel_size = 3
    scaling = 1

    names = ["kings", "vertical"]
    coord_sets = topologies.build_kernel_coordinate_sets(kernel_size, names)

    for name, physical in zip(names, coord_sets):
        edges = compute_edges(physical, scaling)
        display = canonical_layout(len(physical))
        title = f"{name} (grid={kernel_size}, scaling={scaling})"
        plot_topology(display, edges, title)
   


if __name__ == "__main__":
    main()
