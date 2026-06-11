"""Generate a compact TikZ figure of the 8 atom-array topologies.

Each topology is one small panel: the 9 patch pixels as nodes on a 3x3 grid,
with edges weighted by the real Rydberg coupling V_ij = 1/r_ij^6 from each
topology's atom coordinates. Monochrome (black edges, white nodes), tightly
packed 4x2 grid. Writes a standalone docs/paper/figures/atom_topologies.tex
that compiles to atom_topologies.pdf for \\includegraphics.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.physics.kernel_topologies import get_3x3_kernel_set

ORDER = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
NCOL = 4                      # panels per row -> 4x2 layout
PX, PY = 1.75, 1.95          # panel pitch (cm) horizontal / vertical
G = 0.5                       # node grid spacing within a panel (cm)
THRESH = 0.008                # min ABSOLUTE coupling V_ij to draw an edge
LOGMIN, LOGMAX = -2.0, 1.0    # log10(V) span mapped from ghost line to strong bond


def edges(coords):
    """Return [(i, j, V)] for pairs whose ABSOLUTE coupling V=1/r^6 clears the
    threshold. Width/opacity later scale with absolute V (not a per-panel max),
    which is what separates kings from grid: both are the full 3x3 lattice, but
    grid's atoms sit 1.5x apart so every coupling is ~11x weaker. Its orthogonal
    bonds render fainter than kings' and its diagonals (V~0.011) become
    barely-visible ghost lines, present but tiny, rather than dropping out or
    matching kings' diagonals (which a per-panel normalization would force)."""
    out = []
    for i in range(9):
        for j in range(i + 1, 9):
            r = np.linalg.norm(coords[i] - coords[j])
            v = 1.0 / r**6 if r > 0 else 0.0
            if v > THRESH:
                out.append((i, j, v))
    return out


def edge_style(v):
    """Line width (pt) and opacity from ABSOLUTE coupling on a log scale:
    V=1 (a unit-distance bond) reads strong, V~0.01 (grid diagonal) a ghost."""
    t = (np.log10(v) - LOGMIN) / (LOGMAX - LOGMIN)
    t = min(1.0, max(0.0, t))
    return round(0.2 + 1.4 * t, 2), round(0.22 + 0.73 * t, 2)


ks = get_3x3_kernel_set()
lines = [
    r"\documentclass[tikz,border=3pt]{standalone}",
    r"\usepackage{tikz}",
    r"\begin{document}",
    r"\begin{tikzpicture}[x=1cm,y=1cm]",
    r"  \tikzset{atomdot/.style={circle,draw=black,fill=white,line width=0.4pt,"
    r"inner sep=0pt,minimum size=4.2pt}}",
]

for k, name in enumerate(ORDER):
    r, c = divmod(k, NCOL)
    ox, oy = c * PX, -r * PY          # panel origin (top-left node)
    # node absolute position: pixel p at grid (col, row)
    def pos(p):
        return ox + (p % 3) * G, oy - (p // 3) * G

    lines.append(f"  % --- {name} ---")
    for i, j, w in edges(ks[name]):
        lw, op = edge_style(w)
        xi, yi = pos(i); xj, yj = pos(j)
        lines.append(f"  \\draw[black,line width={lw}pt,opacity={op}] "
                     f"({xi:.3f},{yi:.3f}) -- ({xj:.3f},{yj:.3f});")
    for p in range(9):
        x, y = pos(p)
        lines.append(f"  \\node[atomdot] at ({x:.3f},{y:.3f}) {{}};")
    # topology name centered under the panel
    lx = ox + G
    ly = oy - 2 * G - 0.34
    lines.append(f"  \\node[font=\\footnotesize] at ({lx:.3f},{ly:.3f}) {{{name}}};")

lines += [r"\end{tikzpicture}", r"\end{document}", ""]

out = "docs/paper/figures/atom_topologies.tex"
with open(out, "w") as f:
    f.write("\n".join(lines))
print("wrote", out)
