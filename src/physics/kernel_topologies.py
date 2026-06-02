from typing import Dict, List

import numpy as np

# Constant to effectively break interactions (1/r^6 with r=FAR is ~0)
FAR = 100.0

# Spacing constants for anisotropic and 4x4 topologies.
# TIGHT (1 unit)  → J = C6/1^6 = 1.0        — strong entanglement
# LOOSE (5 units) → J = C6/5^6 ≈ 6.4e-5     — negligible cross-talk
# DEAD  (10 units)→ J = C6/10^6 = 1e-6       — physically zero
TIGHT = 1.0
LOOSE = 5.0
DEAD = 10.0


def create_kings_graph_geometry(grid_size=2):
    """Creates coordinates for a square grid (King's Graph topology)."""
    coords = []
    for x in range(grid_size):
        for y in range(grid_size):
            coords.append([x, y])

    return np.array(coords, dtype=float)


def get_2x2_kernel_set() -> Dict[str, np.ndarray]:
    """Manual topologies for a 2x2 grid (4 qubits)."""
    kernels: Dict[str, np.ndarray] = {}

    # King's graph (full)
    kernels["kings"] = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float
    )

    # Horizontal pairs: top row close, bottom row far
    kernels["horizontal"] = np.array(
        [[0.0, 0.0], [0.0, 1.0], [FAR, 0.0], [FAR, 1.0]], dtype=float
    )

    # Vertical pairs: left column close, right column far
    kernels["vertical"] = np.array(
        [[0.0, 0.0], [0.0, FAR], [1.0, 0.0], [1.0, FAR]], dtype=float
    )

    # U-shape path 0-1-3-2 with weakened 0-2 vertical link
    kernels["u_shape"] = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.5, 0.0], [1.5, 1.0]], dtype=float
    )

    return kernels


def get_3x3_kernel_set() -> Dict[str, np.ndarray]:
    """Manual topologies for a 3x3 grid (9 qubits)."""
    kernels: Dict[str, np.ndarray] = {}

    base = []
    for r in range(3):
        for c in range(3):
            base.append([float(r), float(c)])
    base = np.array(base, dtype=float)

    # King's graph (full)
    kernels["kings"] = base.copy()

    # Horizontal lines (rows independent)
    coords_horz = base.copy()
    coords_horz[:, 0] *= FAR
    kernels["horizontal"] = coords_horz

    # Vertical lines (cols independent)
    coords_vert = base.copy()
    coords_vert[:, 1] *= FAR
    kernels["vertical"] = coords_vert

    # Cross (plus sign), corners isolated
    coords_cross = base.copy()
    for idx in [0, 2, 6, 8]:
        coords_cross[idx] = [FAR, FAR]
    kernels["cross"] = coords_cross

    # Ring (hollow center)
    coords_ring = base.copy()
    coords_ring[4] = [FAR, FAR]
    kernels["ring"] = coords_ring

    # Linear chain - qubits arranged in a line 0→1→2→3→4→5→6→7→8
    # Each qubit only couples to its immediate neighbors
    coords_chain = np.zeros((9, 2), dtype=float)
    for i in range(9):
        coords_chain[i] = [0.0, float(i)]  # All on same row, spaced by 1 unit
    kernels["chain"] = coords_chain

    # Star topology - center qubit (4) at origin, all others at distance 1
    # Only hub-spoke connections (no connections between spokes)
    coords_star = np.zeros((9, 2), dtype=float)
    coords_star[4] = [0.0, 0.0]  # Center at origin
    # Place other qubits in a circle around center at distance 1
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    spoke_idx = [0, 1, 2, 3, 5, 6, 7, 8]  # All except center
    for i, idx in enumerate(spoke_idx):
        coords_star[idx] = [np.cos(angles[i]), np.sin(angles[i])]
    kernels["star"] = coords_star

    # 2D Grid (nearest neighbors only - weakened diagonals)
    # Scale up uniformly to increase diagonal distance relative to nearest neighbors
    # At r=1.5: nearest J=1/1.5^6≈0.088 (still decent coupling)
    # At r=√2*1.5≈2.12: diagonal J=1/2.12^6≈0.011 (negligible ~12% of nearest)
    # This creates 12 strong nearest-neighbor pairs + 16 weak diagonal pairs
    coords_grid = base.copy()
    coords_grid *= 1.5
    kernels["grid"] = coords_grid

    # Synthetic placeholder topologies for the classical-nonlinear feature caches
    # (poly-2 / RFF). These are never simulated: the cache is always present so
    # bypass_quantum=True and the quantum layer is skipped. They exist only so the
    # DAQCNN model constructs without error when its config names the cache's
    # synthetic topology. Geometry is irrelevant (a copy of the kings layout).
    for synthetic in ("poly2", "rff45", "rff180"):
        kernels[synthetic] = base.copy()

    return kernels


def get_4x4_kernel_set() -> Dict[str, np.ndarray]:
    """Manual topologies for a 4x4 grid (16 qubits).

    Pixel index → array index mapping (row-major, standard 4x4):
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15

    All five topologies are designed to be structurally orthogonal so that
    the MoT routing gate has genuine per-sample signal to discriminate on.
    Spacing constants: TIGHT=1 (J=1.0), LOOSE=5 (J≈6e-5), DEAD=10 (J≈1e-6).

    Topologies
    ----------
    kings
        Full 4x4 grid at unit spacing.  All nearest- and diagonal-neighbour
        pairs strongly coupled (J=1.0 at r=1, J=0.125 at r=√2).  Captures
        global texture — every pixel influences every other pixel through the
        entanglement network.  58 strong pairs.

    horizontal_chains
        Atoms within each row are TIGHT (x-spacing = TIGHT).
        Rows are separated by LOOSE (y-spacing = LOOSE) so vertical cross-talk
        is negligible (J≈6e-5).  Four independent 4-qubit horizontal chains.
        Sensitive to within-row gradients and horizontal edges.  20 strong pairs.

    vertical_chains
        Transpose of horizontal_chains.  Four independent 4-qubit vertical
        chains.  Columns TIGHT, rows LOOSE.  Sensitive to vertical edges and
        column-wise gradients.  20 strong pairs.

    diagonal_chains
        Two independent diagonal chains of 4 qubits each at r=√2 spacing
        (J=0.125), one along the main diagonal (q0→q5→q10→q15) and one along
        the anti-diagonal (q3→q6→q9→q12).  All remaining 8 atoms are displaced
        to DEAD positions.  Captures diagonal structure that horizontal and
        vertical kernels are blind to.  6 strong pairs.

    block_2x2
        The 4x4 grid is partitioned into four non-overlapping 2x2 quadrants
        (top-left, top-right, bottom-left, bottom-right).  Within each quadrant
        atoms are at unit spacing (J=1.0 at r=1, J=0.125 at r=√2).  Between
        quadrants atoms are separated by DEAD (J≈1e-6).  Each quadrant acts as
        an independent entangled cluster, measuring local 2x2 corner statistics.
        Complementary to kings (local vs global) and to the chain kernels
        (isotropic local vs directional).  24 strong pairs, 0 weak, 96 dead.
    """
    kernels: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 1. Kings — full 4x4 grid at unit spacing
    # ------------------------------------------------------------------
    kings = []
    for r in range(4):
        for c in range(4):
            kings.append([float(r), float(c)])
    kernels["kings"] = np.array(kings, dtype=float)

    # ------------------------------------------------------------------
    # 2. Horizontal chains
    # Pixels 0-3  → row 0, y = 3*LOOSE
    # Pixels 4-7  → row 1, y = 2*LOOSE
    # Pixels 8-11 → row 2, y = 1*LOOSE
    # Pixels 12-15→ row 3, y = 0
    # ------------------------------------------------------------------
    kernels["horizontal_chains"] = np.array(
        [
            [0 * TIGHT, 3 * LOOSE],
            [1 * TIGHT, 3 * LOOSE],
            [2 * TIGHT, 3 * LOOSE],
            [3 * TIGHT, 3 * LOOSE],  # row 0
            [0 * TIGHT, 2 * LOOSE],
            [1 * TIGHT, 2 * LOOSE],
            [2 * TIGHT, 2 * LOOSE],
            [3 * TIGHT, 2 * LOOSE],  # row 1
            [0 * TIGHT, 1 * LOOSE],
            [1 * TIGHT, 1 * LOOSE],
            [2 * TIGHT, 1 * LOOSE],
            [3 * TIGHT, 1 * LOOSE],  # row 2
            [0 * TIGHT, 0 * LOOSE],
            [1 * TIGHT, 0 * LOOSE],
            [2 * TIGHT, 0 * LOOSE],
            [3 * TIGHT, 0 * LOOSE],  # row 3
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 3. Vertical chains
    # Pixels 0,4,8,12  → col 0, x = 0
    # Pixels 1,5,9,13  → col 1, x = 1*LOOSE
    # Pixels 2,6,10,14 → col 2, x = 2*LOOSE
    # Pixels 3,7,11,15 → col 3, x = 3*LOOSE
    # Within each column y-spacing = TIGHT
    # ------------------------------------------------------------------
    kernels["vertical_chains"] = np.array(
        [
            [0 * LOOSE, 3 * TIGHT],
            [1 * LOOSE, 3 * TIGHT],
            [2 * LOOSE, 3 * TIGHT],
            [3 * LOOSE, 3 * TIGHT],  # row 0
            [0 * LOOSE, 2 * TIGHT],
            [1 * LOOSE, 2 * TIGHT],
            [2 * LOOSE, 2 * TIGHT],
            [3 * LOOSE, 2 * TIGHT],  # row 1
            [0 * LOOSE, 1 * TIGHT],
            [1 * LOOSE, 1 * TIGHT],
            [2 * LOOSE, 1 * TIGHT],
            [3 * LOOSE, 1 * TIGHT],  # row 2
            [0 * LOOSE, 0 * TIGHT],
            [1 * LOOSE, 0 * TIGHT],
            [2 * LOOSE, 0 * TIGHT],
            [3 * LOOSE, 0 * TIGHT],  # row 3
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 4. Diagonal chains
    # Main diagonal  : q0→q5→q10→q15  at physical coords with r=√2·TIGHT
    # Anti-diagonal  : q3→q6→q9→q12   offset far from main diag (x += 3*DEAD)
    # All other atoms: displaced to isolated positions so J < 1e-5
    # ------------------------------------------------------------------
    # We need 16 coordinate slots. Build as a list then convert.
    diag_coords: list[list[float]] = [[] for _ in range(16)]

    # Main diagonal atoms — spaced at (TIGHT, TIGHT) steps so r = √2·TIGHT
    diag_coords[0] = [0 * TIGHT, 0 * TIGHT]
    diag_coords[5] = [1 * TIGHT, 1 * TIGHT]
    diag_coords[10] = [2 * TIGHT, 2 * TIGHT]
    diag_coords[15] = [3 * TIGHT, 3 * TIGHT]

    # Anti-diagonal atoms — same local geometry, translated far away in x
    # so main-diag and anti-diag atoms never interact
    _ad_ox = 3 * DEAD  # x-offset for anti-diagonal cluster
    diag_coords[3] = [_ad_ox + 3 * TIGHT, 0 * TIGHT]
    diag_coords[6] = [_ad_ox + 2 * TIGHT, 1 * TIGHT]
    diag_coords[9] = [_ad_ox + 1 * TIGHT, 2 * TIGHT]
    diag_coords[12] = [_ad_ox + 0 * TIGHT, 3 * TIGHT]

    # Off-diagonal atoms — each placed in its own isolated DEAD-spaced slot
    _iso_x = 10 * DEAD
    for _slot, _idx in enumerate([1, 2, 4, 7, 8, 11, 13, 14]):
        diag_coords[_idx] = [_iso_x + _slot * DEAD, 0.0]

    kernels["diagonal_chains"] = np.array(diag_coords, dtype=float)

    # ------------------------------------------------------------------
    # 5. Block 2x2
    # Four non-overlapping 2x2 quadrants, each internally at unit spacing.
    # Quadrants are separated by DEAD so inter-quadrant J ≈ 1e-6.
    #
    #  top-left  : q0, q1, q4, q5   at origin
    #  top-right : q2, q3, q6, q7   at x-offset = DEAD
    #  bot-left  : q8, q9, q12, q13 at y-offset = DEAD
    #  bot-right : q10, q11, q14, q15 at (DEAD, DEAD)
    # ------------------------------------------------------------------
    def _quad(ox: float, oy: float) -> list:
        """Return (x,y) coords for a 2x2 block with bottom-left at (ox, oy)."""
        return [
            [ox, oy + TIGHT],  # top-left of quad
            [ox + TIGHT, oy + TIGHT],  # top-right of quad
            [ox, oy],  # bot-left of quad
            [ox + TIGHT, oy],  # bot-right of quad
        ]

    tl = _quad(0, 0)  # top-left  quadrant
    tr = _quad(DEAD, 0)  # top-right quadrant
    bl = _quad(0, DEAD)  # bot-left  quadrant
    br = _quad(DEAD, DEAD)  # bot-right quadrant

    block_coords: list[list[float]] = [[] for _ in range(16)]
    for _i, _idx in enumerate([0, 1, 4, 5]):
        block_coords[_idx] = tl[_i]
    for _i, _idx in enumerate([2, 3, 6, 7]):
        block_coords[_idx] = tr[_i]
    for _i, _idx in enumerate([8, 9, 12, 13]):
        block_coords[_idx] = bl[_i]
    for _i, _idx in enumerate([10, 11, 14, 15]):
        block_coords[_idx] = br[_i]

    kernels["block_2x2"] = np.array(block_coords, dtype=float)

    return kernels


def build_kernel_coordinate_sets(grid_size: int, names=None) -> List[np.ndarray]:
    """Return manual kernel coordinate layouts for a grid size."""
    if grid_size == 2:
        kernel_map = get_2x2_kernel_set()
    elif grid_size == 3:
        kernel_map = get_3x3_kernel_set()
    elif grid_size == 4:
        kernel_map = get_4x4_kernel_set()
    else:
        kernel_map = {"kings": create_kings_graph_geometry(grid_size)}

    if names is None:
        names = list(kernel_map.keys())

    kernel_set = []
    for n in names:
        if n in kernel_map:
            kernel_set.append(kernel_map[n])
        else:
            raise ValueError(
                f"Kernel '{n}' not found for grid_size={grid_size}. "
                f"Available kernels: {list(kernel_map.keys())}"
            )

    return kernel_set
