from typing import Dict, List

import numpy as np

# Constant to effectively break interactions (1/r^6 with r=FAR is ~0)
FAR = 100.0


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

    return kernels


def build_kernel_coordinate_sets(grid_size: int, names=None) -> List[np.ndarray]:
    """Return manual kernel coordinate layouts for a grid size."""
    if grid_size == 2:
        kernel_map = get_2x2_kernel_set()
    elif grid_size == 3:
        kernel_map = get_3x3_kernel_set()
    else:
        kernel_map = {"kings": create_kings_graph_geometry(grid_size)}

    if names is None:
        names = list(kernel_map.keys())

    kernel_set = []
    for n in names:
        if n in kernel_map:
            kernel_set.append(kernel_map[n])
        else:
            kernel_set.append(kernel_map.get("kings"))

    return kernel_set