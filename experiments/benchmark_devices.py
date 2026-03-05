"""
benchmark_devices.py

Benchmarks all viable device/interface combinations for quantum dataset generation
across 2x2, 3x3, and 4x4 kernel sizes.

Each combination is timed over a small but representative number of patches
(mimicking what create_quantum_dataset.py actually does), then results are
printed in a ranked table so you can pick the fastest setup.

Combinations tested
-------------------
  default.qubit  + torch interface   (baseline)
  default.qubit  + autograd interface
  lightning.qubit + torch interface
  lightning.qubit + autograd interface
  lightning.gpu  + autograd interface
  JAX (default.qubit jit=True)       — GPU-backed JAX
  JAX (default.qubit jit=False)      — GPU-backed JAX, no JIT

Usage:
    python experiments/benchmark_devices.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pennylane as qml

import src.physics.evolution as evo
from src.physics.hamiltonian import get_rydberg_hamiltonian
from src.physics.kernel_topologies import build_kernel_coordinate_sets

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# How many patches to time per combination.
# Kept small so the whole script finishes in reasonable time, but large enough
# to be representative (startup / JIT compile costs amortised).
N_WARMUP = 2  # patches run before timing starts (JIT warm-up, device init)
N_TIMED = 8  # patches actually timed

KERNEL_SIZES = [2, 3, 4]

# Use only the first topology per kernel size to keep runtime manageable.
# The cost per topology scales the same way, so one is representative.
TOPOLOGIES_PER_SIZE = {
    2: ["kings"],
    3: ["kings"],
    4: ["kings"],
}

SCALING_FACTOR = 1.0
EVOLUTION_TIME = 0.2  # same default as the rest of the codebase
MODE = "trotter"
DT = 0.05

# ---------------------------------------------------------------------------
# Candidate combinations
# ---------------------------------------------------------------------------

CANDIDATES = [
    # (label,               device_name,      interface,  use_jit)
    ("default.qubit  | torch", "default.qubit", "torch", False),
    ("default.qubit  | autograd", "default.qubit", "autograd", False),
    ("lightning.qubit| torch", "lightning.qubit", "torch", False),
    ("lightning.qubit| autograd", "lightning.qubit", "autograd", False),
    ("lightning.gpu  | autograd", "lightning.gpu", "autograd", False),
    ("JAX gpu        | no-jit", "default.qubit", "jax", False),
    ("JAX gpu        | jit", "default.qubit", "jax", True),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_qnode(device_name, n_qubits, hamiltonian, interface, use_jit):
    """Build and return a QNode for the given device/interface."""
    dev = qml.device(device_name, wires=n_qubits)
    wires = list(range(n_qubits))

    def circuit(inputs):
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
            qml.Hadamard(wires=i)
        evo.evolve_analog_block(
            hamiltonian,
            time_interval=[0, EVOLUTION_TIME],
            mode=MODE,
            dt=DT,
            interface=interface,
        )
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=None)

    if interface == "jax" and use_jit:
        try:
            import jax as jax_module

            qnode = jax_module.jit(qnode)
        except Exception as e:
            print(f"    [warn] JAX JIT wrapping failed: {e}")

    return qnode


def make_patch_batch(n_qubits, batch_size, interface):
    """Return a random batch of patches in the right format for the interface."""
    raw = np.random.uniform(0, np.pi, size=(batch_size, n_qubits)).astype(np.float64)
    if interface == "torch":
        return torch.tensor(raw, dtype=torch.float64)
    elif interface == "jax":
        import jax.numpy as jnp

        return jnp.array(raw)
    else:  # autograd / numpy
        from pennylane import numpy as pnp

        return pnp.array(raw, requires_grad=False)


def run_qnode(qnode, patch, interface, mode):
    """Execute the qnode and return a numpy array of results."""
    if interface == "jax":
        import jax

        out = qnode(patch)
        # block until GPU kernel finishes so timing is accurate
        jax.block_until_ready(out)
        return np.array(out)
    elif interface == "torch":
        if mode == "exact":
            out = qnode(patch.numpy())
        else:
            out = qnode(patch)
            if isinstance(out, list):
                out = torch.stack(out, dim=1)
        return out.detach().numpy() if isinstance(out, torch.Tensor) else np.array(out)
    else:  # autograd
        out = qnode(patch)
        return np.array(out)


def benchmark_combination(label, device_name, interface, use_jit, kernel_size):
    """
    Time one (device, interface, kernel_size) combination.

    Returns elapsed seconds per patch, or None if the combination failed.
    """
    n_qubits = kernel_size * kernel_size
    topo_names = TOPOLOGIES_PER_SIZE[kernel_size]
    coord_sets = build_kernel_coordinate_sets(kernel_size, topo_names)
    coords = coord_sets[0]  # single topology

    # --- skip combos that are known to be incompatible ---
    if device_name == "lightning.gpu" and interface == "torch":
        return None, "lightning.gpu does not support torch interface"
    if device_name == "lightning.qubit" and interface == "jax":
        return None, "lightning.qubit does not support jax interface"

    try:
        hamiltonian = get_rydberg_hamiltonian(
            list(range(n_qubits)), coords, scaling_factor=SCALING_FACTOR
        )
        qnode = make_qnode(device_name, n_qubits, hamiltonian, interface, use_jit)
    except Exception as e:
        return None, f"setup error: {e}"

    # warm-up
    try:
        for _ in range(N_WARMUP):
            patch = make_patch_batch(n_qubits, 1, interface)
            run_qnode(qnode, patch, interface, MODE)
    except Exception as e:
        return None, f"warmup error: {e}"

    # timed run — one patch at a time (mirrors create_quantum_dataset BATCH_SIZE=1)
    try:
        t0 = time.perf_counter()
        for _ in range(N_TIMED):
            patch = make_patch_batch(n_qubits, 1, interface)
            run_qnode(qnode, patch, interface, MODE)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return None, f"timing error: {e}"

    return elapsed / N_TIMED, None  # seconds per patch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("  QUANTUM DEVICE BENCHMARK")
    print("=" * 72)
    print(f"  Warmup patches : {N_WARMUP}")
    print(f"  Timed patches  : {N_TIMED}")
    print(f"  Kernel sizes   : {KERNEL_SIZES}")
    print(f"  Mode           : {MODE}  (dt={DT}, T={EVOLUTION_TIME})")
    print()

    # results[kernel_size] = list of (label, sec_per_patch, note)
    results = {ks: [] for ks in KERNEL_SIZES}

    for kernel_size in KERNEL_SIZES:
        n_qubits = kernel_size * kernel_size
        hilbert_dim = 2**n_qubits
        print(f"{'─' * 72}")
        print(
            f"  Kernel {kernel_size}x{kernel_size}  |  {n_qubits} qubits  |  Hilbert dim = {hilbert_dim:,}"
        )
        print(f"{'─' * 72}")

        for label, device_name, interface, use_jit in CANDIDATES:
            print(f"  Testing  {label:<30}", end="", flush=True)
            sec, note = benchmark_combination(
                label, device_name, interface, use_jit, kernel_size
            )
            if sec is None:
                results[kernel_size].append((label, None, note))
                print(f"  SKIP  ({note})")
            else:
                results[kernel_size].append((label, sec, None))
                print(f"  {sec * 1000:8.1f} ms/patch")

        print()

    # -----------------------------------------------------------------------
    # Summary tables — one per kernel size, ranked fastest→slowest
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("  SUMMARY  (ranked fastest → slowest, skipped entries excluded)")
    print("=" * 72)

    for kernel_size in KERNEL_SIZES:
        valid = [(l, s) for (l, s, n) in results[kernel_size] if s is not None]
        valid.sort(key=lambda x: x[1])

        if not valid:
            print(f"\n  {kernel_size}x{kernel_size} kernel: no valid results")
            continue

        fastest_sec = valid[0][1]

        print(
            f"\n  Kernel {kernel_size}x{kernel_size}  ({kernel_size * kernel_size} qubits, Hilbert dim = {2 ** (kernel_size * kernel_size):,})"
        )
        print(
            f"  {'Rank':<5} {'Combination':<32} {'ms/patch':>10}  {'×slower':>8}  {'Projected time (breast_mnist train)':>36}"
        )
        print(f"  {'─' * 5} {'─' * 32} {'─' * 10}  {'─' * 8}  {'─' * 36}")

        # breast_mnist train has 546 images; 28x28 with kernel_size stride=kernel_size
        n_images = 546
        patches_per_image = ((28 - kernel_size) // kernel_size + 1) ** 2

        for rank, (lbl, sec) in enumerate(valid, 1):
            total_patches = n_images * patches_per_image
            projected_min = (sec * total_patches) / 60.0
            slower = sec / fastest_sec
            print(
                f"  {rank:<5} {lbl:<32} {sec * 1000:>10.1f}  {slower:>8.1f}x  {projected_min:>34.1f} min"
            )

        skipped = [(l, n) for (l, s, n) in results[kernel_size] if s is None]
        if skipped:
            print(f"\n  Skipped:")
            for lbl, note in skipped:
                print(f"    - {lbl}: {note}")

    print()
    print("=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)

    # Find the fastest overall for 4x4 (the hardest / most important case)
    valid_4x4 = [(l, s) for (l, s, n) in results[4] if s is not None]
    if valid_4x4:
        valid_4x4.sort(key=lambda x: x[1])
        best_label, best_sec = valid_4x4[0]
        print(f"\n  For 4x4 kernels, the fastest combination is:")
        print(f"    >> {best_label}  ({best_sec * 1000:.1f} ms/patch)")
        print()
        print(f"  Use this in create_quantum_dataset.py by setting:")

        # Map label back to device/interface
        for label, device_name, interface, use_jit in CANDIDATES:
            if label == best_label:
                print(f'    quantum_device = "{device_name}"')
                print(f'    interface      = "{interface}"')
                if use_jit:
                    print(f"    use_jit        = True")
                break
    else:
        print("\n  Could not determine a recommendation (no valid 4x4 results).")

    print()


if __name__ == "__main__":
    main()
