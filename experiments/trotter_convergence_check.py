"""R6 (camera-ready): how far are the delta_t = 0.05 Trotterized features from the
exact Hamiltonian evolution, and do they converge as delta_t shrinks?

For a few real test images (patches normalized and unfolded exactly as the
production layer does), kings topology, both encodings, Z and ZZ features:

  prod      production QuantumConv2d(mode="trotter", dt=0.05, default.qubit)
  hand_dt   the same evolve_analog_block Trotter path rebuilt here so dt can vary
            (dt = 0.05, 0.025; also 0.0125 at T = 2.5)
  exact     PennyLane's exact evolution, qml.evolve -> jax odeint (mode="exact")
  scipy     an independent exact reference: dense H(t) built with numpy kron,
            Schroedinger equation integrated with scipy DOP853 at 1e-11

Planted controls (each has a known answer):
  (a) hand_0.05 == prod            (instrument reproduces production, ~1e-12)
  (b) scipy == exact                (two independent exact solvers, ~1e-7)
  (c) exact at default tol == exact at tight tol (ODE converged, ~1e-8)
  (d) T -> 0 digital features == sin x_i and sin x_i sin x_j (encoding sanity)
  (e) err(dt) / err(dt/2) ~ 2      (first-order Trotter, expected order)

Usage:
  python experiments/trotter_convergence_check.py --enc digital [--topo kings --T 2.5]
  python experiments/trotter_convergence_check.py --enc analog  [--topo kings --T 2.5]
  python experiments/trotter_convergence_check.py --summary   # all per-arm CSVs -> table

Writes outputs/camera_ready/trotter_convergence_<enc>_<topo>_T<T>.csv (long format) and
outputs/camera_ready/trotter_convergence/<enc>__<image>__<topo>__T<T>.npz with
every feature array, so nothing has to be recomputed to look at distributions.
"""
import argparse, csv, os, sys, time
from itertools import combinations

import numpy as np
import torch
import pennylane as qml
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.physics.evolution as evo                       # noqa: E402
import src.physics.hamiltonian as phys                    # noqa: E402
import src.physics.kernel_topologies as topologies        # noqa: E402
from src.layers.quantum_convolution import QuantumConv2d  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "camera_ready")
NPZ_DIR = os.path.join(OUT_DIR, "trotter_convergence")

N_QUBITS = 9
WIRES = list(range(N_QUBITS))
ZZ_PAIRS = list(combinations(WIRES, 2))       # production order of the 36 ZZ features
TS = [0.2, 0.5, 1.0, 2.5, 4.0]                # the A9 sweep grid; 2.5 is the paper's operating point
DT_PROD = 0.05

# (label, npz file, split, index). Per-image min-max normalization means each image
# spans [0, pi] on its own, so three images give 243 patches over the full range.
IMAGES = [
    ("breast_test0", "breastmnist.npz", "test_images", 0),
    ("breast_test1", "breastmnist.npz", "test_images", 1),
    ("pneumonia_test0", "pneumoniamnist.npz", "test_images", 0),
]


# ----------------------------------------------------------------------------- patches
def load_patches(conv, npz_name, split, index):
    """Return the (81, 9) normalized patches of one image, built by the production
    layer's own _normalize_inputs and unfold (so ordering and scaling are identical)."""
    img = np.load(os.path.join(ROOT, "data", npz_name))[split][index] / 255.0
    x = torch.tensor(img, dtype=torch.float64)[None, None]          # (1, 1, 28, 28)
    xn = conv._normalize_inputs(x)
    patches = conv.unfold(xn).transpose(1, 2).reshape(-1, N_QUBITS)  # (81, 9)
    return patches


# ----------------------------------------------------------------------------- circuits
def production_features(conv, patches):
    """Production Trotter features, (81, 45), through the DAQKLayer forward."""
    with torch.no_grad():
        out = conv.quantum_kernel(patches)                          # (81, 45)
    return out.detach().cpu().numpy().astype(float)


def make_hand_qnode(coords, enc, T, mode, dt, device="default.qubit", **odeint_kwargs):
    """A QNode that mirrors DAQKLayer.make_circuit but exposes dt and the ODE tolerances.
    Called on ONE patch (a 1-D numpy array of 9 values in [0, pi])."""
    H = phys.get_rydberg_hamiltonian(WIRES, coords, scaling_factor=1.0,
                                     use_local_detuning=(enc == "analog"))
    dev = qml.device(device, wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method=None)
    def circuit(x):
        if enc == "digital":
            for i in range(N_QUBITS):
                qml.RY(x[i], wires=i)
                qml.Hadamard(wires=i)
            params = [1.0, 1.0, 1.0]                                # omega, delta, interaction
        else:
            params = [1.0, 1.0, 1.0] + [float(v) for v in x]        # + local detunings
        if mode == "trotter":
            evo.evolve_analog_block(H, [0.0, T], mode="trotter", dt=dt, params=params)
        else:
            qml.evolve(H)(params=np.asarray(params, dtype=float),
                          t=np.asarray([0.0, T], dtype=float), **odeint_kwargs)
        meas = [qml.expval(qml.PauliZ(i)) for i in WIRES]
        for i, j in ZZ_PAIRS:
            meas.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
        return meas

    return circuit


def hand_features(circuit, patches):
    """Run a hand-built QNode patch by patch, return (n_patches, 45)."""
    rows = []
    for p in patches:
        rows.append(np.asarray(circuit(np.asarray(p, dtype=float)), dtype=float))
    return np.stack(rows)


# ----------------------------------------------------------------------------- scipy reference
def kron_all(single_site_mats):
    """Kronecker product over wires 0..8, wire 0 leftmost (PennyLane's convention)."""
    out = np.array([[1.0 + 0j]])
    for m in single_site_mats:
        out = np.kron(out, m)
    return out


I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
ZDIAG = np.array([1.0, -1.0])            # diag of Pauli Z
NDIAG = np.array([0.0, 1.0])             # diag of n = (1 - Z)/2, Rydberg occupation


def site_diag(single_site_diags):
    """Diagonal of a kron product of diagonal single-site operators."""
    out = np.array([1.0])
    for d in single_site_diags:
        out = np.kron(out, d)
    return out


def build_static_pieces(coords):
    """Dense (512, 512) drive matrix sum_i X_i / 2 and the diagonal vectors of
    sum_i n_i, sum_ij V_ij n_i n_j, each n_i, each Z_i, each Z_i Z_j."""
    hx = np.zeros((2 ** N_QUBITS, 2 ** N_QUBITS), dtype=complex)
    n_i = []
    for i in WIRES:
        hx += 0.5 * kron_all([X if w == i else I2 for w in WIRES])
        n_i.append(site_diag([NDIAG if w == i else np.ones(2) for w in WIRES]))
    n_sum = np.sum(n_i, axis=0)
    v_diag = np.zeros(2 ** N_QUBITS)
    for i, j in ZZ_PAIRS:
        r = np.linalg.norm(np.asarray(coords[i], float) - np.asarray(coords[j], float))
        if r > 1e-6:                                   # same guard as get_rydberg_hamiltonian
            v_diag += (1.0 / r ** 6) * n_i[i] * n_i[j]
    z_i = [site_diag([ZDIAG if w == i else np.ones(2) for w in WIRES]) for i in WIRES]
    return hx, n_sum, v_diag, n_i, z_i


def scipy_features(pieces, enc, T, patches, rtol=1e-11, atol=1e-13):
    """Integrate i d/dt psi = H(t) psi with DOP853 for every patch; return (n, 45)."""
    hx, n_sum, v_diag, n_i, z_i = pieces
    rows = []
    for p in patches:
        p = np.asarray(p, dtype=float)
        if enc == "digital":
            # RY(x)|0> = (cos x/2, sin x/2), then Hadamard
            singles = []
            for x in p:
                c, s = np.cos(x / 2), np.sin(x / 2)
                singles.append(np.array([(c + s), (c - s)]) / np.sqrt(2))
            psi0 = kron_all([v.reshape(2, 1) for v in singles]).reshape(-1).astype(complex)
            static_diag = v_diag.copy()
        else:
            psi0 = np.zeros(2 ** N_QUBITS, dtype=complex)
            psi0[0] = 1.0                                        # |0...0>
            static_diag = v_diag + sum(x * n for x, n in zip(p, n_i))   # + x_i n_i

        def rhs(t, psi):
            # H(t) = t * sum X/2 - t * sum n + V (+ sum x_i n_i)
            return -1j * (t * (hx @ psi) + (static_diag - t * n_sum) * psi)

        sol = solve_ivp(rhs, (0.0, T), psi0, method="DOP853", rtol=rtol, atol=atol)
        psi = sol.y[:, -1]
        prob = np.abs(psi) ** 2
        feats = [float(prob @ z_i[i]) for i in WIRES]
        for i, j in ZZ_PAIRS:
            feats.append(float(prob @ (z_i[i] * z_i[j])))
        rows.append(np.array(feats))
    return np.stack(rows)


# ----------------------------------------------------------------------------- bookkeeping
def diff_stats(a, b):
    """Max-abs and RMS of a - b over the Z block, the ZZ block and all 45 features."""
    d = a - b
    out = {}
    for block, sl in (("Z", slice(0, 9)), ("ZZ", slice(9, 45)), ("all", slice(0, 45))):
        out[block] = (float(np.max(np.abs(d[:, sl]))), float(np.sqrt(np.mean(d[:, sl] ** 2))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", choices=["digital", "analog"])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--n-patches", type=int, default=None, help="smoke test: first n patches per image")
    ap.add_argument("--topo", default=None, help="run only this topology (fan-out)")
    ap.add_argument("--T", type=float, default=None, help="run only this evolution time (fan-out)")
    args = ap.parse_args()
    if args.summary:
        print_summary()
        return
    assert args.enc, "--enc digital|analog"
    enc = args.enc
    os.makedirs(NPZ_DIR, exist_ok=True)
    rows = []
    t_start = time.time()

    # kings at every T (the fully coupled 3x3, the paper's default and the hardest
    # case for a product formula); cross and star at T = 2.5 only, as a spread check.
    arms = [("kings", T) for T in TS] + [("cross", 2.5), ("star", 2.5)]
    if args.topo or args.T is not None:
        arms = [(t, T) for t, T in arms if (args.topo in (None, t)) and (args.T in (None, T))]
    for topo, T in arms:
        csv_path = os.path.join(OUT_DIR, f"trotter_convergence_{enc}_{topo}_T{T}.csv")
        rows = []
        coords = topologies.build_kernel_coordinate_sets(3, [topo])[0]
        conv = QuantumConv2d(kernel_size=3, stride=3, kernel_topology_names=[topo],
                             evolution_time=T, mode="trotter", include_correlators=True,
                             encoding_mode=enc, quantum_device="default.qubit")
        pieces = build_static_pieces(coords)
        dts = [0.05, 0.025] + ([0.0125] if (topo == "kings" and T == 2.5) else [])
        hand = {dt: make_hand_qnode(coords, enc, T, "trotter", dt) for dt in dts}
        exact = make_hand_qnode(coords, enc, T, "exact", None)
        exact_tight = make_hand_qnode(coords, enc, T, "exact", None, atol=1e-11, rtol=1e-11)

        for label, npz_name, split, index in IMAGES:
            patches = load_patches(conv, npz_name, split, index)
            if args.n_patches:
                patches = patches[: args.n_patches]
            feats = {"prod": production_features(conv, patches)}
            for dt in dts:
                feats[f"hand_{dt}"] = hand_features(hand[dt], patches)
            feats["exact"] = hand_features(exact, patches)
            feats["exact_tight"] = hand_features(exact_tight, patches[:10])   # control (c), subset
            feats["scipy"] = scipy_features(pieces, enc, T, patches)

            # comparisons: (name, a, b)
            comps = [
                ("ctrl_a_hand0.05_vs_prod", feats["hand_0.05"], feats["prod"]),
                ("ctrl_b_scipy_vs_exact", feats["scipy"], feats["exact"]),
                ("ctrl_c_exact_tight_vs_exact", feats["exact_tight"], feats["exact"][:10]),
                ("trotter0.05_vs_exact", feats["hand_0.05"], feats["exact"]),
                ("trotter0.025_vs_exact", feats["hand_0.025"], feats["exact"]),
                ("trotter0.05_vs_trotter0.025", feats["hand_0.05"], feats["hand_0.025"]),
            ]
            if 0.0125 in dts:
                comps.append(("trotter0.0125_vs_exact", feats["hand_0.0125"], feats["exact"]))
            if enc == "digital" and T == 0.2:
                # control (d) uses the T -> 0 limit; report the T=0.2 deviation from
                # sin x_i, sin x_i sin x_j here for the record (it is dynamics, not error).
                sx = np.sin(np.asarray(patches, float))
                sin_feats = np.concatenate([sx, np.stack([sx[:, i] * sx[:, j] for i, j in ZZ_PAIRS], 1)], 1)
                comps.append(("trotter0.05_vs_sinx", feats["hand_0.05"], sin_feats))
                comps.append(("exact_vs_sinx", feats["exact"], sin_feats))

            for name, a, b in comps:
                st = diff_stats(a, b)
                for block, (mx, rms) in st.items():
                    rows.append({"enc": enc, "topology": topo, "T": T, "image": label,
                                 "comparison": name, "block": block, "max_abs": mx,
                                 "rms": rms, "n_patches": len(a)})
                print(f"[{time.time()-t_start:7.1f}s] {enc} {topo} T={T} {label:16s} "
                      f"{name:32s} Z {st['Z'][0]:.2e}  ZZ {st['ZZ'][0]:.2e}", flush=True)
            np.savez(os.path.join(NPZ_DIR, f"{enc}__{label}__{topo}__T{T}.npz"),
                     patches=np.asarray(patches, float), **feats)

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {csv_path} ({len(rows)} rows) at {time.time()-t_start:.0f}s", flush=True)

    # control (d): T -> 0 digital limit equals sin x (0 Trotter steps at T=1e-6; the ODE
    # window is negligible). Run on the first image's patches only.
    if enc == "digital" and ("kings", 0.2) in arms:
        coords = topologies.build_kernel_coordinate_sets(3, ["kings"])[0]
        conv0 = QuantumConv2d(kernel_size=3, stride=3, kernel_topology_names=["kings"],
                              evolution_time=1e-6, mode="trotter", include_correlators=True,
                              encoding_mode="digital", quantum_device="default.qubit")
        patches = load_patches(conv0, *IMAGES[0][1:])
        sx = np.sin(np.asarray(patches, float))
        sin_feats = np.concatenate([sx, np.stack([sx[:, i] * sx[:, j] for i, j in ZZ_PAIRS], 1)], 1)
        ex0 = hand_features(make_hand_qnode(coords, "digital", 1e-6, "exact", None), patches)
        print(f"ctrl_d T=1e-6 digital: prod vs sin x max {np.max(np.abs(production_features(conv0, patches) - sin_feats)):.2e}; "
              f"exact vs sin x max {np.max(np.abs(ex0 - sin_feats)):.2e}")


def print_summary():
    """Pool the two CSVs: max over images of max_abs, per (enc, topology, T, comparison, block)."""
    import glob
    from collections import defaultdict
    pooled = defaultdict(lambda: [0.0, 0.0, 0])   # key -> [max_abs, sum of rms^2 * n, n]
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "trotter_convergence_*_T*.csv"))):
        for r in csv.DictReader(open(p)):
            k = (r["enc"], r["topology"], float(r["T"]), r["comparison"], r["block"])
            n = int(r["n_patches"])
            pooled[k][0] = max(pooled[k][0], float(r["max_abs"]))
            pooled[k][1] += float(r["rms"]) ** 2 * n
            pooled[k][2] += n
    print(f"{'enc':8s} {'topo':6s} {'T':>4s} {'comparison':32s} {'blk':4s} {'max_abs':>9s} {'rms':>9s} {'n':>4s}")
    for k in sorted(pooled):
        mx, s2, n = pooled[k]
        print(f"{k[0]:8s} {k[1]:6s} {k[2]:4.2f} {k[3]:32s} {k[4]:4s} {mx:9.2e} {np.sqrt(s2/n):9.2e} {n:4d}")


if __name__ == "__main__":
    main()
