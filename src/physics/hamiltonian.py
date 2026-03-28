# from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.linalg as npl
import pennylane as qml


# Helper to extract scalar from various types of inputs (used in the ramp functions of the Hamiltonian)
def _scalar_param(p, default=1.0):
    # Handles None, python scalars, numpy scalars, jax.DeviceArray scalars, and sequences.
    if p is None:
        return float(default)
    if isinstance(p, (int, float)):
        return float(p)
    arr = np.asarray(p)
    if arr.shape == ():
        return float(arr)
    return float(arr.flat[0]) if arr.size > 0 else float(default)


def projector_eta(wire):
    """
    Returns the projector eta_j = (1 - sigma_z)/2.
    This represents the Rydberg state occupation number n_i.

    if |0> = ground state --> eta|0> = 0
    if |1> = Rydberg state --> eta|1> = 1
    """

    # (I - Z)/2. In PennyLane, Hamiltonian arithmetic handles the identity.
    return 0.5 * (qml.Identity(wire) - qml.PauliZ(wire))


def get_rydberg_hamiltonian(wires, coordinates, scaling_factor=1.0, use_local_detuning=False):
    """
    Constructs the time-dependent Hamiltonian H(t) from the DAQCNN paper.

    H(t) = (Omega(t)/2)*Sum(X) - Delta(t)*Sum(eta) + Sum(J_ij * eta_i * eta_j)
           [+ Sum(delta_i * eta_i) if use_local_detuning is True]

    Args:
        wires: The qubit indices.
        coordinates: List of [x, y] positions for distance calculation.
        scaling_factor: The 'C_6' parameter in J_ij = C_6/r_ij^6.
        use_local_detuning: If True, adds N extra terms to the Hamiltonian,
            each representing the local detuning shift for a qubit, to be
            driven by the input data during evolution.
    """
    n_qubits = len(wires)

    # --- 1. Static Operator Definitions ---
    # Term A: Global Drive (Omega/2 * Sum Sigma_X)
    ops_drive = [qml.PauliX(w) for w in wires]
    H_drive_base = qml.Hamiltonian([0.5] * n_qubits, ops_drive)

    # Term B: Global Detuning (-Delta * Sum eta)
    ops_detuning = [projector_eta(w) for w in wires]
    H_detuning_base = qml.Hamiltonian([-1.0] * n_qubits, ops_detuning)

    # Term C: Interaction (Sum J_ij * eta_i * eta_j)
    coeffs_int = []
    ops_int = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            delta = np.asarray(coordinates[i], dtype=float) - np.asarray(
                coordinates[j], dtype=float
            )
            r_ij = float(npl.norm(delta))
            if r_ij > 1e-6:
                J_ij = scaling_factor / (r_ij**6)
                coeffs_int.append(J_ij)
                ops_int.append(projector_eta(wires[i]) @ projector_eta(wires[j]))

    H_interaction = qml.Hamiltonian(coeffs_int, ops_int)

    # --- 2. Time-Dependent Functions ---
    def drive_ramp(params, t):
        # params[0] is global omega scale
        omega_scale = _scalar_param(params, default=1.0)
        return omega_scale * t

    def detuning_ramp(params, t):
        # params[1] is global delta scale
        delta_scale = _scalar_param(params, default=1.0)
        return delta_scale * t

    def constant_one(params, t):
        return 1.0

    # --- 3. Local Detuning Operators (Data-Driven) ---
    coeffs = [drive_ramp, detuning_ramp]
    ops = [H_drive_base, H_detuning_base]

    if len(coeffs_int) > 0:
        ops.append(H_interaction)
        coeffs.append(constant_one)

    if use_local_detuning:
        # Add a separate term for each qubit's local detuning.
        # These will be driven by the 'inputs' passed as params to the evolution.
        # In make_circuit, we'll pass [global_omega, global_delta, *inputs, 1.0].
        def make_local_coeff(idx):
            def local_coeff(params, t):
                # We expect params to be [omega, delta, p0, p1, ..., pN, const]
                # Index 2 is the start of the data pixels
                return params[2 + idx]
            return local_coeff

        for i in range(n_qubits):
            ops.append(projector_eta(wires[i]))
            coeffs.append(make_local_coeff(i))

    # Combine into a single ParametrizedHamiltonian
    return qml.pulse.ParametrizedHamiltonian(coeffs, ops)
