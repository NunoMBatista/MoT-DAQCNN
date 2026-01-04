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


def get_rydberg_hamiltonian(wires, coordinates, scaling_factor=1.0):
    """
    Constructs the time-dependent Hamiltonian H(t) from Eq. (1) in the
    "Digital-analog quantum convolutional neural networks for image classification" paper.

    H(t) = (Omega(t)/2)*Sum(X) - Delta(t)*Sum(eta) + Sum(J_ij * eta_i * eta_j)

    Args:
        wires: The qubit indices.
        coordinates: List of [x, y] positions for distance calculation.
        scaling_factor: The 'C_6' parameter in J_ij = C_6/r_ij^6.
                        Used to tune interaction strength.
    """
    n_qubits = len(wires)

    # To define an Hamiltonian in pennylane, we need to provide:
    # 1. Static operator definitions (ops list)
    # 2. Time-dependent coefficient functions (coeffs list)
    # And then combine into a ParametrizedHamiltonian

    # --- 1. Static Operator Definitions ---

    # Term A: Global Drive (Omega/2 * Sum Sigma_X)
    # We set coeff to 0.5 because the formula is Omega(t)/2
    ops_drive = [qml.PauliX(w) for w in wires]  # These are the Sigma_X operators
    ops_drive = cast(list[qml.operation.Operator], ops_drive)
    H_drive_base = qml.Hamiltonian(
        [0.5] * n_qubits, ops_drive
    )  # Combine the Sigma_X operators with the coefficients (without taking time into consideration) # type: ignore[arg-type]

    # Term B: Global Detuning (-Delta * Sum eta)
    # Coeff is -1.0 because formula is -Delta(t)
    ops_detuning = [
        projector_eta(w) for w in wires
    ]  # These are the eta operators in the detuning term (0 if ground, 1 if Rydberg)
    H_detuning_base = qml.Hamiltonian(
        [-1.0] * n_qubits, ops_detuning
    )  # Combine them in the Hamiltonian (without taking time into consideration) # type: ignore[arg-type]

    # Term C: Interaction (Sum J_ij * eta_i * eta_j)
    # J_ij = scaling_factor / r_ij^6
    coeffs_int = []  # This will store the interaction coeffs (J_ij)
    ops_int = []  # This will store the interaction operators (eta_i * eta_j)

    # For each pair of qubits, calculate J_ij and the corresponding operator
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # Calculate diference between atom position
            delta = np.asarray(coordinates[i], dtype=float) - np.asarray(
                coordinates[j], dtype=float
            )
            # The Euclidean distance r_ij (norm of the delta vector)
            r_ij = float(npl.norm(delta))

            # Avoid division by zero
            if r_ij > 1e-6:
                J_ij = scaling_factor / (r_ij**6)  # Interaction strength
                coeffs_int.append(J_ij)
                # The operator is the tensor product eta_i * eta_j
                ops_int.append(projector_eta(wires[i]) @ projector_eta(wires[j]))

    H_interaction = qml.Hamiltonian(coeffs_int, ops_int)  # noqa: F841

    # --- 2. Time-Dependent Functions ---
    # The paper states that in each timestep t: Omega(t) = t, Delta(t) = t

    # The time dependent stuff needs to be functions so that the
    # ParametrizedHamiltonian can call them with (params, t) at any timestep t
    # but this can be tuned with the parameters

    def drive_ramp(params, t):
        omega_scale = _scalar_param(params, default=1.0)
        return omega_scale * t  # Omega(t)

    def detuning_ramp(params, t):
        delta_scale = _scalar_param(params, default=1.0)
        return delta_scale * t  # Delta(t)

    # --- 3. Combine into ParametrizedHamiltonian ---
    # The coeffs list correspond to the coefficients of the operators in the ops list
    coeffs = [drive_ramp, detuning_ramp]
    ops = [H_drive_base, H_detuning_base]

    # if len(coeffs_int) > 0:
    #    def constant_one(params, t):
    #        return 1.0
    #
    #    ops.append(H_interaction)
    #    coeffs.append(constant_one)

    # Combine all terms into a single ParametrizedHamiltonian
    H_total = qml.pulse.ParametrizedHamiltonian(coeffs, ops)
    # default_params = [0.0 for _ in coeffs]
    # cast(Any, H_total).default_params = default_params
    return H_total
