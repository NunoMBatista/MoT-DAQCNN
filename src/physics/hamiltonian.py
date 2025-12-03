from typing import Any, cast

import numpy as np
import numpy.linalg as npl

import pennylane as qml

def projector_eta(wire):
    """
    Returns the projector eta_j = (1 - sigma_z)/2.
    This represents the Rydberg state occupation number n_i.
    
    if |0> = ground state --> eta|0> = 0
    if |1> = Rydberg state --> eta|1> = 1
    """
    
    # (I - Z)/2. In PennyLane, Hamiltonian arithmetic handles the identity.
    return 0.5 * (qml.Identity(wire) - qml.PauliZ(wire))


def create_kings_graph_geometry(grid_size=2):
    """
    Creates coordinates for a square grid (King's Graph topology ).
    For a 2x2 grid, this returns 4 coordinates.
    """
    
    coords = []
    for x in range(grid_size):
        for y in range(grid_size):
            coords.append([x, y])
    return np.array(coords, dtype=float)


def get_rydberg_hamiltonian(wires, coordinates, scaling_factor=1.0):
    """
    Constructs the time-dependent Hamiltonian H(t) from Eq. (1) in the 
    "Digital-analog quantum convolutional neural networks for image classification" paper.
    
    H(t) = (Omega(t)/2)*Sum(X) - Delta(t)*Sum(eta) + Sum(J_ij * eta_i * eta_j)
    
    Parameters:
        wires: The qubit indices.
        coordinates: List of [x, y] positions for distance calculation.
        scaling_factor: The 's' parameter in J_ij = s*tau/pi.
                        Used to tune interaction strength.
    """
    n_qubits = len(wires)
    
    # --- 1. Static Operator Definitions ---
    
    # Term A: Global Drive (Omega/2 * Sum Sigma_X)
    # We set coeff to 0.5 because the formula is Omega(t)/2
    ops_drive = [qml.PauliX(w) for w in wires]
    H_drive_base = qml.Hamiltonian([0.5] * n_qubits, ops_drive)  # type: ignore[arg-type]

    # Term B: Global Detuning (-Delta * Sum eta)
    # Coeff is -1.0 because formula is -Delta(t)
    ops_detuning = [projector_eta(w) for w in wires]
    H_detuning_base = qml.Hamiltonian([-1.0] * n_qubits, ops_detuning)  # type: ignore[arg-type]

    # Term C: Interaction (Sum J_ij * eta_i * eta_j)
    # J_ij = scaling_factor / r_ij^6
    coeffs_int = []
    ops_int = []
    
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # Calculate Euclidean distance r_ij
            delta = np.asarray(coordinates[i], dtype=float) - np.asarray(
                coordinates[j], dtype=float
            )
            r_ij = float(npl.norm(delta))
            
            # Avoid division by zero
            if r_ij > 1e-6:
                J_ij = scaling_factor / (r_ij**6)
                coeffs_int.append(J_ij)
                ops_int.append(projector_eta(wires[i]) @ projector_eta(wires[j]))
    
    if len(coeffs_int) > 0:
        H_interaction = qml.Hamiltonian(coeffs_int, ops_int)
    else:
        # Fallback for 1x1 case (no interaction)
        H_interaction = qml.Hamiltonian([], [])


    # --- 2. Time-Dependent Functions ---
    # As per: Omega(t) = t, Delta(t) = t
    
    def drive_ramp(params, t):
        return t  # Omega(t)

    def detuning_ramp(params, t):
        return t  # Delta(t)


    # --- 3. Combine into ParametrizedHamiltonian ---
    coeffs = [drive_ramp, detuning_ramp]
    ops = [H_drive_base, H_detuning_base]

    if len(coeffs_int) > 0:
        def constant_one(params, t):
            return 1.0

        ops.append(H_interaction)
        coeffs.append(constant_one)

    H_total = qml.pulse.ParametrizedHamiltonian(coeffs, ops)
    default_params = [0.0 for _ in coeffs]
    cast(Any, H_total).default_params = default_params
    return H_total