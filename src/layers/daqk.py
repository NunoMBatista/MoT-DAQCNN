import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# Import your physics engine
import src.physics.hamiltonian as phys
import src.physics.evolution as evo


class DAQKLayer(nn.Module):
    """
    Digital-Analog Quantum Kernel Layer.
    Wraps the Rydberg physics engine into a PyTorch module.

    Args:
        n_qubits (int): Number of qubits (4 for 2x2, 9 for 3x3).
        grid_size (int): 2 or 3.
        scaling_factor (float): The 's' parameter for interaction strength.
        mode (str): 'trotter' (default) or 'exact'.
    """
    def __init__(self, n_qubits=4, grid_size=2, scaling_factor=1.0, mode="trotter"):
        super().__init__()

        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.mode = mode

        # 1. Pre-compute Geometry and Hamiltonian
        # We do this once to save overhead.
        self.coords = phys.create_geometry(grid_size)

        # Note: Hamiltonian is time-dependent, but the structure is static.
        self.hamiltonian = phys.get_rydberg_hamiltonian(
            self.wires,
            self.coords,
            scaling_factor=scaling_factor,
        )

        # 2. Define the Quantum Device
        # 'default.qubit' is standard for simulation.
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # 3. Create the QNode
        # interface='torch' ensures gradients can flow (even if we don't train it yet)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")


    def _circuit(self, inputs):
        """
        The actual quantum circuit execution.
        inputs: Tensor of shape (n_qubits,) containing pixel angles.
        """
        # --- A. Encoding Layer  ---
        # The paper specifies H * Ry(phi).
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.Hadamard(wires=i)

        # --- B. Analog Evolution ---
        # Evolve for tau=0.2 using the specified mode (trotter/exact)
        evo.evolve_analog_block(
            self.hamiltonian,
            time_interval=[0, 0.2],
            mode=self.mode,
            dt=0.05,
        )

        # --- C. Measurement  ---
        # Return expectation value of Sigma_Z for every qubit
        return [qml.expval(qml.PauliZ(i)) for i in self.wires]


    def forward(self, x):
        """
        PyTorch Forward Pass.

        Args:
            x (Tensor): Input batch of image patches. Shape (Batch, n_qubits).
                        Values should be normalized to [0, pi].

        Returns:
            Tensor: Output features. Shape (Batch, n_qubits).
        """
        # PennyLane's torch interface usually handles batching automatically
        # if the first dimension is batch_size. However, explicit looping
        # or vmap is safer for custom Hamiltonians.

        # For simplicity/safety in simulation:
        batch_size = x.shape[0]
        outputs = []

        # Loop over batch (parallelize this later if slow)
        for i in range(batch_size):
            # The QNode returns a list of tensors/floats
            # Stack them into a single tensor
            out = torch.stack(self.qnode(x[i]))
            outputs.append(out)

        return torch.stack(outputs)
