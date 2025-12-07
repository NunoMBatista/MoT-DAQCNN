import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# Import defined physics
import src.physics.hamiltonian as phys
import src.physics.evolution as evo
import src.physics.kernel_topologies as topologies

class DAQKLayer(nn.Module):
    """
    Digital-Analog Quantum Kernel Layer.

    Supports one or more kernel topologies. Outputs are concatenated along the feature
    dimension: shape (batch, num_kernels * n_qubits).
    
    Attributes:
        n_qubits (int): Number of qubits (must match kernel topology size).
        grid_size (int): Size of the kernel grid (e.g., 2 for 2x2).
        scaling_factor (float): Scaling factor for the Hamiltonian terms.
        mode (str): 'trotter' (default) or 'exact'.
        coordinates (list | None): Explicit coordinates for a single kernel.
        kernel_topology_names (list | None): Names of topologies to use; ignored if coordinates provided.        
    """
    def __init__(
        self,
        n_qubits=4,
        grid_size=2,
        scaling_factor=1.0,
        mode="trotter",
        coordinates=None,
        kernel_topology_names=None,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.mode = mode

        # Resolve coordinate sets: explicit coordinates take priority, else use named kernels.
        #if coordinates is not None:
        #    coord_sets = [coordinates]
        #else:
         
        # Default to single kings kernel to preserve previous single-output behavior.
        names = ("kings",) if kernel_topology_names is None else kernel_topology_names
        coord_sets = topologies.build_kernel_coordinate_sets(grid_size, names)

        self.num_kernels = len(coord_sets)

        self._build_kernels(coord_sets, scaling_factor)


    def _build_kernels(self, coord_sets, scaling_factor):
        """Create one Hamiltonian/QNode per topology and cache them."""
        self.hamiltonians = []
        self.devices = []
        self.qnodes = []

        for coords in coord_sets:
            hamiltonian = phys.get_rydberg_hamiltonian(
                self.wires,
                coords,
                scaling_factor=scaling_factor,
            )
            dev = qml.device("default.qubit", wires=self.n_qubits)

            def _circuit(inputs, H=hamiltonian):
                for i in range(self.n_qubits):
                    qml.RY(inputs[i], wires=i)
                    qml.Hadamard(wires=i)

                evo.evolve_analog_block(
                    H,
                    time_interval=[0, 0.2],
                    mode=self.mode,
                    dt=0.05,
                )

                return [qml.expval(qml.PauliZ(i)) for i in self.wires]

            qnode = qml.QNode(_circuit, dev, interface="torch")

            self.hamiltonians.append(hamiltonian)
            self.devices.append(dev)
            self.qnodes.append(qnode)


    def forward(self, x):
        """Forward pass over one or more kernels.

        Args:
            x: Tensor of shape (batch, n_qubits) with values in [0, pi].
        Returns:
            Tensor of shape (batch, num_kernels * n_qubits).
        """
        batch_size = x.shape[0]
        per_kernel = [] 

        for qnode in self.qnodes:
            outputs = []
            for i in range(batch_size):
                out = torch.stack(qnode(x[i]))
                outputs.append(out)
            per_kernel.append(torch.stack(outputs))  # (B, n_qubits)

        return torch.cat(per_kernel, dim=1)
