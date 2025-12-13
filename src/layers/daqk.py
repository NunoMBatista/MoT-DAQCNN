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
        quantum_device (str): Pennylane device name (e.g., "default.qubit", "lightning.gpu").
        quantum_device_kwargs (dict | None): Extra kwargs for the device (e.g., {"batch_size": B}).        
    """
    def __init__(
        self,
        n_qubits=4,
        grid_size=2,
        scaling_factor=1.0,
        mode="trotter",
        coordinates=None,
        kernel_topology_names=None,
        quantum_device="default.qubit",
        quantum_device_kwargs=None,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.mode = mode


        # Default to single kings kernel to preserve previous single-output behavior.
        names = ("kings",)
        if kernel_topology_names is not None:
            names = kernel_topology_names
        else:
            print("DAQKLayer: No kernel_topology_names provided, defaulting to ('kings',)")
        
        coord_sets = topologies.build_kernel_coordinate_sets(grid_size, names)

        self.num_kernels = len(coord_sets)

        self._build_kernels(coord_sets, scaling_factor, quantum_device, quantum_device_kwargs or {})


    def _build_kernels(self, coord_sets, scaling_factor, quantum_device, quantum_device_kwargs):
        """Create one Hamiltonian/QNode per topology and cache them."""
        self.hamiltonians = []
        self.devices = []
        self.kernel_circuits = []

        # Go through the coordinates of every kernel
        for coords in coord_sets:
            # build the Hamiltonian corresponding to each coordinate set
            hamiltonian = phys.get_rydberg_hamiltonian(
                self.wires,
                coords,
                scaling_factor=scaling_factor,
            )
            
            dev = qml.device(
                quantum_device, 
                wires=self.n_qubits, 
                **quantum_device_kwargs
            )

            def _circuit(inputs, H=hamiltonian):
                for i in range(self.n_qubits):
                    
                    # [..., i] to handle both single inputs (N) and batches (B, N)
                    qml.RY(inputs[..., i], wires=i)
                    #qml.RY(inputs[i], wires=i)
                    qml.Hadamard(wires=i)

                evo.evolve_analog_block(
                    H,
                    time_interval=[0, 0.2],
                    mode=self.mode,
                    dt=0.05,
                )

                return [qml.expval(qml.PauliZ(i)) for i in self.wires]

            qnode = qml.QNode(
                _circuit, 
                dev, 
                interface="torch"
            )

            self.hamiltonians.append(hamiltonian)
            self.devices.append(dev)
            self.kernel_circuits.append(qnode)


    def forward(self, x):
        """Forward pass over one or more kernels.

        Args:
            x: Tensor of shape (batch, n_qubits) with values in [0, pi].
        Returns:
            Tensor of shape (batch, num_kernels * n_qubits).
        """
        batch_size = x.shape[0] # This is every patch from a batch (batch * n_patches)
        #print(x.shape)
        per_kernel = [] # This will store the outputs from each kernel
        
        #print(len(self.kernel_circuits))

        # For each kernel topology
        # for kernel_circuit in self.kernel_circuits:
            
        #     outputs = []
        #     # run every patch from every batch
        #     for i in range(batch_size):
        #         # get the output from a single quantum patch
        #         quantum_output = kernel_circuit(x[i])
        #         #print(quantum_output)
                
        #         # make a tensor with the outputs from each atom in the patch
        #         out = torch.stack(quantum_output)
        #         #print(out)
        #         #exit(0)
        #         outputs.append(out)
                
        #         #print(len(outputs))
                
                
        #     per_kernel.append(torch.stack(outputs))  # (B, n_qubits)

        # For each kernel topology
        for kernel_circuit in self.kernel_circuits:
            # Input every patch from every batch
            circuit_output = kernel_circuit(x) 
            # circuit output outputs a list of length n_qubits with (B,) tensors
            out = torch.stack(circuit_output, dim=1)  # (B, n_qubits)
            per_kernel.append(out)

        return torch.cat(per_kernel, dim=1)
