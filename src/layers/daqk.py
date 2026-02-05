import pennylane as qml
import torch
import torch.nn as nn
from pennylane import numpy as np

import src.physics.evolution as evo

# Import defined physics
import src.physics.hamiltonian as phys
import src.physics.kernel_topologies as topologies

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


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
        evolution_time=0.2,
        mode="trotter",
        coordinates=None,
        kernel_topology_names=None,
        quantum_device="default.qubit",
        quantum_device_kwargs=None,
        interface="torch",
        use_jit=False,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.mode = mode
        self.evolution_time = evolution_time
        self.interface = interface
        self.use_jit = use_jit

        # Default to single kings kernel to preserve previous single-output behavior.
        names = ("kings",)
        if kernel_topology_names is not None:
            names = kernel_topology_names
        else:
            print(
                "DAQKLayer: No kernel_topology_names provided, defaulting to ('kings',)"
            )

        # Store the kernel topology names for metadata access
        self.kernel_topology_names = names

        coord_sets = topologies.build_kernel_coordinate_sets(grid_size, names)

        self.num_kernels = len(coord_sets)

        self._build_kernels(
            coord_sets,
            scaling_factor,
            evolution_time,
            quantum_device,
            quantum_device_kwargs or {},
            interface,
            use_jit,
        )

    def _build_kernels(
        self,
        coord_sets,
        scaling_factor,
        evolution_time,
        quantum_device,
        quantum_device_kwargs,
        interface,
        use_jit,
    ):
        """Create one Hamiltonian/QNode per topology and cache them."""
        self.hamiltonians = []
        self.devices = []
        self.kernel_circuits = []

        def make_circuit(H, n_qubits, wires, mode, evo_time, iface):
            def _circuit(inputs):
                for i in range(n_qubits):
                    qml.RY(inputs[..., i], wires=i)
                    qml.Hadamard(wires=i)

                evo.evolve_analog_block(
                    H,
                    time_interval=[0, evo_time],
                    mode=mode,
                    dt=0.05,
                    interface=iface,
                )

                return [qml.expval(qml.PauliZ(i)) for i in wires]

            return _circuit

        # Go through the coordinates of every kernel
        for coords in coord_sets:
            # build the Hamiltonian corresponding to each coordinate set
            hamiltonian = phys.get_rydberg_hamiltonian(
                self.wires,
                coords,
                scaling_factor=scaling_factor,
            )

            dev = qml.device(
                quantum_device, wires=self.n_qubits, **quantum_device_kwargs
            )

            circuit = make_circuit(
                hamiltonian,
                self.n_qubits,
                self.wires,
                self.mode,
                evolution_time,
                interface,
            )

            qnode = qml.QNode(
                circuit,
                dev,
                interface=interface,
                diff_method=None,
            )

            if interface == "jax" and JAX_AVAILABLE and use_jit:
                import jax as jax_module

                qnode = jax_module.jit(qnode)

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
        per_kernel = []

        for kernel_circuit in self.kernel_circuits:
            if self.interface == "jax" and JAX_AVAILABLE:
                import jax.numpy as jnp_module

                x_jax = jnp_module.array(x.detach().cpu().numpy())
                circuit_output = kernel_circuit(x_jax)
                circuit_output = np.array(circuit_output)
                out = torch.as_tensor(circuit_output, device=x.device, dtype=x.dtype)
                per_kernel.append(out)
            elif self.interface == "autograd":
                circuit_output = kernel_circuit(x.detach().cpu().numpy())
                circuit_output = np.array(circuit_output)
                out = torch.as_tensor(circuit_output, device=x.device, dtype=x.dtype)
                per_kernel.append(out)
            else:  # torch
                if self.mode == "exact":
                    circuit_output = kernel_circuit(x.detach().cpu().numpy())
                    circuit_output = np.array(circuit_output)
                    out = torch.as_tensor(
                        circuit_output, device=x.device, dtype=x.dtype
                    )
                else:
                    circuit_output = kernel_circuit(x)
                    out = torch.stack(circuit_output, dim=1)
                per_kernel.append(out)

        return torch.cat(per_kernel, dim=1)
