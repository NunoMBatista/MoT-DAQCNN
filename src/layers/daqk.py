from itertools import combinations

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

    jax.config.update("jax_enable_x64", True)
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
                              Overridden to "default.mixed" when noise_model is provided.
        quantum_device_kwargs (dict | None): Extra kwargs for the device (e.g., {"batch_size": B}).
        noise_model (qml.NoiseModel | None): Optional noise model from src.physics.noise_model.
                                             When set, forces device to "default.mixed" and wraps
                                             each QNode with qml.noise.add_noise.
                                             Only feasible for kernel_size <= 3 (9 qubits).
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
        include_correlators=False,
        encoding_mode="digital",
        noise_model=None,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.mode = mode
        self.evolution_time = evolution_time
        self.interface = interface
        self.use_jit = use_jit
        self.include_correlators = include_correlators
        self.encoding_mode = encoding_mode  # "digital" or "analog"

        # Number of measurements per kernel: n_qubits Z + C(n_qubits, 2) ZZ
        n_zz = len(list(combinations(self.wires, 2))) if include_correlators else 0
        self.measurements_per_kernel = n_qubits + n_zz

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
            include_correlators,
            encoding_mode,
            noise_model,
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
        include_correlators,
        encoding_mode,
        noise_model=None,
    ):
        """Create one Hamiltonian/QNode per topology and cache them."""
        self.hamiltonians = []
        self.devices = []
        self.kernel_circuits = []

        def make_circuit(H, n_qubits, wires, mode, evo_time, iface, correlators, enc_mode):
            def _circuit(inputs):
                if enc_mode == "digital":
                    # Traditional DAQCNN: encode into state with RY gates
                    for i in range(n_qubits):
                        qml.RY(inputs[..., i], wires=i)
                        qml.Hadamard(wires=i)
                    
                    # Evolve with fixed Hamiltonian (default params 1.0)
                    evo.evolve_analog_block(
                        H,
                        time_interval=[0, evo_time],
                        mode=mode,
                        dt=0.05,
                        interface=iface,
                    )
                else:
                    # Analog encoding: skip initial gates, start in |0>
                    # Encode pixels directly into the Hamiltonian's local detuning.
                    # Build param list: [omega_scale, delta_scale, interaction_const, p0, p1, ..., pN]
                    if torch.is_tensor(inputs):
                        p_list = [1.0, 1.0, 1.0] + inputs.tolist()
                    else:
                        p_list = [1.0, 1.0, 1.0] + list(inputs)

                    evo.evolve_analog_block(
                        H,
                        time_interval=[0, evo_time],
                        mode=mode,
                        dt=0.05,
                        params=p_list,
                        interface=iface,
                    )

                measurements = [qml.expval(qml.PauliZ(i)) for i in wires]
                if correlators:
                    for i, j in combinations(wires, 2):
                        measurements.append(
                            qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                        )
                return measurements

            return _circuit

        # Go through the coordinates of every kernel
        for coords in coord_sets:
            # build the Hamiltonian corresponding to each coordinate set
            # If analog mode, we tell the Hamiltonian to add local detuning terms
            hamiltonian = phys.get_rydberg_hamiltonian(
                self.wires,
                coords,
                scaling_factor=scaling_factor,
                use_local_detuning=(encoding_mode == "analog")
            )

            # When a noise model is provided we need density-matrix simulation;
            # default.mixed is the only PennyLane device that supports it.
            # Note: default.mixed is ~100× slower and only feasible for ≤ 9 qubits
            # (kernel_size ≤ 3).  For 4×4 kernels (16 qubits) use noise_model=None.
            effective_device = "default.mixed" if noise_model is not None else quantum_device
            dev = qml.device(effective_device, wires=self.n_qubits, **quantum_device_kwargs)

            circuit = make_circuit(
                hamiltonian,
                self.n_qubits,
                self.wires,
                self.mode,
                evolution_time,
                interface,
                include_correlators,
                encoding_mode,
            )

            qnode = qml.QNode(
                circuit,
                dev,
                interface=interface,
                diff_method=None,
            )

            # Wrap with noise channels after each matching gate type
            if noise_model is not None:
                qnode = qml.noise.add_noise(qnode, noise_model)

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
            Tensor of shape (batch, num_kernels * measurements_per_kernel).
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
                    if self.encoding_mode == "analog" and x.ndim > 1:
                        # Manual batching for analog mode because ApproxTimeEvolution 
                        # doesn't support batched Hamiltonian coefficients well.
                        batch_out = []
                        for i in range(x.shape[0]):
                            res = kernel_circuit(x[i])
                            # res is a list of 0D tensors. Stack them into a 1D tensor
                            res_tensor = torch.stack(res, dim=0)
                            batch_out.append(res_tensor)
                        out = torch.stack(batch_out, dim=0)
                    else:
                        circuit_output = kernel_circuit(x)
                        # In digital mode with batching, PennyLane returns a list of 1D tensors (B,)
                        # We stack them along dim=1 to get (B, num_measurements)
                        out = torch.stack(circuit_output, dim=1)
                per_kernel.append(out)

        return torch.cat(per_kernel, dim=1)
