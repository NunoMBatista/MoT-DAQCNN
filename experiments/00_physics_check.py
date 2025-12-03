import sys
import os
from pathlib import Path


# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Keep this before the rest of the imports
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pennylane as qml
import src.physics.hamiltonian as phys
import src.physics.evolution as evo


def main():
    print("--- MoT-DAQC Physics Engine Verification ---")
    
    # 1. Setup
    wires = [0, 1, 2, 3] # 4 qubits for 2x2 kernel
    grid_size = 2
    
    # Create Geometry (King's Graph) 
    coords = phys.create_kings_graph_geometry(grid_size)
    print(f"Geometry created: {len(coords)} atoms (2x2 grid)")
    
    
    # 2. Define Parameters from Paper 
    tau = 0.2 # total evolution time
    dt = 0.05 # trotter step size
    
    # NOTE: Tuning 'scaling_factor' is crucial. 
    # Too high = Trotter diverges. Too low = No entanglement.
    test_scaling = 1.0 
    
    # Get Hamiltonian
    H_sys = phys.get_rydberg_hamiltonian(wires, coords, scaling_factor=test_scaling)
    
    # Define Device
    dev = qml.device("default.qubit", wires=wires)
    
    
    # 3. Define QNodes for both modes
    @qml.qnode(dev)
    def circuit(mode, inputs):
        # Encoding Layer: H * Ry(phi)
        for i in range(4):
            qml.RY(inputs[i], wires=i)
            qml.Hadamard(wires=i)
            
        # Analog Evolution Layer
        evo.evolve_analog_block(H_sys, [0, tau], mode=mode, dt=dt)
        
        # Measurement: Expectation of Z
        return [qml.expval(qml.PauliZ(i)) for i in wires]


    # 4. Run Comparisons
    # Input: Random angles normalized to [0, pi] (like in the paper)
    dummy_input = np.array([0.5, 1.2, 2.5, 0.1])
    
    print(f"\nRunning Simulation (tau={tau}, dt={dt}, scale={test_scaling})...")
    
    res_trotter = np.array(circuit(mode="trotter", inputs=dummy_input))
    print(f"Trotter Mode (Paper): {res_trotter}")
    
    res_exact = np.array(circuit(mode="exact", inputs=dummy_input))
    print(f"Exact Mode (Ideal):   {res_exact}")
    
    # 5. Analysis
    diff = np.linalg.norm(res_trotter - res_exact)
    print(f"\nL2 Difference: {diff:.6f}")
    
    if diff > 1.0:
        print("WARNING: Divergence is very high. Reduce scaling_factor.")
    elif diff < 1e-9:
        print("WARNING: Results identical. Interaction might be effectively zero.")
    else:
        print("SUCCESS: Physics engine operational. Trotter error is present but stable.")


if __name__ == "__main__":
    main()