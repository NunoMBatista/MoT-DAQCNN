import pennylane as qml


def evolve_analog_block(hamiltonian, time_interval, mode="trotter", dt=0.05, params=None):
    """
    Applies the unitary evolution of the Hamiltonian to the active circuit.
    
    Args:
        hamiltonian (ParametrizedHamiltonian): The system H(t).
        time_interval (list): [t_start, t_end]. Paper uses [0, 0.2].
        mode (str): "exact" for ODE solver, "trotter" for discrete steps.
        dt (float): Step size for trotterization. Paper uses 0.05.
        params (list | float | int | None): Optional parameters passed to H(t).
    """
    if params is None:
        params = [1.0 for _ in hamiltonian.coeffs]
    elif isinstance(params, (int, float)):
        params = [float(params) for _ in hamiltonian.coeffs]
    
    t_start, t_end = time_interval
    
    if mode == "exact":
        # Use PennyLane's built-in ODE solver
        # This provides the ideal continuous physics.
        qml.evolve(hamiltonian)(params=params, t=time_interval)
        
    elif mode == "trotter":
        # Replicate the paper's discrete time steps 
        # We hold the Hamiltonian constant for each slice dt.
        
        total_time = t_end - t_start
        # Calculate number of steps (e.g. 0.2 / 0.05 = 4 steps)
        steps = int(round(total_time / dt))

        for step in range(steps):
            # Time at the beginning of the step
            t_current = t_start + step * dt
            
            # 1. Freeze the Hamiltonian at this time
            # H_static is now a concrete Hamiltonian operator, not a function
            H_static = hamiltonian(params=params, t=t_current)
            
            # 2. Evolve this static Hamiltonian for duration dt
            # e^{-i * H(t) * dt}
            qml.ApproxTimeEvolution(H_static, dt, n=1)

    else:
        raise ValueError(f"Unknown evolution mode: {mode}")