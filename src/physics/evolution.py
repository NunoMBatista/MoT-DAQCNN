import pennylane as qml


def _default_params(hamiltonian):
    if hasattr(hamiltonian, "default_params"):
        return list(getattr(hamiltonian, "default_params"))
    coeffs = getattr(hamiltonian, "coeffs", None)
    if coeffs is not None:
        return [0.0 for _ in coeffs]
    return []


def evolve_analog_block(hamiltonian, time_interval, mode="trotter", dt=0.05):
    """
    Applies the unitary evolution of the Hamiltonian to the active circuit.
    
    Args:
        hamiltonian (ParametrizedHamiltonian): The system H(t).
        time_interval (list): [t_start, t_end]. Paper uses [0, 0.2].
        mode (str): "exact" for ODE solver, "trotter" for discrete steps.
        dt (float): Step size. Paper uses 0.05.
    """
    t_start, t_end = time_interval
    
    if mode == "exact":
        params = _default_params(hamiltonian)
        # Use PennyLane's built-in ODE solver (Dormand-Prince 5 usually)
        # This provides the ideal continuous physics.
        qml.evolve(hamiltonian)(params=params, t=time_interval)
        
    elif mode == "trotter":
        # Replicate the paper's discrete time steps 
        # We hold the Hamiltonian constant for each slice dt.
        
        total_time = t_end - t_start
        # Calculate number of steps (e.g. 0.2 / 0.05 = 4 steps)
        steps = int(round(total_time / dt))

        params = _default_params(hamiltonian)
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