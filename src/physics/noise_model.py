"""
Rydberg neutral-atom noise model for DAQCNN.

Builds a qml.NoiseModel that reflects hardware-realistic error budgets from
recent neutral-atom benchmarks (see docs/noise_model.md for derivation).

Three channels are applied:
  1. DepolarizingChannel after each RY  gate (digital-mode encoding).
  2. DepolarizingChannel after each Hadamard gate (digital-mode encoding).
  3. ThermalRelaxationError per qubit after every ApproxTimeEvolution block
     (one Trotter step), modelling T1 Rydberg decay + T2 dephasing jointly.
"""

import numpy as np
import pennylane as qml

# ---------------------------------------------------------------------------
# Literature-backed defaults
# Sources (see docs/noise_model.md for full citations):
#   T1/T2 : Rydberg state lifetime ~200 μs (n≈60, 87Rb); T2 conservative 100 μs
#   p_gate : Evered et al. 2023, Nature — single-qubit fidelity 99.77 %
#   omega  : typical Rabi frequency for optical-tweezer Rydberg experiments
# ---------------------------------------------------------------------------
RYDBERG_NOISE_DEFAULTS: dict = {
    "T1_us": 200.0,      # Rydberg-state T1 relaxation time [μs]
    "T2_us": 100.0,      # qubit T2 dephasing time [μs]  (conservative)
    "p_gate_1q": 0.002,  # single-qubit depolarizing error per gate
    "omega_mhz": 1.0,    # Rabi frequency Ω [MHz] — sets τ = 1/Ω as the sim time unit
}


def make_rydberg_noise_model(
    T1_us: float = 200.0,
    T2_us: float = 100.0,
    p_gate_1q: float = 0.002,
    omega_mhz: float = 1.0,
    trotter_dt: float = 0.05,
) -> tuple:
    """
    Build a qml.NoiseModel for a Rydberg neutral-atom device.

    Time-unit conversion
    --------------------
    One simulation unit equals one Rabi period τ = 1 / (2π Ω).
    At omega_mhz = 1.0:  τ ≈ 160 ns per sim-unit.
    Physical times T1_us / T2_us are converted to sim-units before being
    passed to ThermalRelaxationError so all three time arguments (t1, t2, tg)
    share the same unit.

    Args:
        T1_us: Rydberg-state T1 relaxation time in microseconds.
        T2_us: Qubit T2 dephasing time in microseconds.  Must satisfy T2 ≤ 2·T1.
        p_gate_1q: Per-gate depolarizing error for single-qubit encoding gates.
        omega_mhz: Rabi frequency Ω in MHz; determines τ (ns/sim-unit).
        trotter_dt: Duration of one Trotter step in simulation units (matches
                    evolution.py default of 0.05).

    Returns:
        noise_model: qml.NoiseModel ready for qml.noise.add_noise.
        meta: dict of derived rates useful for logging or sanity checks.
    """
    # --- Convert physical times to simulation units -------------------------
    tau_ns = 1000.0 / (2.0 * np.pi * omega_mhz)  # ns per simulation unit
    T1_sim = T1_us * 1000.0 / tau_ns
    T2_sim = T2_us * 1000.0 / tau_ns

    # Physics requires T2 ≤ 2·T1; clamp to avoid numerical issues in the channel
    T2_sim = min(T2_sim, 2.0 * T1_sim * (1.0 - 1e-9))

    # --- Noise function for Trotter steps -----------------------------------
    # Applied once after each ApproxTimeEvolution block (= one Trotter step).
    # ThermalRelaxationError(pe=0, t1, t2, tg) is the standard Lindblad channel
    # for combined T1 decay and T2 dephasing; pe=0 assumes cold-atom ground-state
    # preparation (thermal excitation negligible at μK temperatures).
    def _trotter_noise(op, **_kwargs):
        for wire in op.wires:
            qml.ThermalRelaxationError(0.0, T1_sim, T2_sim, trotter_dt, wires=wire)

    noise_model = qml.NoiseModel(
        {
            # Encoding-gate errors (only fire in digital mode where RY + H are used)
            qml.noise.op_eq(qml.RY):       qml.noise.partial_wires(qml.DepolarizingChannel, p_gate_1q),
            qml.noise.op_eq(qml.Hadamard): qml.noise.partial_wires(qml.DepolarizingChannel, p_gate_1q),
            # Trotter-step decoherence (fires for both digital and analog modes)
            qml.noise.op_eq(qml.ApproxTimeEvolution): _trotter_noise,
        }
    )

    meta = {
        "T1_us": T1_us,
        "T2_us": T2_us,
        "p_gate_1q": p_gate_1q,
        "omega_mhz": omega_mhz,
        "tau_ns_per_sim_unit": round(tau_ns, 3),
        "T1_sim_units": round(T1_sim, 1),
        "T2_sim_units": round(T2_sim, 1),
        "trotter_dt_sim_units": trotter_dt,
    }

    return noise_model, meta


def noise_model_from_cfg(cfg: dict) -> tuple:
    """
    Read the optional ``noise:`` block from a loaded YAML config dict and
    build a noise model if noise is enabled.

    Returns (None, {}) when ``noise.enabled`` is False or the section is absent,
    preserving the existing noiseless behaviour everywhere.
    """
    noise_cfg = cfg.get("noise", {})
    if not noise_cfg.get("enabled", False):
        return None, {}

    kwargs = {
        key: noise_cfg.get(key, RYDBERG_NOISE_DEFAULTS[key])
        for key in ("T1_us", "T2_us", "p_gate_1q", "omega_mhz")
    }
    # Allow config to override the Trotter step size for noise purposes
    kwargs["trotter_dt"] = noise_cfg.get("trotter_dt", 0.05)

    return make_rydberg_noise_model(**kwargs)
