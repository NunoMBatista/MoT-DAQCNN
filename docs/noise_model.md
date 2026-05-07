# Rydberg Noise Model

This document describes the physics-grounded noise model available in DAQCNN and explains how to configure it.

---

## Physical Motivation

The DAQCNN quantum kernels simulate Rydberg atom arrays.  Real hardware introduces three dominant error sources:

| Error source | Physical origin | Model |
|---|---|---|
| **T1 relaxation** | Spontaneous emission from the Rydberg state; typical lifetime 100–300 μs for n ≈ 60, 87Rb | `ThermalRelaxationError` |
| **T2 dephasing** | Laser phase noise and atomic motion during evolution | `ThermalRelaxationError` |
| **Gate error** | Finite pulse fidelity for single-qubit encoding gates (RY, H) | `DepolarizingChannel` |

---

## Default Parameter Values

The defaults are grounded in recent neutral-atom benchmarks:

| Parameter | Default | Source |
|---|---|---|
| `T1_us = 200` μs | Rydberg state (n ≈ 60, 87Rb) lifetime ≈ 100–300 μs | [1] |
| `T2_us = 100` μs | Conservative lab estimate for dephasing during Rydberg evolution | [2] |
| `p_gate_1q = 0.002` | Single-qubit gate fidelity 99.77 % → ε ≈ 0.002 | [3] |
| `omega_mhz = 1.0` | Typical Rabi frequency Ω = 2π × 1 MHz for optical-tweezer arrays | [2, 3] |

**References**

[1] Rydberg state lifetime: standard atomic physics result for principal quantum number n ≈ 60.

[2] Graham et al. (2022), *Nature* — "Multi-qubit entanglement and algorithms on a neutral-atom quantum computer" `arXiv:2112.14589`. T2 echo ≈ 0.28 s (ground state); T2 during Rydberg gates ≈ tens of μs.

[3] Evered et al. (2023), *Nature* — "High-fidelity parallel entangling gates on a neutral-atom quantum computer" `arXiv:2304.05420`. Single-qubit fidelity 99.77 %; CZ gate fidelity 99.5 %.

---

## Time-Unit Conversion

The DAQCNN Hamiltonian is written in dimensionless simulation units where the Rabi frequency sets the natural timescale:

```
1 simulation unit  =  τ  =  1 / (2π Ω)
                          ≈  160 ns   (at Ω = 2π × 1 MHz)
```

Physical T1 and T2 values (in μs) are converted to simulation units before being passed to the noise channel:

```
T1_sim  =  T1_us × 1000 ns/μs  /  τ_ns
T2_sim  =  T2_us × 1000 ns/μs  /  τ_ns
```

At the defaults (T1 = 200 μs, omega = 1 MHz):
- `T1_sim ≈ 1257 simulation units`
- `T2_sim ≈ 628 simulation units`
- Each Trotter step `dt = 0.05` sim-units ≈ 8 ns — much shorter than both T1 and T2.

Over a full evolution of `evolution_time = 2.5` (50 Trotter steps), the cumulative per-qubit error is small but measurable:
- T1 loss per qubit: `1 − exp(−2.5 / T1_sim) ≈ 0.2 %`
- T2 dephasing per qubit: `1 − exp(−2.5 / T2_sim) ≈ 0.4 %`

---

## Noise Channels

### 1. `ThermalRelaxationError` — Trotter steps

Applied **once per qubit per Trotter step** after each `ApproxTimeEvolution` block.  This is the standard quantum channel for combined T1/T2 noise, parameterised by:

```
ThermalRelaxationError(pe=0, t1=T1_sim, t2=T2_sim, tg=dt)
```

- `pe = 0`: cold ground-state preparation (thermal population negligible at μK).
- `t1`, `t2`, `tg` are all in simulation units, ensuring consistent scaling.

### 2. `DepolarizingChannel` — encoding gates (digital mode only)

Applied **once per qubit** after each `RY` and `Hadamard` gate:

```
DepolarizingChannel(p = p_gate_1q)
```

In analog mode there are no encoding gates, so this channel never fires.

---

## Configuration

Add a `noise:` section to any YAML config.  Omitting the section (or setting `enabled: false`) leaves all existing behaviour unchanged.

```yaml
noise:
  enabled: false      # master switch — flip to true to enable
  T1_us: 200.0        # Rydberg-state T1 relaxation time [μs]
  T2_us: 100.0        # qubit T2 dephasing time [μs]
  p_gate_1q: 0.002    # per-gate single-qubit depolarizing error
  omega_mhz: 1.0      # Rabi frequency Ω [MHz] — sets τ (ns/sim-unit)
```

All five fields are optional; any omitted field falls back to its literature default.

---

## Generating a Noisy Feature Cache

```bash
# 1. Enable noise in the cache-generation config
#    Set noise.enabled: true in configs/breast_mnist/cache_generation/analog_zz.yml

python experiments/create_quantum_dataset.py \
    --config configs/breast_mnist/cache_generation/analog_zz.yml
```

The output file will include the T1/T2 values in its name for instant identification:

```
breast_mnist__k3_s3_tkin-hor-..._gray_zz_analog_noisy-T1=200-T2=100.npz
```

The JSON sidecar stores all five noise parameters so the cache matcher can find and validate it unambiguously.

---

## Running a Noisy Experiment

```bash
# Enable noise in the experiment config (noise.enabled: true)
python experiments/robust_test_original_daqcnn.py \
    --config configs/breast_mnist/original_daqcnn_best.yml
```

If a matching noisy cache exists it is loaded automatically and `bypass_quantum = True` is set (no live simulation).  If no cache is found, the noise model runs live — which is significantly slower.

---

## Performance Caveat

Noise simulation requires density-matrix (mixed-state) simulation via `default.mixed`.  Memory scales as the **square** of the Hilbert-space dimension:

| kernel_size | qubits | DM size | Feasibility |
|---|---|---|---|
| 2 × 2 | 4 | 16 × 16 ≈ 2 kB | ✓ fast |
| 3 × 3 | 9 | 512 × 512 ≈ 2 MB | ✓ feasible (~100× slower than default.qubit) |
| 4 × 4 | 16 | 65 536 × 65 536 ≈ 34 GB | ✗ impractical |

**Noise simulation is only supported for `kernel_size ≤ 3`.**

For 4 × 4 kernels leave `noise.enabled: false` (or use noise only at the cache-generation stage with a smaller kernel).
