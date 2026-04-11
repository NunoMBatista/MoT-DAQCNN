# ZZ-AQCNN: Comprehensive Hyperparameter Search (BreastMNIST)

## Recommended Execution Order
1. **Digital (1-Kern)**: Establishing the gate-based baseline.
2. **Digital (4-Kern)**: Replicating your previous setup with ZZ.
3. **Digital (8-Kern)**: Testing maximum gate-based diversity.
4. **Analog (1-Kern)**: First test of the grounded-physics approach.
5. **Analog (4-Kern)**: Fair comparison vs Digital (4-Kern).
6. **Analog (8-Kern)**: The "Ultimate" grounded-physics model.

---

## Common Parameters for All Runs
- **Trials:** 1,000
- **Trial Seeds (Robustness):** 11 seeds (0 through 10)
- **Validation:** Top 5 trials
- **Validation Seeds (Generalization):** 10 seeds (11 through 20)
- **Logging:** Weights & Biases with Trials Table enabled

---

## Phase 0: Single-Kernel Baseline (The "Atomic" Comparison)
Tunes which **single** quantum topology is best (`kings`, `star`, etc.) alongside classical HPs.

### 1. Digital Encoding (RY Gates) + ZZ (1-Kernel)
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/1_kernel_3x3_digital_zz.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_single_kernel_search.yml \
    --n-trials 1000 \
    --trial-seeds 0 1 2 3 4 5 6 7 8 9 10 \
    --validate-top-k 5 \
    --validation-seeds 11 12 13 14 15 16 17 18 19 20 \
    --study-name hp_digital_1kern_zz \
    --wandb \
    --wandb-log-trials-table
```

### 2. Analog Encoding (Local Detuning) + ZZ (1-Kernel)
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/1_kernel_3x3_analog_zz.yml \
    --search-config configs/breast_mnist/hp_search/analog_zz_single_kernel_search.yml \
    --n-trials 1000 \
    --trial-seeds 0 1 2 3 4 5 6 7 8 9 10 \
    --validate-top-k 5 \
    --validation-seeds 11 12 13 14 15 16 17 18 19 20 \
    --study-name hp_analog_1kern_zz \
    --wandb \
    --wandb-log-trials-table
```

---

## Phase 1: 4-Kernel Comparison (Isolating the Physics)
Uses three fixed kernels (`grid`, `chain`, `horizontal`) and **tunes the 4th kernel** from a diverse list (`kings`, `vertical`, `cross`, `ring`, `star`), exactly matching your previous TS-MoE methodology.

### 3. Digital Encoding (RY Gates) + ZZ (4-Kernels)
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/4_kernels_3x3_digital_zz.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_4kern_search.yml \
    --n-trials 1000 \
    --trial-seeds 0 1 2 3 4 5 6 7 8 9 10 \
    --validate-top-k 5 \
    --validation-seeds 11 12 13 14 15 16 17 18 19 20 \
    --study-name hp_digital_4kern_zz \
    --wandb \
    --wandb-log-trials-table
```

### 4. Analog Encoding (Local Detuning) + ZZ (4-Kernels)
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/4_kernels_3x3_analog_zz.yml \
    --search-config configs/breast_mnist/hp_search/analog_zz_4kern_search.yml \
    --n-trials 1000 \
    --trial-seeds 0 1 2 3 4 5 6 7 8 9 10 \
    --validate-top-k 5 \
    --validation-seeds 11 12 13 14 15 16 17 18 19 20 \
    --study-name hp_analog_4kern_zz \
    --wandb \
    --wandb-log-trials-table
```

---

## Phase 2: 8-Kernel Comparison (Isolating Diversity)
Uses all 8 fixed topologies to provide the maximum physical feature density.

### 5. Digital Encoding (RY Gates) + ZZ (8-Kernels)
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/all_kernels_3x3_digital_zz.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_8kern_search.yml \
    --n-trials 1000 \
    --trial-seeds 0 1 2 3 4 5 6 7 8 9 10 \
    --validate-top-k 5 \
    --validation-seeds 11 12 13 14 15 16 17 18 19 20 \
    --study-name hp_digital_8kern_zz \
    --wandb \
    --wandb-log-trials-table
```

### 6. Analog Encoding (Local Detuning) + ZZ (8-Kernels)
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/all_kernels_3x3_analog_zz.yml \
    --search-config configs/breast_mnist/hp_search/analog_zz_8kern_search.yml \
    --n-trials 1000 \
    --trial-seeds 0 1 2 3 4 5 6 7 8 9 10 \
    --validate-top-k 5 \
    --validation-seeds 11 12 13 14 15 16 17 18 19 20 \
    --study-name hp_analog_8kern_zz \
    --wandb \
    --wandb-log-trials-table
```

---

## Notes on Performance
- **Cache Generation:** The first trial of each run will generate the quantum features cache. 
    - **Digital caches** typically take 20-30 minutes.
    - **Analog caches** typically take 60-120 minutes (due to manual batching).
- **GPU Acceleration:** To speed up the quantum part, you can edit the YAML configs to set `quantum_device: lightning.qubit` or `lightning.gpu` if your environment supports them.
