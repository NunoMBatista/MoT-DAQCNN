# MoT-DAQCNN: Comprehensive Hyperparameter Search (BreastMNIST)

This document contains the sequential commands to run a large-scale, robust comparison between **Digital vs. Analog** encoding and **4-Kernel vs. 8-Kernel** configurations, all utilizing **ZZ correlator measurements**.

## Common Parameters for All Runs
- **Trials:** 1,000
- **Trial Seeds (Robustness):** 11 seeds (0 through 10)
- **Validation:** Top 5 trials
- **Validation Seeds (Generalization):** 10 seeds (11 through 20)
- **Logging:** Weights & Biases with Trials Table enabled

---

## Phase 1: 4-Kernel Comparison (Isolating the Physics)
Using the kernels: `grid`, `chain`, `horizontal`, `kings`.

### 1. Digital Encoding (RY Gates) + ZZ
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/4_kernels_3x3_digital_zz.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_search.yml \
    --n-trials 1000 \
    --trial-seeds 0 1 2 3 4 5 6 7 8 9 10 \
    --validate-top-k 5 \
    --validation-seeds 11 12 13 14 15 16 17 18 19 20 \
    --study-name hp_digital_4kern_zz \
    --wandb \
    --wandb-log-trials-table
```

### 2. Analog Encoding (Local Detuning) + ZZ
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/4_kernels_3x3_analog_zz.yml \
    --search-config configs/breast_mnist/hp_search/analog_zz_search.yml \
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
Using all 8 topologies: `kings`, `horizontal`, `vertical`, `cross`, `ring`, `chain`, `star`, `grid`.

### 3. Digital Encoding (RY Gates) + ZZ
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/all_kernels_3x3_digital_zz.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_search.yml \
    --n-trials 1000 \
    --trial-seeds 0 1 2 3 4 5 6 7 8 9 10 \
    --validate-top-k 5 \
    --validation-seeds 11 12 13 14 15 16 17 18 19 20 \
    --study-name hp_digital_8kern_zz \
    --wandb \
    --wandb-log-trials-table
```

### 4. Analog Encoding (Local Detuning) + ZZ
```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/all_kernels_3x3_analog_zz.yml \
    --search-config configs/breast_mnist/hp_search/analog_zz_search.yml \
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
