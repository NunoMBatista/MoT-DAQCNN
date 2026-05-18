# Experiments TODO

All HP searches must be re-run: previous runs optimised against `test_acc`
(now fixed to `val_acc`). The `--validate-top-k 1 --validation-seeds 0..9`
flag runs the full 10-seed test evaluation inline after the search.

---

## 1. BreastMNIST

### 1a. DAQCNN — Digital, no ZZ (original) - RTX

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/1_kernel_3x3.yml \
    --search-config configs/breast_mnist/hp_search/single_kernel.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1b. DAQCNN — Digital + ZZ 

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/1_kernel_3x3_digital_zz.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_single_kernel_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1c. DAQCNN — Analog + ZZ

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/1_kernel_3x3_analog_zz.yml \
    --search-config configs/breast_mnist/hp_search/analog_zz_single_kernel_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1d. Classical — Random filters (1 kernel)

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/classical_baseline_base_1kern.yml \
    --search-config configs/breast_mnist/hp_search/classical_random_1kern_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1e. Classical — Trainable filters (1 kernel)

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/classical_baseline_base_1kern.yml \
    --search-config configs/breast_mnist/hp_search/classical_trainable_1kern_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1f. Vanilla CNN

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/vanilla_cnn_base.yml \
    --search-config configs/breast_mnist/hp_search/vanilla_cnn_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1g. Raw pixels baseline

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/classical_raw_image_base.yml \
    --search-config configs/breast_mnist/hp_search/classical_raw_image_search.yml \
    --n-trials 100 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1h. DAQCNN — Digital + ZZ, 4-kernel

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/4_kernels_3x3_digital_zz.yml \
    --search-config configs/breast_mnist/hp_search/digital_zz_4kern_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1i. DAQCNN — Analog + ZZ, 4-kernel

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/original/4_kernels_3x3_analog_zz.yml \
    --search-config configs/breast_mnist/hp_search/analog_zz_4kern_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1j. Classical — Random filters, 4-kernel (180 filters)

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/classical_baseline_base.yml \
    --search-config configs/breast_mnist/hp_search/classical_random_4kern_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

### 1k. Classical — Trainable filters, 4-kernel (180 filters)

```bash
python experiments/hyperparameter_search.py \
    --config configs/breast_mnist/classical_baseline_base.yml \
    --search-config configs/breast_mnist/hp_search/classical_trainable_4kern_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --wandb --wandb-project MoT-DAQCNN
```

---

## 2. PneumoniaMNIST

> **Prerequisite:** Create base configs for each model variant by copying the
> corresponding `configs/breast_mnist/original/` configs and setting
> `dataset.name: pneumonia_mnist`. The breast_mnist hp_search configs can
> be reused as-is (search spaces are dataset-agnostic).

### 2a–2k. Same 11 models as BreastMNIST (1g excluded — raw pixels is dataset-agnostic)

Same commands as Section 1, replacing:

- `--config configs/breast_mnist/original/1_kernel_3x3.yml` → pneumonia equivalent
- `--wandb-project MoT-DAQCNN` (same project, W&B tags will distinguish runs)

Add `--wandb-group pneumonia_mnist` to keep runs grouped in W&B.

---

## 3. TissueMNIST *(secondary — run if time permits)*

Existing search configs: `configs/tissue_mnist/hp_search/original.yml`,
`single_kernel.yml`. Need analog and ZZ variants (adapt from breast_mnist).

Same workflow as above once configs are in place.

---

## 4. Quantum Feature Cache Generation

Required before feature probing and before any DAQCNN HP search on a new
dataset (BreastMNIST caches likely already exist; generate for Pneumonia/Tissue).

```bash
# Digital ZZ (BreastMNIST — likely cached already, verify)
python experiments/create_quantum_dataset.py \
    --config configs/breast_mnist/cache_generation/digital_zz.yml

# Analog ZZ (BreastMNIST)
python experiments/create_quantum_dataset.py \
    --config configs/breast_mnist/cache_generation/analog_zz.yml

# Repeat for pneumonia_mnist and tissue_mnist once configs exist
```

---

## 5. Feature Probing

Run after HP searches are done so the `--quantum-1kern` config points to
the updated best config. BreastMNIST results already exist but should be
re-run with the corrected best configs.

```bash
# BreastMNIST
python experiments/feature_probing_ablation.py \
    --classical-1kern configs/breast_mnist/classical_baseline_base_1kern.yml \
    --quantum-1kern configs/breast_mnist/digital_zz_best.yml \
    --classical-4kern configs/breast_mnist/classical_baseline_base.yml \
    --quantum-4kern configs/breast_mnist/original/4_kernels_3x3_digital_zz.yml \
    --seeds 0 1 2 3 4

# PneumoniaMNIST (once configs and caches exist)
python experiments/feature_probing_ablation.py \
    --classical-1kern configs/pneumonia_mnist/classical_baseline_base_1kern.yml \
    --quantum-1kern configs/pneumonia_mnist/digital_zz_best.yml \
    --classical-4kern configs/pneumonia_mnist/classical_baseline_base.yml \
    --quantum-4kern configs/pneumonia_mnist/original/4_kernels_3x3_digital_zz.yml \
    --seeds 0 1 2 3 4
```

---

## 6. CKA Similarity

No training required — reads from cached `.npz` files. Run once caches exist.

```bash
python experiments/kernel_cka_similarity.py \
    --datasets breast_mnist pneumonia_mnist \
    --split train
```
