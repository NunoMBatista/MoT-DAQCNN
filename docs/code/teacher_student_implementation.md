### Document 2: Teacher-Student Approach (Recommended)

# Implementation Specification: Approach 2 (Teacher-Student Distillation)

## 1. System Overview
**Goal:** A robust 3-phase pipeline.
1.  **Teacher:** A heavy model runs *all* kernels and learns which is best using a Grouped SE Block.
2.  **Student:** A tiny model learns to mimic the Teacher's choice from raw pixels.
3.  **Inference:** The Student routes inputs to specific hardware kernels for sparse execution.

## 2. Phase 1: The Teacher Model

### 2.1 Class: `GroupedSEBlock` (The Judge)
* **Concept:** Like Squeeze-and-Excitation, but outputs a choice between *groups* of channels (Kernels), not individual channels.
* **Input:** Tensor `(Batch, 18, 1, 1)` (After Global Pooling).
* **Architecture:**
    * `Linear(18, 9)` + `ReLU` (Squeeze).
    * `Linear(9, 2)` (Excitation) -> **Important:** Output is size 2 (Num Kernels), not 18.
    * `Softmax(dim=1)` -> Output weights $\alpha_A, \alpha_B$.
* **Broadcasting:**
    * $\alpha_A$ scales Channels 0-8.
    * $\alpha_B$ scales Channels 9-17.

### 2.2 Class: `TeacherNet`
* **Forward Pass:**
    1.  Run `QuantumKernel_A` -> `Feat_A` (9 ch).
    2.  Run `QuantumKernel_B` -> `Feat_B` (9 ch).
    3.  Concatenate -> `Feats` (18 ch).
    4.  Apply `GlobalAvgPool` -> Pass to `GroupedSEBlock`.
    5.  Get Weights `W` (Batch, 2).
    6.  Multiply `Feat_A * W[:,0]` and `Feat_B * W[:,1]`.
    7.  Pass weighted features to `StandardCNNHead`.
    8.  **Return:** Class predictions AND the SE Weights `W` (needed for loss).

### 2.3 Phase 1 Loss Function (Entropy Regularization)
We must force the Teacher to choose ONE kernel, not mix them.
```python
def entropy_loss(alpha_weights):
    # alpha_weights shape: (Batch, 2)
    # Formula: - sum( p * log(p) )
    epsilon = 1e-8
    entropy = -torch.sum(alpha_weights * torch.log(alpha_weights + epsilon), dim=1)
    return torch.mean(entropy)

# Training Loop
lambda_reg = 0.1 # Increase this over epochs
loss = cross_entropy(pred, target) + lambda_reg * entropy_loss(se_weights)
```

## 3. Phase 2: The Student (Distillation)

### 3.1 Data Preparation (Offline Step)
* **Freeze trained TeacherNet.**
* Pass validation/training set through Teacher.
* **Extract Labels:** For every patch, look at the SE Weights $W$.
    * If $W[0] > 0.5 \rightarrow$ Label = 0 (Kernel A).
    * Else $\rightarrow$ Label = 1 (Kernel B).
* **Save dataset:** `(Raw_Patch_3x3, Best_Kernel_Index)`.

### 3.2 Class: StudentRouter
* **Input:** `(Batch, 1, 3, 3)`
* **Architecture:** Ultra-lightweight CNN.
    * `Conv2d(1, 8, 3, padding=1)` + ReLU.
    * `Conv2d(8, 16, 3, padding=1)` + ReLU.
    * `GlobalMaxPool2d`.
    * `Linear(16, 2)`.
* **Task:** Standard Classification (`CrossEntropyLoss` against the labels generated in 3.1).

---

## 4. Phase 3: Sparse Inference Pipeline

### 4.1 Class: HybridInferenceModel
This puts it all together for the final application.

* **Components:** Trained `StudentRouter`, Fixed `QuantumKernels`, Fresh `CNNHead`.
* **Forward Pass:**
    * **Route:** Pass patch $x$ into `StudentRouter` $\rightarrow$ Get Index $k$.
    * **Dispatch:**
        * If $k == 0$: Run `QuantumKernel_A(x)`. Output `[OutA, Zeros]`.
        * If $k == 1$: Run `QuantumKernel_B(x)`. Output `[Zeros, OutB]`.
    * **Assemble:** Construct the sparse tensor `(18, 3, 3)`.
    * **Classify:** Pass to `CNNHead`.

> **Note:** The `CNNHead` in Phase 3 should be fine-tuned on these "Sparse Tensors" (containing zeros) for a few epochs to ensure it handles the zero-padding correctly.
