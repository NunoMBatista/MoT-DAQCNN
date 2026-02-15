## 1. System Overview
**Goal:** Implement a single-stage Quantum Analog CNN where a lightweight "Router" network dynamically selects which quantum kernel to execute for a given input patch.
**Mechanism:** Use the Gumbel-Softmax reparameterization trick to allow backpropagation through the discrete "Kernel Selection" decision.

## 2. Global Constants
- **Input Patch Size:** $3 \times 3$
- **Num Quantum Kernels ($K$):** 2 (Example: Topology A, Topology B)
- **Atoms per Kernel ($N$):** 9
- **Channels per Kernel:** 9
- **Total Potential Channels:** 18

## 3. Component Specifications

### 3.1 Class: `RouterNetwork`
A lightweight classical CNN to estimate kernel relevance from raw pixels.
* **Input:** Tensor `(Batch, 1, 3, 3)`
* **Architecture:**
    * `Conv2d(1, 4, kernel_size=3, padding=1)` + `ReLU`
    * `Conv2d(4, 8, kernel_size=3, padding=1)` + `ReLU`
    * `AdaptiveAvgPool2d(1)` -> Flatten
    * `Linear(8, K)` -> Output Logits (Shape: `(Batch, K)`)
* **Output:** Unnormalized logits (do **not** apply Softmax here).

### 3.2 Class: `GumbelRouterBlock`
The logic handling the sampling and masking.
* **Init Args:** `temperature` (float), `hard` (bool).
* **Forward Pass:**
    1.  Receive `logits` from `RouterNetwork`.
    2.  Apply **Gumbel-Softmax**: `F.gumbel_softmax(logits, tau=temperature, hard=True, dim=1)`.
    3.  **Output:** A binary mask tensor `M` of shape `(Batch, K)`.
        * Example: `[1, 0]` means "Run Kernel A, Skip Kernel B".

### 3.3 Class: `DynamicQuantumLayer` (The Wrapper)
* **Components:**
    * `RouterNetwork` instance.
    * `QuantumKernel_A` (Pre-existing module).
    * `QuantumKernel_B` (Pre-existing module).
* **Forward Pass Logic:**
    1.  Generate Mask `M` using the Router.
    2.  **Conditional Execution (Simulation):**
        * Run `QuantumKernel_A(input)`. Output shape: `(Batch, 9, 3, 3)`.
        * Run `QuantumKernel_B(input)`. Output shape: `(Batch, 9, 3, 3)`.
    3.  **Apply Mask:**
        * Expand Mask `M[:, 0]` to shape `(Batch, 9, 3, 3)`. Multiply Kernel A output by this mask.
        * Expand Mask `M[:, 1]` to shape `(Batch, 9, 3, 3)`. Multiply Kernel B output by this mask.
    4.  **Concatenate:** Stack results to get `(Batch, 18, 3, 3)`.
    5.  **Return:** Feature map (sparse) and `mean_sparsity` (for loss calculation).

## 4. Loss Function Implementation
The training loop must minimize both classification error and computational cost.

```python
# Pseudo-code for Training Loop
criterion = nn.CrossEntropyLoss()
sparsity_weight = 0.01  # Lambda

# Forward
outputs, routing_decisions = model(images) 
# routing_decisions shape: (Batch, K) (Softmax probabilities)

# 1. Task Loss
loss_task = criterion(outputs, labels)

# 2. Budget Loss (L1 Penalty on activation)
# We want to minimize the number of active kernels.
# Target: minimize the sum of probabilities for the "expensive" choices.
loss_budget = torch.mean(torch.sum(routing_decisions, dim=1)) 

total_loss = loss_task + (sparsity_weight * loss_budget)
```

5. Annealing StrategyImplement a scheduler to decay the temperature ($\tau$) of the Gumbel-Softmax over epochs.Start: $\tau = 1.0$ (High exploration).End: $\tau = 0.1$ (Approaching Argmax).
