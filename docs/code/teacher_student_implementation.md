# Teacher-Student Training Protocol for the Gatekeeper (TS-MoE)

## 1. System Definitions & Constants
To avoid ambiguity regarding feature map dimensions and hardware constraints:
- **Input Patch Size ($P$):** $3 \times 3$ pixels (Matches hardware atom array size).
- **Hardware Qubits ($N$):** 9 atoms (arranged in $3 \times 3$).
- **Quantum Kernel Output ($C_{out}$):** 9 Channels per kernel (State of every atom is measured).
- **Available Topologies ($M$):** 2 (e.g., $K_A$: Star-Graph, $K_B$: Ring-Graph).
- **Total Teacher Channels:** $M \times N = 18$ channels.

**IMPORTANT NOTES ON SCALABILITY:**
- The dimensions shown above (2 kernels, 3×3 size, 18 channels) are **example values only**
- This architecture is **fully scalable** to:
  - Any number of kernel topologies (M can be 2, 3, 4, or more)
  - Different kernel sizes (2×2 with 4 qubits, 3×3 with 9 qubits, etc.)
  - Different strides (1, 2, 3, etc.)
  - Total channels = M × N where N = kernel_size²
- All tensor dimension examples in this document should be generalized accordingly

---
## Phase 1: Teacher
**Goal:** Train a heavy model to discover which kernel topology ($K_A$ or $K_B$) minimizes classification error for each specific image patch.

### 1.0 Using Cached Quantum Datasets (IMPORTANT)
**The project already has quantum dataset caching utilities in `src/utils/quantum_dataset_cache.py`:**
- **Cache Location:** `data/quantum_datasets/`
- **Metadata Files:** Each `.npz` file has an accompanying `.json` metadata file

**Existing Functions to Use:**
- `find_cached_quantum_dataset(cfg)` - Searches for matching cached dataset
  - Matches: `dataset_name`, `kernel_size`, `stride`, `kernel_topology_names`, `scaling_factor`, `evolution_time`, `color_space`
  - Returns path to `.npz` file or `None` if not found
- `load_cached_quantum_dataset(npz_path, batch_size, num_workers)` - Loads cached dataset
  - Returns: `(train_loader, val_loader, test_loader, n_classes, metadata)`
  - The `metadata` dict is already parsed from the JSON string in the `.npz` file

**Channel-to-Kernel Mapping in Cached Datasets:**
- Each cached `.npz` file contains a `metadata` field (JSON string)
- The metadata includes `channel_kernel_map`: a list of dictionaries, one per output channel
- **Structure:** `[{"channel": 0, "kernel": "kings", "qubit": 0}, {"channel": 1, "kernel": "kings", "qubit": 1}, ...]`
- **Usage:** This mapping tells you which channels belong to which kernel topology
- **Example:** If `channel_kernel_map` shows channels 0-8 → "kings" and channels 9-17 → "horizontal", then:
  - Group A (Kernel "kings") = channels 0-8
  - Group B (Kernel "horizontal") = channels 9-17

**Implementation Strategy:**
1. Use `find_cached_quantum_dataset(cfg)` to search for matching dataset
2. If found: Use `load_cached_quantum_dataset(npz_path)` to get loaders and metadata
3. Extract `channel_kernel_map` from metadata to build kernel-to-channels mapping
4. If not found: Error and instruct user to run `experiments/create_quantum_dataset.py` first

### 1.1 Architecture
- **Input:** Single Image sliced into $N_{patches}$. Tensor: `(N_patches, 1, 3, 3)`.
  - **Note:** Patch size and count depend on kernel_size and stride (generalize accordingly)
- **Quantum Layer (Brute Force):**
    - Execute **ALL** kernel topologies (e.g., $K_A$, $K_B$, ..., $K_M$) on every patch.
    - **Output per kernel:** N values (where N = kernel_size²)
    - **Example for M=2, N=9:** Output A: 9 values (Channels 0-8), Output B: 9 values (Channels 9-17)
    - **Stacked Tensor:** `(N_patches, M×N, H_out, W_out)` (for sliding) or `(N_patches, M×N)` (for non-overlapping)
    - **General formula:** Total channels = M (num_kernels) × N (qubits per kernel)
- **Grouped Squeeze-and-Excitation Block (The Judge):**
    - **CRITICAL:** SE operates **patch-wise**, not image-wise! Each patch gets its own weights.
    - **Pooling:** For each patch, compute statistics (e.g., mean, max) per kernel group
      - If spatial dimension exists: Global Average Pooling over (H_out, W_out)
      - Result: One summary value per kernel group per patch
    - **Gating Network:** 
      - Input: Pooled features from all M kernel groups (one value per group per patch)
      - Output: M weights (one per kernel group): $[\alpha_1, \alpha_2, ..., \alpha_M]$
      - **IMPORTANT:** Outputs M weights (one per kernel group), NOT M×N weights (not per channel!)
      - Apply softmax or sigmoid to ensure weights are normalized
      - **Process separately for EACH patch** in the batch
    - **Action:** For each patch, multiply $\alpha_k$ with ALL N channels of kernel group k
      - Example: $\alpha_A$ multiplies all channels 0-8, $\alpha_B$ multiplies all channels 9-17
      - This is done patch-wise across the entire batch
- **Classification Head (The Driver):**
    - Standard classical CNN trained on the **weighted (M×N)-channel volume**.
    - This Head provides the backpropagation gradients that teach the SE-Block which group is useful.

### 1.2 Sharp Loss Function
To ensure the Teacher makes decisive choices (instead of choosing multiple kernels) we add Entropy Regularization.
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}} + \lambda \cdot \mathcal{L}_{\text{Entropy}}$$
Where:
$$\mathcal{L}_{\text{Entropy}} = - \sum_{k \in \{A, B\}} \alpha_k \log(\alpha_k)$$
- **Effect:** Penalizes uncertainty ($\alpha \approx 0.5$). Rewards decisiveness ($\alpha \approx 1.0$ or $0.0$).
- **Target:** $\lambda$ should be ramped up during training to force hard routing behavior by the final epoch.

---
## Phase 2: Student/Gatekeeper (Distillation)
**Goal:** Train a lightweight classical network to predict the Teacher's choice without running the quantum hardware.
### 2.1 Label Generation
1. Freeze the trained Teacher (Phase 1).
2. Pass the training dataset through the Teacher.
3. Extract the $\alpha$ values (kernel group weights) for every patch.
4. **Hard Labeling:**
    - Find $k^* = \arg\max_k \alpha_k$ for each patch
    - Assign $y_{target} = k^*$ (the index of the winning kernel)
    - **Example for M=2:** If $\alpha_A > \alpha_B \rightarrow y_{target} = 0$; else $y_{target} = 1$
    - **General for M kernels:** $y_{target} \in \{0, 1, ..., M-1\}$
    - _Optional:_ If confidence is low ($\max(\alpha) < \text{threshold}$), mark as "Skip/Classical Bypass".
### 2.2 Student Training
- **Input:** Raw pixel patches `(N_patches, 1, kernel_size, kernel_size)`.
- **Model:** Lightweight CNN (e.g., 3 layers, <5k parameters).
- **Output:** Logits for M kernel classes (one per kernel topology).
- **Loss:** `CrossEntropyLoss(Student_Pred, Teacher_Label)`.
- **Success Metric:** Student must achieve >90% agreement with Teacher on high-confidence patches.
- **Optional Enhancement:** Feed additional classical features (variance, FFT, edge statistics) to help the student

---
## Phase 3: Final CNN Head Training and Inference
**Goal:** Route patches to the static hardware tiles and reconstruct the image for final diagnosis.
### 3.1 Router Loop
For a new input image:
1. **Slice:** Break image into grid of patches (size determined by kernel_size and stride).
2. **Gatekeeper Inference:**    
    - Batch all patches through the Student.
    - Generate Routing Map: `[Patch 0: k_0, Patch 1: k_1, Patch 2: k_2, ...]` where $k_i \in \{0, 1, ..., M-1\}$
    - **Example for M=2:** `[Patch 0: A, Patch 1: A, Patch 2: B, ...]`
3. **Quantum Hardware Execution:**
    - **Simulation (what we implement):** Load from cached quantum dataset, extract only selected channels per patch
    - **Hardware Dispatch (future deployment only, not needed now):**
      - Send patches assigned to kernel $k$ to hardware tile configured with topology $k$
      - **Execute:** Simultaneous shot across all tiles
      - **Return:** N scalar values per patch (where N = kernel_size²)
### 3.2 Sparse Tensor Reconstruction
We must assemble the disparate hardware outputs into a single tensor for the final classifier.
- **Global Tensor Shape:** `(Batch_Images, M×N Channels, Height, Width)`.
- **General Logic:** For each patch at position $(i, j)$ assigned to kernel $k$:
    - Fill channels $[k \times N : (k+1) \times N]$ with quantum output data
    - Fill all other channel groups with **Zeros**
- **Example for M=2, N=9:**
    - If Patch $(i, j)$ used **Kernel A (k=0)**: Fill Channels 0-8 with data, Channels 9-17 with Zeros
    - If Patch $(i, j)$ used **Kernel B (k=1)**: Fill Channels 0-8 with Zeros, Channels 9-17 with data
- **Using Channel Mapping:** Use the `channel_kernel_map` from cached dataset metadata to identify which channels correspond to which kernel

### 3.3 Final Classifier (Head Replacement)
- **Note:** The Classification Head from Phase 1 (Teacher) cannot be used here because it was trained on "Soft Mixed" data.
- **Action:** Train a **NEW** Classification Head from scratch (or fine-tune) on the **Sparse Tensors** generated by the Phase 3 pipeline.
- **Why:** This new head learns to interpret "Zeros" as "Not Selected" rather than "Low Signal."
- **Optional Enhancement:** Add a (M×N + 1)-th "Mask Channel" that explicitly indicates which patches have data (1) vs are masked (0)

## Upgrades & Extensions
1. **Student Enhancement:** Feed additional classical statistics (variance, FFT features, edge density) to give it a better chance of correlating with the teacher's choice (the student needs more help because it lacks quantum performance information).
2. **Masked Channel:** Add a (M×N + 1)-th channel that explicitly indicates data presence:
   - Value = 1.0 for channels with active data
   - Value = 0.0 for masked/unused channels
   - This helps the classifier focus on classification rather than learning to ignore zeros
3. **Lambda Annealing:** Use $\lambda$-annealing for entropy regularization (start with $\lambda=0$) so the network explores before converging to hard decisions.
4. **Multi-kernel Generalization:** Scale to M > 2 kernels by generalizing all binary decisions to M-way classification.

---

## Implementation Guidelines for LLM Assistant

### Code Style & Conventions
- **Make the implementation minimalistic and human-readable**
- Write code that looks natural and maintainable, not over-engineered
- Add useful comments explaining key decisions, but avoid excessive documentation
- Keep functions focused and modular - each function should do one thing well
- Avoid complex nested structures where simpler alternatives exist

### Project Structure & Conventions
- **Follow existing `src/` folder conventions:**
  - New models go in `src/models/` (e.g., `src/models/teacher_student_moe.py`)
  - New layers/modules go in `src/layers/` if reusable
  - Training logic can go in `src/models/` alongside model definitions
  - Utility functions go in `src/utils/` if they're general-purpose
- **Look at existing files** (`src/models/daqcnn.py`, `src/models/daqcnn_training.py`) to match the code style
- Use PyTorch conventions (nn.Module, forward(), etc.)

### IMPORTANT: Use Existing Utilities
- **DO NOT create a new quantum dataset loader!** 
- The project already has `src/utils/quantum_dataset_cache.py` with:
  - `find_cached_quantum_dataset(cfg)` - finds matching cached datasets
  - `load_cached_quantum_dataset(npz_path)` - loads features, labels, and metadata
- **You only need to create** a small helper function to parse `channel_kernel_map` from the metadata
- See the "Using Cached Quantum Datasets" section above for details

### Configuration & Integration
- **Config File Integration:**
  - Add a new field to YAML configs: `model.architecture: "original"` or `"TS-MoE"`
  - When `architecture: "original"` → use existing DAQCNN model
  - When `architecture: "TS-MoE"` → use new Teacher-Student MoE model
  - Add TS-MoE specific parameters under a new `ts_moe:` section in config if needed
  - Example new parameters: `lambda_entropy`, `student_features`, `use_mask_channel`, etc.

- **Modify `experiments/robust_test_original_daqcnn.py`:**
  - Read `model.architecture` from config
  - Branch to appropriate model initialization based on this value
  - Keep the script name unchanged (it's now a general experiment runner)
  - Naturally, the name of this script should change to "robust_experiment_runner.py" 

- **TUI Compatibility:**
  - The new model MUST be compatible with the existing TUI app
  - Ensure the model interface matches (same forward signature, output format)
  - TUI should be able to select "original" vs "TS-MoE" through config

### Cached Dataset Usage
- **Check for cached datasets FIRST** before creating quantum layers
- **Implementation Steps:**
  1. Parse config to extract: `kernel_size`, `stride`, `kernel_topology_names`, `scaling_factor`, `evolution_time`, `dataset_name`
  2. Search `data/quantum_datasets/*.json` for matching metadata
  3. If found: Load `.npz` file and extract `channel_kernel_map` from metadata
  4. Use `channel_kernel_map` to group channels by kernel topology
  5. If not found: Either raise error asking user to run `create_quantum_dataset.py` OR auto-generate and cache

- **Channel Mapping Usage:**
  - **Using Existing Cached Dataset Utilities:**
    ```python
    from src.utils.quantum_dataset_cache import find_cached_quantum_dataset, load_cached_quantum_dataset
  
    # Find matching cached dataset
    npz_path = find_cached_quantum_dataset(config)
    if npz_path is None:
        raise FileNotFoundError("No cached quantum dataset found. Run create_quantum_dataset.py first.")
  
    # Load dataset and metadata
    train_loader, val_loader, test_loader, n_classes, metadata = load_cached_quantum_dataset(npz_path, batch_size=32)
  
    # Extract channel-to-kernel mapping
    channel_kernel_map = metadata['channel_kernel_map']
  
    # Build reverse mapping: kernel_name -> list of channel indices
    kernel_to_channels = {}
    for entry in channel_kernel_map:
        kernel_name = entry['kernel']
        channel_idx = entry['channel']
        if kernel_name not in kernel_to_channels:
            kernel_to_channels[kernel_name] = []
        kernel_to_channels[kernel_name].append(channel_idx)
    ```

### Implementation Phases
**Recommend implementing in this order:**

1. **Phase 1a: Grouped SE Block** (core component)
   - Create a modular SE block that works patch-wise
   - Input: `(batch, M*N, H, W)` or `(batch, M*N)`
   - Output: M weights per patch
   - Test independently before integrating

2. **Phase 1b: Teacher Model** (full model)
   - Integrate quantum layer (or cached dataset loader)
   - Add grouped SE block
   - Add classification head
   - Add entropy loss
   - Train and verify it learns to select kernels

3. **Phase 2: Student/Gatekeeper**
   - Simple lightweight CNN
   - Distillation from teacher's alpha values
   - Can be trained independently once teacher is ready

4. **Phase 3a: Sparse Reconstruction**
   - Implement the routing logic
   - Build sparse tensors from student predictions
   - Test with dummy data first

5. **Phase 3b: New Classification Head**
   - Train new head on sparse tensors
   - Compare performance with teacher

6. **Integration: Config & TUI**
   - Add architecture selection to configs
   - Modify experiment runner to branch on model type
   - Test with TUI

### Testing Strategy
- **Unit test each component** before integration (SE block, student, router, etc.)
- **Use small datasets** (like `quick_test.yml`) for rapid iteration
- **Verify shapes** at each stage (print tensor shapes liberally during development)
- **Compare with teacher** when validating student and final classifier
- **Test both architectures** ("original" and "TS-MoE") to ensure config switching works

### Common Pitfalls to Avoid
- Don't make SE block operate image-wise (it's patch-wise!)
- Don't output weights per channel (it's per kernel group!)
- Don't reuse teacher's classification head in Phase 3 (train new one!)
- Don't hardcode M=2 or N=9 (make it general for any M, N)
- Don't forget to check for cached datasets before running quantum circuits
- Don't create overly complex abstractions - keep it simple and readable

### Success Criteria
- Code follows existing project style and conventions  
- Works with cached quantum datasets when available  
- Generalizes to any number of kernels (M) and kernel sizes (N)  
- SE block operates patch-wise and outputs M weights per patch  
- Config file can switch between "original" and "TS-MoE"  
- Compatible with TUI and `robust_test_original_daqcnn.py`  
- Code is readable, well-commented, and maintainable  
- Successfully trains teacher, student, and final classifier

---

## Step-by-Step Implementation Checklist

### Preparation Steps
- [ ] **Step 0.1:** Read existing code in `src/models/daqcnn.py` and `src/models/daqcnn_training.py` to understand project conventions
- [ ] **Step 0.2:** Review a config file (e.g., `configs/pneumonia_mnist.yml`) to understand the structure
- [ ] **Step 0.3:** Check if quantum datasets exist in `data/quantum_datasets/` - if not, run `experiments/create_quantum_dataset.py` first
- [ ] **Step 0.4:** Review existing `src/utils/quantum_dataset_cache.py` to understand how cached datasets are loaded
- [ ] **Step 0.5:** Create a test config file (e.g., `configs/ts_moe_test.yml`) with `architecture: "TS-MoE"` for development

### Phase 1: Core Components (Teacher Model)

- [ ] **Step 1.1:** Create `src/layers/grouped_se_block.py`
  - [ ] Implement patch-wise squeeze-and-excitation
  - [ ] Input: `(batch, num_patches, M*N_channels)` or `(batch, M*N_channels, H, W)`
  - [ ] Output: `(batch, num_patches, M)` - M weights per patch
  - [ ] Test with dummy data to verify shape transformations

- [ ] **Step 1.2:** Create utility function for kernel-to-channels mapping
  - [ ] Create `src/utils/kernel_mapping.py` with function `build_kernel_to_channels_map(channel_kernel_map)`
  - [ ] Converts list of `{"channel": idx, "kernel": name, "qubit": q}` into dict: `{"kernel_name": [channel_indices]}`
  - [ ] This works with metadata already loaded by existing `quantum_dataset_cache.py`
  - [ ] Test with real `channel_kernel_map` from a cached dataset

- [ ] **Step 1.3:** Create `src/models/teacher_moe.py`
  - [ ] Implement `TeacherMoE` class (inherits from `nn.Module`)
  - [ ] Constructor uses `find_cached_quantum_dataset(config)` to locate cached data
  - [ ] Uses `load_cached_quantum_dataset()` to load features and metadata
  - [ ] Extracts `channel_kernel_map` from metadata and builds kernel-to-channels mapping
  - [ ] Forward pass: quantum features → grouped SE → weighted features → classification head
  - [ ] Add method to extract alpha weights for distillation
  - [ ] Test forward pass with dummy images

- [ ] **Step 1.4:** Implement entropy regularization loss
  - [ ] Create `src/utils/losses.py` if it doesn't exist
  - [ ] Implement `entropy_loss(alpha_weights)` function
  - [ ] Test with dummy alpha values

- [ ] **Step 1.5:** Create `src/models/teacher_moe_training.py`
  - [ ] Implement training loop for teacher (similar to `daqcnn_training.py`)
  - [ ] Combine classification loss + lambda * entropy loss
  - [ ] Add lambda annealing schedule
  - [ ] Test on small dataset subset

### Phase 2: Student/Gatekeeper

- [ ] **Step 2.1:** Create `src/models/student_gatekeeper.py`
  - [ ] Implement lightweight CNN (3-5 layers, <5k parameters)
  - [ ] Input: raw image patches `(batch, 1, kernel_size, kernel_size)`
  - [ ] Output: logits for M kernel classes
  - [ ] Optional: Add classical feature extractors (variance, FFT, etc.)
  - [ ] Test forward pass

- [ ] **Step 2.2:** Implement label generation from teacher
  - [ ] Add method in `teacher_moe_training.py` to extract alpha values
  - [ ] Create hard labels: `argmax(alpha)` per patch
  - [ ] Save labels to disk for student training
  - [ ] Test label extraction on small batch

- [ ] **Step 2.3:** Create `src/models/student_training.py`
  - [ ] Implement training loop for student
  - [ ] Use CrossEntropyLoss with teacher labels
  - [ ] Track agreement metric with teacher predictions
  - [ ] Test on small dataset

### Phase 3: Sparse Reconstruction & Final Classifier

- [ ] **Step 3.1:** Create `src/utils/sparse_reconstruction.py`
  - [ ] Implement patch routing based on student predictions
  - [ ] Build sparse tensor: fill selected kernel channels, zero others
  - [ ] Use `channel_kernel_map` to identify which channels to fill
  - [ ] Test with dummy student predictions and quantum features

- [ ] **Step 3.2:** Optional: Implement mask channel
  - [ ] Add (M*N + 1)-th channel indicating data presence
  - [ ] Set to 1.0 for active channels, 0.0 for masked
  - [ ] Test tensor shape is correct

- [ ] **Step 3.3:** Create `src/models/final_classifier.py`
  - [ ] New classification head (separate from teacher's head)
  - [ ] Input: sparse tensors `(batch, M*N [+1], H, W)`
  - [ ] Train from scratch on routed data
  - [ ] Test on validation set

### Phase 4: Configuration & Integration

- [ ] **Step 4.1:** Update config schema
  - [ ] Add `model.architecture: "original"` or `"TS-MoE"` field
  - [ ] Add `ts_moe:` section with parameters:
    - `lambda_entropy`: initial entropy weight
    - `lambda_max`: maximum entropy weight
    - `lambda_anneal_epochs`: epochs to anneal lambda
    - `student_hidden_dims`: architecture for student
    - `use_mask_channel`: boolean
    - `confidence_threshold`: for low-confidence filtering
  - [ ] Create example config: `configs/pneumonia_mnist_ts_moe.yml`

- [ ] **Step 4.2:** Modify `experiments/robust_test_original_daqcnn.py`
  - [ ] Read `model.architecture` from config
  - [ ] Add conditional: if "original" → use DAQCNN, if "TS-MoE" → use TeacherMoE
  - [ ] Ensure both paths work with same experiment runner
  - [ ] Test both configurations

- [ ] **Step 4.3:** Create unified training script
  - [ ] `experiments/train_ts_moe.py` or integrate into existing script
  - [ ] Phase 1: Train teacher
  - [ ] Phase 2: Extract labels and train student
  - [ ] Phase 3: Build sparse tensors and train final classifier
  - [ ] Save all checkpoints appropriately

### Phase 5: Testing & Validation

- [ ] **Step 5.1:** Unit tests
  - [ ] Test grouped SE block with various M and N values
  - [ ] Test quantum dataset loader with existing cached files
  - [ ] Test sparse reconstruction logic
  - [ ] Test all models forward passes

- [ ] **Step 5.2:** Integration tests
  - [ ] Train teacher on quick_test dataset
  - [ ] Verify teacher learns to select kernels (check alpha distributions)
  - [ ] Train student and verify >90% agreement
  - [ ] Build sparse tensors and train final classifier
  - [ ] Compare final accuracy with baseline

- [ ] **Step 5.3:** Full experiment
  - [ ] Run full pipeline on `pneumonia_mnist` or `tissue_mnist`
  - [ ] Compare performance: original DAQCNN vs TS-MoE
  - [ ] Analyze which kernels are selected for different patches
  - [ ] Generate visualizations (kernel selection heatmaps, etc.)

### Phase 6: TUI Integration & Polish

- [ ] **Step 6.1:** Verify TUI compatibility
  - [ ] Test loading TS-MoE model in TUI
  - [ ] Ensure inference works correctly
  - [ ] Add UI elements to display kernel selection info (optional)

- [ ] **Step 6.2:** Documentation
  - [ ] Add docstrings to all new classes and functions
  - [ ] Update README if needed
  - [ ] Create usage examples in config files
  - [ ] Document performance comparisons

- [ ] **Step 6.3:** Code cleanup
  - [ ] Remove debug print statements
  - [ ] Ensure consistent code style
  - [ ] Add helpful comments where needed
  - [ ] Run linter/formatter if project uses one

### Final Checklist
- [ ] All unit tests pass
- [ ] Both "original" and "TS-MoE" architectures work via config switching
- [ ] Cached quantum datasets are used when available
- [ ] System generalizes to arbitrary M kernels and N qubits
- [ ] Student achieves >90% agreement with teacher
- [ ] Final classifier performance is competitive with baseline
- [ ] Code is clean, readable, and well-commented
- [ ] TUI integration works
- [ ] Documentation is complete

---

## Quick Start Example

Once implemented, running the TS-MoE should be as simple as:

```bash
# 1. Generate cached quantum dataset (if not exists)
python experiments/create_quantum_dataset.py
# (Edit parameters in script to match your config)

# 2. Run TS-MoE training
python experiments/robust_test_original_daqcnn.py --config configs/pneumonia_mnist_ts_moe.yml

# 3. Compare with original
python experiments/robust_test_original_daqcnn.py --config configs/pneumonia_mnist.yml
```

The config file determines which architecture is used - everything else is automatic!
