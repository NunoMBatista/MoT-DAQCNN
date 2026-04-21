# Experiments TODO

## Classical Baseline

### Methodology: Feature Probing / Linear Probing

We will extract features from three distinct sources at the exact same bottleneck (post-convolution, post-flattening, pre-dense classification head) and evaluate their intrinsic information content using a neutral, task-agnostic arbiter (e.g., a Random Forest or regularized SVM).

This ablation study answers the critical question: *"Is the quantum kernel actually doing useful work, or is the classical head just large enough to memorize the dataset?"*

**The Three Competitors:**

1. **Raw Pixels:** The baseline information content of the image.
2. **Trained Classical Features (Task-Aware):** Features extracted from a classical CNN (`out_channels` matching the quantum model) trained end-to-end via backpropagation on the BreastMNIST dataset.
3. **Fixed Quantum Features (Task-Agnostic):** Raw $\langle Z \rangle$ and $\langle ZZ \rangle$ correlators extracted from the un-trained Hamiltonian evolution of the Rydberg atoms.

**Success Criteria:**
If the task-agnostic Quantum Features outperform the task-aware Classical Features on the neutral ML model, it provides definitive proof that the multi-partite entanglement dynamics natively map morphological complexity into a superior, more generalizable feature space.

*(Note: Due to the high dimensionality of the flattened features relative to the BreastMNIST sample size, consider applying PCA or using a heavily regularized/depth-restricted Random Forest to prevent the curse of dimensionality).*

### TODO List

**1-Kernel Experiments (45 output features: 9 Z + 36 ZZ)**

- [ ] Extract flattened feature vectors from the optimally trained 1-Kernel Classical Baseline CNN.
- [ ] Extract flattened feature vectors from the 1-Kernel ZZ-AQCNN.
- [ ] Extract flattened raw pixel vectors from BreastMNIST.
- [ ] Train a Random Forest (or SVM) on the Raw Pixel vectors and record test metrics (Accuracy/AUC).
- [ ] Train a Random Forest (or SVM) on the 1-Kernel Classical Features and record test metrics.
- [ ] Train a Random Forest (or SVM) on the 1-Kernel Quantum Features and record test metrics.
- [ ] Compare results.

**4-Kernel Experiments (180 output features: 4 x 45)**

- [ ] Extract flattened feature vectors from the optimally trained 4-Kernel Classical Baseline CNN.
- [ ] Extract flattened feature vectors from the 4-Kernel ZZ-AQCNN (Best 3 + 1 Diverse).
- [ ] Train a Random Forest (or SVM) on the 4-Kernel Classical Features and record test metrics.
- [ ] Train a Random Forest (or SVM) on the 4-Kernel Quantum Features and record test metrics.
- [ ] Compare results and generate a final bar chart for the AWS proposal/publication.
