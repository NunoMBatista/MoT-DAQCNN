import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

def analyze_importance(file_path):
    print(f"Loading dataset: {file_path}")
    data = np.load(file_path)
    
    # NPZ files from create_quantum_dataset.py typically have keys like:
    # 'train_features', 'train_labels', 'test_features', 'test_labels'
    # Features are (N, num_kernels * features_per_kernel)
    X = data['train_features']
    y = data['train_labels']
    
    # For simplicity, let's look at the first kernel block (e.g., Kings)
    # A 3x3 kernel has 9 Z + 36 ZZ = 45 features
    n_z = 9
    n_zz = 36
    features_per_kernel = n_z + n_zz
    
    # Slice the first kernel's features and apply global average pooling
    X_kernel = X[:, :features_per_kernel, :, :].mean(axis=(2, 3))
    
    print(f"Training Random Forest on {X_kernel.shape[0]} samples...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_kernel, y)
    
    importances = rf.feature_importances_
    
    z_importance = np.sum(importances[:n_z])
    zz_importance = np.sum(importances[n_z:])
    
    print("\n--- Feature Importance Summary ---")
    print(f"Total Importance from Z features (9):  {z_importance:.3f} ({z_importance*100:.1f}%)")
    print(f"Total Importance from ZZ features (36): {zz_importance:.3f} ({zz_importance*100:.1f}%)")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    feature_names = [f"Z_{i}" for i in range(n_z)] + [f"ZZ_{i}" for i in range(n_zz)]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    top_n = 15
    
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45)
    plt.title(f"Top {top_n} Quantum Features (Z vs ZZ)")
    plt.ylabel("MDI Importance")
    
    output_path = "feature_importance_plot.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    # Point this to your local NPZ file
    path = "data/quantum_datasets/breast_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz.npz"
    if os.path.exists(path):
        analyze_importance(path)
    else:
        print(f"File not found: {path}")
        print("Please ensure the .npz file is in the data/quantum_datasets/ directory.")
