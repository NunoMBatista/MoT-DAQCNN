"""
Feature Probing (Linear Probing) Ablation Study

This script runs the monolithic experiment to compare the intrinsic representational
power of Raw Pixels, Task-Aware Classical Features, and Task-Agnostic Quantum Features.

Workflow:
1. Trains Classical Baseline CNNs (both 1-kernel and 4-kernel) end-to-end to generate task-aware features.
2. Trains Classical Baseline CNNs with FIXED RANDOM filters to generate random projection features.
3. Loads the pre-computed quantum features from the cache and trains the classical head for the Quantum Model.
4. Extracts features from multiple points: Raw Pixels, Post-Conv (After Kernel), and Pre-Dense (Before classification).
5. Trains a Random Forest Classifier AND an SVM on all feature sets.
6. Compares and reports Test Accuracy and ROC-AUC in a massive symmetry table.

Usage:
    python experiments/feature_probing_ablation.py \
        --classical-1kern configs/breast_mnist/classical_baseline_base_1kern.yml \
        --quantum-1kern configs/breast_mnist/digital_zz_best.yml \
        --classical-4kern configs/breast_mnist/classical_baseline_base.yml \
        --quantum-4kern configs/breast_mnist/original/4_kernels_3x3_digital_zz.yml \
        --seed 42
"""

import argparse
import copy
import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tabulate import tabulate

from src.models.classical_baseline_training import run_classical_baseline
from src.models.classical_baseline import ClassicalBaselineCNN
from src.models.daqcnn_training import run_single_seed
from src.models.daqcnn import DAQCNN
from src.utils.data import get_dataloaders
from src.utils.quantum_dataset_cache import find_cached_quantum_dataset, load_cached_quantum_dataset
from src.utils.training_utils import resolve_device

def extract_features(model, loader, device, extract_point="post_conv", is_quantum_model=False, is_raw=False):
    """
    Extract flattened features from a model.
    extract_point can be:
      - "raw": Raw image pixels.
      - "post_conv": Right after the conv layer (equivalent to output of quantum kernel).
      - "pre_dense": Right before the final Linear layer in the classification head.
    """
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            
            if is_raw:
                feats = imgs.view(imgs.size(0), -1)
            elif extract_point == "post_conv":
                if is_quantum_model:
                    # The loader yields quantum features directly!
                    feats = imgs.view(imgs.size(0), -1)
                else:
                    feats = model.conv(imgs.float()).view(imgs.size(0), -1)
            elif extract_point == "pre_dense":
                if is_quantum_model:
                    x = imgs.float() # Already quantum features
                else:
                    x = model.conv(imgs.float())
                
                # Pass through head until Linear
                for layer in model.head:
                    if isinstance(layer, (nn.Linear, nn.LazyLinear)):
                        break
                    x = layer(x)
                feats = x.view(x.size(0), -1)
            else:
                raise ValueError(f"Unknown extract_point {extract_point}")
                
            all_feats.append(feats.cpu().numpy())
            
            lbl = labels.squeeze().numpy()
            if lbl.ndim == 0:
                lbl = np.expand_dims(lbl, axis=0)
            all_labels.append(lbl)
            
    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)

def train_and_evaluate_models(X_train, y_train, X_test, y_test, seed, name=""):
    """Train a Random Forest and an SVM, then evaluate both."""
    print(f"Training Models on {name} (Features: {X_train.shape[1]})...")
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)
    
    if y_prob_rf.shape[1] == 2:
        auc_rf = roc_auc_score(y_test, y_prob_rf[:, 1])
    else:
        auc_rf = roc_auc_score(y_test, y_prob_rf, multi_class="ovr")
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    # 2. Support Vector Machine (SVM)
    # SVMs require standardized features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    t0 = time.time()
    svm.fit(X_train_scaled, y_train)
    t_svm = time.time() - t0
    
    y_pred_svm = svm.predict(X_test_scaled)
    y_prob_svm = svm.predict_proba(X_test_scaled)
    
    if y_prob_svm.shape[1] == 2:
        auc_svm = roc_auc_score(y_test, y_prob_svm[:, 1])
    else:
        auc_svm = roc_auc_score(y_test, y_prob_svm, multi_class="ovr")
    acc_svm = accuracy_score(y_test, y_pred_svm)
    
    print(f"  -> RF Done  in {t_rf:.2f}s | Acc: {acc_rf:.4f} | AUC: {auc_rf:.4f}")
    print(f"  -> SVM Done in {t_svm:.2f}s | Acc: {acc_svm:.4f} | AUC: {auc_svm:.4f}\n")
    
    return acc_rf, auc_rf, acc_svm, auc_svm

def process_pipeline(classical_config_path, quantum_config_path, seed, output_dir, label_prefix):
    """Run the training, extraction, and probing for a single configuration pair."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(classical_config_path, "r") as f:
        class_cfg = yaml.safe_load(f)
    with open(quantum_config_path, "r") as f:
        quant_cfg = yaml.safe_load(f)

    device = resolve_device(class_cfg, verbose=False)

    # --- 1. Train Trainable Classical ---
    print(f"\nPHASE 1A: Training Task-Aware Classical CNN ({label_prefix})")
    class_train_dir = os.path.join(output_dir, "class_train")
    os.makedirs(class_train_dir, exist_ok=True)
    class_results = run_classical_baseline(class_cfg, seed, class_train_dir, verbose=False)
    
    best_model_path = os.path.join(class_train_dir, f"best_model_seed_{seed}.pt")
    checkpoint = torch.load(best_model_path, map_location=device)
    
    model_cfg = class_cfg.get("model", {})
    out_channels = model_cfg.get("out_channels", 180)
    
    model_class = ClassicalBaselineCNN(
        num_classes=class_results["num_classes"],
        in_channels=model_cfg.get("in_channels", 1),
        kernel_size=model_cfg.get("kernel_size", 3),
        stride=model_cfg.get("stride", 3),
        out_channels=out_channels,
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
        head_hidden_channels=model_cfg.get("head_hidden_channels", 64),
        fixed_random_filters=False,
        raw_features=False
    )
    model_class.load_state_dict(checkpoint["model_state_dict"])
    model_class.to(device)

    # --- 2. Train Random Classical ---
    print(f"\nPHASE 1B: Training Random Fixed Classical CNN ({label_prefix})")
    class_rand_cfg = copy.deepcopy(class_cfg)
    if "model" not in class_rand_cfg: class_rand_cfg["model"] = {}
    class_rand_cfg["model"]["fixed_random_filters"] = True
    
    class_rand_dir = os.path.join(output_dir, "class_rand")
    os.makedirs(class_rand_dir, exist_ok=True)
    class_rand_results = run_classical_baseline(class_rand_cfg, seed, class_rand_dir, verbose=False)
    best_model_path_rand = os.path.join(class_rand_dir, f"best_model_seed_{seed}.pt")
    checkpoint_rand = torch.load(best_model_path_rand, map_location=device)
    
    model_rand = ClassicalBaselineCNN(
        num_classes=class_rand_results["num_classes"],
        in_channels=class_rand_cfg["model"].get("in_channels", 1),
        kernel_size=class_rand_cfg["model"].get("kernel_size", 3),
        stride=class_rand_cfg["model"].get("stride", 3),
        out_channels=class_rand_cfg["model"].get("out_channels", out_channels),
        dropout=class_rand_cfg["model"].get("dropout", 0.1),
        activation=class_rand_cfg["model"].get("activation", "relu"),
        head_hidden_channels=class_rand_cfg["model"].get("head_hidden_channels", 64),
        fixed_random_filters=True,
        raw_features=False
    )
    model_rand.load_state_dict(checkpoint_rand["model_state_dict"])
    model_rand.to(device)

    # --- 3. Train Quantum Head ---
    print(f"\nPHASE 1C: Training Quantum Model Head ({label_prefix})")
    quant_dir = os.path.join(output_dir, "quant")
    os.makedirs(quant_dir, exist_ok=True)
    quant_results = run_single_seed(quant_cfg, seed, quant_dir, verbose=False)
    
    best_model_path_quant = os.path.join(quant_dir, f"best_model_seed_{seed}.pt")
    checkpoint_quant = torch.load(best_model_path_quant, map_location=device)
    
    cached_path = find_cached_quantum_dataset(quant_cfg)
    if cached_path is None:
        raise FileNotFoundError(f"Could not find cached quantum dataset for: {quantum_config_path}")
        
    batch_size = quant_cfg.get("dataset", {}).get("batch_size", 32)
    requested_kernels = quant_cfg.get("model", {}).get("kernel_topology_names")
    
    q_train_loader, q_val_loader, q_test_loader, n_classes, cache_meta = load_cached_quantum_dataset(
        cached_path, batch_size, 2, requested_kernels=requested_kernels
    )
    
    model_quant = DAQCNN(
        num_classes=n_classes,
        kernel_size=quant_cfg.get("model", {}).get("kernel_size", 3),
        stride=quant_cfg.get("model", {}).get("stride", 3),
        kernel_topology_names=requested_kernels,
        scaling_factor=quant_cfg.get("model", {}).get("scaling_factor", 1.0),
        evolution_time=quant_cfg.get("model", {}).get("evolution_time", 2.5),
        mode=quant_cfg.get("model", {}).get("mode", "trotter"),
        dropout=quant_cfg.get("model", {}).get("dropout", 0.6),
        activation=quant_cfg.get("model", {}).get("activation", "gelu"),
        quantum_device=quant_cfg.get("model", {}).get("quantum_device", "default.qubit"),
        classical_device=device,
        in_channels=quant_cfg.get("model", {}).get("in_channels", 1),
        include_correlators=quant_cfg.get("model", {}).get("include_correlators", False),
        override_quantum_out_channels=cache_meta.get("out_channels") if cache_meta else None
    )
    model_quant.bypass_quantum = True
    model_quant.load_state_dict(checkpoint_quant["model_state_dict"])
    model_quant.to(device)

    # --- PHASE 2: EXTRACT FEATURES ---
    print(f"\nPHASE 2: Extracting Features ({label_prefix})")
    
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(class_cfg)
    
    # Rebuild loaders with shuffle=False
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, shuffle=False, num_workers=0)
    
    q_train_loader = torch.utils.data.DataLoader(q_train_loader.dataset, batch_size=q_train_loader.batch_size, shuffle=False, num_workers=0)
    q_test_loader = torch.utils.data.DataLoader(q_test_loader.dataset, batch_size=q_test_loader.batch_size, shuffle=False, num_workers=0)
    
    # Raw Pixels
    X_train_raw, y_train_raw = extract_features(model_class, train_loader, device, extract_point="raw", is_raw=True)
    X_test_raw, y_test_raw = extract_features(model_class, test_loader, device, extract_point="raw", is_raw=True)
    
    # Check alignment
    _, y_train_quant_check = extract_features(model_quant, q_train_loader, device, extract_point="post_conv", is_quantum_model=True)
    assert np.array_equal(y_train_raw, y_train_quant_check), "Train labels do not match between classical and quantum loaders!"
    
    # --- AFTER KERNEL ---
    X_train_cr_k, y_train_cr_k = extract_features(model_rand, train_loader, device, extract_point="post_conv")
    X_test_cr_k, y_test_cr_k = extract_features(model_rand, test_loader, device, extract_point="post_conv")
    
    X_train_q_k, y_train_q_k = extract_features(model_quant, q_train_loader, device, extract_point="post_conv", is_quantum_model=True)
    X_test_q_k, y_test_q_k = extract_features(model_quant, q_test_loader, device, extract_point="post_conv", is_quantum_model=True)
    
    X_train_ct_k, y_train_ct_k = extract_features(model_class, train_loader, device, extract_point="post_conv")
    X_test_ct_k, y_test_ct_k = extract_features(model_class, test_loader, device, extract_point="post_conv")
    
    # --- PRE DENSE ---
    X_train_cr_d, y_train_cr_d = extract_features(model_rand, train_loader, device, extract_point="pre_dense")
    X_test_cr_d, y_test_cr_d = extract_features(model_rand, test_loader, device, extract_point="pre_dense")
    
    X_train_q_d, y_train_q_d = extract_features(model_quant, q_train_loader, device, extract_point="pre_dense", is_quantum_model=True)
    X_test_q_d, y_test_q_d = extract_features(model_quant, q_test_loader, device, extract_point="pre_dense", is_quantum_model=True)
    
    X_train_ct_d, y_train_ct_d = extract_features(model_class, train_loader, device, extract_point="pre_dense")
    X_test_ct_d, y_test_ct_d = extract_features(model_class, test_loader, device, extract_point="pre_dense")

    # --- PHASE 3: EVALUATE ---
    print(f"\nPHASE 3: Feature Probing ({label_prefix})")
    pipeline_results = []
    
    def _eval(x_tr, y_tr, x_te, y_te, name):
        acc_rf, auc_rf, acc_svm, auc_svm = train_and_evaluate_models(x_tr, y_tr, x_te, y_te, seed, name)
        pipeline_results.append([name, x_tr.shape[1], f"{acc_rf:.4f}", f"{auc_rf:.4f}", f"{acc_svm:.4f}", f"{auc_svm:.4f}"])
        
    _eval(X_train_raw, y_train_raw, X_test_raw, y_test_raw, "Raw Pixels")
    
    _eval(X_train_cr_k, y_train_cr_k, X_test_cr_k, y_test_cr_k, f"{label_prefix} Classical Random (After Kernel)")
    _eval(X_train_q_k, y_train_q_k, X_test_q_k, y_test_q_k, f"{label_prefix} Quantum (Fixed)")
    _eval(X_train_ct_k, y_train_ct_k, X_test_ct_k, y_test_ct_k, f"{label_prefix} Classical Trainable (After Kernel)")
    
    _eval(X_train_cr_d, y_train_cr_d, X_test_cr_d, y_test_cr_d, f"{label_prefix} Classical Random (Pre-Dense)")
    _eval(X_train_q_d, y_train_q_d, X_test_q_d, y_test_q_d, f"{label_prefix} Quantum (Pre-Dense)")
    _eval(X_train_ct_d, y_train_ct_d, X_test_ct_d, y_test_ct_d, f"{label_prefix} Classical Trainable (Pre-Dense)")
    
    return pipeline_results

def main():
    parser = argparse.ArgumentParser(description="Feature Probing Ablation Study")
    parser.add_argument("--classical-1kern", type=str, default=None, help="Path to 1-kernel classical baseline config")
    parser.add_argument("--quantum-1kern", type=str, default=None, help="Path to 1-kernel quantum config")
    parser.add_argument("--classical-4kern", type=str, default=None, help="Path to 4-kernel classical baseline config")
    parser.add_argument("--quantum-4kern", type=str, default=None, help="Path to 4-kernel quantum config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs/feature_probing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    # Process 1-Kernel Pipeline
    if args.classical_1kern and args.quantum_1kern:
        out_dir = os.path.join(args.output_dir, "1kern")
        res = process_pipeline(args.classical_1kern, args.quantum_1kern, args.seed, out_dir, "1-Kernel")
        all_results.extend(res)

    # Process 4-Kernel Pipeline
    if args.classical_4kern and args.quantum_4kern:
        out_dir = os.path.join(args.output_dir, "4kern")
        res = process_pipeline(args.classical_4kern, args.quantum_4kern, args.seed, out_dir, "4-Kernel")
        # To avoid duplicating Raw Pixels in the final table if both are run
        if args.classical_1kern and args.quantum_1kern:
            res = res[1:] 
        all_results.extend(res)

    # Final Report
    if all_results:
        print("\n" + "="*110)
        print("FINAL RESULTS: FEATURE PROBING ABLATION (THE PERFECT SYMMETRY)")
        print("="*110)
        headers = ["Feature Source", "Dimensions", "RF Acc", "RF AUC", "SVM Acc", "SVM AUC"]
        print(tabulate(all_results, headers=headers, tablefmt="grid"))
    else:
        print("No configurations provided! Please pass --classical-1kern / --quantum-1kern and/or their 4-kernel counterparts.")

if __name__ == "__main__":
    main()
