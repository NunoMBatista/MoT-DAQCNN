"""Utility to quickly view experiment results from robust_test.py outputs.

Usage:
    python src/utils/view_results.py outputs/run_YYYYMMDD_HHMMSS
"""

import argparse
import json
import os
import sys

import numpy as np
from plotting import plot_confusion_matrix


def print_separator(char="=", length=70):
    print(char * length)


def view_results(output_dir):
    """View results from an experiment run."""
    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' does not exist")
        sys.exit(1)

    print_separator()
    print(f"Experiment Results: {output_dir}")
    print_separator()
    print()

    # Load and display config
    config_path = os.path.join(output_dir, "config.yml")
    if os.path.exists(config_path):
        print("Configuration:")
        print_separator("-")
        with open(config_path, "r") as f:
            print(f.read())
        print()

    # Load aggregate metrics
    aggregate_path = os.path.join(output_dir, "aggregate_metrics.json")
    if os.path.exists(aggregate_path):
        with open(aggregate_path, "r") as f:
            aggregate = json.load(f)

        print_separator()
        print("Aggregate Metrics (mean ± std)")
        print_separator()

        for metric_name, stats in aggregate.items():
            # Skip model_metadata (displayed separately)
            if metric_name == "model_metadata":
                continue
            print(f"{metric_name:25s}: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"{'':25s}  [min: {stats['min']:.6f}, max: {stats['max']:.6f}]")
        print()

        # Display model metadata if available
        if "model_metadata" in aggregate:
            metadata = aggregate["model_metadata"]
            print_separator()
            print("Model Architecture")
            print_separator()
            print(f"Total parameters:      {metadata.get('total_params', 'N/A'):,}")
            print(f"Trainable parameters:  {metadata.get('trainable_params', 'N/A'):,}")

            # Display quantum kernel information
            kernel_size = metadata.get("quantum_kernel_size")
            kernel_topologies = metadata.get("quantum_kernel_topologies", [])
            if kernel_size is not None:
                print(f"Quantum kernel size:   {kernel_size}×{kernel_size}")
            if kernel_topologies:
                topologies_str = ", ".join(kernel_topologies)
                print(f"Kernel topologies:     {topologies_str}")

            head_structure = metadata.get("head_structure", [])
            if head_structure:
                conv_layers = [l for l in head_structure if l["type"] == "Conv2d"]
                linear_layers = [
                    l for l in head_structure if l["type"] in ["Linear", "LazyLinear"]
                ]
                print(f"CNN layers:            {len(conv_layers)}")
                print(f"FC layers:             {len(linear_layers)}")
                print(f"\nCNN Head Structure ({len(head_structure)} layers):")
                for layer in head_structure:
                    layer_type = layer["type"]
                    if layer_type == "Conv2d":
                        print(
                            f"  [{layer['index']}] Conv2d: {layer['in_channels']}→{layer['out_channels']}, kernel={layer['kernel_size']}, stride={layer['stride']}"
                        )
                    elif layer_type == "Linear":
                        print(
                            f"  [{layer['index']}] Linear: {layer['in_features']}→{layer['out_features']}"
                        )
                    elif layer_type == "LazyLinear":
                        print(
                            f"  [{layer['index']}] LazyLinear: ?→{layer['out_features']}"
                        )
                    elif layer_type == "BatchNorm2d":
                        print(
                            f"  [{layer['index']}] BatchNorm2d: {layer['num_features']} features"
                        )
                    elif layer_type == "Dropout":
                        print(f"  [{layer['index']}] Dropout: p={layer['p']}")
                    elif layer_type == "MaxPool2d":
                        print(
                            f"  [{layer['index']}] MaxPool2d: kernel={layer['kernel_size']}, stride={layer['stride']}"
                        )
                    else:
                        print(f"  [{layer['index']}] {layer_type}")
            print()

    # Load individual results
    individual_path = os.path.join(output_dir, "individual_results.json")
    if os.path.exists(individual_path):
        with open(individual_path, "r") as f:
            individual = json.load(f)

        print_separator()
        print(f"Individual Results (n={len(individual)} runs)")
        print_separator()

        for i, result in enumerate(individual, 1):
            seed = result.get("seed", "N/A")
            test_acc = result.get("test_acc", 0)
            test_loss = result.get("test_loss", 0)
            test_auc = result.get("test_auc", 0)
            test_f1 = result.get("test_f1", 0)
            test_recall = result.get("test_recall", 0)
            print(
                f"Run {i} (seed={seed}): acc={test_acc:.6f}, loss={test_loss:.6f}, "
                f"auc={test_auc:.6f}, f1={test_f1:.6f}, recall={test_recall:.6f}"
            )
        print()

        # Plot confusion matrices if available
        print_separator()
        print("Generating Confusion Matrix Plots...")
        print_separator()

        for result in individual:
            seed = result.get("seed", "N/A")
            cm = result.get("test_confusion_matrix")
            num_classes = result.get("num_classes", 2)

            if cm is not None:
                cm_array = np.array(cm)
                cm_plot_path = os.path.join(
                    output_dir, f"confusion_matrix_seed_{seed}.png"
                )
                plot_confusion_matrix(cm_array, cm_plot_path, num_classes=num_classes)
                print(f"  - Generated: confusion_matrix_seed_{seed}.png")
        print()

    # List available plots
    plots = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if plots:
        print_separator()
        print("Available Plots:")
        print_separator()
        for plot in sorted(plots):
            print(f"  - {plot}")
        print()

    print_separator()
    print("Summary Complete")
    print_separator()


def main():
    parser = argparse.ArgumentParser(
        description="View experiment results from robust_test.py"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory (e.g., outputs/run_YYYYMMDD_HHMMSS)",
    )
    args = parser.parse_args()

    view_results(args.output_dir)


if __name__ == "__main__":
    main()
