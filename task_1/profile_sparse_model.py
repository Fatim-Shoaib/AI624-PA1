# new_root/task_1/profile_sparse_model.py

import sys
import os
import torch
import torch.nn as nn
import pandas as pd

# --- Add parent directory to Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Imports from Task 0 ---
from task_0.config import (
    DEVICE,
    MODEL_CIFAR10,
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    CIFAR10_STATS,
    NUM_PROFILING_BATCHES,
)
from task_0.utils.data import get_dataloaders
from task_0.utils.profiler import evaluate_and_profile, get_macs
from task_0.utils.models import load_pretrained_model

# --- Imports from this task's previous step ---
# We need these to correctly build the model architecture before loading the state_dict
try:
    from create_sparse_model import SparseConv2d, SparseLinear, convert_model_to_sparse
except ImportError:
    print("Error: `create_sparse_model.py` not found.")
    print("Please ensure it is in the same directory (`task_1/`).")
    sys.exit(1)


def get_effective_model_size_mb(model: nn.Module) -> float:
    """
    Calculates the effective model size by counting only non-zero parameters.
    This correctly handles the size of our custom sparse modules.
    """
    total_params = 0
    for module in model.modules():
        # For our custom sparse layers, count the stored non-zero values
        if isinstance(module, (SparseConv2d, SparseLinear)):
            if hasattr(module, "weight_sparse"):
                total_params += module.weight_sparse._values().numel()
            if module.bias is not None:
                total_params += module.bias.numel()
        # For standard layers like BatchNorm that were not pruned
        elif isinstance(module, (nn.BatchNorm2d)):
            total_params += module.weight.numel()
            total_params += module.bias.numel()

    # Assuming float32 (4 bytes per parameter)
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb


def main():
    """Main function to profile the sparse COO model."""

    # --- Load the Sparse COO Model ---
    sparse_model_path = "vgg16_cifar10_sparse_coo.pth"
    if not os.path.exists(sparse_model_path):
        print(f"Error: Sparse model file not found at '{sparse_model_path}'")
        print("Please run `create_sparse_model.py` first.")
        return

    print(f"--- Loading sparse COO model from '{sparse_model_path}' ---")

    # 1. Create the base VGG16 architecture
    sparse_model = load_pretrained_model(MODEL_CIFAR10, DEVICE)

    # 2. Convert its layers to our sparse-aware versions
    convert_model_to_sparse(sparse_model)

    # 3. Now, load the state_dict which contains the sparse weight buffers
    sparse_model.load_state_dict(torch.load(sparse_model_path, map_location=DEVICE))
    sparse_model.eval()
    print("Sparse model loaded and configured successfully.\n")

    # --- Load Data ---
    _, test_loader = get_dataloaders(
        "cifar10", DATA_DIR, BATCH_SIZE, NUM_WORKERS, CIFAR10_STATS
    )

    # --- Profile the Sparse Model ---
    print("\n--- Profiling Sparse Model Performance ---")

    # Calculate effective model size (non-zero parameters)
    effective_size_mb = get_effective_model_size_mb(sparse_model)

    # Calculate MACs using torchprofile
    # Note: This will report the dense MACs because SparseConv2d converts to dense.
    # This is an important point for the final analysis/report.
    macs_g = get_macs(sparse_model, (3, 32, 32), DEVICE) / 1e9

    # Profile latency, memory, energy, and accuracy
    metrics = evaluate_and_profile(
        model=sparse_model,
        loader=test_loader,
        device=DEVICE,
        num_profiling_batches=NUM_PROFILING_BATCHES,
        description="Profiling Sparse Model",
    )

    sparse_results = {
        "Phase": "Magnitude Pruned (Part a)",
        "Model Size (MB)": effective_size_mb,
        "MACs (G)": macs_g,
        "Peak Memory (MB)": metrics["peak_memory_mb"],
        "Latency (ms/batch)": metrics["avg_latency_ms"],
        "Energy (mJ/batch)": metrics["avg_energy_mj"],
        "Test Top-1 Acc (%)": metrics["top1_acc"],
        "Test Top-5 Acc (%)": metrics["top5_acc"],
    }

    # --- Load Baseline Results for Comparison ---
    baseline_results_path = "../task_0/task0_baseline_results.md"
    baseline_df = pd.DataFrame()
    try:
        # Read the markdown table from Task 0 results
        # We skip the first 2 rows (header and separator) and use the 3rd as header
        df_list = pd.read_html(baseline_results_path, header=0)
        baseline_df = df_list[0]
        # Filter for CIFAR-10
        baseline_cifar10 = (
            baseline_df[baseline_df["Dataset"] == "CIFAR-10"].iloc[0].to_dict()
        )

        baseline_results = {
            "Phase": "Baseline (Task 0)",
            "Model Size (MB)": baseline_cifar10.get("Model Size (MB)"),
            "MACs (G)": baseline_cifar10.get("MACs (G)"),
            "Peak Memory (MB)": baseline_cifar10.get("Peak Memory (MB)"),
            "Latency (ms/batch)": baseline_cifar10.get("Latency (ms/batch)"),
            "Energy (mJ/batch)": baseline_cifar10.get("Energy (mJ/batch)"),
            "Test Top-1 Acc (%)": baseline_cifar10.get("Test Top-1 Acc (%)"),
            "Test Top-5 Acc (%)": baseline_cifar10.get("Test Top-5 Acc (%)"),
        }
    except Exception as e:
        print(
            f"\nCould not load baseline results for comparison from '{baseline_results_path}': {e}"
        )
        baseline_results = {
            "Phase": "Baseline (Task 0)"
        }  # Empty dict if file not found

    # --- Display and Save Final Report ---
    results_df = pd.DataFrame([baseline_results, sparse_results])

    print("\n\n--- Comparison: Baseline vs. Magnitude Pruning ---")
    print(results_df.round(2).to_string(index=False))

    report_filename = "task1a_magnitude_pruning_results.md"
    with open(report_filename, "w") as f:
        f.write("# Task 1(a): Magnitude-Based Pruning Results\n\n")
        f.write(
            "This table compares the performance of the baseline VGG16-BN model against the fine-tuned, magnitude-pruned sparse model.\n\n"
        )
        f.write(results_df.to_markdown(index=False))

    print(f"\nResults have been successfully saved to '{report_filename}'")


if __name__ == "__main__":
    main()
