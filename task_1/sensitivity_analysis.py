# new_root/task_1/sensitivity_analysis.py

import sys
import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# --- Add the parent directory (new_root) to the Python path ---
# This allows us to import from the 'task_0' directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Imports from Task 0 ---
from task_0.config import DEVICE, MODEL_CIFAR10, DATA_DIR, BATCH_SIZE, NUM_WORKERS, CIFAR10_STATS
from task_0.utils.models import load_pretrained_model
from task_0.utils.data import get_dataloaders
from task_0.utils.profiler import evaluate_accuracy


def prune_layer_by_magnitude(layer: nn.Module, sparsity_percentage: float):
    """
    Prunes a single layer's weight tensor by setting the weights with the 
    lowest absolute magnitude to zero. This function modifies the layer in-place.
    
    Args:
        layer (nn.Module): The layer to be pruned (e.g., Conv2d or Linear).
        sparsity_percentage (float): The percentage of weights to prune (0.0 to 100.0).
    """
    # Ensure we are only pruning layers with a 'weight' attribute
    if not hasattr(layer, 'weight'):
        return

    weights = layer.weight.data
    num_elements = weights.numel()
    
    # Calculate the number of weights to set to zero
    num_to_prune = int(round(sparsity_percentage / 100.0 * num_elements))
    
    if num_to_prune == 0:
        return
        
    # Find the threshold for pruning. We use the k-th smallest absolute value.
    # The tensor must be flattened to use kthvalue
    threshold = torch.kthvalue(weights.abs().view(-1), num_to_prune).values
    
    # Create a mask where 1s indicate weights to keep
    mask = weights.abs() > threshold
    
    # Apply the mask to zero out the low-magnitude weights
    layer.weight.data.mul_(mask.float())


def perform_sensitivity_analysis(model, test_loader, device):
    """
    Performs sensitivity analysis on the model by individually pruning each
    target layer at different sparsity levels and evaluating the impact on accuracy.

    Returns:
        pd.DataFrame: A DataFrame containing accuracies for each layer at each sparsity level.
    """
    # Sparsity levels to test for each layer
    sparsity_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    # Identify target layers (Conv2d and Linear) for pruning
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            target_layers.append(name)

    print(f"Found {len(target_layers)} target layers for sensitivity analysis.")
    
    # Store results: { "layer_name": [acc_at_sparsity_0, acc_at_sparsity_1, ...], ... }
    results = {}

    # Main analysis loop
    for layer_name in tqdm(target_layers, desc="Analyzing Layers"):
        layer_accuracies = []
        for sparsity in tqdm(sparsity_levels, desc=f"Sparsity for {layer_name}", leave=False):
            # 1. CRITICAL: Create a deep copy of the original model for each run to ensure isolation.
            model_copy = copy.deepcopy(model).to(device)
            
            # 2. Get the specific layer from the copied model by its name
            # This logic navigates the model's module hierarchy
            submodule = model_copy
            for part in layer_name.split('.'):
                submodule = getattr(submodule, part)

            # 3. Prune the single, targeted layer in the copied model
            prune_layer_by_magnitude(submodule, sparsity)

            # 4. Evaluate the accuracy of the model with one layer pruned
            # The progress bar for evaluation is disabled to keep the output clean
            top1_acc, _ = evaluate_accuracy(
                model=model_copy,
                loader=test_loader,
                device=device,
                description=f"Eval {layer_name} @ {sparsity}%" # This description won't show
            )
            layer_accuracies.append(top1_acc)

        results[layer_name] = layer_accuracies

    # Create a pandas DataFrame for easy plotting and saving
    results_df = pd.DataFrame(results, index=sparsity_levels)
    results_df.index.name = "Sparsity (%)"
    return results_df


def plot_sensitivity_curves(results_df: pd.DataFrame, save_path: str):
    """
    Plots the sensitivity curves from the results DataFrame and saves the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # The plot() method of a DataFrame automatically creates a legend
    results_df.plot(ax=ax, marker='o', linestyle='-')
    
    ax.set_title("VGG16-BN Layer Sensitivity to Pruning on CIFAR-10", fontsize=18, weight='bold')
    ax.set_xlabel("Sparsity Level of Single Layer (%)", fontsize=14)
    ax.set_ylabel("Model Top-1 Accuracy (%)", fontsize=14)
    ax.legend(title="Layer Name", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adjust layout to prevent the legend from being cut off
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    
    # Ensure the plot directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nSensitivity plot saved to '{save_path}'")
    plt.show()


def main():
    """Main function to orchestrate the sensitivity analysis workflow."""
    print("--- Starting Task 1(a), Step 1: Sensitivity Analysis ---")
    
    # --- Load Pre-trained Model and Test Data ---
    print(f"Using device: {DEVICE}")
    baseline_model = load_pretrained_model(MODEL_CIFAR10, DEVICE)
    
    _, test_loader = get_dataloaders(
        "cifar10", DATA_DIR, BATCH_SIZE, NUM_WORKERS, CIFAR10_STATS
    )
    
    # --- Perform the Core Analysis ---
    results_df = perform_sensitivity_analysis(baseline_model, test_loader, DEVICE)
    
    # --- Print and Save Numerical Results ---
    print("\n--- Sensitivity Analysis Results ---")
    # Display the results table in the console
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df.round(2))
    
    # Save the DataFrame to a CSV for future use (e.g., for step 2)
    results_df.to_csv("sensitivity_results.csv")
    print("\nResults table saved to 'sensitivity_results.csv'")

    # --- Plot the Curves ---
    plot_sensitivity_curves(results_df, "plots/sensitivity_curves.png")


if __name__ == "__main__":
    main()