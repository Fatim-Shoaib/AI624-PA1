# new_root/task_1/finetune_pruned_model.py

import sys
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm
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
)
from task_0.utils.models import load_pretrained_model
from task_0.utils.data import get_dataloaders
from task_0.utils.profiler import evaluate_accuracy

# --- Fine-tuning Configuration ---
NUM_FINETUNE_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ==============================================================================
# --- STEP 2: DEFINE YOUR SPARSITY RECIPE ---
# TODO: YOU MUST REPLACE THESE VALUES with the ones you decided on from your
#       sensitivity analysis results (`sensitivity_results.csv`).
#
# Guideline for choosing values:
# - Layers that showed a sharp drop in accuracy are SENSITIVE. Use a LOW sparsity (e.g., 0.1 - 0.4).
# - Layers that maintained high accuracy are ROBUST. Use a HIGH sparsity (e.g., 0.7 - 0.9).
# - The first and last layers are often the most sensitive.
# ==============================================================================
SPARSITY_DICT = {
    "features.0": 0.3,  # First conv layer, often sensitive
    "features.3": 0.6,
    "features.7": 0.7,
    "features.10": 0.7,
    "features.14": 0.8,
    "features.17": 0.8,
    "features.20": 0.8,
    "features.24": 0.9,  # Deeper conv layers, often more robust
    "features.27": 0.9,
    "features.30": 0.9,
    "features.34": 0.9,
    "features.37": 0.9,
    "features.40": 0.9,
    "classifier.0": 0.9,  # First linear layer can be pruned heavily
    "classifier.3": 0.9,
    "classifier.6": 0.4,  # Final linear layer, very sensitive
}


def apply_layerwise_pruning(model: nn.Module, sparsity_dict: dict):
    """Applies l1 unstructured pruning to specified layers of the model."""
    print("Applying layer-wise unstructured pruning...")
    for name, module in model.named_modules():
        if name in sparsity_dict:
            prune.l1_unstructured(module, name="weight", amount=sparsity_dict[name])
    print("Pruning hooks applied.")


def make_pruning_permanent(model: nn.Module):
    """Removes pruning re-parameterization from the model."""
    print("Making pruning permanent by removing hooks...")
    for name, module in model.named_modules():
        if prune.is_pruned(module):
            prune.remove(module, "weight")
    print("Pruning made permanent.")


def calculate_sparsity(model: nn.Module):
    """Calculates and prints the sparsity of each layer and the entire model."""
    total_zeros = 0
    total_elements = 0
    print("\n--- Model Sparsity Report ---")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_zeros = torch.sum(module.weight == 0).item()
            layer_elements = module.weight.numel()
            layer_sparsity = 100.0 * layer_zeros / layer_elements
            print(f"- Layer '{name}': {layer_sparsity:.2f}% sparse")
            total_zeros += layer_zeros
            total_elements += layer_elements

    total_sparsity = 100.0 * total_zeros / total_elements
    print("-" * 30)
    print(f"Overall Model Sparsity: {total_sparsity:.2f}%")
    print("-" * 30)
    return total_sparsity


def main():
    """Main function to prune and fine-tune the model."""

    # --- Load Model and Data ---
    print("--- Loading baseline model and data ---")
    model = load_pretrained_model(MODEL_CIFAR10, DEVICE)
    train_loader, test_loader = get_dataloaders(
        "cifar10", DATA_DIR, BATCH_SIZE, NUM_WORKERS, CIFAR10_STATS
    )

    # --- Baseline Accuracy ---
    baseline_acc, _ = evaluate_accuracy(
        model, test_loader, DEVICE, "Evaluating Baseline"
    )
    print(f"\nBaseline Model Top-1 Accuracy: {baseline_acc:.2f}%\n")

    # --- Apply Pruning ---
    apply_layerwise_pruning(model, SPARSITY_DICT)

    # --- Check Accuracy Immediately After Pruning (Before Fine-tuning) ---
    acc_before_finetune, _ = evaluate_accuracy(
        model, test_loader, DEVICE, "Eval After Pruning"
    )
    print(f"\nAccuracy After Pruning (Before Fine-tuning): {acc_before_finetune:.2f}%")
    calculate_sparsity(model)

    # --- Step 3: Fine-tune the Pruned Model ---
    print("\n--- Starting fine-tuning process ---")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(NUM_FINETUNE_EPOCHS):
        model.train()
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_FINETUNE_EPOCHS} [Training]"
        )
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # NOTE: We DO NOT need to manually re-apply the mask. The hook
            # attached by `prune.l1_unstructured` handles this automatically.

        # Evaluate after each epoch
        val_acc, _ = evaluate_accuracy(model, test_loader, DEVICE, "Epoch Validation")
        print(f"Epoch {epoch+1} | Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_pruned_model.pth")

    print(f"Fine-tuning complete. Best validation accuracy: {best_acc:.2f}%")

    # --- Step 4: Finalize the Pruned Model ---
    print("\n--- Finalizing the pruned model ---")
    # Load the best performing model
    model.load_state_dict(torch.load("best_pruned_model.pth"))

    # Make the pruning permanent
    make_pruning_permanent(model)

    # --- Final Verification ---
    print("\nVerifying final pruned model...")
    final_acc, _ = evaluate_accuracy(model, test_loader, DEVICE, "Final Evaluation")
    final_sparsity = calculate_sparsity(model)

    print(f"\nFinal Pruned Model Accuracy: {final_acc:.2f}%")
    print(f"Final Pruned Model Sparsity: {final_sparsity:.2f}%")

    # Save the final, permanent state_dict
    final_model_path = "vgg16_cifar10_pruned_permanent.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal permanently pruned model saved to '{final_model_path}'")


if __name__ == "__main__":
    main()
