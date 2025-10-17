# main_file.py (Corrected Version)

import torch
import pandas as pd
from config import *
from utils.models import load_pretrained_model
from utils.data import get_dataloaders
from utils.profiler import evaluate_and_profile, get_macs, get_model_size_mb


def main():
    """Main function to execute the entire Task 0 workflow."""

    # --- Task 0(a): Load Pretrained Models ---
    print("--- Starting Task 0(a): Loading Pretrained Models ---")
    print(f"Using device: {DEVICE}\n")

    model_cifar10 = load_pretrained_model(MODEL_CIFAR10, DEVICE)
    model_cifar100 = load_pretrained_model(MODEL_CIFAR100, DEVICE)
    print("-" * 50)

    # --- Task 0(b) & 0(c): Load Datasets and Build Dataloaders ---
    print("\n--- Starting Task 0(b) & 0(c): Loading Datasets and Dataloaders ---")

    # --- THIS SECTION IS CORRECTED ---
    train_loader_cifar10, test_loader_cifar10 = get_dataloaders(
        "cifar10", DATA_DIR, BATCH_SIZE, NUM_WORKERS, CIFAR10_STATS
    )
    print("-" * 50)
    train_loader_cifar100, test_loader_cifar100 = get_dataloaders(
        "cifar100", DATA_DIR, BATCH_SIZE, NUM_WORKERS, CIFAR100_STATS
    )
    # --- END OF CORRECTION ---
    print("-" * 50)

    # ... The rest of the file is the same ...
    print("\n--- Starting Task 0(d) & 0(e): Profiling and Accuracy Verification ---")

    results_data = []
    all_models_info = [
        {
            "dataset": "CIFAR-10",
            "model": model_cifar10,
            "train_loader": train_loader_cifar10,
            "test_loader": test_loader_cifar10,
        },
        {
            "dataset": "CIFAR-100",
            "model": model_cifar100,
            "train_loader": train_loader_cifar100,
            "test_loader": test_loader_cifar100,
        },
    ]

    for item in all_models_info:
        model, dataset = item["model"], item["dataset"]
        print(f"\n--- Profiling VGG16-BN on {dataset} ---")

        model_size = get_model_size_mb(model)
        macs = get_macs(model, (3, 32, 32), DEVICE)

        test_metrics = evaluate_and_profile(
            model=model,
            loader=item["test_loader"],
            device=DEVICE,
            num_profiling_batches=NUM_PROFILING_BATCHES,
            description=f"Profiling {dataset} Test Set",
        )
        train_metrics = evaluate_and_profile(
            model=model,
            loader=item["train_loader"],
            device=DEVICE,
            num_profiling_batches=0,
            description=f"Evaluating {dataset} Train Set",
        )

        results_data.append(
            {
                "Model": "VGG16-BN",
                "Dataset": dataset,
                "Model Size (MB)": model_size,
                "MACs (G)": macs / 1e9,
                "Peak Memory (MB)": test_metrics["peak_memory_mb"],
                "Latency (ms/batch)": test_metrics["avg_latency_ms"],
                "Energy (mJ/batch)": test_metrics["avg_energy_mj"],
                "Test Top-1 Acc (%)": test_metrics["top1_acc"],
                "Test Top-5 Acc (%)": test_metrics["top5_acc"],
                "Train Top-1 Acc (%)": train_metrics["top1_acc"],
            }
        )

    print("\nProfiling and evaluation complete.")
    print("\n\n--- Baseline Profiling and Accuracy Results ---")

    results_df = pd.DataFrame(results_data)
    pd.options.display.float_format = "{:.2f}".format
    print(results_df.to_string())

    report_filename = "task0_baseline_results.md"
    with open(report_filename, "w") as f:
        f.write("# Task 0: Baseline Profiling Results\n\n")
        f.write(results_df.to_markdown(index=False))

    print(f"\nResults have been successfully saved to '{report_filename}'")


if __name__ == "__main__":
    main()