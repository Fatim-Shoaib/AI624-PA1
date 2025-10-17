# utils/profiler.py (Final Working Version)

import io
import time
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from tqdm.auto import tqdm
from pyJoules.energy_meter import EnergyContext
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.nvidia_device import NvidiaGPUDomain


def get_model_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 * 1024)
    return size_mb


def get_macs(model: nn.Module, input_res: tuple, device: torch.device) -> float:
    dummy_input = torch.randn(1, *input_res).to(device)
    macs = profile_macs(model, dummy_input)
    return macs


def profile_performance(model, loader, device, num_profiling_batches):
    """
    A dedicated function to measure latency and energy over multiple batches.
    This ensures the measurement duration is long enough for pyJoules.
    """
    latencies = []
    total_energy_uj = 0
    temp_energy_file = "temp_energy_profile.csv"
    energy_csv_handler = CSVHandler(temp_energy_file)
    domains_to_monitor = [NvidiaGPUDomain(0)] if torch.cuda.is_available() else []

    # Get a fixed set of data for profiling
    profiling_data = [next(iter(loader)) for _ in range(num_profiling_batches)]

    try:
        # --- THIS IS THE KEY: Measure the entire profiling loop in one go ---
        with EnergyContext(
            handler=energy_csv_handler,
            domains=domains_to_monitor,
            start_tag="profiling_run",
        ) as ctx:
            for images, labels in profiling_data:
                images, labels = images.to(device), labels.to(device)

                start_time = time.time()
                _ = model(images)
                torch.cuda.synchronize()
                end_time = time.time()

                latencies.append((end_time - start_time) * 1000)
    except Exception as e:
        print(f"\nCould not profile energy. Error: {e}\n")
        # Rerun latency measurement without energy context if it fails
        latencies = []
        for images, labels in profiling_data:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            _ = model(images)
            torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)

    # --- Calculate average energy from the single, total measurement ---
    avg_energy_mj = 0
    if os.path.exists(temp_energy_file):
        energy_csv_handler.save_data()
        try:
            energy_df = pd.read_csv(temp_energy_file, sep=";")
            if not energy_df.empty:
                total_energy_uj = energy_df.iloc[0, 4:].sum()
                avg_energy_mj = (total_energy_uj / num_profiling_batches) / 1000
        except pd.errors.EmptyDataError:
            pass
        os.remove(temp_energy_file)

    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
    return avg_latency_ms, avg_energy_mj


def evaluate_accuracy(model, loader, device, description):
    """A dedicated function to evaluate only accuracy."""
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    progress_bar = tqdm(loader, desc=description)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            total += labels.size(0)
            correct_top1 += (predicted_top5[:, 0] == labels).sum().item()
            correct_top5 += (predicted_top5 == labels.view(-1, 1)).sum().item()
            progress_bar.set_postfix(
                {"Top-1 Acc": f"{100 * correct_top1 / total:.2f}%"}
            )

    return 100 * correct_top1 / total, 100 * correct_top5 / total


def evaluate_and_profile(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_profiling_batches: int,
    description: str = "Evaluating",
) -> dict:
    """
    Main function that separates profiling from accuracy evaluation.
    """
    avg_latency_ms, avg_energy_mj = 0, 0
    if num_profiling_batches > 0:
        avg_latency_ms, avg_energy_mj = profile_performance(
            model, loader, device, num_profiling_batches
        )

    top1_acc, top5_acc = evaluate_accuracy(model, loader, device, description)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if torch.cuda.is_available()
        else 0
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "avg_latency_ms": avg_latency_ms,
        "peak_memory_mb": peak_memory_mb,
        "avg_energy_mj": avg_energy_mj,
    }
