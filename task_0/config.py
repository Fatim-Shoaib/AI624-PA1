# config.py

import torch

# --- General ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"  # Directory to save datasets
BATCH_SIZE = 256
NUM_WORKERS = 2  # Number of parallel workers for data loading
NUM_PROFILING_BATCHES = 20  # Number of batches for latency/energy profiling

# --- Model Names ---
MODEL_CIFAR10 = "cifar10_vgg16_bn"
MODEL_CIFAR100 = "cifar100_vgg16_bn"

# --- Dataset Statistics ---
# Found in the chenyaofo/pytorch-cifar-models repository
CIFAR10_STATS = {
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.2023, 0.1994, 0.2010],
}

CIFAR100_STATS = {
    "mean": [0.5071, 0.4867, 0.4408],
    "std": [0.2675, 0.2565, 0.2761],
}
