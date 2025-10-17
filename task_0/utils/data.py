# utils/data.py

from typing import Tuple
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


def get_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    stats: dict,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns the training and test dataloaders for a given dataset.
    """
    print(f"Loading '{dataset_name.upper()}' dataset...")

    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=stats["mean"], std=stats["std"]),
        ]
    )

    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=stats["mean"], std=stats["std"]),
        ]
    )

    dataset_class = {"cifar10": CIFAR10, "cifar100": CIFAR100}.get(dataset_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_dataset = dataset_class(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = dataset_class(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader
