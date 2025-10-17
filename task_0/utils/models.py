# utils/models.py

import torch
import torch.nn as nn

# Suppress verbose output from the model hub
import logging

logging.getLogger("torch.hub").setLevel(logging.ERROR)


def load_pretrained_model(model_name: str, device: torch.device) -> nn.Module:
    """
    Loads a pretrained VGG16-BN model for CIFAR10/100 from the specified hub.

    Args:
        model_name (str): The name of the model to load (e.g., 'cifar10_vgg16_bn').
        device (torch.device): The device to move the model to (CPU or CUDA).

    Returns:
        nn.Module: The loaded pretrained model, moved to the specified device
                   and set to evaluation mode.
    """
    print(f"Loading pretrained model '{model_name}'...")
    try:
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", model_name, pretrained=True
        )
        model.to(device)
        model.eval()
        print(f"Model '{model_name}' loaded successfully and moved to {device}.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None
