# new_root/task_1/create_sparse_model.py

import sys
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Add parent directory to Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Imports from Task 0 ---
from task_0.config import DEVICE, MODEL_CIFAR10
from task_0.utils.models import load_pretrained_model


def dense_to_coo(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a dense tensor to a sparse COO tensor.

    Args:
        tensor (torch.Tensor): The dense input tensor.

    Returns:
        torch.Tensor: The sparse COO representation of the input tensor.
    """
    # Get the indices of non-zero elements.
    # .t() is required because nonzero() returns (nnz, dims) and sparse_coo_tensor expects (dims, nnz)
    indices = torch.nonzero(tensor).t()

    # Get the values of the non-zero elements
    values = tensor[tensor.nonzero(as_tuple=True)]

    # Create the sparse COO tensor
    return torch.sparse_coo_tensor(indices, values, tensor.size())


def verify_coo_conversion(original_tensor: torch.Tensor):
    """
    (Step 6) Verifies that the COO conversion is correct by converting a tensor
    to sparse and back to dense, ensuring it matches the original.
    """
    print("--- Starting Verification of COO Conversion (Step 6) ---")

    if original_tensor.ndim > 2:
        print(
            "Verification tensor is high-dimensional. Showing stats instead of full tensor."
        )
        print(f"Original Tensor Shape: {original_tensor.shape}")
    else:
        print("Original Pruned Tensor:\n", original_tensor)

    # Convert to sparse COO format
    sparse_tensor = dense_to_coo(original_tensor)
    print("\nConverted Sparse COO Tensor:\n", sparse_tensor)

    # Convert back to dense for verification
    reconstructed_dense = sparse_tensor.to_dense()

    # Check if they are identical
    are_equal = torch.equal(original_tensor, reconstructed_dense)

    print(f"\nVerification successful: {are_equal}")
    assert (
        are_equal
    ), "Verification failed: Reconstructed tensor does not match original."
    print("--- Verification Complete ---")


# --- Custom Modules for Sparse Inference ---


class SparseConv2d(nn.Module):
    """
    Custom Conv2d module to handle sparse weights.
    As per assignment, we convert the weight to dense for the convolution operation.
    """

    def __init__(self, dense_conv: nn.Conv2d):
        super().__init__()
        self.in_channels = dense_conv.in_channels
        self.out_channels = dense_conv.out_channels
        self.kernel_size = dense_conv.kernel_size
        self.stride = dense_conv.stride
        self.padding = dense_conv.padding
        self.dilation = dense_conv.dilation
        self.groups = dense_conv.groups

        # Convert weight to sparse and register as a buffer (not a parameter)
        self.register_buffer("weight_sparse", dense_to_coo(dense_conv.weight.data))

        if dense_conv.bias is not None:
            self.bias = nn.Parameter(dense_conv.bias.data)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert sparse weight to dense on-the-fly for convolution
        dense_weight = self.weight_sparse.to_dense()
        return F.conv2d(
            x,
            dense_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SparseLinear(nn.Module):
    """
    Custom Linear module to handle sparse weights.
    Uses `torch.sparse.mm` for potential speedup as suggested.
    """

    def __init__(self, dense_linear: nn.Linear):
        super().__init__()
        # Convert weight to sparse and register as a buffer
        self.register_buffer("weight_sparse", dense_to_coo(dense_linear.weight.data))

        if dense_linear.bias is not None:
            self.bias = nn.Parameter(dense_linear.bias.data)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The operation y = xA^T + b can be computed as (A(x.T)).T + b
        # Here, A is our sparse weight matrix. torch.sparse.mm computes sparse @ dense.
        output = torch.sparse.mm(self.weight_sparse, x.t()).t()
        if self.bias is not None:
            output += self.bias
        return output


def convert_model_to_sparse(model: nn.Module):
    """
    Recursively iterates through a model and replaces Conv2d and Linear
    layers with their sparse counterparts.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d)):
            sparse_module = SparseConv2d(module)
            setattr(model, name, sparse_module)
        elif isinstance(module, (nn.Linear)):
            sparse_module = SparseLinear(module)
            setattr(model, name, sparse_module)
        elif len(list(module.children())) > 0:
            # Recurse for container modules like nn.Sequential
            convert_model_to_sparse(module)


def main():
    """Main function to perform COO conversion and verification."""

    # --- Load the Permanently Pruned Model from the Previous Step ---
    pruned_model_path = "vgg16_cifar10_pruned_permanent.pth"
    if not os.path.exists(pruned_model_path):
        print(f"Error: Pruned model file not found at '{pruned_model_path}'")
        print("Please run `finetune_pruned_model.py` first.")
        return

    print(f"--- Loading permanently pruned model from '{pruned_model_path}' ---")
    # Load the architecture first
    pruned_dense_model = load_pretrained_model(MODEL_CIFAR10, DEVICE)
    # Then load the pruned weights
    pruned_dense_model.load_state_dict(
        torch.load(pruned_model_path, map_location=DEVICE)
    )
    pruned_dense_model.eval()

    # --- Step 6: Verify COO Conversion on a sample layer ---
    # We pick the first convolutional layer for demonstration
    sample_layer_weight = pruned_dense_model.features[0].weight.data
    verify_coo_conversion(sample_layer_weight)

    # --- Step 5: Convert the Entire Model to Use Sparse Tensors ---
    print("\n--- Converting full model to sparse format (Step 5) ---")

    # Create a copy to modify
    sparse_model = copy.deepcopy(pruned_dense_model)
    convert_model_to_sparse(sparse_model)

    print("\nModel converted successfully. New module structure:")
    print(sparse_model)

    # --- Sanity Check: Run one inference pass to ensure no errors ---
    print("\nPerforming a sanity check inference pass...")
    try:
        dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
        _ = sparse_model(dummy_input)
        print("Sanity check passed. Forward pass is executable.")
    except Exception as e:
        print(f"Sanity check FAILED. Error during forward pass: {e}")

    # --- Save the Sparse Model State ---
    sparse_model_path = "vgg16_cifar10_sparse_coo.pth"
    torch.save(sparse_model.state_dict(), sparse_model_path)
    print(f"\nFinal sparse COO model saved to '{sparse_model_path}'")
    print("This model is now ready for profiling in the next step.")


if __name__ == "__main__":
    main()
