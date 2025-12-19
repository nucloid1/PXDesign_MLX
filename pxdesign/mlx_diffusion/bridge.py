# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch-MLX tensor bridge for hybrid execution.

Provides efficient conversion between PyTorch and MLX tensors, leveraging
shared memory when possible to minimize overhead.
"""

import torch
import mlx.core as mx
import numpy as np
from typing import Union, Tuple


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """
    Convert PyTorch tensor to MLX array.

    Uses NumPy as intermediate format for safe cross-framework transfer.
    For MPS tensors, first moves to CPU.

    Args:
        tensor: PyTorch tensor (CPU, CUDA, or MPS)

    Returns:
        MLX array with same data and shape
    """
    # Convert to CPU if on MPS/CUDA
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    # Convert to numpy, then to MLX
    # Detach to avoid gradient tracking
    np_array = tensor.detach().numpy()
    mlx_array = mx.array(np_array)

    return mlx_array


def mlx_to_torch(array: mx.array, device: Union[str, torch.device] = "cpu", dtype: torch.dtype = None) -> torch.Tensor:
    """
    Convert MLX array to PyTorch tensor.

    Args:
        array: MLX array
        device: Target PyTorch device ("cpu", "cuda", "mps")
        dtype: Target PyTorch dtype (default: infer from MLX array)

    Returns:
        PyTorch tensor with same data and shape
    """
    # Convert to numpy first
    np_array = np.array(array)

    # Create PyTorch tensor
    tensor = torch.from_numpy(np_array)

    # Convert dtype if specified
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    # Move to target device
    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


def convert_batch_to_mlx(
    *tensors: torch.Tensor,
) -> Union[mx.array, Tuple[mx.array, ...]]:
    """
    Convert multiple PyTorch tensors to MLX arrays.

    Args:
        *tensors: Variable number of PyTorch tensors

    Returns:
        Single MLX array if one tensor, tuple of MLX arrays otherwise
    """
    mlx_arrays = tuple(torch_to_mlx(t) for t in tensors)

    if len(mlx_arrays) == 1:
        return mlx_arrays[0]
    return mlx_arrays


def convert_batch_to_torch(
    *arrays: mx.array,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Convert multiple MLX arrays to PyTorch tensors.

    Args:
        *arrays: Variable number of MLX arrays
        device: Target PyTorch device
        dtype: Target PyTorch dtype

    Returns:
        Single PyTorch tensor if one array, tuple of tensors otherwise
    """
    tensors = tuple(mlx_to_torch(a, device=device, dtype=dtype) for a in arrays)

    if len(tensors) == 1:
        return tensors[0]
    return tensors


class HybridExecutor:
    """
    Manages hybrid PyTorch/MLX execution with automatic conversion.

    Handles device detection, tensor conversion, and fallback logic.
    """

    def __init__(self, use_mlx: bool = None, verbose: bool = True):
        """
        Args:
            use_mlx: Force MLX usage (True), force PyTorch (False), or auto-detect (None)
            verbose: Print device selection info
        """
        if use_mlx is None:
            # Auto-detect: use MLX if on Apple Silicon
            try:
                import platform
                self.use_mlx = platform.system() == "Darwin" and platform.machine() == "arm64"
                if self.use_mlx:
                    # Test MLX availability
                    _ = mx.array([1.0])
            except Exception:
                self.use_mlx = False
        else:
            self.use_mlx = use_mlx

        self.verbose = verbose

        if self.verbose:
            if self.use_mlx:
                print("✓ MLX acceleration enabled for diffusion transformer")
            else:
                print("ℹ Using PyTorch for diffusion transformer (MLX not available)")

    def execute_mlx(
        self,
        mlx_module,
        *torch_inputs: torch.Tensor,
        output_device: Union[str, torch.device] = None,
        output_dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Execute MLX module with PyTorch tensors.

        Automatically handles:
        1. PyTorch → MLX tensor conversion
        2. MLX module execution
        3. MLX → PyTorch tensor conversion

        Args:
            mlx_module: MLX module to execute
            *torch_inputs: PyTorch input tensors
            output_device: Device for output tensor
            output_dtype: Dtype for output tensor

        Returns:
            PyTorch tensor result
        """
        if not self.use_mlx:
            raise RuntimeError("MLX not enabled, cannot execute MLX module")

        # Infer output device/dtype from first input if not specified
        if output_device is None:
            output_device = torch_inputs[0].device
        if output_dtype is None:
            output_dtype = torch_inputs[0].dtype

        # Convert inputs to MLX
        mlx_inputs = convert_batch_to_mlx(*torch_inputs)
        if not isinstance(mlx_inputs, tuple):
            mlx_inputs = (mlx_inputs,)

        # Execute MLX module
        mlx_output = mlx_module(*mlx_inputs)

        # Evaluate the computation (MLX is lazy)
        mx.eval(mlx_output)

        # Convert output back to PyTorch
        torch_output = mlx_to_torch(mlx_output, device=output_device, dtype=output_dtype)

        return torch_output
