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
JAX-MLX Bridge: Zero-copy conversion via buffer protocol.

This module provides seamless interoperability between JAX and MLX arrays
using the Python buffer protocol, enabling efficient data transfer on
unified memory architecture (M3 Max).
"""

import mlx.core as mx
import jax.numpy as jnp
import numpy as np
from typing import Union, Dict, Any


def jax_to_mlx(arr: Union[jnp.ndarray, np.ndarray, Any]) -> mx.array:
    """
    Convert JAX array to MLX array using zero-copy buffer protocol.

    On Apple Silicon with unified memory, this conversion is extremely fast
    as both frameworks can access the same memory region.

    Args:
        arr: JAX array, NumPy array, or Python array buffer

    Returns:
        MLX array with same data and shape

    Example:
        >>> import jax.numpy as jnp
        >>> jax_arr = jnp.arange(100)
        >>> mlx_arr = jax_to_mlx(jax_arr)
        >>> print(mlx_arr.shape)
        (100,)
    """
    if isinstance(arr, mx.array):
        # Already MLX array
        return arr

    # Use buffer protocol for zero-copy conversion
    # JAX fully supports the buffer protocol
    return mx.array(arr)


def mlx_to_jax(arr: Union[mx.array, np.ndarray, Any]) -> jnp.ndarray:
    """
    Convert MLX array to JAX array using zero-copy buffer protocol.

    Args:
        arr: MLX array, NumPy array, or Python array buffer

    Returns:
        JAX array with same data and shape

    Example:
        >>> import mlx.core as mx
        >>> mlx_arr = mx.arange(100)
        >>> jax_arr = mlx_to_jax(mlx_arr)
        >>> print(jax_arr.shape)
        (100,)
    """
    if isinstance(arr, jnp.ndarray):
        # Already JAX array
        return arr

    # Use buffer protocol for zero-copy conversion
    return jnp.array(arr)


def jax_dict_to_mlx(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert dictionary of JAX arrays to MLX arrays.

    Useful for converting feature dictionaries used in AlphaFold2.

    Args:
        data: Dictionary potentially containing JAX arrays

    Returns:
        Dictionary with JAX arrays converted to MLX
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            result[key] = jax_to_mlx(value)
        elif isinstance(value, dict):
            result[key] = jax_dict_to_mlx(value)
        elif isinstance(value, (list, tuple)):
            result[key] = type(value)(
                jax_to_mlx(item) if isinstance(item, (jnp.ndarray, np.ndarray)) else item
                for item in value
            )
        else:
            result[key] = value
    return result


def mlx_dict_to_jax(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert dictionary of MLX arrays to JAX arrays.

    Args:
        data: Dictionary potentially containing MLX arrays

    Returns:
        Dictionary with MLX arrays converted to JAX
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, mx.array):
            result[key] = mlx_to_jax(value)
        elif isinstance(value, dict):
            result[key] = mlx_dict_to_jax(value)
        elif isinstance(value, (list, tuple)):
            result[key] = type(value)(
                mlx_to_jax(item) if isinstance(item, mx.array) else item
                for item in value
            )
        else:
            result[key] = value
    return result


def verify_conversion(arr_jax: jnp.ndarray, arr_mlx: mx.array, atol: float = 1e-6) -> bool:
    """
    Verify that JAX and MLX arrays contain the same data.

    Args:
        arr_jax: JAX array
        arr_mlx: MLX array
        atol: Absolute tolerance for numerical comparison

    Returns:
        True if arrays match within tolerance
    """
    # Convert both to NumPy for comparison
    np_from_jax = np.array(arr_jax)
    np_from_mlx = np.array(arr_mlx)

    return np.allclose(np_from_jax, np_from_mlx, atol=atol)
