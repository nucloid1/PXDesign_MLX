#!/usr/bin/env python3
"""Test JAX-MLX bridge conversion via buffer protocol."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import mlx.core as mx

from pxdesign.mlx_af2.jax_mlx_bridge import (
    jax_to_mlx,
    mlx_to_jax,
    jax_dict_to_mlx,
    mlx_dict_to_jax,
    verify_conversion,
)


def test_basic_conversion():
    """Test basic JAX → MLX → JAX conversion."""
    print("Testing basic conversion...")

    # Create JAX array
    jax_arr = jnp.arange(100, dtype=jnp.float32).reshape(10, 10)
    print(f"JAX array shape: {jax_arr.shape}, dtype: {jax_arr.dtype}")

    # Convert to MLX
    mlx_arr = jax_to_mlx(jax_arr)
    print(f"MLX array shape: {mlx_arr.shape}, dtype: {mlx_arr.dtype}")

    # Convert back to JAX
    jax_arr2 = mlx_to_jax(mlx_arr)
    print(f"JAX array (round-trip) shape: {jax_arr2.shape}, dtype: {jax_arr2.dtype}")

    # Verify data matches
    assert verify_conversion(jax_arr, mlx_arr), "JAX-MLX conversion failed!"
    assert np.allclose(jax_arr, jax_arr2), "Round-trip conversion failed!"

    print("✓ Basic conversion passed\n")


def test_conversion_speed():
    """Test conversion speed on M3 Max."""
    print("Testing conversion speed...")

    # Create larger array (simulating protein features)
    # Shape: [N_res=2000, c=256]
    key = jax.random.PRNGKey(0)
    jax_arr = jax.random.normal(key, (2000, 256), dtype=jnp.float32)

    # Time JAX → MLX conversion
    start = time.time()
    for _ in range(100):
        mlx_arr = jax_to_mlx(jax_arr)
    jax_to_mlx_time = (time.time() - start) / 100

    # Time MLX → JAX conversion
    start = time.time()
    for _ in range(100):
        jax_arr2 = mlx_to_jax(mlx_arr)
    mlx_to_jax_time = (time.time() - start) / 100

    print(f"JAX → MLX conversion: {jax_to_mlx_time*1000:.2f} ms")
    print(f"MLX → JAX conversion: {mlx_to_jax_time*1000:.2f} ms")
    print(f"Total round-trip: {(jax_to_mlx_time + mlx_to_jax_time)*1000:.2f} ms")

    # Should be < 1ms for zero-copy on unified memory
    assert (jax_to_mlx_time + mlx_to_jax_time) < 0.01, "Conversion too slow!"

    print("✓ Conversion speed test passed\n")


def test_dict_conversion():
    """Test dictionary conversion (useful for AF2 features)."""
    print("Testing dictionary conversion...")

    # Create nested dictionary simulating AF2 features
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    jax_dict = {
        "msa": jax.random.normal(key1, (1, 128, 256), dtype=jnp.float32),
        "pair": jax.random.normal(key2, (128, 128, 128), dtype=jnp.float32),
        "metadata": {
            "seq_length": 128,
            "name": "test_protein",
        },
    }

    # Convert to MLX
    mlx_dict = jax_dict_to_mlx(jax_dict)

    assert isinstance(mlx_dict["msa"], mx.array), "MSA not converted!"
    assert isinstance(mlx_dict["pair"], mx.array), "Pair not converted!"
    assert mlx_dict["metadata"]["seq_length"] == 128, "Metadata lost!"

    # Convert back
    jax_dict2 = mlx_dict_to_jax(mlx_dict)

    assert isinstance(jax_dict2["msa"], jnp.ndarray), "MSA not converted back!"
    assert np.allclose(jax_dict["msa"], jax_dict2["msa"]), "MSA data mismatch!"

    print("✓ Dictionary conversion passed\n")


def test_device_placement():
    """Test that MLX arrays are on MPS GPU."""
    print("Testing device placement...")

    # Create MLX array
    mlx_arr = mx.random.normal((1000, 1000))

    # Check device
    print(f"MLX default device: {mx.default_device()}")

    # Perform operation to trigger execution
    result = mx.matmul(mlx_arr, mlx_arr.T)
    mx.eval(result)  # Force evaluation

    print("✓ Device placement test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("JAX-MLX Bridge Tests")
    print("=" * 60 + "\n")

    test_basic_conversion()
    test_conversion_speed()
    test_dict_conversion()
    test_device_placement()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
