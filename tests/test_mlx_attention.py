#!/usr/bin/env python3
"""Test MLX attention implementation."""

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from pxdesign.mlx_af2.attention import MLXAttention, MLXAttentionPairBias


def test_basic_attention():
    """Test basic multi-head attention."""
    print("Testing basic MLX attention...")

    batch_size = 2
    N_res = 128
    c_in = 256
    c_hidden = 32
    num_heads = 8

    # Create random input
    x = mx.random.normal((batch_size, N_res, c_in))

    # Create attention module
    attn = MLXAttention(
        c_q=c_in,
        c_kv=c_in,
        c_hidden=c_hidden,
        num_heads=num_heads,
        c_out=c_in,
        gating=True,
    )

    # Forward pass
    output = attn(x, x)

    # Check shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Basic attention test passed\n")


def test_attention_with_bias():
    """Test attention with bias."""
    print("Testing attention with bias...")

    batch_size = 2
    N_res = 64
    c_in = 128
    num_heads = 4

    x = mx.random.normal((batch_size, N_res, c_in))
    bias = mx.random.normal((batch_size, num_heads, N_res, N_res))

    attn = MLXAttention(c_q=c_in, c_hidden=32, num_heads=num_heads)

    output = attn(x, x, bias=bias)

    assert output.shape == x.shape
    print(f"✓ Attention with bias passed\n")


def test_attention_pair_bias():
    """Test attention with pair representation bias."""
    print("Testing attention with pair bias...")

    batch_size = 2
    N_res = 64
    c_in = 256
    c_pair = 128
    num_heads = 8

    x = mx.random.normal((batch_size, N_res, c_in))
    pair_rep = mx.random.normal((batch_size, N_res, N_res, c_pair))

    attn = MLXAttentionPairBias(
        c_in=c_in,
        c_hidden=32,
        num_heads=num_heads,
        c_pair=c_pair,
    )

    output = attn(x, pair_rep)

    assert output.shape == x.shape
    print(f"✓ Attention with pair bias passed\n")


def test_attention_speed():
    """Benchmark attention speed on MPS."""
    print("Benchmarking attention speed on M3 Max MPS...")

    N_res = 256
    c_in = 256
    num_heads = 8

    x = mx.random.normal((1, N_res, c_in))
    bias = mx.random.normal((1, num_heads, N_res, N_res))

    attn = MLXAttention(c_q=c_in, c_hidden=32, num_heads=num_heads, gating=True)

    # Warmup
    for _ in range(5):
        output = attn(x, x, bias=bias)
        mx.eval(output)

    # Benchmark
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        output = attn(x, x, bias=bias)
        mx.eval(output)  # Force evaluation
    elapsed = time.time() - start

    avg_time = elapsed / num_iters
    print(f"✓ Attention forward pass: {avg_time*1000:.2f} ms")
    print(f"✓ Throughput: {num_iters/elapsed:.1f} iterations/sec")
    print(f"✓ Device: {mx.default_device()}\n")


def test_numerical_stability():
    """Test numerical stability with large inputs."""
    print("Testing numerical stability...")

    N_res = 512
    c_in = 384

    x = mx.random.normal((1, N_res, c_in)) * 10.0  # Large values
    attn = MLXAttention(c_q=c_in, c_hidden=64, num_heads=16)

    output = attn(x, x)
    mx.eval(output)

    # Check for NaN/Inf
    has_nan = mx.any(mx.isnan(output))
    has_inf = mx.any(mx.isinf(output))

    assert not has_nan, "Output contains NaN!"
    assert not has_inf, "Output contains Inf!"

    print(f"✓ No NaN or Inf in output")
    print(f"✓ Numerical stability test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MLX Attention Tests")
    print("=" * 60 + "\n")

    test_basic_attention()
    test_attention_with_bias()
    test_attention_pair_bias()
    test_attention_speed()
    test_numerical_stability()

    print("=" * 60)
    print("All attention tests passed! ✓")
    print("=" * 60)
