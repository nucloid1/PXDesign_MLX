#!/usr/bin/env python3
"""Test MLX Evoformer implementation."""

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from pxdesign.mlx_af2.evoformer import (
    LayerNorm,
    Transition,
    TriangleMultiplication,
    TriangleAttention,
    MSARowAttentionWithPairBias,
    MSAColumnAttention,
    OuterProductMean,
    EvoformerIteration,
)


def test_layer_norm():
    """Test LayerNorm module."""
    print("Testing LayerNorm...")

    x = mx.random.normal((10, 20, 64))
    ln = LayerNorm(64, affine=True)

    output = ln(x)

    assert output.shape == x.shape
    # Check normalization: mean~0, std~1
    mean = mx.mean(output, axis=-1)
    std = mx.sqrt(mx.var(output, axis=-1))

    print(f"✓ LayerNorm output shape: {output.shape}")
    print(f"✓ Mean (should be ~0): {float(mx.mean(mean)):.4f}")
    print(f"✓ Std (should be ~1): {float(mx.mean(std)):.4f}\n")


def test_transition():
    """Test Transition (MLP) module."""
    print("Testing Transition...")

    x = mx.random.normal((10, 128, 256))
    transition = Transition(c_in=256, num_intermediate_factor=4)

    output = transition(x)

    assert output.shape == x.shape
    print(f"✓ Transition output shape: {output.shape}\n")


def test_triangle_multiplication():
    """Test TriangleMultiplication module."""
    print("Testing TriangleMultiplication...")

    N_res = 64
    c_z = 128

    pair_act = mx.random.normal((N_res, N_res, c_z))
    pair_mask = mx.ones((N_res, N_res))

    # Test outgoing
    tri_mul_out = TriangleMultiplication(
        c_z=c_z,
        c_hidden=128,
        equation='ikc,jkc->ijc'
    )
    output_out = tri_mul_out(pair_act, pair_mask)
    assert output_out.shape == pair_act.shape

    # Test incoming
    tri_mul_in = TriangleMultiplication(
        c_z=c_z,
        c_hidden=128,
        equation='kjc,kic->ijc'
    )
    output_in = tri_mul_in(pair_act, pair_mask)
    assert output_in.shape == pair_act.shape

    print(f"✓ TriangleMultiplication (outgoing) shape: {output_out.shape}")
    print(f"✓ TriangleMultiplication (incoming) shape: {output_in.shape}\n")


def test_triangle_attention():
    """Test TriangleAttention module."""
    print("Testing TriangleAttention...")

    N_res = 64
    c_z = 128

    pair_act = mx.random.normal((N_res, N_res, c_z))
    pair_mask = mx.ones((N_res, N_res))

    # Test starting node (per_row)
    tri_att_start = TriangleAttention(
        c_z=c_z,
        c_hidden=32,
        num_heads=4,
        orientation='per_row'
    )
    output_start = tri_att_start(pair_act, pair_mask)
    assert output_start.shape == pair_act.shape

    # Test ending node (per_column)
    tri_att_end = TriangleAttention(
        c_z=c_z,
        c_hidden=32,
        num_heads=4,
        orientation='per_column'
    )
    output_end = tri_att_end(pair_act, pair_mask)
    assert output_end.shape == pair_act.shape

    print(f"✓ TriangleAttention (starting) shape: {output_start.shape}")
    print(f"✓ TriangleAttention (ending) shape: {output_end.shape}\n")


def test_msa_row_attention():
    """Test MSARowAttentionWithPairBias module."""
    print("Testing MSARowAttentionWithPairBias...")

    N_seq = 8
    N_res = 64
    c_m = 256
    c_z = 128

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))
    msa_mask = mx.ones((N_seq, N_res))

    msa_row_att = MSARowAttentionWithPairBias(
        c_m=c_m,
        c_z=c_z,
        c_hidden=32,
        num_heads=8
    )

    output = msa_row_att(msa_act, pair_act, msa_mask)

    assert output.shape == msa_act.shape
    print(f"✓ MSARowAttentionWithPairBias shape: {output.shape}\n")


def test_msa_column_attention():
    """Test MSAColumnAttention module."""
    print("Testing MSAColumnAttention...")

    N_seq = 8
    N_res = 64
    c_m = 256

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    msa_mask = mx.ones((N_seq, N_res))

    msa_col_att = MSAColumnAttention(
        c_m=c_m,
        c_hidden=32,
        num_heads=8
    )

    output = msa_col_att(msa_act, msa_mask)

    assert output.shape == msa_act.shape
    print(f"✓ MSAColumnAttention shape: {output.shape}\n")


def test_outer_product_mean():
    """Test OuterProductMean module."""
    print("Testing OuterProductMean...")

    N_seq = 8
    N_res = 64
    c_m = 256
    c_z = 128

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    msa_mask = mx.ones((N_seq, N_res))

    opm = OuterProductMean(
        c_m=c_m,
        c_z=c_z,
        c_hidden=32
    )

    output = opm(msa_act, msa_mask)

    assert output.shape == (N_res, N_res, c_z)
    print(f"✓ OuterProductMean shape: {output.shape}\n")


def test_evoformer_iteration():
    """Test complete EvoformerIteration module."""
    print("Testing EvoformerIteration...")

    N_seq = 8
    N_res = 128
    c_m = 256
    c_z = 128

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))
    msa_mask = mx.ones((N_seq, N_res))
    pair_mask = mx.ones((N_res, N_res))

    evoformer = EvoformerIteration(
        c_m=c_m,
        c_z=c_z,
        c_hidden_msa_att=32,
        c_hidden_opm=32,
        c_hidden_tri_mul=128,
        c_hidden_tri_att=32,
        num_heads_msa=8,
        num_heads_tri=4
    )

    msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)

    assert msa_out.shape == msa_act.shape
    assert pair_out.shape == pair_act.shape

    print(f"✓ MSA output shape: {msa_out.shape}")
    print(f"✓ Pair output shape: {pair_out.shape}\n")


def test_evoformer_speed():
    """Benchmark EvoformerIteration speed on MPS."""
    print("Benchmarking EvoformerIteration speed on M3 Max MPS...")

    N_seq = 4
    N_res = 128
    c_m = 256
    c_z = 128

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))
    msa_mask = mx.ones((N_seq, N_res))
    pair_mask = mx.ones((N_res, N_res))

    evoformer = EvoformerIteration(c_m=c_m, c_z=c_z)

    # Warmup
    for _ in range(3):
        msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)
        mx.eval(msa_out)
        mx.eval(pair_out)

    # Benchmark
    num_iters = 20
    start = time.time()
    for _ in range(num_iters):
        msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)
        mx.eval(msa_out)
        mx.eval(pair_out)
    elapsed = time.time() - start

    avg_time = elapsed / num_iters
    print(f"✓ Single Evoformer iteration: {avg_time*1000:.2f} ms")
    print(f"✓ Throughput: {num_iters/elapsed:.1f} iterations/sec")
    print(f"✓ Estimated 48-block stack: {avg_time * 48 * 1000:.2f} ms")
    print(f"✓ Device: {mx.default_device()}\n")


def test_numerical_stability():
    """Test numerical stability of EvoformerIteration."""
    print("Testing numerical stability...")

    N_seq = 4
    N_res = 256
    c_m = 256
    c_z = 128

    # Large values
    msa_act = mx.random.normal((N_seq, N_res, c_m)) * 10.0
    pair_act = mx.random.normal((N_res, N_res, c_z)) * 10.0
    msa_mask = mx.ones((N_seq, N_res))
    pair_mask = mx.ones((N_res, N_res))

    evoformer = EvoformerIteration(c_m=c_m, c_z=c_z)

    msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)
    mx.eval(msa_out)
    mx.eval(pair_out)

    # Check for NaN/Inf
    has_nan_msa = mx.any(mx.isnan(msa_out))
    has_inf_msa = mx.any(mx.isinf(msa_out))
    has_nan_pair = mx.any(mx.isnan(pair_out))
    has_inf_pair = mx.any(mx.isinf(pair_out))

    assert not has_nan_msa, "MSA output contains NaN!"
    assert not has_inf_msa, "MSA output contains Inf!"
    assert not has_nan_pair, "Pair output contains NaN!"
    assert not has_inf_pair, "Pair output contains Inf!"

    print(f"✓ No NaN or Inf in MSA output")
    print(f"✓ No NaN or Inf in pair output")
    print(f"✓ Numerical stability test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MLX Evoformer Tests")
    print("=" * 60 + "\n")

    test_layer_norm()
    test_transition()
    test_triangle_multiplication()
    test_triangle_attention()
    test_msa_row_attention()
    test_msa_column_attention()
    test_outer_product_mean()
    test_evoformer_iteration()
    test_evoformer_speed()
    test_numerical_stability()

    print("=" * 60)
    print("All Evoformer tests passed! ✓")
    print("=" * 60)
