#!/usr/bin/env python3
"""Test MLX Evoformer stack implementation."""

import time
import mlx.core as mx
import numpy as np

from pxdesign.mlx_af2.evoformer_stack import (
    EvoformerStack,
    RecyclingEmbedding,
    EvoformerWithRecycling,
)


def test_evoformer_stack_small():
    """Test EvoformerStack with small number of blocks."""
    print("Testing EvoformerStack (8 blocks)...")

    N_seq = 4
    N_res = 64
    c_m = 256
    c_z = 128

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))
    msa_mask = mx.ones((N_seq, N_res))
    pair_mask = mx.ones((N_res, N_res))

    # Small stack for testing
    evoformer = EvoformerStack(num_blocks=8, c_m=c_m, c_z=c_z)

    msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)

    assert msa_out.shape == msa_act.shape
    assert pair_out.shape == pair_act.shape

    print(f"✓ MSA output shape: {msa_out.shape}")
    print(f"✓ Pair output shape: {pair_out.shape}\n")


def test_evoformer_stack_full():
    """Test full 48-block EvoformerStack."""
    print("Testing full EvoformerStack (48 blocks)...")

    N_seq = 4
    N_res = 128
    c_m = 256
    c_z = 128

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))
    msa_mask = mx.ones((N_seq, N_res))
    pair_mask = mx.ones((N_res, N_res))

    # Full 48-block stack
    evoformer = EvoformerStack(num_blocks=48, c_m=c_m, c_z=c_z)

    # Warmup
    msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)
    mx.eval(msa_out)
    mx.eval(pair_out)

    # Time it
    start = time.time()
    msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)
    mx.eval(msa_out)
    mx.eval(pair_out)
    elapsed = time.time() - start

    assert msa_out.shape == msa_act.shape
    assert pair_out.shape == pair_act.shape

    print(f"✓ MSA output shape: {msa_out.shape}")
    print(f"✓ Pair output shape: {pair_out.shape}")
    print(f"✓ Full 48-block stack time: {elapsed*1000:.2f} ms ({elapsed:.2f} sec)")
    print(f"✓ Device: {mx.default_device()}\n")


def test_recycling_embedding():
    """Test RecyclingEmbedding module."""
    print("Testing RecyclingEmbedding...")

    N_seq = 4
    N_res = 64
    c_m = 256
    c_z = 128

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))
    prev_positions = mx.random.normal((N_res, 3)) * 10.0  # Positions in Angstroms

    recycling = RecyclingEmbedding(c_m=c_m, c_z=c_z)

    msa_out, pair_out = recycling(msa_act, pair_act, prev_positions)

    assert msa_out.shape == msa_act.shape
    assert pair_out.shape == pair_act.shape

    print(f"✓ Recycling MSA shape: {msa_out.shape}")
    print(f"✓ Recycling pair shape: {pair_out.shape}\n")


def test_evoformer_with_recycling():
    """Test EvoformerWithRecycling module."""
    print("Testing EvoformerWithRecycling...")

    N_seq = 4
    N_res = 64
    c_m = 256
    c_z = 128

    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))
    msa_mask = mx.ones((N_seq, N_res))
    pair_mask = mx.ones((N_res, N_res))

    # Use small stack for testing (8 blocks instead of 48)
    evoformer = EvoformerWithRecycling(
        num_recycles=3,
        num_blocks=8,
        c_m=c_m,
        c_z=c_z
    )

    msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)

    assert msa_out.shape == msa_act.shape
    assert pair_out.shape == pair_act.shape

    print(f"✓ MSA output shape: {msa_out.shape}")
    print(f"✓ Pair output shape: {pair_out.shape}\n")


def test_numerical_stability_stack():
    """Test numerical stability of full stack."""
    print("Testing numerical stability of full stack...")

    N_seq = 4
    N_res = 128
    c_m = 256
    c_z = 128

    # Large values to test stability
    msa_act = mx.random.normal((N_seq, N_res, c_m)) * 5.0
    pair_act = mx.random.normal((N_res, N_res, c_z)) * 5.0
    msa_mask = mx.ones((N_seq, N_res))
    pair_mask = mx.ones((N_res, N_res))

    # Use smaller stack for faster testing
    evoformer = EvoformerStack(num_blocks=16, c_m=c_m, c_z=c_z)

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


def benchmark_different_sizes():
    """Benchmark Evoformer stack with different protein sizes."""
    print("Benchmarking different protein sizes...")

    sizes = [64, 128, 256]
    N_seq = 4
    c_m = 256
    c_z = 128
    num_blocks = 48

    evoformer = EvoformerStack(num_blocks=num_blocks, c_m=c_m, c_z=c_z)

    for N_res in sizes:
        msa_act = mx.random.normal((N_seq, N_res, c_m))
        pair_act = mx.random.normal((N_res, N_res, c_z))
        msa_mask = mx.ones((N_seq, N_res))
        pair_mask = mx.ones((N_res, N_res))

        # Warmup
        msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)
        mx.eval(msa_out)
        mx.eval(pair_out)

        # Benchmark
        start = time.time()
        msa_out, pair_out = evoformer(msa_act, pair_act, msa_mask, pair_mask)
        mx.eval(msa_out)
        mx.eval(pair_out)
        elapsed = time.time() - start

        print(f"  N_res={N_res:3d}: {elapsed*1000:7.2f} ms ({elapsed:.2f} sec)")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("MLX Evoformer Stack Tests")
    print("=" * 60 + "\n")

    test_evoformer_stack_small()
    test_recycling_embedding()
    test_evoformer_with_recycling()
    test_numerical_stability_stack()
    test_evoformer_stack_full()
    benchmark_different_sizes()

    print("=" * 60)
    print("All Evoformer stack tests passed! ✓")
    print("=" * 60)
