#!/usr/bin/env python3
"""Test MLX Structure Module implementation."""

import time
import mlx.core as mx
import numpy as np

from pxdesign.mlx_af2.quat_affine import (
    QuatAffine,
    quat_to_rot,
    quat_multiply,
    apply_rot_to_vec,
    squared_difference
)
from pxdesign.mlx_af2.structure_module import (
    InvariantPointAttention,
    BackboneUpdate,
    StructureModuleIteration,
    StructureModule
)


def test_quat_affine():
    """Test QuatAffine class."""
    print("Testing QuatAffine...")

    N_res = 10

    # Create identity transformation
    affine = QuatAffine.identity((N_res,))

    assert affine.quaternion.shape == (N_res, 4)
    assert affine.translation.shape == (N_res, 3)
    assert affine.rotation.shape == (N_res, 3, 3)

    # Test apply_to_point
    point = mx.random.normal((N_res, 3))
    transformed = affine.apply_to_point(point)

    # Identity should not change the point
    assert mx.allclose(transformed, point, atol=1e-5).item()

    print(f"✓ QuatAffine identity works correctly")
    print(f"✓ QuatAffine shapes: quat={affine.quaternion.shape}, trans={affine.translation.shape}, rot={affine.rotation.shape}\n")


def test_quat_operations():
    """Test quaternion operations."""
    print("Testing quaternion operations...")

    # Create random quaternions
    quat1 = mx.random.normal((5, 4))
    quat1 = quat1 / mx.sqrt(mx.sum(mx.square(quat1), axis=-1, keepdims=True))

    quat2 = mx.random.normal((5, 4))
    quat2 = quat2 / mx.sqrt(mx.sum(mx.square(quat2), axis=-1, keepdims=True))

    # Test quaternion multiplication
    quat_prod = quat_multiply(quat1, quat2)

    # Result should be normalized
    norm = mx.sqrt(mx.sum(mx.square(quat_prod), axis=-1))
    assert mx.allclose(norm, mx.ones_like(norm), atol=1e-5).item()

    # Test quaternion to rotation
    rot = quat_to_rot(quat1)
    assert rot.shape == (5, 3, 3)

    # Rotation matrices should be orthogonal (R^T R = I)
    rot_t = mx.transpose(rot, (0, 2, 1))
    identity = mx.matmul(rot_t, rot)
    eye = mx.broadcast_to(mx.eye(3), (5, 3, 3))
    assert mx.allclose(identity, eye, atol=1e-4).item()

    print(f"✓ Quaternion multiplication preserves normalization")
    print(f"✓ Quaternion to rotation produces orthogonal matrices\n")


def test_invariant_point_attention():
    """Test InvariantPointAttention module."""
    print("Testing InvariantPointAttention...")

    N_res = 32
    c_s = 384
    c_z = 128

    inputs_1d = mx.random.normal((N_res, c_s))
    inputs_2d = mx.random.normal((N_res, N_res, c_z))
    mask = mx.ones((N_res, 1))

    # Create random affine transformations
    affine = QuatAffine.identity((N_res,))

    ipa = InvariantPointAttention(
        c_s=c_s,
        c_z=c_z,
        num_heads=12,
        num_qk_points=4,
        num_v_points=8,
        num_scalar_qk=16,
        num_scalar_v=16
    )

    output = ipa(inputs_1d, inputs_2d, mask, affine)

    assert output.shape == inputs_1d.shape
    print(f"✓ IPA output shape: {output.shape}\n")


def test_backbone_update():
    """Test BackboneUpdate module."""
    print("Testing BackboneUpdate...")

    N_res = 32
    c_s = 384

    s = mx.random.normal((N_res, c_s))
    affine = QuatAffine.identity((N_res,))

    backbone_update = BackboneUpdate(c_s=c_s)

    new_affine = backbone_update(s, affine)

    assert new_affine.quaternion.shape == (N_res, 4)
    assert new_affine.translation.shape == (N_res, 3)

    # Quaternions should still be normalized
    quat_norm = mx.sqrt(mx.sum(mx.square(new_affine.quaternion), axis=-1))
    assert mx.allclose(quat_norm, mx.ones_like(quat_norm), atol=1e-5).item()

    print(f"✓ BackboneUpdate produces valid quaternions")
    print(f"✓ Backbone shapes: quat={new_affine.quaternion.shape}, trans={new_affine.translation.shape}\n")


def test_structure_module_iteration():
    """Test StructureModuleIteration."""
    print("Testing StructureModuleIteration...")

    N_res = 32
    c_s = 384
    c_z = 128

    s = mx.random.normal((N_res, c_s))
    z = mx.random.normal((N_res, N_res, c_z))
    mask = mx.ones((N_res, 1))
    affine = QuatAffine.identity((N_res,))

    iteration = StructureModuleIteration(c_s=c_s, c_z=c_z)

    s_out, affine_out = iteration(s, z, affine, mask)

    assert s_out.shape == s.shape
    assert affine_out.quaternion.shape == (N_res, 4)
    assert affine_out.translation.shape == (N_res, 3)

    print(f"✓ Structure iteration output shapes correct")
    print(f"✓ s: {s_out.shape}, affine: quat={affine_out.quaternion.shape}, trans={affine_out.translation.shape}\n")


def test_structure_module():
    """Test complete StructureModule."""
    print("Testing StructureModule...")

    N_res = 64
    c_s = 384
    c_z = 128

    s = mx.random.normal((N_res, c_s))
    z = mx.random.normal((N_res, N_res, c_z))
    mask = mx.ones((N_res, 1))

    structure_module = StructureModule(
        num_iterations=8,
        c_s=c_s,
        c_z=c_z
    )

    s_out, affine_out = structure_module(s, z, mask)

    assert s_out.shape == s.shape
    assert affine_out.quaternion.shape == (N_res, 4)
    assert affine_out.translation.shape == (N_res, 3)

    # Check that positions have changed from identity
    assert not mx.allclose(affine_out.translation, mx.zeros_like(affine_out.translation), atol=1e-3).item()

    print(f"✓ StructureModule (8 iterations) output shapes correct")
    print(f"✓ Final positions have non-zero values")
    print(f"✓ Translation range: [{float(mx.min(affine_out.translation)):.3f}, {float(mx.max(affine_out.translation)):.3f}]\n")


def test_structure_module_speed():
    """Benchmark StructureModule speed."""
    print("Benchmarking StructureModule speed on M3 Max MPS...")

    N_res = 128
    c_s = 384
    c_z = 128

    s = mx.random.normal((N_res, c_s))
    z = mx.random.normal((N_res, N_res, c_z))
    mask = mx.ones((N_res, 1))

    structure_module = StructureModule(num_iterations=8, c_s=c_s, c_z=c_z)

    # Warmup
    for _ in range(2):
        s_out, affine_out = structure_module(s, z, mask)
        mx.eval(s_out)
        mx.eval(affine_out.translation)

    # Benchmark
    num_iters = 10
    start = time.time()
    for _ in range(num_iters):
        s_out, affine_out = structure_module(s, z, mask)
        mx.eval(s_out)
        mx.eval(affine_out.translation)
    elapsed = time.time() - start

    avg_time = elapsed / num_iters
    print(f"✓ Structure module (8 iterations): {avg_time*1000:.2f} ms")
    print(f"✓ Throughput: {num_iters/elapsed:.1f} predictions/sec")
    print(f"✓ Device: {mx.default_device()}\n")


def test_numerical_stability():
    """Test numerical stability of structure module."""
    print("Testing numerical stability...")

    N_res = 128
    c_s = 384
    c_z = 128

    # Large values to test stability
    s = mx.random.normal((N_res, c_s)) * 5.0
    z = mx.random.normal((N_res, N_res, c_z)) * 5.0
    mask = mx.ones((N_res, 1))

    structure_module = StructureModule(num_iterations=8, c_s=c_s, c_z=c_z)

    s_out, affine_out = structure_module(s, z, mask)
    mx.eval(s_out)
    mx.eval(affine_out.translation)

    # Check for NaN/Inf
    has_nan_s = mx.any(mx.isnan(s_out))
    has_inf_s = mx.any(mx.isinf(s_out))
    has_nan_trans = mx.any(mx.isnan(affine_out.translation))
    has_inf_trans = mx.any(mx.isinf(affine_out.translation))
    has_nan_quat = mx.any(mx.isnan(affine_out.quaternion))
    has_inf_quat = mx.any(mx.isinf(affine_out.quaternion))

    assert not has_nan_s, "s output contains NaN!"
    assert not has_inf_s, "s output contains Inf!"
    assert not has_nan_trans, "Translation contains NaN!"
    assert not has_inf_trans, "Translation contains Inf!"
    assert not has_nan_quat, "Quaternion contains NaN!"
    assert not has_inf_quat, "Quaternion contains Inf!"

    # Check quaternions are normalized
    quat_norm = mx.sqrt(mx.sum(mx.square(affine_out.quaternion), axis=-1))
    assert mx.allclose(quat_norm, mx.ones_like(quat_norm), atol=1e-4).item()

    print(f"✓ No NaN or Inf in outputs")
    print(f"✓ Quaternions remain normalized")
    print(f"✓ Numerical stability test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MLX Structure Module Tests")
    print("=" * 60 + "\n")

    test_quat_affine()
    test_quat_operations()
    test_invariant_point_attention()
    test_backbone_update()
    test_structure_module_iteration()
    test_structure_module()
    test_structure_module_speed()
    test_numerical_stability()

    print("=" * 60)
    print("All structure module tests passed! ✓")
    print("=" * 60)
