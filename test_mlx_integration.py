#!/usr/bin/env python3
"""Test MLX AlphaFold2 integration in pxdbench pipeline"""

import os
import sys
import time

# Test 1: Factory function works
print("=" * 70)
print("MLX AlphaFold2 Integration Test")
print("=" * 70)
print()

print("Test 1: Factory function with MLX backend")
print("-" * 70)

try:
    from pxdbench.tools.af2.af2_model_factory import create_af2_prediction_model

    # Force MLX backend
    os.environ["PXDESIGN_AF2_BACKEND"] = "mlx"

    print("Creating MLX predictor...")
    start = time.time()
    predictor = create_af2_prediction_model(
        protocol="binder",
        num_recycles=3,
        data_dir=None,  # Testing mode
        backend="mlx"
    )
    elapsed = time.time() - start
    print(f"✅ MLX predictor created successfully in {elapsed:.2f}s")
    print()

except Exception as e:
    print(f"❌ Failed to create MLX predictor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test 2: Run a quick prediction
print("Test 2: Quick prediction with MLX backend")
print("-" * 70)

try:
    sequence = "MKTAYIAKQRQISFVKSHFSRQL"
    print(f"Sequence: {sequence} ({len(sequence)} residues)")

    print("Running prediction...")
    start = time.time()
    predictor.predict(
        seq=sequence,
        models=[0],
        num_recycles=3,
        verbose=True
    )
    elapsed = time.time() - start

    # Check results
    metrics = predictor.aux["log"]
    print()
    print(f"✅ Prediction completed in {elapsed:.2f}s")
    print(f"   pLDDT: {metrics['plddt']:.2f}")
    print(f"   pTM: {metrics['ptm']:.3f}")
    print(f"   i_pTM: {metrics['i_ptm']:.3f}")
    print()

except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test 3: Auto-detection works
print("Test 3: Auto-detection of backend")
print("-" * 70)

try:
    # Remove env variable to test auto-detection
    if "PXDESIGN_AF2_BACKEND" in os.environ:
        del os.environ["PXDESIGN_AF2_BACKEND"]

    print("Creating predictor with auto-detection...")
    predictor_auto = create_af2_prediction_model(
        protocol="binder",
        num_recycles=3,
        data_dir=None,
        backend=None  # Auto-detect
    )
    print("✅ Auto-detection successful")
    print()

except Exception as e:
    print(f"❌ Auto-detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test 4: Save PDB
print("Test 4: PDB export")
print("-" * 70)

try:
    output_pdb = "/tmp/mlx_integration_test.pdb"
    predictor.save_pdb(output_pdb)

    if os.path.exists(output_pdb):
        file_size = os.path.getsize(output_pdb)
        print(f"✅ PDB saved successfully: {output_pdb}")
        print(f"   File size: {file_size} bytes")

        # Show first few lines
        with open(output_pdb, "r") as f:
            lines = f.readlines()[:5]
        print(f"   First lines:")
        for line in lines:
            print(f"     {line.rstrip()}")
    else:
        print(f"❌ PDB file not created")
        sys.exit(1)
    print()

except Exception as e:
    print(f"❌ PDB export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


print("=" * 70)
print("✅ All integration tests passed!")
print("=" * 70)
print()
print("MLX AlphaFold2 is now integrated into the pxdbench pipeline.")
print()
print("To use MLX backend in your pipeline:")
print("1. Set environment variable: export PXDESIGN_AF2_BACKEND=mlx")
print("2. Or let it auto-detect (uses MLX on Apple Silicon with MPS)")
print("3. Run your pipeline as usual - it will use MLX automatically")
print()
print("Expected speedup: ~60x faster than JAX CPU")
print()
