#!/usr/bin/env python3
"""Test MLX AlphaFold2 predictor end-to-end."""

import time
import os
import tempfile
import mlx.core as mx

from pxdesign.mlx_af2.model import MLXAlphaFold2, create_mlx_alphafold2
from pxdesign.mlx_af2.predictor import MLXAlphaFold2Predictor, mk_mlx_afdesign_model


def test_mlx_alphafold2_model():
    """Test complete MLX AlphaFold2 model."""
    print("Testing MLX AlphaFold2 model...")

    N_seq = 4
    N_res = 64
    c_m = 256
    c_z = 128

    # Create model with smaller configuration for faster testing
    model = MLXAlphaFold2(
        c_m=c_m,
        c_z=c_z,
        c_s=384,
        num_evoformer_blocks=8,  # Reduced from 48 for testing
        num_structure_iterations=4,  # Reduced from 8 for testing
        num_recycles=2  # Reduced from 3 for testing
    )

    # Create random inputs
    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))

    print(f"  Input shapes: msa={msa_act.shape}, pair={pair_act.shape}")

    # Run forward pass
    output = model(msa_act, pair_act, num_recycles=2)

    # Check outputs
    assert 'final_atom_positions' in output
    assert 'quaternions' in output
    assert 'translations' in output
    assert 'single' in output
    assert 'pair' in output

    positions = output['final_atom_positions']
    assert positions.shape == (N_res, 3)

    print(f"  ✓ Output positions shape: {positions.shape}")
    print(f"  ✓ Position range: [{float(mx.min(positions)):.2f}, {float(mx.max(positions)):.2f}]")
    print(f"  ✓ All output keys present\n")


def test_mlx_predictor_interface():
    """Test MLX predictor with ColabDesign-compatible interface."""
    print("Testing MLX predictor interface...")

    # Create predictor
    predictor = mk_mlx_afdesign_model(
        protocol="binder",
        num_recycles=2,
        data_dir=None,  # Will use dummy data
        use_multimer=False
    )

    # Test sequence (small protein)
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

    # Run prediction
    predictor.predict(seq=seq, models=[0], num_recycles=2, verbose=True)

    # Check that aux contains metrics
    assert "log" in predictor.aux
    assert "plddt" in predictor.aux["log"]
    assert "ptm" in predictor.aux["log"]

    print(f"  ✓ Predictor interface working")
    print(f"  ✓ Metrics computed: {list(predictor.aux['log'].keys())}\n")


def test_pdb_output():
    """Test PDB file generation."""
    print("Testing PDB output...")

    predictor = mk_mlx_afdesign_model(num_recycles=1)

    # Short test sequence
    seq = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"

    # Run prediction
    predictor.predict(seq=seq, models=[0], num_recycles=1, verbose=False)

    # Save PDB
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        pdb_path = f.name

    try:
        predictor.save_pdb(pdb_path)

        # Check file was created
        assert os.path.exists(pdb_path)

        # Read and check PDB content
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        # Check for CA atoms
        ca_atoms = [l for l in lines if l.startswith('ATOM')]
        assert len(ca_atoms) == len(seq)

        print(f"  ✓ PDB file created: {pdb_path}")
        print(f"  ✓ Number of CA atoms: {len(ca_atoms)}")
        print(f"  ✓ First line: {ca_atoms[0].strip()}")

    finally:
        # Clean up
        if os.path.exists(pdb_path):
            os.remove(pdb_path)

    print("  ✓ PDB output test passed\n")


def test_performance_full_model():
    """Test performance with full-size model."""
    print("Testing performance with full-size model...")

    N_seq = 4
    N_res = 128
    c_m = 256
    c_z = 128

    # Create full model
    model = create_mlx_alphafold2(
        model_name="model_1_ptm",
        num_recycles=3
    )

    # Create inputs
    msa_act = mx.random.normal((N_seq, N_res, c_m))
    pair_act = mx.random.normal((N_res, N_res, c_z))

    # Warmup
    output = model(msa_act, pair_act, num_recycles=1)
    mx.eval(output['final_atom_positions'])

    # Benchmark
    start = time.time()
    output = model(msa_act, pair_act, num_recycles=3)
    mx.eval(output['final_atom_positions'])
    elapsed = time.time() - start

    print(f"  ✓ Full model (48 Evoformer + 8 Structure, 3 recycles)")
    print(f"  ✓ Protein size: {N_res} residues")
    print(f"  ✓ Time: {elapsed:.2f} seconds ({elapsed*1000:.0f} ms)")
    print(f"  ✓ Device: {mx.default_device()}")

    # Estimate throughput for 200 predictions
    estimated_total = elapsed * 200
    print(f"  ✓ Estimated time for 200 predictions: {estimated_total/60:.1f} minutes\n")


def test_multiple_models():
    """Test using multiple model indices."""
    print("Testing multiple model indices...")

    predictor = mk_mlx_afdesign_model(num_recycles=1)
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSR"

    # Test different model indices
    for model_idx in [0, 1]:
        print(f"  Testing model {model_idx}...")
        predictor.predict(seq=seq, models=[model_idx], num_recycles=1, verbose=False)
        assert "plddt" in predictor.aux["log"]

    print("  ✓ Multiple models test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MLX AlphaFold2 Predictor Tests")
    print("=" * 60 + "\n")

    test_mlx_alphafold2_model()
    test_mlx_predictor_interface()
    test_pdb_output()
    test_multiple_models()
    test_performance_full_model()

    print("=" * 60)
    print("All predictor tests passed! ✓")
    print("=" * 60)
    print("\n🚀 MLX AlphaFold2 is ready for integration!")
