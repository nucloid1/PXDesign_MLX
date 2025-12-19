# MLX vs JAX AlphaFold2 Validation Plan

## Objective
Validate that the MLX implementation produces equivalent results to the JAX ColabDesign implementation.

## Validation Strategy

### Phase 1: Component-Level Validation
Compare individual components with controlled inputs:
1. Attention mechanisms
2. Evoformer iterations
3. Structure module outputs
4. Quaternion operations

### Phase 2: End-to-End Validation
Compare complete predictions on the same inputs:
1. Use identical random seeds
2. Feed same features to both implementations
3. Compare final outputs

### Phase 3: Structural Validation
Compare predicted structures:
1. RMSD between predicted coordinates
2. TM-score for structural similarity
3. Metric comparison (pLDDT, pTM, PAE)
4. Visual inspection

## Metrics for Comparison

### Quantitative Metrics
1. **Coordinate RMSD**: Should be < 0.5 Å
2. **TM-score**: Should be > 0.95
3. **Metric Differences**:
   - pLDDT difference: < 1.0
   - pTM difference: < 0.01
   - PAE difference: < 0.5 Å

### Numerical Validation
1. **Array shapes**: Must match exactly
2. **Value ranges**: Should be similar
3. **Relative errors**: < 1e-3 for most operations
4. **Outliers**: Check for NaN/Inf

## Test Cases

### Test Case 1: Small Protein (Fast Validation)
- Sequence: 50 residues
- Purpose: Quick sanity check
- Expected time: ~5 seconds total

### Test Case 2: Medium Protein (Representative)
- Sequence: 128 residues
- Purpose: Real-world performance
- Expected time: ~30 seconds total

### Test Case 3: Different Sequences
- 5 diverse sequences
- Purpose: Ensure consistency across different inputs
- Expected time: ~2 minutes total

## Implementation Steps

### Step 1: Create Comparison Framework
```python
def compare_predictions(jax_output, mlx_output):
    """Compare JAX and MLX outputs."""
    results = {}

    # Compare coordinates
    results['rmsd'] = compute_rmsd(jax_output['positions'], mlx_output['positions'])
    results['tm_score'] = compute_tm_score(jax_output['positions'], mlx_output['positions'])

    # Compare metrics
    for metric in ['plddt', 'ptm', 'pae']:
        results[f'{metric}_diff'] = abs(jax_output[metric] - mlx_output[metric])

    return results
```

### Step 2: Extract Features from JAX
```python
def get_jax_features(sequence):
    """Get MSA and pair features from JAX ColabDesign."""
    jax_model = mk_afdesign_model(...)
    jax_model.prep_inputs(seq=sequence)

    # Extract internal features
    features = {
        'msa': jax_model._inputs['msa'],
        'pair': jax_model._inputs['pair'],
        # ... other features
    }
    return features
```

### Step 3: Run Both Models with Same Features
```python
# Get features from JAX
features = get_jax_features(sequence)

# Run JAX prediction
jax_output = run_jax_prediction(features)

# Convert features to MLX
mlx_features = jax_dict_to_mlx(features)

# Run MLX prediction
mlx_output = run_mlx_prediction(mlx_features)

# Compare
results = compare_predictions(jax_output, mlx_output)
```

## Success Criteria

### Must Pass (Critical)
- ✅ RMSD < 1.0 Å for same inputs
- ✅ No NaN or Inf in outputs
- ✅ Shapes match exactly
- ✅ TM-score > 0.90

### Should Pass (Important)
- ✅ RMSD < 0.5 Å for same inputs
- ✅ TM-score > 0.95
- ✅ Metric differences < 1%
- ✅ Consistent across multiple sequences

### Nice to Have (Optimization)
- ✅ RMSD < 0.1 Å
- ✅ TM-score > 0.99
- ✅ Identical metrics (within floating point precision)

## Known Differences to Account For

### Acceptable Differences
1. **Floating point precision**: JAX vs MLX may use different precision
2. **Random initialization**: If using random features
3. **Optimization differences**: Different backends may optimize differently

### Unacceptable Differences
1. **Large coordinate deviations** (> 2 Å RMSD)
2. **NaN or Inf values**
3. **Completely different structures** (TM-score < 0.5)
4. **Metrics off by orders of magnitude**

## Fallback Plans

### If Large Differences Found
1. **Debug component by component**: Find which layer differs
2. **Check intermediate outputs**: Validate Evoformer outputs
3. **Verify quaternion operations**: Check rotation matrices
4. **Review attention mechanisms**: Ensure proper broadcasting

### If Small Differences Found (< 1 Å)
1. **Acceptable**: Document and proceed
2. **Investigate**: Understand source (precision, optimization)
3. **Optimize if possible**: Match JAX behavior more closely

## Reporting

### Generate Report Including:
1. **Summary Statistics**: RMSD, TM-score, metric comparisons
2. **Visualizations**: Overlay structures, difference heatmaps
3. **Performance Comparison**: Timing for both implementations
4. **Recommendations**: Whether to proceed with MLX integration

---

## Next Steps

1. ✅ Implement comparison utilities
2. ✅ Run validation tests
3. ✅ Generate comparison report
4. ✅ Make go/no-go decision on MLX integration
