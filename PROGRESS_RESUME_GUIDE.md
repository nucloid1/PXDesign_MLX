# Progress Tracking & Resume Capability

PXDesign now includes built-in progress tracking and resume capabilities for interrupted pipeline runs.

## Features

### 1. Real-Time Progress Counter

The pipeline now displays detailed progress during diffusion sampling:

```
============================================================
Starting chunked diffusion: 500 samples in chunks of 100
============================================================

[Chunk 1/5] Generating samples 1-100/500
  Diffusion step [1/400] for samples [1-100/500]
  Diffusion step [51/400] for samples [1-100/500]
  Diffusion step [101/400] for samples [1-100/500]
  ...
  Diffusion step [400/400] for samples [1-100/500]
✓ Completed chunk 1/5 (samples 1-100)

[Chunk 2/5] Generating samples 101-200/500
  Diffusion step [1/400] for samples [101-200/500]
  ...
```

**Progress indicators show:**
- Current chunk being processed
- Sample range (e.g., 1-100/500)
- Diffusion step (e.g., 51/400)
- Completion status for each chunk

### 2. Automatic Resume Capability

If the pipeline is interrupted (crash, Ctrl+C, system shutdown), you can resume from where it stopped.

#### How It Works

1. **CIF files saved incrementally**: Samples are saved as soon as they're generated
2. **Automatic detection**: On restart, the pipeline detects existing samples
3. **Skip completed work**: Already-generated samples are skipped
4. **Continue generation**: Only missing samples are generated

#### Resume Example

**Initial Run (interrupted after 150/500 samples):**
```bash
pxdesign pipeline --preset extended --input_json_path design.json --dump_dir results/

============================================================
Starting chunked diffusion: 500 samples in chunks of 100
============================================================

[Chunk 1/5] Generating samples 1-100/500
✓ Completed chunk 1/5

[Chunk 2/5] Generating samples 101-200/500
  Diffusion step [200/400] for samples [101-200/500]
^C  # Interrupted!
```

**Resume Run (continues from 150):**
```bash
# Run the exact same command
pxdesign pipeline --preset extended --input_json_path design.json --dump_dir results/

⚠ Resuming sample=my_design: 150 samples already exist, continuing generation...

  Saving CIF files: 350/500 (skipped 150 existing)
✓ Saved 350 new samples, skipped 150 existing samples
```

### 3. CIF File Progress

When saving CIF files, progress is displayed:

```
  Saving CIF files: 50/500 (skipped 0 existing)
  Saving CIF files: 100/500 (skipped 0 existing)
  Saving CIF files: 150/500 (skipped 0 existing)
  ...
  Saving CIF files: 500/500 (skipped 0 existing)
✓ Saved all 500 samples
```

## Usage

### Normal Run

No changes needed - progress tracking is automatic:

```bash
pxdesign pipeline --preset extended \
    --input_json_path my_design.json \
    --dump_dir results/
```

### Resume After Interruption

Simply re-run the **exact same command**:

```bash
# Same command as before
pxdesign pipeline --preset extended \
    --input_json_path my_design.json \
    --dump_dir results/  # Same output directory!
```

**Important**: Use the same `--dump_dir` to resume from existing files.

## How Resume Works Internally

### Detection Logic

1. **Check for existing samples**: Counts `.cif` files in `global_run_*/predictions/`
2. **Compare with total**: Determines how many samples are missing
3. **Resume or skip**:
   - If all samples exist: Skip task entirely
   - If some samples exist: Resume generation
   - If no samples exist: Start fresh

### File Structure

```
results/
└── global_run_0/
    └── my_design/
        └── seed_12345/
            ├── predictions/
            │   ├── my_design_sample_0.cif   ✓ Exists (skip)
            │   ├── my_design_sample_1.cif   ✓ Exists (skip)
            │   ├── ...
            │   ├── my_design_sample_149.cif ✓ Exists (skip)
            │   ├── my_design_sample_150.cif ✗ Missing (generate)
            │   └── ...
            └── SUCCESS_FILE  # Created when all done
```

### Smart Skipping

- **During generation**: All N samples generated in memory (can't skip)
- **During saving**: Existing CIF files are detected and skipped
- **Efficient**: Only writes new samples to disk

## Advanced Usage

### Force Fresh Start

To ignore existing samples and start over:

```bash
# Delete the output directory
rm -rf results/

# Run pipeline
pxdesign pipeline --preset extended \
    --input_json_path my_design.json \
    --dump_dir results/
```

### Check Progress Without Running

```bash
# Count existing samples
ls results/global_run_0/*/seed_*/predictions/*.cif | wc -l

# Check for completion marker
ls results/global_run_0/*/seed_*/SUCCESS_FILE
```

### Multiple Runs (Different Seeds)

Each seed creates a separate run directory:

```bash
# First run with seed 123
pxdesign pipeline --preset extended \
    --input_json_path design.json \
    --dump_dir results/ \
    --seeds 123

# Second run with seed 456 (separate directory)
pxdesign pipeline --preset extended \
    --input_json_path design.json \
    --dump_dir results/ \
    --seeds 456
```

Results are isolated:
```
results/
└── global_run_0/
    └── my_design/
        ├── seed_123/  # First run
        └── seed_456/  # Second run (independent)
```

## Benefits

### 1. No Wasted Computation
- Interrupted runs don't lose progress
- Resume exactly where you left off
- Save hours of GPU/CPU time

### 2. Better Visibility
- See exactly what's happening
- Estimate completion time
- Monitor long-running jobs

### 3. Fault Tolerance
- Survive crashes and interruptions
- Handle system restarts
- Robust for production workloads

## Limitations

### Current Behavior

**Generation is all-or-nothing**: The diffusion model generates all N samples in one pass. If interrupted during generation:
- ✗ Generated samples are lost (not saved yet)
- ✓ But you can restart immediately
- ✓ Previously saved CIF files are preserved

**Why**: PyTorch/diffusion models generate batches atomically in GPU memory.

### Workaround for Very Long Runs

Use smaller chunks to get more frequent saves:

```bash
pxdesign pipeline --preset extended \
    --N_sample 500 \
    --sample_diffusion_chunk_size 50  # Save every 50 samples
```

This means:
- Generation happens in 10 chunks of 50 samples
- Each chunk is saved immediately after generation
- Max loss if interrupted: 50 samples (~3 min on MLX)

## Troubleshooting

### "Resuming but still generating all samples"

**Reason**: Diffusion generates all samples before saving.

**Solution**: The samples will be generated, but only new CIFs will be written to disk (faster).

### "Samples counted incorrectly"

**Check**: Ensure you're using the same `--dump_dir` path.

```bash
# Wrong: Different directories
pxdesign pipeline --dump_dir results/
pxdesign pipeline --dump_dir ./results/  # Different path!

# Correct: Same directory
pxdesign pipeline --dump_dir results/
pxdesign pipeline --dump_dir results/   # Resumes ✓
```

### "Want to restart but samples still skipped"

**Solution**: Delete the output directory:

```bash
rm -rf results/
pxdesign pipeline --dump_dir results/  # Fresh start
```

## Example Session

```bash
# Start pipeline
$ pxdesign pipeline --preset extended --input_json_path binder.json --dump_dir run1/

Starting chunked diffusion: 500 samples in chunks of 100
[Chunk 1/5] Generating samples 1-100/500
  Diffusion step [1/400] for samples [1-100/500]
  ...
  Diffusion step [400/400] for samples [1-100/500]
✓ Completed chunk 1/5
  Saving CIF files: 100/500
^C

# Resume after interruption
$ pxdesign pipeline --preset extended --input_json_path binder.json --dump_dir run1/

⚠ Resuming sample=my_binder: 100 samples already exist, continuing generation...

[Chunk 2/5] Generating samples 101-200/500
  ...
```

---

**Generated**: 2025-12-19
**Pipeline Version**: PXDesign with MLX integration
