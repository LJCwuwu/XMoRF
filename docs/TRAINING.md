# Training

The final ensemble uses four configuration files in `configs/training/`:

| File | Predictor |
| --- | --- |
| `nested_r.json` | Region network of the nested model |
| `nested_q.json` | Conditional core network of the nested model |
| `core.json` | Independent core predictor |
| `region.json` | Independent region predictor |

Each configuration is shared across four folds and six seeds. The training
command sets `fold` and `seed`; all other parameters match the corresponding
original run configurations. The nested model uses S0 with `k_stop=0` and no
local attention. The independent core predictor masks non-core labels within
40 residues of a core. The language models remain frozen, and the predictors
are trained without warm starts.

## Data and folds

Prepare the input files from the original experiment directory:

```bash
python scripts/prepare_data.py --source-project /path/to/MoRF_project --output data
python scripts/train_final.py --role nested --fold 0 --seed 0 --data data --check-only
```

The preparation script uses the source paths and checksums in
`configs/data_sources.json`. See [Data](DATA.md) for benchmark sources and masks.

`configs/train421_folds.json` defines the TRAIN421 split: 421 records representing
375 unique sequences, with 103, 102, 120 and 96 records in the validation folds.
Identical sequences share a fold. MC2-ext uses its four supplied folds. The
historical 342-record `data/multitask/cl` subset is not used for this ensemble.

## Run training

Provide cached ProtT5 and ProstT5 embeddings for every training protein. Their
layout is the same as `examples/features/`. Run one fold and seed with:

```bash
python scripts/train_final.py --role nested --fold 0 --seed 0 \
  --data data --features /path/to/features --output training_outputs --device cuda
```

Run each role (`nested`, `core`, `region`) for folds 0–3 and seeds 0–5 to obtain
72 checkpoints. Each command runs one job; GPU scheduling is handled by the
caller. Existing run directories are not overwritten.

The wrapper sets local input/output paths and disables post-training test-set
reporting. Training architecture, losses, optimization, stopping rules and
validation-selected smoothing follow the implementation in `training/morf/`.
`training/source_manifest.json` records the original source file hashes.

Retrained weights need a new model manifest with their own checksums, smoothing
windows and development-set thresholds. The supplied ensemble manifest describes
the original checkpoints. CUDA/PyTorch versions and nondeterministic GPU kernels
can affect the result of retraining.
