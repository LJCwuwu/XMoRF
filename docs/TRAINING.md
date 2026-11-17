# Reproduce final-model training

`configs/original_runs/` contains the original configuration for every fold and
seed. `training_reference/` contains the original training implementation and a
SHA256 manifest. The final nested configuration uses S0, k_stop=0 and no local
attention. The independent core predictor excludes non-core labels within 40
residues of a core; this is the meaning of FT40 in the historical run names.
No predictor warm start is enabled and the language models remain frozen.

The final TRAIN421 split is `configs/train421_folds.json`: 421 benchmark records,
375 unique sequences, with 103/102/120/96 records in the four validation folds.
Identical sequences share the same fold. MC2-ext retains its four supplied folds.
The older 342-record `data/multitask/cl` directory is **not** the final TRAIN421
split and is not included in this release.

Prepare the input files:

```bash
python scripts/prepare_data.py --source-project /path/to/MoRF_project --output data
python scripts/train_final.py --role nested --fold 0 --seed 0 --data data --check-only
```

Provide cached ProtT5 and ProstT5 embeddings for every training protein, then run
one job, for example:

```bash
python scripts/train_final.py --role nested --fold 0 --seed 0 \
  --data data --features /path/to/features_v2 --output training_outputs --device cuda
```

Run each of `nested`, `core` and `region` for folds 0–3 and seeds 0–5 to obtain
72 checkpoint files. Use separate GPU processes as appropriate for the hardware;
the wrapper does not schedule GPUs automatically. Existing run directories are
not overwritten.

The wrapper changes input/output locations and disables post-training test-set
reporting. It preserves the training architecture, loss, fold assignment,
optimizer, stopping rule and validation-selected smoothing. The original source
modules are preserved byte for byte. Exact weight identity after retraining is
not guaranteed across CUDA/PyTorch versions or nondeterministic GPU kernels.

The published ensemble manifest describes the supplied trained weights. Newly
trained weights require a new manifest with their own hashes, smoothing windows,
development-set thresholds and validation results; do not substitute them into
the original manifest silently.
