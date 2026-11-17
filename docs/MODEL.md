# Model details

## Ensemble

The model bundle contains 72 checkpoints: 24 nested models, 24 independent core
predictors and 24 independent region predictors. Each group covers four folds
and six seeds. A nested checkpoint contains two networks, `r` and `q`.

Each checkpoint's scores are smoothed with its validation-selected window before
averaging. Smoothing is applied within each protein.

```text
core       = (mean(smoothed r*q) + mean(smoothed q')) / 2
region     = (mean(smoothed r)   + mean(smoothed r')) / 2
any_length = sigmoid(0.35*logit(core) + 0.65*logit(region))
```

Scores are clipped to `[1e-6, 1-1e-6]` before the logit transform. There is no
additional smoothing after ensemble averaging. The raw nested score `r*q` is
bounded by `r`; the separately smoothed and combined final outputs need not
retain this ordering.

Encoder extraction uses 1,000-residue chunks, a 750-residue stride and weighted
overlap-add. The prediction networks process each full protein, without a
1,024-residue inference crop.

## Weight bundle

The predictor reads `weights/manifest.json`. Model paths are relative to
`weights/`; the manifest specifies each checkpoint's SHA256 checksum, role,
fold, seed and smoothing window, along with the development-set thresholds.

To assemble a bundle from checkpoint files stored in a local directory, run:

```bash
python scripts/prepare_weights.py --source /path/to/checkpoints --output weights
```

The command searches recursively for `.pt` and `.pth` files and matches them to
the SHA256 checksums in `configs/ensemble.json`. It copies the matching files
into the bundle layout and writes the manifest. Filenames and source folder
names may differ. Missing files or conflicting output files stop the import.
