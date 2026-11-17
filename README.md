# XMoRF

XMoRF predicts molecular recognition features (MoRFs) from protein sequences. It
produces three residue-level scores: **core**, **region**, and **any-length**.
Use the any-length output when the extent of the binding segment is unknown.

This release candidate implements the final 72-checkpoint ensemble: 24 nested
models, 24 independent core predictors and 24 independent region predictors.
Each group comprises four folds and six seeds. The two frozen encoders are ProtT5
and ProstT5. A nested checkpoint contains two predictor networks, r and q.

## Installation

Use Python 3.10 or later. Install a PyTorch build appropriate for your CPU/GPU,
then install the package from this directory:

```bash
python -m pip install -e .
```

To extract new language-model embeddings, also install:

```bash
python -m pip install -e '.[features]'
```

The numerical verification included here used Python 3.12 and PyTorch 2.5.1+cpu.
The original experiment package versions are recorded in
`docs/original_environment.tsv`. These are provenance records, not a claim that
every dependency combination has been tested.

## Check the released model

Place the final model bundle in `weights/` (with `weights/manifest.json`). For an
existing experiment project, assemble it without retraining:

```bash
python scripts/prepare_weights.py --source-project /path/to/MoRF_project --output weights
python scripts/verify_examples.py --weights weights
```

The example proteins have lengths 126, 362 and 1,061. Their supplied float16
embeddings allow this check to run without downloading either language model.
The expected predictions were assembled from the original experiment caches,
independently of the new prediction entry point.

## Predict a FASTA file

For the supplied examples:

```bash
xmorf-predict --fasta examples/proteins.fasta --features examples/features \
  --weights weights --output predictions.tsv --device cpu
```

For new protein sequences, first extract both embeddings:

```bash
python scripts/extract_features.py --fasta proteins.fasta --output features --device cuda
xmorf-predict --fasta proteins.fasta --features features \
  --weights weights --output predictions.tsv --device cuda
```

The feature extraction command downloads the encoders on first use. Existing
local encoder directories can be supplied through `--prott5` and `--prostt5`.
Encoder extraction uses 1,000-residue chunks, a 750-residue stride and the
original weighted overlap-add procedure. Prediction networks then process each
whole protein; they do not crop it to 1,024 residues at inference.

The TSV contains the protein identifier, a **1-based residue position**, amino
acid, three continuous scores and three binary labels. Binary labels use the
fixed development-set thresholds recorded in the model manifest. These differ
from a threshold chosen retrospectively to maximize test-set F1.

## Model composition

Each checkpoint's outputs are smoothed using its original validation-selected
window before averaging. Smoothing never crosses protein boundaries.

```text
core       = (mean(smoothed r*q) + mean(smoothed q')) / 2
region     = (mean(smoothed r)   + mean(smoothed r')) / 2
any_length = sigmoid(0.35*logit(core) + 0.65*logit(region))
```

Probabilities are clipped to [1e-6, 1-1e-6] before the logit transform. There is
no additional smoothing after ensemble averaging. The raw nested r*q score is
bounded by r; independently smoothed and combined final outputs need not retain
that ordering.

## Training and data

See [docs/TRAINING.md](docs/TRAINING.md) for the exact configurations, original
training code and training commands. See [docs/DATA.md](docs/DATA.md) for dataset
provenance, masks and evaluation subsets. Original input files can be prepared
from the reviewed experiment project with `scripts/prepare_data.py`.

The original 48-checkpoint releases and the old single-output prediction script
are not part of this implementation. Historical optional branches remain only
in `training_reference/` to preserve the original training implementation;
`train_final.py` accepts the final configurations only.

## Verification and scope

All 72 checkpoints load with strict key/shape matching and contain finite tensor
values. For the three example proteins, all 96 individual predictor networks
produce bitwise-identical logits to the original classes on the same CPU.
Against the saved GPU predictions, all four ensemble components and all three
final outputs agree within 1e-5 absolute error. Detailed reports are under
`examples/`.

This check covers inference from the supplied embeddings. Regenerating the
embeddings from downloaded encoders and retraining all 72 checkpoints have not
been rerun in this release-preparation environment. No claim of full retraining
or reproduction of every manuscript table is implied by the example test.

## Citation and release status

Citation metadata are provided in `CITATION.cff`. No paper DOI is assigned here.
This is a private release candidate in [LJCwuwu/XMoRF](https://github.com/LJCwuwu/XMoRF).
Access is restricted to the repository owner and explicitly authorized accounts.
Model weights and full data archives are not included in the Git history.
The public release tag, weight-download location and software/model license are
to be finalized by the authors before public release.
Third-party encoders and data retain their original terms and citations.
