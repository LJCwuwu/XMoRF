# XMoRF

XMoRF predicts molecular recognition features (MoRFs) from protein sequences.
It returns residue-level scores for binding **cores**, broader binding
**regions**, and **any-length** MoRFs. The predictor combines frozen ProtT5 and
ProstT5 embeddings with an ensemble of nested, core and region models.

## Installation

Use Python 3.10 or later and a PyTorch build suitable for your CPU or GPU.

```bash
git clone https://github.com/LJCwuwu/XMoRF.git
cd XMoRF
python -m pip install -e .
```

To generate embeddings for new sequences, install the feature dependencies:

```bash
python -m pip install -e '.[features]'
```

## Model weights

Prediction requires the separate model bundle containing 72 checkpoints and
`manifest.json`. Extract the bundle into `weights/`, so that the manifest is at
`weights/manifest.json`. The checkpoints are not stored in this repository;
a public download link is not yet available.

The bundle layout and preparation command are described in
[Model details](docs/MODEL.md).

## Prediction

The three example proteins include cached embeddings and reference predictions.
After installing the model weights, run:

```bash
xmorf-predict --fasta examples/proteins.fasta --features examples/features \
  --weights weights --output predictions.tsv --device cpu
```

For a new protein FASTA file, extract embeddings and then predict:

```bash
python scripts/extract_features.py --fasta proteins.fasta --output features --device cuda
xmorf-predict --fasta proteins.fasta --features features \
  --weights weights --output predictions.tsv --device cuda
```

Feature extraction downloads ProtT5 and ProstT5 on first use. To use local encoder
directories, pass `--prott5` and `--prostt5`. Use `--device cpu` for CPU execution.

The output is a tab-separated file with one row per residue:

| Columns | Description |
| --- | --- |
| `protein_id`, `position`, `residue` | FASTA identifier, 1-based position and amino acid |
| `core`, `region`, `any_length` | Continuous prediction scores |
| `core_label`, `region_label`, `any_length_label` | Binary predictions at the development-set thresholds in the model manifest |

Use the any-length score when the extent of a binding segment is unknown.

## Training

The ensemble contains 24 nested, 24 core and 24 region checkpoints, covering
four folds and six seeds. See [Training](docs/TRAINING.md) for data preparation,
fold assignments and commands. Dataset sources and label conventions are listed
in [Data](docs/DATA.md).

## Repository contents

| Directory | Contents |
| --- | --- |
| `src/xmorf/` | Prediction models and command-line interface |
| `training/` | Training and feature-extraction implementation |
| `configs/` | Four training configurations, fold assignments and model/data manifests |
| `scripts/` | Feature extraction, data preparation, training and example checks |
| `examples/` | Three protein sequences, cached embeddings and expected scores |
| `docs/` | Model, data, training and validation documentation |
| `tests/` | Resource-import regression tests |

To compare the example predictions with the stored reference scores:

```bash
python scripts/verify_examples.py --weights weights
```

Recorded results and the scope of this check are in
[Validation](docs/VALIDATION.md). Citation metadata are provided in
[CITATION.cff](CITATION.cff).
