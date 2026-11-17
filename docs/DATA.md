# Data

XMoRF uses TRAIN421, TEST464, EXP53, the MC2 benchmarks, homology-filtered
MC2-ext, cross-benchmark filtered subsets and an IDEAL evaluation set.
`configs/data_sources.json` records source paths, file sizes and SHA256 checksums.
`scripts/prepare_data.py` copies these files from the original experiment
directory without changing their labels or masks.

## Benchmarks

| Dataset | Records | Positive residues | Negative residues |
| --- | ---: | ---: | ---: |
| TRAIN421 | 421 | 5,396 | 240,588 |
| TEST464, full input | 464 | 5,779 | 290,583 |
| EXP53 | 53 | 2,432 | 22,754 |
| MC2-ext, high-quality labels | 444 | 23,425 | 100,712 |
| IDEAL, final filtered evaluation | 38 | 1,072 | 10,797 |

Masked residues are excluded from the positive and negative counts. EXP53 labels
1 and 2 both denote MoRF residues. MC2 label `-` is ignored, except in the
explicitly unmasked benchmark variant.

The IDEAL source pool contains 39 records; the filtered evaluation uses 38.
Cross-method comparisons may use a further intersection of prediction coverage.
For example, TEST464C has 196 input records, while the common-method comparison
uses 195.

The benchmark files were obtained through
[MoRFs_AdapTrans](https://github.com/zhang403/MoRFs_AdapTrans) and the original
sources cited in the manuscript. The full datasets are not bundled here.
Third-party datasets retain their original attribution and usage terms.

## Protein embeddings

The frozen encoders are:

- [ProtT5](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)
- [ProstT5](https://huggingface.co/Rostlab/ProstT5_fp16)

Feature extraction is implemented in `training/morf/features/plm.py`. Cache
filenames are the MD5 of the uppercase full protein sequence. Arrays have shape
`(sequence_length, 1024)` and dtype `float16`, stored under `prott5/` and
`prostt5/`.

`examples/` contains three benchmark proteins, their cached embeddings and
reference predictions. `examples/provenance.json` records their sources. Encoder
checkpoints and the complete training embedding cache are stored separately.
