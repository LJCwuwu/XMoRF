# Data and feature provenance

The reviewed project contains TRAIN421, TEST464, EXP53, the MC2 benchmarks,
homology-filtered MC2-ext, cross-benchmark filtered subsets and the IDEAL pool.
`configs/data_sources.json` records the exact source paths, sizes and SHA256
digests. The preparation script preserves file bytes and label masks.

| Dataset | Records | Positive residues | Negative residues |
|---|---:|---:|---:|
| TRAIN421 | 421 | 5,396 | 240,588 |
| TEST464, full input | 464 | 5,779 | 290,583 |
| EXP53 | 53 | 2,432 | 22,754 |
| MC2-ext, high-quality labels | 444 | 23,425 | 100,712 |
| IDEAL, final filtered evaluation | 38 | 1,072 | 10,797 |

Masked residues are excluded from those positive/negative counts. EXP53 labels
1 and 2 both represent MoRF residues. MC2 label '-' is ignored; it must not be
converted to a negative except in the explicitly unmasked benchmark variant.
The IDEAL source pool contains 39 records; the final filtered evaluation uses
38. Cross-method comparisons can use a further coverage intersection. In
particular, the full TEST464C input contains 196 records, whereas the manuscript's
common-method comparison uses 195. Do not interchange these denominators.

The raw benchmark files were obtained through
https://github.com/zhang403/MoRFs_AdapTrans and the original benchmark sources
cited in the manuscript. The public distribution route for third-party datasets
must preserve their original attribution and redistribution terms. This private
candidate therefore supplies the exact preparation manifest, rather than
declaring a new license for those data.

The frozen encoder identifiers recorded by the project are:

- ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc
- ProstT5: https://huggingface.co/Rostlab/ProstT5_fp16

The extraction implementation is preserved in
`training_reference/morf/features/plm.py`. Cache filenames are the MD5 of the
uppercase full protein sequence; each array has shape (residue count, 1024) and
dtype float16. The example cache contains only three benchmark proteins. The
encoder checkpoints and the complete training embedding cache are not bundled.
