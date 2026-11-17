# Validation

The recorded inference checks use three proteins of lengths 126, 362 and 1,061
(1,549 residues), their cached float16 embeddings and the 72 original checkpoints.
The reference scores were assembled from the original experiment caches.

| Check | Recorded result | Report |
| --- | --- | --- |
| Predictor equivalence | Identical logits for all 96 network instances on the same CPU | [Architecture](validation/architecture_equivalence.json) |
| CPU inference against GPU reference scores | Maximum absolute error below `1e-5` for all four components and three final outputs | [Inference](validation/verification_cpu.json) |
| Command-line output | 1,549 rows; maximum final-score error below `5.30e-6` | [CLI](validation/cli_verification.json) |

These checks used Python 3.12 and PyTorch 2.5.1+cpu. The original training
environment is recorded in [original_environment.tsv](original_environment.tsv).

Run the example comparison after installing the weight bundle:

```bash
python scripts/verify_examples.py --weights weights --report verification.json
```

This is an inference regression check from cached embeddings. It does not cover
encoder re-extraction, full model retraining or regeneration of manuscript
tables and bootstrap statistics.
