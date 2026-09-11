# E5 TDVP Implementation Audit

Dataset: `DIVE_main6_opcode_process01`; validation-only; `test_checked=false`.

## Verified implementation

- B2 vulnerability-specific representation: label-conditioned attention over BiGRU states.
- Context: label-agnostic masked mean of BiGRU states.
- Dictionary: train-loader-only KMeans, K=16 in the current code path.
- Dictionary: frozen buffer after construction; not optimized by backprop.
- Joint dimension: 128.
- Fusion: `Linear(512 -> 256)` followed by the original label scorer; no GELU or residual gate in the current artifact.

## Reproducibility findings

- Train samples: `16474`; valid samples: `2032`.
- Train/valid normalized-opcode hash overlap: `0`.
- Additional trainable parameters: `196864`.
- Existing E5 result: `0.809403315` Macro-F1, best epoch `22`.
- Existing artifact does not persist split hashes, train IDs hash, centroid hash, or cluster assignments.

## Terminology boundary

The current method should be described as context adjustment inspired by deconfounding. A causal identification claim is not verified.
