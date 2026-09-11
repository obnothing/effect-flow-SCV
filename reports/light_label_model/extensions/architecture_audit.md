# B2 Extension Audit: E1--E5

## Scope

This audit covers the five mutually exclusive validation candidates built on
the local B2 label-specific attention model. The route uses only
`DIVE_main6_opcode_process01`, seed 42, and train/valid data. Test is locked
(`test_checked=false`) and is not loaded by the training or diagnostic code.

## B2 reference

The reference model is `b2_label_attention` from
`configs/light_label/b2_label_attention.yaml`:

```text
opcode ids [B, T]
  -> Embedding(vocab, 128) [B, T, 128]
  -> 1-layer bidirectional GRU(hidden=128) [B, T, 256]
  -> shared projection W_a [B, T, 256]
  -> six label queries q_l [6, 256]
  -> masked label-specific softmax attention [B, 6, T]
  -> six representations z_l [B, 6, 256]
  -> six label scorers and biases [B, 6]
```

Sequences are tokenized with the existing EVM opcode vocabulary, capped at
`max_len=8192`, and padded per batch. The padding mask is used by attention;
the GRU uses packed sequences. The common protocol is weighted BCE with the
existing square-root positive-ratio weights, AdamW, AMP when CUDA is
available, batch size 64 from the resolved 8 GiB runtime, accumulation 4,
30 epochs, and patience 5.

The current B2 validation reference is tuned Macro-F1 `0.799591`, Micro-F1
`0.818213`, and Detection-F1 `0.926323`. It is a validation reference only.

## Candidate mapping

| Variant | Retained idea | SCVD adaptation | Extra trainable mechanism |
|---|---|---|---|
| E1 VCFM | ACoL-style complementary search | erase each label's cumulative top attention mass `rho=0.30`, then run one masked second pass | one scalar beta, initialized `0.1` |
| E2 VROP | representative-feature propagation | select four detached-ranked intra-contract anchors and cosine-propagate them at temperature `0.2` | one scalar gamma and LayerNorm |
| E3 VASM | segment-level modeling | threshold attention at mean plus one standard deviation, merge gaps up to 2, retain up to 8 segments, one 4-head MHA | segment MHA/projection and scalar gamma |
| E4 PGVR | local prior refinement | Gaussian prior around each label's attention peak, six learned widths and gates, consistency weight `0.01` | six sigma, six eta |
| E5 TDVP | train-only context dictionary | K-means over label-agnostic masked means from train contracts only, `K=16`, 128-d context matching and fusion | dictionary plus context projections/fusion |

The candidates are never composed. Every extension is initialized from the
same B2 checkpoint, so the comparison isolates the added mechanism rather
than a fresh random B2 initialization.

## Data and leakage audit

- Training input: `data/processed/DIVE_main6_opcode_process01/train.jsonl`.
- Validation input: `data/processed/DIVE_main6_opcode_process01/valid.jsonl`.
- Test path exists in the dataset directory but is not opened by this route.
- E5 clusters only the train-loader representations and stores the resulting
  centroids as a fixed model buffer before optimization.
- No influence pseudo-label, retrieval memory, prototype, contrastive loss,
  graph feature, external model, or test artifact is used.

## Hardware estimate

The resolved local runtime reports an approximately 8 GiB GPU, batch size 64,
and B2 peak allocation around 3.7 GiB in the existing cache-based run. E1,
E2, and E4 add small tensor operations; E3 adds one small attention layer;
E5 adds a short train-only K-means preprocessing pass. The runner records
peak allocated memory, epoch time, and inference time for each candidate, so
these estimates are checked rather than treated as measured results.

## Decision rule

An extension is a serious candidate only when its validation tuned Macro-F1
beats B2 by at least `0.01`, its diagnostics support the intended mechanism,
its per-label behavior is not an isolated tradeoff, and it remains within
the local memory/runtime budget. Gains below `0.01` are reported as
performance fluctuation under the project decision rule.

