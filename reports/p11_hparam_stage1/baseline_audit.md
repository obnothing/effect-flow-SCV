# P11 Hyperparameter Stage 1 Baseline Audit

## Protocol

- Dataset: `data/processed/DIVE_main6_opcode_process01`; train and validation only; test remains locked.
- P11 encoder: embedding 128, one-layer bidirectional GRU with 384 units per direction, output dimension 768.
- PDVQ head: query tensor `[6,2,768]`, shared four-head query-to-token cross-attention, label-wise shared scorer and `s=e+ - e-` competition.
- Optimizer: AdamW, learning rate 0.001, no scheduler, weight decay 0.0001, gradient clipping at norm 1.0 and AMP enabled.
- Epoch checkpoint selection uses the highest validation Tuned Macro-F1. All reported metrics are then recomputed from that single checkpoint.
- Per-label validation thresholds use the existing grid from 0.05 to 0.95 in increments of 0.05. Test is not read.

## Audited hyperparameters

### Representation dropout

Before this study, `PolarityQueryNet` inherited a dropout module from the older label-attention model but did not apply it in the polarity scorer, so the parameter had no effect on P11. Stage 1 adds the minimal intended operation: dropout is applied only to the positive/negative evidence representations immediately before the unchanged shared label scorer. No embedding, recurrent, attention or classifier dropout is added.

### Polarity coefficient

The configuration field `auxiliary_weight` is the effective `lambda_pol`. The implemented objective is:

`L = weighted_BCE(s, y) + lambda_pol * 0.5 * (BCE(e+, y_pol+) + BCE(e-, y_pol-))`.

P11 retains DoS soft targets 0.8/0.1 and hard complementary targets for the other five labels.

### Positive-class weights

Weights are computed from train labels only as `(N_negative / N_positive) ** pos_weight_power`, then clamped to a minimum of 1.0 to preserve the established P11 `sqrt_ratio` behavior and capped at 5.0. No validation or test label distribution is used. Each trial audit stores all six realized values.

### Effective batch

The physical batch remains 64. Effective batch 128 uses two gradient-accumulation steps; effective batch 256 uses four. Learning rate remains 0.001. The requested and effective values are both written to every trial audit.

## Loss and evaluation boundaries

- Training classification loss is weighted BCEWithLogitsLoss on `s=e+ - e-`.
- Training total loss includes the polarity auxiliary term; validation loss is classification-only.
- Validation Tuned Macro-F1 is the sole optimization objective.
- Trial 0 retrains the exact P11 hyperparameters for 40 epochs; it is not a reuse of the historical 30-epoch score.
