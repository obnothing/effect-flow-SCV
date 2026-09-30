# P11 DIVE-8 Full Supervision Comparison

Dataset: `data/processed/DIVE_8_opcode_random_split`, full original train
(17,864 contracts) and valid (2,233 contracts). Seed 42; test remains locked.
Both models are initialized from scratch with eight positive/negative query
pairs. The prior LOVO models and support episodes are not initialization inputs.

| Variant | Embedding | BiGRU hidden/direction | BiGRU output | Query | Heads |
|---|---:|---:|---:|---:|---:|
| original | 128 | 384 | 768 | 768 | 4 x 192 |
| e3 | 512 | 512 | 1024 | 512 | 8 x 64 |

Common protocol: max length 8,192; batch 16, accumulation 8 (effective 128);
AdamW, fixed LR 0.001, weight decay 0.0001; 30 epochs, patience 5;
weighted main BCE with sqrt-ratio positive weights capped at 5;
polarity auxiliary coefficient 0.1, DoS soft targets 0.8/0.1.
These shared loss settings are the original P11 settings. The earlier E3
smoke config used auxiliary 0.08 and weight exponent 0.6; they are deliberately
not used here so this comparison isolates the requested architecture changes.

Selection uses valid per-label tuned Macro-F1 with the existing 0.05-step
threshold grid. Fixed 0.5 metrics are also reported. Metrics are validation
results, not final test performance. Each variant saves actual dimensions,
parameter count, source/data hashes, epoch history, best/last checkpoints,
valid predictions and polarity scores, thresholds, per-label metrics and
TP/FP/FN/TN. Separate labeled valid caches preserve the LOVO cache protocol.

Entrypoint: `scripts/train_p11_dive8.py train --variant original|e3`.
Slurm: `scripts/slurm_p11_dive8.slurm original|e3` (one GPU/two CPUs per job).
