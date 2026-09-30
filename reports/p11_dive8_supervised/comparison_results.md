# P11 DIVE-8 Full Supervision Results

Both Slurm jobs completed with exit code 0: original 235397 (01:18:21),
E3 235398 (01:50:18). Dataset: `data/processed/DIVE_8_opcode_random_split`,
full train=17,864 and valid=2,233. Seed 42, length 8,192, batch 16 with
accumulation 8, fixed LR 0.001, 30-epoch cap and patience 5. Both use the
original P11 loss (auxiliary 0.1, sqrt-ratio weights capped at 5, DoS soft
targets 0.8/0.1). Test remains locked.

## Overall Metrics

| Model | Best / stopped epoch | Fixed Macro-F1 | Fixed Micro-F1 | Tuned Macro-F1 | Tuned Micro-F1 | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Original | 12 / 17 | 0.748317 | 0.821463 | 0.779406 | 0.837512 | 3,710,608 |
| E3 | 15 / 20 | 0.777591 | 0.834529 | 0.793290 | 0.844264 | 5,331,472 |

E3 - original: tuned Macro-F1 +0.013885, tuned Micro-F1 +0.006753,
fixed Macro-F1 +0.029274. These are seed-42 validation measurements;
multi-seed stability and test performance have not been measured.

## Per-Label Results

F1 uses validation-selected per-label thresholds. AP is threshold-independent.

| Label | Valid positives | Original F1 | E3 F1 | Delta F1 | Original AP | E3 AP |
|---|---:|---:|---:|---:|---:|---:|
| Reentrancy | 1139 | 0.838459 | 0.841721 | +0.003262 | 0.876321 | 0.876470 |
| Access Control | 1674 | 0.894635 | 0.899225 | +0.004589 | 0.932643 | 0.933340 |
| Arithmetic | 950 | 0.790652 | 0.789841 | -0.000812 | 0.834885 | 0.832065 |
| Unchecked Return Values | 592 | 0.849462 | 0.865648 | +0.016185 | 0.898796 | 0.902800 |
| DoS | 375 | 0.659924 | 0.697318 | +0.037394 | 0.686152 | 0.741061 |
| Bad Randomness | 63 | 0.743802 | 0.709677 | -0.034124 | 0.761757 | 0.715314 |
| Front Running | 62 | 0.582524 | 0.661538 | +0.079014 | 0.578741 | 0.659322 |
| Time manipulation | 637 | 0.875786 | 0.881356 | +0.005570 | 0.937026 | 0.931703 |

The strongest gains are Front Running and DoS; both also improve AP.
Bad Randomness declines in F1 and AP. The two rare labels have only 62/63
validation positives, so their changes require cautious interpretation.

## Overfitting and Cost

Compare training classification BCE to validation classification BCE;
total training loss additionally contains auxiliary polarity loss.

| Model | Minimum valid BCE (epoch) | Best-F1 epoch train cls / valid BCE | Last epoch train cls / valid BCE | Epoch time | Peak allocated CUDA memory |
|---|---:|---:|---:|---:|---:|
| Original | 0.327402 (11) | 0.264175 / 0.331247 | 0.196577 / 0.351432 | about 4.52 min | about 2.60 GiB |
| E3 | 0.321338 (6) | 0.123277 / 0.442846 | 0.080890 / 0.555325 | about 5.46 min | about 3.75 GiB |

Both show declining training BCE and increasing validation BCE after its
minimum; this is substantially stronger for E3. Its minimum validation BCE
is at epoch 6, best tuned F1 at 15, and early stopping at 20. F1 can improve
while BCE worsens as confidence on remaining errors increases. The retained
checkpoint is the best-F1 epoch, not the final epoch.

## Artifact Verification

Metrics, history, per-label CSV, predictions and actual-model audit exist for
both variants. Local recomputation reproduces fixed/tuned Macro/Micro-F1.
Logits have shape [2233,8], polarity scores [2233,8,2]; values are finite,
ordered validation IDs and labels agree. Data hashes and shared configuration
match. All test flags are false.

Actual original: embedding128, BiGRU384/direction -> 768, queries [8,2,768],
attention4x192. Actual E3: embedding512, BiGRU512/direction -> 1024,
queries [8,2,512], attention8x64.

Small results were retrieved to `results/p11_dive8_supervised/{original,e3}/seed_42`.
Best/last checkpoints remain under the server directory
`/home/oywater/workspace/smart_1/checkpoints/p11_dive8_supervised`.

## Interpretation

E3 is the stronger candidate in this comparison, with +0.013885 tuned
Macro-F1. This experiment jointly changes embedding, encoder, query dimension
and head count; it cannot attribute the gain to one factor. It is a measured
architecture/configuration comparison, not validation of a new query mechanism.
No further training or test evaluation was started.
