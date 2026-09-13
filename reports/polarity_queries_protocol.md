# Polarity query experiment protocol

The route uses process01 train/valid only, seed 42, embedding 128, a one-layer
bidirectional GRU with hidden size 384, and at most 8192 opcode tokens.
Training uses AdamW (LR 0.001, weight decay 0.0001), 30 epochs, patience 5,
and train-derived sqrt-ratio BCE positive weights capped at 5.
No test input is read. No historical checkpoint is used for initialization.

P0 is the existing B2/R0 model. P1 reads tokens with six queries and one shared
four-head cross-attention module. P2-P4 use six positive/negative query pairs,
the same shared attention module and per-label scorer shared across polarities.
P2 averages representations; P3 subtracts branch scores; P4 additionally uses
0.1 times the average unweighted BCE of positive scores against y and negative
scores against 1-y. Both branches run for every contract, regardless of labels.
Negative labels denote dataset negatives for that particular label, not proof
of whole-contract safety. Attention is not ground-truth localization.

Resource selection checks every model with alternating longest/median batches,
forward/backward, optimizer updates and diagnostic attention. It tries 64, 32,
then 16 and chooses one common batch; accumulation maintains effective batch 256.
CPU benchmarking compares 2/4/8 threads on P4 with the common batch. If the
8-thread run uses at least 6.4 CPU-core equivalents and improves throughput by
more than 5% over 4 threads, 16/20 threads are also checked within host affinity.
The lowest thread count within 5% of the best measured throughput wins.

`resources.json` records the measured selection. Every variant has audit,
history, best/last checkpoint, validation predictions, metrics and diagnostic
artifacts under its own directory in `polarity_queries_v1`. Encoder initial
weights are identical across variants and P2/P3/P4 full initial states match.
Initialization is repeated after preflight and prior to any training. Each
epoch checkpoint preserves optimizer, scaler and Python/NumPy/torch RNG states.
Resume requires matching configuration and code/data fingerprints. OOM stops
the queue; it never silently changes only one model's batch or architecture.

P4 must improve validation Macro-F1 over P0 by at least 0.01 and exceed P1/P2
to pass the performance gate. Branch score distributions, attention/representation
similarities, and zero-branch interventions at frozen validation thresholds are
required to interpret the mechanism. Single-seed screening is not a claim of
statistical stability. Hyperparameter search and prior automations stay paused.

Run: `bash scripts/run_polarity_queries.sh`. Use nohup with stdin redirected
from /dev/null for disconnection-safe operation. Repeating the command skips
matching completed models and resumes an unfinished model from its last epoch.
