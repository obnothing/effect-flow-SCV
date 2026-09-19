# P11 Hyperparameter Stage 1 Search Space

Exactly 18 successful single-seed validation trials are run. Trial 0 is fixed at the original P11 values and 40 epochs. Trials 1-17 use a seed-42 Optuna TPE sampler over four discrete fields:

- representation dropout: 0.00, 0.03, 0.05, 0.08, 0.10
- polarity coefficient: 0.03, 0.05, 0.08, 0.10, 0.15, 0.20
- positive-weight power: 0.40, 0.45, 0.50, 0.55, 0.60
- effective batch: 128 or 256

The harness prevents duplicate trained configurations. If TPE proposes an already completed tuple, the proposal is deterministically mapped to the nearest unused tuple and both values are recorded in Optuna user attributes. No pruning or early stopping is used. All trials run 40 epochs, and test remains locked.
