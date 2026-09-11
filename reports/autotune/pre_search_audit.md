# SCVD AutoTune Pre-search Audit

Dataset: `DIVE_main6_opcode_process01`; validation-only; `test_checked=false`.

## Baseline

- Configured max length: `8192`.
- Train coverage: `0.7958`.
- Valid coverage: `0.7830`.
- Existing tuned Macro-F1: `0.799591`.

## Blocking findings

- `gru_layers` is currently a dormant configuration field; multi-layer trials are prohibited until the implementation is wired and smoke-tested.
- Optuna available in the current Python environment: `False`.
- No test file is opened by this audit.
