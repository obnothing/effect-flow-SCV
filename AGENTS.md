# EVEF-MVD Main-6 Guidance

## Scope

This repository is exclusively for opcode-only DIVE Main-6 experiments.
Do not reintroduce BJUT, legacy DIVE-8, CodeBERT variants, Front Running,
Bad Randomness, MLSMOTE, ASL, graph evidence, retrieval-label overrides, or
legacy Slurm pipelines.

## Dataset and Protocol

- Downstream data: `data/processed/DIVE_main6_random_split`.
- Labels: Reentrancy, Access Control, Arithmetic, Unchecked Return Values,
  DoS, Time manipulation.
- Fixed downstream seed: `42`.
- MLM + multi-role ETP pretraining may use only the random train split plus
  `ethereum_public_pretrain_19143_unique_runtime`.
- Never use valid/test labels or opcodes for model selection, pretraining, or
  feature construction.
- Do not use train-label lookup for duplicate test opcodes.

## Main Route

```text
public opcode MLM
-> train-only MLM + 17-role multi-label ETP pretraining
-> 64 x 768 chunk caches and token-level Top-2 ETP caches
-> label-decoupled ETP cross-attention MIL controls
-> validation-selected per-label thresholds
```

Use `configs/train_main6_random_090.yaml` and
`scripts/run_main6_random_090.sh`. Do not use vulnerability templates or
EFPP/ERR/VEP/VTM in this route.

## Validation

- Run `python -m py_compile` for changed Python modules.
- Run focused tests under the project torch environment.
- Check feature/cache alignment, shape, ID order, masks, label width, and
  NaN/Inf before training.
- Select a single candidate by validation macro-F1. Test exactly once with
  validation-frozen thresholds and `ALLOW_TEST=1`.
- Save config, manifest, thresholds, per-label metrics, TP/FP/FN and
  predictions for every final run.

## Reporting

Report DIVE Main-6 random split, seed 42, train-only pretraining, the absence
of label retrieval, and all six per-label F1 values. Do not claim macro-F1
greater than 0.9 until the final test artifact exists.
