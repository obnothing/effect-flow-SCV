# EVEF-MVD Main-6 Guidance

## Scope

This repository has two isolated experiment routes: historical opcode-only
DIVE Main-6, and DIVE Source-Main6. Do not mix their data, checkpoints,
metrics, or claims. Do not reintroduce BJUT, legacy DIVE-8 labels, Front
Running, Bad Randomness, MLSMOTE, ASL, retrieval-label overrides, or legacy
Slurm pipelines.

## Dataset and Protocol

- Downstream data: `data/processed/DIVE_main6_random_split`.
- Labels: Reentrancy, Access Control, Arithmetic, Unchecked Return Values,
  DoS, Time manipulation.
- Fixed downstream seed: `42`.
- Main-6 continued MLM may use only the random train split, initialized from
  the public `ethereum_public_pretrain_19143_unique_runtime` MLM checkpoint.
- Never use valid/test labels or opcodes for model selection, pretraining, or
  feature construction.
- Do not use train-label lookup for duplicate test opcodes.
- DIVE Source-Main6 may use only source files mapped from `DIVE_Raw_Data/Raw`.
  It must retain exactly the same six Main-6 labels, audit runtime alignment,
  group duplicate normalized sources, and split groups with seed 42.
- Source-Main6 DAPT is train-source-only. Valid/test source is inference input,
  never a pretraining or model-selection corpus. Test remains single-use.

## Main Route

```text
public opcode MLM
-> train-only continued MLM
-> 64 x 8 x 768 pure-MLM chunk-view caches
-> label-conditioned multi-slot hierarchical MIL
-> validation-selected per-label thresholds
```

Use `configs/train_main6_random_090.yaml` and
`scripts/run_main6_random_090.sh`. Do not use ETP, vulnerability templates,
EFPP/ERR/VEP/VTM, graph evidence, or label retrieval in this route.

## Source-Main6 Route

```text
audited Solidity source groups
-> train-only GraphCodeBERT DAPT
-> Solidity data-flow graph attention
-> label-conditioned multi-slot MIL
-> validation-selected thresholds
```

Use `configs/train_dive_source_main6_graphcodebert.yaml` and
`scripts/run_dive_source_main6_graphcodebert.sh`. This route may use
GraphCodeBERT and conservative Solidity def-use graphs only; it must not use
opcode caches, ETP, templates, hard-negative mining, ASL, MLSMOTE, or labels
outside the six Main-6 labels.

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

Report the exact route and dataset name, seed 42, train-only pretraining, the
absence of label retrieval, and all six per-label F1 values. Do not claim
macro-F1 greater than 0.9 until the corresponding final test artifact exists.
