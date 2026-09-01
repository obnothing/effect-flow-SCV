# EVEF-MVD Main-6 Guidance

## Scope

This repository has six isolated experiment routes: historical opcode-only
DIVE Main-6, DIVE Source-Main6, DIVE Main6 Opcode-CSDG, DIVE Main6
Opcode-Execution-Aware, DIVE Main6 Opcode-Stack-Relational, and the Four Vulnerability Dataset route. Do not mix their
data, checkpoints, metrics, or claims. Do not reintroduce BJUT, legacy DIVE-8 labels, Front
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

## Opcode-CSDG Route

```text
Main-6 train-only continued MLM
-> existing 64 x 8 x 768 sequence branch
-> opcode-native basic-block CFG + conservative stack def-use graph
-> label-conditioned graph residual MIL
-> optional validation-qualified Gumbel-TopK node selection
```

Use `configs/train_main6_opcode_csdg.yaml` and the route-specific Slurm
scripts. This route may use only generic EVM control-flow and stack
dependencies; it must not use vulnerability templates, label retrieval, ETP,
EFPP/ERR/VEP/VTM, hard-negative mining, ASL, MLSMOTE, or PPO/REINFORCE.
Its graph caches, checkpoints, reports, and metrics live under the
`main6_opcode_csdg` directories and must not overwrite historical opcode-only
artifacts. Test remains locked until validation selection and threshold
freezing are complete.

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

## Opcode-Execution-Aware Route

```text
existing MLM8 opcode chunk views
-> conservative local EVM stack lineage/role analysis
-> chunk execution summaries
-> gated execution-aware multi-slot MIL
```

Use `configs/train_main6_execution_aware_mil.yaml` and its route-specific
scripts. This route uses generic EVM stack semantics only: producer/consumer
lineage, stack effects, stack height, role categories, and provenance. It does
not use vulnerability labels to construct features, and does not use AST/CFG
graphs, ETP, templates, label retrieval, ASL, MLSMOTE, PPO, or REINFORCE.
Its artifacts live under `main6_opcode_execution_aware`. The initial feature
implementation is bounded local stack analysis; unresolved underflow and
unsupported effects become unknown values rather than guessed dependencies.

## Opcode-Stack-Relational Route

```text
train-only continued EVM MLM
-> conservative global stack-value lineage
-> token stack-state embeddings + sparse relation-aware BERT attention
-> existing eight-view MLM chunk pooling
-> label-conditioned multi-slot MIL
```

The Stack-Relational route must use the v2 relation cache after structural
changes. Its recommended downstream protocol is train-only Stack-Aware MLM,
one-time frozen train/valid encoder feature extraction, then MIL-only training;
the live encoder training path is retained only for diagnostics. The v2 cache
passes cross-chunk lineage through CLS summary edges, uses per-head relation
gates, and preserves bounded producers at conservative control-flow joins.

An isolated `adapter_v3` diagnostic variant adds a lightweight Stack
Structural Encoder, gated embedding fusion, and Q/V LoRA adapters while
freezing the original EVM-BERT. Its artifacts use separate `adapter_v3` paths
and must not be mixed with v2 metrics.

Use `configs/train_main6_stack_relational.yaml` and
`scripts/run_main6_stack_relational.sh`. This is an independent input-encoding
route, not a post-BERT execution branch and not a graph/GNN route. It uses only
generic EVM stack semantics: operand-slot binding, producer-consumer lineage,
`DUP` aliases, `SWAP` reordering, conservative cross-block propagation, and
unknown states for unresolved execution. It must not use vulnerability labels
to construct relations, templates, retrieval, ETP, ASL, MLSMOTE, PPO, or
REINFORCE. Its artifacts live under `main6_opcode_stack_relational` and never
overwrite other routes. The default commands create only train/valid caches;
test remains locked until validation selection and threshold freezing.

## Four Vulnerability Dataset Route

The added `dataset_preprocessing_for_vulnerabilities` source is an independent
four-label benchmark: Delegatecall, Integer Overflow/Underflow, Reentrancy,
and Timestamp Dependence. Use `scripts/audit_four_vulnerability_dataset.py`
to merge the per-vulnerability name/label files, extract opcode from compiled
bytecode, and create a seed-42 contract-ID grouped split. Use only its
route-specific `four_vulnerability_*` data, checkpoints, reports, and results.
Do not mix its metrics with Main-6 metrics. The raw dataset and generated
JSONL/cache files are not Git artifacts. Test remains locked until validation
selection and threshold freezing.

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
