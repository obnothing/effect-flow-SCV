# Effect-Flow Guided Smart Contract Vulnerability Detection

This project implements an EVM behavior effect-flow framework for multi-label
smart contract vulnerability detection.

The current research direction is:

```text
opcode chunks
-> EVM-BERT representation
-> effect-flow behavior pretraining
-> EFPP/ETP semantic evidence
-> evidence-guided MIL
-> contract-level multi-label vulnerability detection
```

## Core Idea

The model explicitly decomposes smart contract vulnerability mechanisms into:

- risky behavior, such as external calls and value/gas-sensitive calls;
- protective behavior, such as guards and checked returns;
- missing-check behavior, such as unchecked call return and unguarded state writes.

These behaviors are learned through effect-flow pretraining tasks and then used
as semantic evidence in a Multiple Instance Learning detector.

## Main Components

```text
src/
  effect_flow_schema.py              # effect types and EFPP pattern rules
  effect_flow_pretraining_dataset.py # MOM + ETP + EFPP pretraining data
  effect_flow_pretraining_model.py   # EVM-BERT with effect-flow heads
  pretrain_effect_flow_evm_bert.py   # effect-flow continued pretraining
  evm_chunk_mil_model.py             # chunk context + guided MIL detector
  train_chunk_mil.py                 # detector training
  evaluate_chunk_mil.py              # detector evaluation and threshold search

scripts/
  build_effect_flow_pretraining_corpus.py
  audit_effect_flow_annotations.py
  audit_label_pattern_relevance.py
  extract_effect_flow_semantic_features.py
  train_eval_effect_flow_guided_mil.slurm

configs/
  pretrain_effect_flow_evm_bert_bjut_dive_train_balanced.yaml
  train_bjut_effect_flow_guided_mil.yaml
  train_dive_effect_flow_guided_mil.yaml
```

## Main Experiments

Effect-flow pretraining:

```bash
sbatch scripts/pretrain_effect_flow_evm_bert_bjut_dive_train_balanced.slurm
```

Effect-flow guided MIL on BJUT and DIVE:

```bash
sbatch scripts/train_eval_effect_flow_guided_mil.slurm
```

The guided MIL script runs the two datasets sequentially on two GPUs:

1. checks/generates BJUT chunk and semantic caches;
2. trains and evaluates BJUT guided MIL;
3. checks/generates DIVE 64-chunk caches;
4. trains and evaluates DIVE guided MIL;
5. writes semantic contribution summaries.

## Current Key Results

BJUT random split, effect-flow guided MIL:

```text
micro-F1:    0.6875
macro-F1:    0.5712
detection F1: 0.7970
```

DIVE random split, effect-flow guided MIL:

```text
micro-F1:    0.8309
macro-F1:    0.7424
detection F1: 0.9459
```

Detailed reports are kept under:

```text
results/train_bjut_effect_flow_guided_mil/
results/train_dive_effect_flow_guided_mil/
data/reports/
```

## Notes

- BJUT uses `max_chunks=32`.
- DIVE uses `max_chunks=64`.
- Chunk labels are not manually assigned. The detector uses contract-level
  labels and learns chunk-level evidence through MIL attention.
- EFPP and ETP outputs are semantic evidence, not final vulnerability labels.
