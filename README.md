# Effect-Flow Guided Smart Contract Vulnerability Detection

This project implements an EVM behavior effect-flow framework for multi-label
smart contract vulnerability detection.

The current research direction is:

```text
opcode chunks
-> EVM-BERT representation
-> effect-flow behavior pretraining
-> ETP/EFPP/ERR/VEP/VTM semantic evidence
-> template-aware evidence-guided MIL
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
  effect_flow_schema.py              # effect types, EFPP patterns, and relations
  effect_flow_pretraining_dataset.py # MOM + ETP + EFPP + ERR + VEP + VTM data
  effect_flow_pretraining_model.py   # EVM-BERT with shared and vulnerability-specific heads
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

Unified EVEF-MVD one-click server pipeline:

```bash
sbatch scripts/stage_evef_mvd_pipeline.slurm
```

Or run selected segments directly:

```bash
bash scripts/stage_evef_mvd_pipeline.sh --mode status
bash scripts/stage_evef_mvd_pipeline.sh --mode corpus
bash scripts/stage_evef_mvd_pipeline.sh --mode pretrain_sanity
bash scripts/stage_evef_mvd_pipeline.sh --mode pretrain_full
bash scripts/stage_evef_mvd_pipeline.sh --mode downstream --dataset both
```

The guided MIL script runs the two datasets sequentially on two GPUs:

1. checks/generates BJUT chunk and semantic caches;
2. trains and evaluates BJUT guided MIL;
3. checks/generates DIVE 64-chunk caches;
4. trains and evaluates DIVE guided MIL;
5. writes semantic contribution summaries.

The unified pipeline goes one level higher and runs:

1. ontology/template-driven corpus build and audits;
2. Stage P1/P2 sanity pretraining;
3. Stage P1/P2 full pretraining;
4. BJUT semantic cache v2 extraction + template-aware guided MIL;
5. DIVE semantic cache v2 extraction + template-aware guided MIL;
6. threshold calibration and semantic contribution summaries.

## Current Stage

The repository is currently at:

```text
EVEF-MVD mainline code-complete, sanity-ready
```

Meaning:

- ontology and vulnerability templates are now first-class config files;
- corpus v2 now includes relation/template/evidence pseudo supervision;
- pretraining has been upgraded from MOM+ETP+EFPP to MOM+ETP+EFPP+ERR+VEP+VTM;
- semantic cache v2 now includes relation / evidence / template scores;
- downstream MIL has been upgraded to template-aware guided MIL;
- the unified pipeline checks stale v1 corpus/cache artifacts before reusing them;
- local syntax and minimal forward checks pass;
- the next hard boundary is server-side end-to-end runtime validation.

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
- ETP, EFPP, ERR, VEP, and VTM outputs are semantic evidence, not final
  vulnerability labels.
