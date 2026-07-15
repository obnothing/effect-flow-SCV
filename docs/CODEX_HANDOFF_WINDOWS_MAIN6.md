# EVEF-MVD Windows Main-6 Handoff

## Current Objective

The only active route is local Windows/RTX 4060 experimentation:

```text
19,143 unique public-Ethereum runtime opcode sequences
-> effect-flow-aware EVM-BERT pretraining
-> new chunk features and six-label semantic cache
-> DIVE Main-6 multi-label vulnerability detection
```

Keep the project opcode-only. Do not use Slurm, server worktrees, BJUT, the
legacy eight-label DIVE split, or generic CodeBERT variants for this route.

## Preserved Local Data

These data directories are ignored by Git and must be backed up separately.

| Purpose | Path | Contents |
| --- | --- | --- |
| Pretraining | `data/processed/ethereum_public_pretrain_19143_unique_runtime/runtime_opcode.jsonl` | 19,143 public Ethereum contracts with unique runtime opcode hashes. |
| Downstream | `data/processed/DIVE_main6_access4000_clean3952/{train,valid,test}.jsonl` | New Main-6 benchmark: 17,865 / 2,208 / 2,209 rows. |
| Labels | `data/processed/DIVE_main6_access4000_clean3952/label_mapping.json` | Reentrancy, Access Control, Arithmetic, Unchecked Return Values, DoS, Time manipulation. |

The downstream train, valid and test splits have zero exact opcode-hash
overlap. This is a new benchmark and cannot be compared directly with legacy
`DIVE_random_split` metrics.

## Construction Facts

The Main-6 benchmark was constructed by projecting all 22,330 legacy DIVE
rows to six labels, removing exactly 4,000 Access Control positive rows with
seed 42, adding 3,952 SmartBugs Wild multi-tool-negative runtime contracts,
and then re-splitting by opcode-hash group.

The clean pool is multi-tool-negative, not formally proven safe. DIVE lacks
compiler metadata, so exact DIVE-clean compiler-era matching is unavailable.

The public pretraining corpus came from Blockscout verified-contract pages and
public Ethereum JSON-RPC runtime retrieval. Exact opcode overlap with the new
downstream benchmark was excluded. The original 100k-address collection had
only 19,143 unique runtime opcodes; the strict unique-opcode corpus is retained.

Read these reports before making claims:

- `data/reports/dive_main6_access4000_clean3952_report.txt`
- `data/reports/ethereum_public_pretrain_19143_unique_report.txt`
- `data/reports/ethereum_public_pretrain_100k_report.txt`

## Removed Assets

The following were intentionally removed from local disk and current master:

- BJUT data and results.
- Legacy raw/processed DIVE and DIVE random splits.
- SmartBugs raw source/results and address-level crawl intermediates.
- Old feature caches, graph/typed-mechanism caches, checkpoints and results.
- Generic CodeBERT, GraphCodeBERT and CodeBERTa model directories.

Many old configs still reference these paths. Do not run them unchanged.

## Core Code

| File | Responsibility |
| --- | --- |
| `src/pretrain_effect_flow_evm_bert.py` | Effect-flow-aware EVM-BERT pretraining. |
| `src/evm_pretrain_dataset.py` | Opcode pretraining dataset preparation. |
| `src/evm_chunk_mil_model.py` | Chunk Transformer, label-wise MIL, heads and side-evidence fusion. |
| `src/chunk_feature_dataset.py` | Chunk/semantic cache loading and label-name slicing. |
| `src/train_chunk_mil.py` | Downstream multi-label training. |
| `src/evaluate_chunk_mil.py` | Threshold calibration and strict evaluation. |
| `src/evm_opcode.py` | Runtime bytecode to project-standard opcode tokens. |

The established downstream shape is:

```text
chunk feature [B, 64, 768]
-> LayerNorm + 768-to-512 projection
-> chunk-context Transformer
-> label-wise gated MIL attention
-> label-specific representation and recognition head

attention += beta[label] * semantic_chunk_evidence
logit += gamma[label] * semantic_evidence_logit
```

Single-task recognition means one shared backbone plus six label-wise heads,
not six independent full backbones.

## Required Next Steps

1. Verify CUDA PyTorch on the RTX 4060 and obtain the intended EVM-BERT base
   checkpoint/tokenizer. The deleted generic models are not substitutes.
2. Create a new pretraining config for the 19,143-row corpus only. Split it
   by `opcode_hash` into pretraining train/valid before checkpoint selection.
3. Run effect-flow pretraining. Save config, checkpoint summary, loss history
   and validation metrics under a new local result directory.
4. Create fresh feature-extraction configs for the new Main-6 dataset:
   chunk size 512, stride 256, max chunks 64, feature dimension 768.
5. Create a fresh six-label semantic cache. Do not retain Bad Randomness or
   Front Running channels from legacy cache/config files.
6. Train a recognition-only six-label MIL baseline with weighted BCE and
   validation-selected per-label thresholds. Start batch size at 32 or 64 on
   the 8 GB RTX 4060 and use gradient accumulation if needed.
7. Tune only on validation. Run test once for the selected configuration and
   save TP/FP/FN, AP/AUC, predicted positives, thresholds and per-label F1.

Do not initially enable MLSMOTE, ASL, graph evidence, contrastive loss, hard
negatives or rare-negative subsampling. They were legacy diagnostic routes,
not this baseline.

## Guardrails

- Never tune thresholds or hyperparameters on test labels.
- Audit opcode-hash overlap after every new split or corpus append.
- Report clean samples as multi-tool-negative, not ground-truth secure.
- The 19,143-row corpus is small. Compare continued pretraining against an
  otherwise identical no-continued-pretraining baseline.
- Generated data, `.pt` caches, checkpoints and large predictions remain
  ignored. Commit code, configs, manifests and small reports only.

## Git State

Current branch: `master`.

```text
55e6e8d Remove retired datasets models and experiment artifacts
7dba8a1 Build external Ethereum runtime opcode pretraining corpus
33934cc Build balanced DIVE Main-6 clean benchmark
```
