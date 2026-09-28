# E3/P11 architecture and SPOR cache interface

## Verified model dimensions

The existing `configs/p11_width_study/e3.yaml` sets `embedding_dim=512`,
`gru_hidden_size=512`, `bidirectional=true`, `attention_heads=8`, and
`query_dim=512`. `src/polarity_query_model.py` constructs the P11 variant
using `src/light_label_model.py`'s single-layer BiGRU. There is no Block
Transformer in this E3 route.

| Stage | Shape for one batch |
|---|---|
| Opcode IDs | `[B, T]` |
| Opcode embedding | `[B, T, 512]` |
| One-layer bidirectional GRU | `[B, T, 1024]` |
| Eight positive and eight negative queries | `[B, 16, 512]` |
| Shared Q projection | `512 -> 512` |
| Shared K/V projections | `1024 -> 512` each |
| Attention scores | `[B, 8, 16, T]`, with 64 dimensions per head |
| Evidence output | `[B, 8, 2, 512]` |
| Positive and negative energies | `[B, 8, 2]` |
| Competition logits | `[B, 8]` |

The checked-in E3 YAML is for six labels (`DIVE_main6_opcode_process01`).
Eight-label SPOR experiments require an eight-label E3 configuration and
fresh eight-label model training; a six-label checkpoint is not an eight-label
baseline. No locally verifiable full E3 checkpoint was found under
`checkpoints/p11_width_study/E3` during this Phase 2 audit. Phase 2 does not
modify model code or train a model.

## Feature contract

The cache builder reads only DIVE-8 train and valid opcode text and raw
disassembled runtime opcode text. It does not read vulnerability labels to
derive features and does not include labels in the feature cache. Each split
cache provides:

- `ids`: original dataset IDs in original split order;
- `offsets`: ragged row boundaries, length `len(ids)+1`;
- `token_ids`: original tokenizer IDs without special tokens;
- `provenance`: `uint8 [N_tokens, 13]` multi-hot bits;
- `operation_roles`: operation-role IDs per token;
- `analysis_status`: IDs from the quality report's `status_names` list;
- `valid_mask`: true only for instruction tokens with supported, known sources;
- `instruction_index` and `derived_pc`: per-token mapping; PC is nullable via
  sentinel `-1` after malformed PUSH data;
- `original_lengths`: token length per contract.

For contract row `r`, slice each flat token tensor with
`offsets[r]:offsets[r+1]`. Match `ids[r]` to the original JSONL row before
using features. A PUSH operand token points to the same instruction index as
its PUSH instruction, carries an `operand_alias` status and is not executed
again. Later batching must set provenance to zero and valid mask to false on
padding positions. A `false` valid mask is not proof of absent provenance:
check `analysis_status` to distinguish unknown entry, analysis failure,
operand alias, and not applicable.

The underlying runtime bytecode is unavailable. Derived PCs and source
provenance have basic-block-local text-level validity only.
