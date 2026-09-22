# SPOR Phase 2: Local Stack Provenance Reconstruction

## Scope

- Dataset: `DIVE_8_opcode_random_split`
- Splits processed: train and valid only
- Test: locked; `test_checked=false`
- Input: trusted disassembled runtime-opcode text aligned with the existing
  `EVMOpcodeTokenizer`
- Analysis scope: conservative basic-block-local abstract stack analysis
- This report does not claim full runtime-bytecode recovery, complete CFG data
  flow, or real execution-path semantics.

## Implementation

The parser records instruction index, derived program counter, opcode,
immediate operand, source-token positions, model-token positions, parse status,
and basic-block membership. `PUSH0` and `PUSH1`--`PUSH32` consume their
immediate bytes as part of the instruction; the immediate is not treated as a
second EVM instruction. Unknown-opcode annotations such as `'FE' (Unknown
Opcode)` are retained as explicit parse-status records.

The local abstract stack tracks multi-hot provenance categories for constants,
caller/call value/calldata, block environment, storage, memory, call results,
arithmetic, comparison, other, and unknown values. `DUP`, `SWAP`, `POP`, basic
arithmetic/comparison operations, `CALL`, `JUMPI`, `SLOAD`, and `SSTORE` are
handled. At every basic-block entry the stack is reset to unknown rather than
being incorrectly carried across a control-flow boundary. Unsupported stack
effects clear the local stack and emit an explicit analysis-failure value.

## Correctness checks

`python scripts/test_spor.py` passed 10/10 checks, including:

1. PUSH width and derived-PC accounting;
2. no independent instruction for PUSH immediate data;
3. DUP/SWAP order;
4. multi-source comparison provenance;
5. separation of JUMPI destination and condition;
6. EVM CALL argument order;
7. SSTORE value/key order;
8. basic-block stack isolation;
9. unsupported-instruction failure handling;
10. tokenizer alignment for ordinary and unknown-opcode annotation text.

`python scripts/smoke_spor_real.py` also completed on representative DIVE-8
contracts containing PUSH0, CALL, JUMPI, SLOAD, SSTORE, and long opcode
sequences. Its traces are in `real_data_smoke_traces.json` and
`real_data_smoke_traces.md`.

## Cache and quality summary

The generated caches are:

- `data/features/spor_dive8/train.pt`
- `data/features/spor_dive8/valid.pt`

The cache stores token-aligned provenance `[N_tokens, 13]`, operation-role IDs,
analysis-status IDs, valid masks, instruction indices, derived PCs, token IDs,
contract offsets, IDs, and eight-label targets. The quality report is
`spor_feature_quality.json`.

| split | contracts | mean instructions | mean model tokens | mean basic blocks | parse width mismatches | unknown annotations |
|---|---:|---:|---:|---:|---:|---:|
| train | 17,864 | 4,343.77 | 5,523.91 | 402.91 | 313 | 159,359 |
| valid | 2,233 | 4,332.11 | 5,509.64 | 401.44 | 33 | 21,295 |

The token-level valid mask is true for all stored non-padding model tokens. The
cache is ragged and offset-indexed, so padding is introduced only by a later
consumer; that consumer must set provenance features to zero at padding
positions. Unknown provenance is represented by the `UNKNOWN` bit, while
`unknown_entry`, `analysis_failure`, and parse-status failure remain separate
status values.

## Limitations and decision

The original runtime bytecode is not present in the current project inputs.
Therefore program counters are derived from parsed PUSH widths and cannot be
independently verified against byte offsets. The input is sufficiently aligned
for an explicitly bounded, disassembled-opcode-based Phase 2 experiment, but
not sufficient to claim exact bytecode-level provenance or complete inter-block
data-flow analysis. The 313/33 PUSH-width mismatches and the high number of
unsupported/underflow states must remain visible in downstream reports.

**Phase 2 status: READY FOR A SMALL SPOR CONSUMER SMOKE TEST, NOT YET READY
FOR A FULL PERFORMANCE CLAIM.** Any P11 integration must preserve the raw
P11 path when SPOR is disabled and must report coverage and failure states.
