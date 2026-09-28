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

`python scripts/test_spor.py` passed 18/18 checks, including:

1. PUSH width and derived-PC accounting;
2. no independent instruction for PUSH immediate data;
3. DUP/SWAP order;
4. multi-source comparison provenance;
5. separation of JUMPI destination and condition;
6. EVM CALL argument order (gas is the top stack item);
7. SSTORE key/value and MSTORE offset/value order (key/offset is the top stack item);
8. basic-block stack isolation;
9. unsupported-instruction failure handling;
10. tokenizer alignment for ordinary and unknown-opcode annotation text;
11. propagation of unknown-entry and failed states to later operations;
12. malformed PUSH does not invent a zero value or a verified PC.

Operand order was cross-checked against the [Ethereum opcode reference](https://ethereum.org/developers/docs/evm/opcodes)
and the [Ethereum Yellow Paper](https://ethereum.github.io/yellowpaper/paper.pdf).
The top of stack is the first argument: `MSTORE(offset, value)`,
`SSTORE(key, value)`, and `CALL(gas, to, value, in_offset, in_size,
out_offset, out_size)`.

`python scripts/smoke_spor_real.py` also completed on representative DIVE-8
contracts containing PUSH0, CALL, JUMPI, SLOAD, SSTORE, and long opcode
sequences. Its traces are in `real_data_smoke_traces.json` and
`real_data_smoke_traces.md`.

One manually inspected example is contract `20346` (compiler 0.8.20): its
first JUMPI has destination constant 308 and a condition produced by `LT`
from calldata and a constant. The trace preserves those operands separately.
In the same contract, a later SSTORE has a known constant key 9 but an
unknown-entry value, so the whole instruction is not counted as fully known.
Many CALLs in this example remain analysis failures because unsupported stack
effects occur earlier in their blocks; these are not presented as reliable
call-target provenance.

## Cache and quality summary

The generated caches are:

- `data/features/spor_dive8/train.pt`
- `data/features/spor_dive8/valid.pt`

The cache stores token-aligned provenance `[N_tokens, 13]`, operation-role IDs,
analysis-status IDs, valid masks, instruction indices, derived PCs, token IDs,
contract offsets, and IDs. Vulnerability labels are not stored in the cache.
The quality report is
`spor_feature_quality.json`.

| split | contracts | mean instructions | mean model tokens | mean basic blocks | parse width mismatches | unknown annotations |
|---|---:|---:|---:|---:|---:|---:|
| train | 17,864 | 4,343.77 | 5,523.91 | 402.91 | 313 | 159,359 |
| valid | 2,233 | 4,332.11 | 5,509.64 | 401.44 | 33 | 21,295 |

The token-level valid mask is true only when an instruction token has a
supported, known source. It is false for operand aliases, unknown-entry
sources, failures, and not-applicable tokens. The cache is ragged and
offset-indexed; later batching must zero padding features and masks. Unknown
provenance uses the `UNKNOWN` bit; status IDs distinguish unknown entry,
analysis failure, operand alias, and not applicable.

| Split | Known instruction | Unknown entry | Analysis failure | Strictly valid token | Unknown-source token |
|---|---:|---:|---:|---:|---:|
| train | 55.29% | 30.32% | 14.40% | 38.36% | 38.04% |
| valid | 55.15% | 30.26% | 14.60% | 38.26% | 38.19% |

Known-source coverage at important instructions (train / valid):

| Instruction | train | valid |
|---|---:|---:|
| JUMPI | 28.08% | 28.19% |
| CALL | 1.40% | 1.40% |
| SLOAD | 58.07% | 58.38% |
| SSTORE | 18.70% | 18.98% |

All 20,922,729 train and 2,608,133 valid PUSH operand model tokens have an
instruction mapping. No model tokens were left unmapped. The two caches occupy
about 3.26 GB and 0.41 GB respectively. The quality JSON provides per-compiler
coverage (181 train and 98 valid compiler-version groups), p50/p95 processing
times, failure examples, and target-status counts.

## Limitations and decision

The original runtime bytecode is not present in the current project inputs.
Therefore program counters are derived from parsed PUSH widths and cannot be
independently verified against byte offsets. The input is sufficiently aligned
for an explicitly bounded, disassembled-opcode-based Phase 2 experiment, but
not sufficient to claim exact bytecode-level provenance or complete inter-block
data-flow analysis. The 313/33 PUSH-width mismatches and the high number of
unsupported/underflow states must remain visible in downstream reports.

**Phase 2 decision: NO-GO for Phase 3 at present.** The parser/tokenizer
alignment gate passes, but the requested stopping condition is met: important
target instructions, especially CALL, overwhelmingly lack fully known local
operands. The full model input would receive valid SPOR features at only about
38% of tokens. Improve verified opcode semantics or add a separately audited
conservative cross-block mechanism before using SPOR as a performance feature.
This result is a data-quality diagnosis, not a vulnerability-detection result.
