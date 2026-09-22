# SPOR Phase 1 Input Audit

- Dataset: `data\processed\DIVE_main6_opcode_process01`; splits: `{'train': 16474, 'valid': 2032, 'test': 2025}`.
- Runtime opcode rows: `22330`; source files: `22330`; compiler metadata rows: `22330`.
- Runtime/source ID alignment across process01: `False`.
- Normalized opcode mismatches: `409`.
- Label widths in process01: `{6: 20531}`; complete eight-label data: `False`.
- Original runtime bytecode available: `False`.
- Trusted disassembled opcode sequence available: `True`.
- Program counter independently verifiable against bytecode: `False`.
- Tokenizer alignment failures in audit sample: `0` examples.

## Tokenizer finding
The tokenizer recognizes PUSH0 and PUSH1-PUSH32, consumes the following hex operand as the PUSH immediate, and emits a normalized operand token. That operand token is a model token but is not an independently executed EVM instruction.

## SPOR boundary
Basic-block-local instruction order, PUSH width, instruction index and derived local program counters can be reconstructed from the trusted mnemonic/operand text. The original runtime bytes and independently verifiable program counters are absent, so the result would be a bounded text-level reconstruction rather than byte-verified or complete EVM execution semantics.

## Label boundary
process01 contains six labels, not the eight-label DIVE target. The eight-label DIVE labels exist separately in DIVE_Labels.csv, but they are not the labels currently attached to process01. The SPOR model experiment must keep the chosen dataset label definition explicit.

Test data was audited only for schema/alignment and remains locked for model selection: `test_checked=false`.
