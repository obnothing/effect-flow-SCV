# SPOR Phase 1 Input Audit

- Dataset: `data\processed\DIVE_8_opcode_random_split`; splits: `{'train': 17864, 'valid': 2233, 'test': 2233}`.
- Runtime opcode rows: `22330`; source files: `22330`; compiler metadata rows: `22330`.
- Runtime/source ID alignment across selected DIVE_8 dataset: `True`.
- Normalized opcode mismatches: `0`.
- Label widths: `{8: 22330}`; complete expected-label data: `True`.
- Original runtime bytecode available: `False`.
- Trusted disassembled opcode sequence available: `True`.
- Program counter independently verifiable against bytecode: `False`.
- Tokenizer alignment failures in audit sample: `0` examples.

## Tokenizer finding
The tokenizer recognizes PUSH0 and PUSH1-PUSH32, consumes the following hex operand as the PUSH immediate, and emits a normalized operand token. That operand token is a model token but is not an independently executed EVM instruction.

## SPOR boundary
Basic-block-local instruction order, PUSH width, instruction index and derived local program counters can be reconstructed from the trusted mnemonic/operand text. The original runtime bytes and independently verifiable program counters are absent, so the result would be a bounded text-level reconstruction rather than byte-verified or complete EVM execution semantics.

## Label boundary
The selected DIVE_8 dataset contains the expected eight labels from the official `DIVE_Labels.csv` assignments. The label order is recorded in the dataset manifest and remains fixed for SPOR.

Test data was audited only for schema/alignment and remains locked for model selection: `test_checked=false`.
