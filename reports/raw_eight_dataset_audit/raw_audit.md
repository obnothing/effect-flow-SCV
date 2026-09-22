# Raw DIVE 8-label Dataset Audit

- Runtime rows: `22330`; unique contract IDs: `22330`.
- Solidity source files: `22330`; runtime/source ID overlap: `22330`.
- Runtime rows without source file: `0`; empty opcode rows: `0`.
- Runtime label-like fields: `[]`.
- DIVE_Samples fields: `['Address']`; label-like fields: `[]`.

The raw files contain opcode, source and metadata, but no 8-label vulnerability mapping.
A label mapping keyed by contractID/address is required before creating a supervised opcode dataset and a supervised source dataset.
Any 8:1:1 split produced before that mapping would be only an unlabeled membership split.
test_checked=false.

Potential mapping keys are `contractID` or the address mapping in `DIVE_Samples.csv`; the label names and per-contract assignments must come from the authoritative annotation source.
