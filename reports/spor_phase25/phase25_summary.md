# SPOR Phase 2.5 Summary and Decision

Dataset: `DIVE_8_opcode_random_split`; seed 42 baseline preflight; train/valid only; test locked.

## Findings

Strict coverage rose from 38.36% to 51.54% on train, and from 38.26% to 51.60% on valid.

Known-source rates improved for all four tracked opcodes, but CALL and SSTORE remain limited. Missingness alone is predictive for several labels, so a downstream SPOR model could use analysis availability as a label shortcut.

The metadata-only classifier has ROC-AUC >= 0.75 for: Front Running.

E3_P11_8_A0 forward/backward smoke passed; parameters=5,331,472, eight labels, 16 polarity queries, query/evidence 512, BiGRU output 1024, K/V 1024->512, 8x64 attention. No training or test evaluation was run.

## Decision: NO-GO for Phase 3

Do not integrate the full SPOR feature stream into P11 yet. 48.46% of train tokens still fail the strict criterion, CALL known-source coverage is 27.63%, SSTORE is 28.38%, and missingness-only prediction exposes a measurable shortcut. A later controlled phase may evaluate an explicitly selected reliable subset after adding missingness controls; current evidence does not support a clean full-SPOR causal test.

Safe interpretation today: retain the improved provenance cache for diagnostics and use the per-instruction coverage flags in future controlled experiments. Do not treat coverage availability as evidence of vulnerability semantics. Phase 3 remains pending human decision.
