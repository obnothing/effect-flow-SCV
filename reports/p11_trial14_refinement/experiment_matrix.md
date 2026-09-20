# Trial 14 Controlled Refinement

All runs use process01, seed 42, validation-only model selection, 40 epochs and the unchanged P11 encoder/PDVQ mechanism. Trial 14 is reused as the reference. Each new experiment changes exactly one field.

| Experiment | Single change |
|---|---|
| R1 | lambda_pol 0.05 |
| R2 | lambda_pol 0.10 |
| R3 | pos_weight_power 0.55 |
| R4 | pos_weight_power 0.65 |
| R5 | effective batch 64 |
| R6 | effective batch 256 |
| R7 | representation dropout 0.02 |
| E1 | opcode embedding 256 |
| E2 | opcode embedding 512 |
| E3 | opcode embedding 768 |
| Q1 | polarity query/evidence dimension 512 |
| Q2 | polarity query/evidence dimension 1024 |

Trial 2 from Stage 1 is reused as the existing dropout 0.05 comparison. Query-dimension variants retain `H:[B,T,768]`; only query, K/V projection outputs, evidence representations and the scorer use 512 or 1024 dimensions. Memory preflight starts at physical batch 64 and may reduce it while preserving the requested effective batch.
