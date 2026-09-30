# Few-shot Pilot Split Audit

Dataset: `data/processed/DIVE_8_opcode_random_split`; strict_novel=True; seed=42.
Only official valid/test IDs were extracted for set-intersection checks; no valid/test labels or opcode content were read during split construction.

| Novel label | Original train positive | Base train | Base dev | Positive support pool | Negative support pool | status |
|---|---:|---:|---:|---:|---:|---|
| Reentrancy | 9119 | 7781 | 864 | 100 | 100 | ready |
| Arithmetic | 7634 | 9117 | 1013 | 100 | 100 | ready |
| DoS | 3026 | 13264 | 1474 | 100 | 100 | ready |
| Bad Randomness | 509 | 15529 | 1726 | 100 | 100 | ready |

Every LOVO split JSON contains the exact ID sets, zero pairwise intersections, 7-label mapping, and five support episodes.
