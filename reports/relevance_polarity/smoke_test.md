# Relevance–Polarity Smoke Test

Dataset: process01 train subset 512 / valid subset 256; seed 42; test locked.

## B1

- shapes: `{"H": [64, 8192, 768], "alpha_rel": [64, 6, 4, 8192], "z_rel": [64, 6, 4, 192], "gates": [6, 2, 4, 192], "representations": [64, 6, 2, 768], "energies": [64, 6, 2], "logits": [64, 6]}`
- alpha sum maximum error: `2.384e-07`
- maximum padding attention: `0.000e+00`
- parameters: `3705996`; delta vs P11: `0`
- positive and negative branches consume the same `alpha_rel` and `z_rel` by construction.

## B2

- shapes: `{"H": [64, 8192, 768], "alpha_rel": [64, 6, 4, 8192], "z_rel": [64, 6, 4, 192], "gates": [6, 2, 4, 192], "representations": [64, 6, 2, 768], "energies": [64, 6, 2], "logits": [64, 6]}`
- alpha sum maximum error: `2.384e-07`
- maximum padding attention: `0.000e+00`
- parameters: `3710604`; delta vs P11: `4608`
- positive and negative branches consume the same `alpha_rel` and `z_rel` by construction.
