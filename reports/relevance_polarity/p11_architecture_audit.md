# P11 Architecture Audit

Route: `P11 — Polarity-Decoupled Vulnerability Query Network`

Dataset: `data/processed/DIVE_main6_opcode_process01`

Protocol: seed 42; train/valid only; `test_checked=false`.

## Backbone

- Input is a padded opcode token sequence `[B,T]`, with `T <= 8192`.
- `nn.Embedding(vocab_size, 128, padding_idx=pad_id)` maps tokens to `[B,T,128]`.
- A packed one-layer bidirectional GRU uses hidden size 384 per direction and returns `H:[B,T,768]`.
- Padding is excluded by the boolean token mask. Packed sequence encoding also prevents padded suffixes from entering the GRU state updates.
- Representation dropout is 0.0 and the GRU dropout is 0.0.

## Independent polarity retrieval

P11 stores `queries:[6,2,768]`, initialized with `Normal(0,0.02)`. The second axis contains the positive and negative query for each label. The queries are flattened to 12 query tokens before shared four-head cross-attention.

The shared attention module contains exactly one set of bias-free projections:

- `W_Q:[768,768]`
- `W_K:[768,768]`
- `W_V:[768,768]`
- `W_O:[768,768]`

For each head, `d_h=192`. P11 computes token scores independently for every positive and negative query:

```text
r_(b,l,p,h,t) = <W_Q q_(l,p), W_K H_(b,t)> / sqrt(192)
alpha_(b,l,p,h,:) = softmax_t(r_(b,l,p,h,:))
```

Padding positions are filled with the minimum finite value before softmax. Consequently, `alpha+` and `alpha-` use shared projection parameters but are separate token distributions and are not structurally aligned.

The attended head values are concatenated and passed through the shared output projection. This yields `representations:[B,6,2,768]`.

## Scoring and loss

Each label has one scorer `w_l:[768]`, shared by its positive and negative branches, plus two branch biases. P11 computes:

```text
e_l+ = <w_l,z_l+> + b_l+
e_l- = <w_l,z_l-> + b_l-
s_l  = e_l+ - e_l-
```

The main objective is weighted BCE on `s`, using `sqrt_ratio` positive weights capped at 5. The auxiliary polarity objective is `0.5*(BCE(e+,y+) + BCE(e-,y-))`, multiplied by 0.1. Only the DoS auxiliary targets are softened to 0.8/0.1; the main classification targets remain hard labels.

Validation thresholds are selected independently per label from the fixed candidate grid. The verified P11 checkpoint reports tuned Macro-F1 0.8335071028, fixed Macro-F1 0.8270858271, Micro-F1 0.8439933916, and `test_checked=false`.

## Audit conclusion

P11 compares positive and negative energies produced from independently located token supports. Parameter sharing aligns the projection and scoring spaces, but it does not enforce positional evidence alignment. This is the precise mechanism changed by B1/B2.
