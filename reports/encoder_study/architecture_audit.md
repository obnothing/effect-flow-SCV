# PDVQ Encoder Architecture Audit

- Dataset: `data/processed/DIVE_main6_opcode_process01`; train/valid only; test locked.
- Reference: P11 PDVQ-Net, validation tuned Macro-F1 0.833507, seed 42.
- Shared head: six labels x two polarity queries, shape `[6,2,768]`; four-head query-to-token cross-attention; shared label scorer; `s=e+ - e-`.
- Shared training: weighted BCE plus 0.1 polarity auxiliary loss, AdamW, LR 0.001, no scheduler, weight decay 0.0001, effective batch 256, 30 epochs, patience 5, AMP.
- A0: embedding 128 -> one-layer BiGRU 384 per direction -> 768.
- E1: embedding 128 -> projection 384 -> two pre-LN local Transformer layers -> projection 768. Window attention is explicitly partitioned and has `O(TW)` score complexity; it does not construct a full token-by-token matrix. Position is factorized into learnable window-index and in-window embeddings.
- E2: unchanged A0 BiGRU -> layer-normalized 384-dimensional one-layer local Transformer refinement -> projection 768 -> zero-initialized scalar residual gate.
- E3: unchanged A0 BiGRU -> masked mean pooling over contiguous 64-token blocks -> two-layer global Transformer over at most 128 blocks -> broadcast to tokens -> zero-initialized scalar residual gate.
- All encoder outputs are `[B,T,768]`; the existing P11 PDVQ head is reused rather than copied.
- Padding tokens, empty blocks and broadcast positions are masked. E1/E2 use plain non-overlapping windows, not shifted windows.
- E1-256, E1-512, E2 and E3 change only the encoder architecture. All other protocol fields remain fixed.
