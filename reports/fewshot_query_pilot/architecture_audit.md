# Few-shot Query Pilot Architecture Audit

Dataset: `data/processed/DIVE_8_opcode_random_split`. Model instantiated from actual `PolarityQueryNet` and E3/P11 configuration.

- Additional Block Transformer: **no**.
- Opcode embedding: **512**.
- BiGRU: **512 per direction**, output **1024**.
- Shared K/V: **1024 → 512**; attention **8 × 64**.
- Positive and negative query shape: **[8, 2, 512]**.
- Label scorer: **[8, 512]**, shared between positive/negative branches for each label; branch bias **[8, 2]**.
- Total 8-label parameters: **5,331,472**.
- Per-label independent parameters: q+=512, q-=512, shared scorer=512, branch biases=2.
- In M2 adaptation, 14 mixture logits plus 514 scorer/bias parameters are trainable: scorer/bias are 97.3% of that count. The scorer does not dominate M0's query+scorer count (33.4%).
- Eight-label config exists: **True**; trained checkpoint for this pilot: **False**.
- Shared parameters: opcode embedding, bidirectional GRU, shared query/key/value/output projections. Independent rows: each label's q+/q-, scorer vector, and two branch biases.
- Loss: weighted main BCE on e+−e− plus polarity auxiliary BCE; E3 defaults use auxiliary weight 0.08, sqrt-ratio^0.6 positive weighting capped at 5, and DoS soft auxiliary targets 0.8/0.1.
- Pilot primary evaluation is Average Precision; thresholded F1 uses fixed 0.5 and no threshold is tuned on official valid.
- Official test accessed: **false**.
