# PDVQ-Net Visualization Summary

- `pdvq_tsne_label_combination`: displays six-label `Z+ - Z-` evidence differences by frequent label combination; suitable for the main paper if visual structure is legible.
- `baseline_vs_pdvq_tsne`: compares masked mean pooling and PDVQ evidence-difference aggregation from the same P11 token encoder; suitable for the main paper as a representation comparison.
- `polarity_space/*`: compares positive and negative query evidence spaces for Reentrancy, DoS and Access Control; suitable for supplementary material or a focused main-paper panel.
- `query_attention_heatmap`: shows polarity-specific query attention and top attended opcode positions for six validation examples; suitable for supplementary material.
All artifacts are validation-only, derive from an existing checkpoint, and should be described as model representation or attention diagnostics rather than ground-truth evidence localization.
