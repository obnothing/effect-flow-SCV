# Local Label Model Audit

- Dataset: `D:\effect-flow-scv\data\processed\DIVE_main6_opcode_process01`
- Train/valid/test: 16474 / 2032 / 2025
- Labels: Reentrancy, Access Control, Arithmetic, Unchecked Return Values, DoS, Time manipulation
- Tokenizer: `D:\effect-flow-scv\data\processed\ethereum_public_pretrain_19143_unique_runtime\evm_vocab.json`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- VRAM: 7.96 GiB
- PyTorch/CUDA: 2.10.0+cu130 / 13.0
- Train length p50/p75/p90/p95/p99/max: 4674 / 7159 / 9793 / 11870 / 15085 / 30982
- Coverage 4096/8192/16384: 0.4475 / 0.7958 / 0.9939
- Selected max_len: 8192; train truncation ratio: 0.2042

Only the number of test records was counted. Test opcode and labels were not used for max_len selection, training, or threshold tuning.
