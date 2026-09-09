# VulProbe-V1 Architecture Audit

## Scope

- Route: DIVE Main-6 process01 VulProbe-V1
- Dataset: `data/processed/DIVE_main6_opcode_process01`
- Seed: 42
- Training/model selection: train and validation only
- Test status: locked and not inspected by the VulProbe pipeline

## Existing Backbone and Input

- EVM-BERT: `checkpoints/pretrain_evm_bert_main6_random_train_mlm8/hf_model`
- Tokenizer: `checkpoints/pretrain_evm_bert_main6_random_train_mlm8/evm_vocab.json`
- Existing M0 input: frozen pooled feature cache with shape `[B, 64, 8, 768]`
- Existing eight views: CLS, masked mean, masked max, and five positional segment means
- Process01 M0 validation Macro-F1 reference: 0.822535
- No verifiable process01 M0 test artifact is present in this workspace; no test result is claimed.

## Token-Level VulProbe Path

The pooled M0 cache cannot answer the probe hypothesis. VulProbe therefore reconstructs chunks from raw opcode and reads EVM-BERT `last_hidden_state` directly:

```text
opcode -> tokenizer -> up to 64 overlapping chunks
       -> [CLS] + 510 content tokens + [SEP]
       -> EVM-BERT hidden states [C, 512, 768]
       -> read-only probe-to-code cross-attention
       -> chunk score per label
       -> masked LogMeanExp MIL
       -> six contract logits
```

The chunk stride is 256. Padding, CLS and SEP are excluded from normal probe attention. Empty opcode chunks use CLS only as a numerical fallback.

## Information-Flow Boundary

EVM-BERT encodes opcode tokens before a vulnerability probe reads them. Probe vectors are not inserted into the BERT sequence, token representations never attend to probes, and isolated B2 probes do not communicate with one another.

## Training Feasibility

Full token hidden caches would require hundreds of gigabytes at process01 scale, so the implementation encodes chunks online. A contract batch contains one contract. EVM-BERT processes four chunks at a time and automatically falls back to two or one after a CUDA OOM. Gradient accumulation gives an effective batch of eight contracts. Stage 1 freezes EVM-BERT; Stage 2 unfreezes only its last two encoder layers. Exact memory and throughput are recorded during the server smoke and full runs.

## Fairness and Boundaries

B0, B1 and B2 share data, chunking, EVM-BERT initialization, BCE loss, class weighting, MIL temperature, optimization budget and threshold protocol. B2 adds five probe vectors relative to B1; the report exposes the exact parameter increment. M0 remains a strong reference rather than a causal baseline because it additionally uses eight-view pooling, a cross-chunk Transformer and multi-slot aggregation.
