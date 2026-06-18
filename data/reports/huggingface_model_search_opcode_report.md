# Hugging Face Model Search For BJUT Opcode-Based Vulnerability Detection

Date: 2026-06-18

## Task Fit

The current BJUT SC01 pipeline uses EVM opcode sequences, not Solidity source code:

```text
PUSH1 MSTORE CALLVALUE JUMPI ...
```

Models trained on Solidity source code may understand Solidity semantics, but their tokenizer and representations are not necessarily suitable for opcode sequences.

## Checked Candidates

| Model | Link | Fit For Current Opcode Pipeline | Notes |
|---|---|---|---|
| ZOSCARZ/CodeBERT-Solidity | https://huggingface.co/ZOSCARZ/CodeBERT-Solidity | Medium-low | Solidity-oriented RoBERTa/CodeBERT feature extractor. Relevant to smart contracts, but not opcode-specific. |
| angusleung100/GraphCodeBERT-Base-Solidity-Vulnerability | https://huggingface.co/angusleung100/GraphCodeBERT-Base-Solidity-Vulnerability | Low | Fine-tuned Solidity text-classification model. Useful as source-code baseline, not opcode encoder. |
| angusleung100/CodeBERT-Base-Solidity-Vulnerability | https://huggingface.co/angusleung100/CodeBERT-Base-Solidity-Vulnerability | Low | Solidity source classifier, not multi-label opcode model. |
| angusleung100/CodeT5-Base-Solidity-Vulnerability | https://huggingface.co/angusleung100/CodeT5-Base-Solidity-Vulnerability | Low | Solidity source text-classification model; architecture mismatch with current encoder pipeline. |
| priftil/ethereum-smart-contract-vulnerability-detection | https://huggingface.co/priftil/ethereum-smart-contract-vulnerability-detection | Low | Smart-contract vulnerability model, but not a drop-in Transformer encoder for opcode MIL. |
| demeleww/evm-bytecode-to-solidity-qwen3.6-27b | https://huggingface.co/demeleww/evm-bytecode-to-solidity-qwen3.6-27b | Low | EVM bytecode decompilation/generation direction, too large and not a classification encoder. |

## Recommendation

Do not switch the current BJUT opcode pipeline to CodeBERT-Solidity yet.

The most suitable model for the current task remains the locally pretrained EVM-BERT because:

1. It uses the project EVM opcode-aware vocabulary.
2. It is trained on opcode chunks.
3. It works directly with the current stride=256 MIL feature cache.

If Solidity source code becomes available for BJUT or DIVE, CodeBERT-Solidity becomes worth testing as a separate source-code baseline.

