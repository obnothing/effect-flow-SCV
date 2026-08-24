"""Build train/valid-only execution summaries aligned to the MLM8 cache."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_execution_analysis import FEATURE_NAMES, analyze_opcode_execution  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def iter_jsonl(path):
    with resolve(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line.strip():
                yield line_no, json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_execution_aware_mil.yaml")
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train", "valid"])
    parser.add_argument("--allow-test-cache", action="store_true")
    args = parser.parse_args()
    payload = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    config = payload.get("common", payload)
    if "test" in args.splits and not (args.allow_test_cache and str(__import__("os").environ.get("ALLOW_TEST", "0")) == "1"):
        raise RuntimeError("Test execution cache is locked; use --allow-test-cache with ALLOW_TEST=1 after validation selection")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    output_dir = resolve(config["execution_feature_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_chunks = int(config.get("max_chunks", 64))
    content_size = int(config.get("chunk_content_size", 510))
    stride = int(config.get("chunk_stride", 256))
    feature_dim = len(FEATURE_NAMES)
    reports = {}
    for split in args.splits:
        input_path = config[f"{split}_path"]
        rows = list(iter_jsonl(input_path))
        n = len(rows)
        chunk_features = torch.zeros((n, max_chunks, feature_dim * 2), dtype=torch.float32)
        chunk_mask = torch.zeros((n, max_chunks), dtype=torch.bool)
        ids, multi_labels, binary_labels, metadata = [], [], [], []
        dependency_total = 0
        truncated = 0
        for row_idx, (line_no, item) in enumerate(tqdm(rows, desc=f"execution:{split}")):
            opcode = item.get("opcode", "")
            tokens = tokenizer.tokenize(opcode, add_special_tokens=False)
            analysis = analyze_opcode_execution(opcode, tokenizer=tokenizer)
            token_features = analysis["token_features"]
            if token_features.shape[0] != len(tokens):
                raise ValueError(f"Token alignment mismatch at {split}:{line_no + 1}")
            num_chunks = max(1, math.ceil(len(tokens) / stride)) if tokens else 1
            kept = min(max_chunks, num_chunks)
            if num_chunks > max_chunks:
                truncated += 1
            for chunk_idx in range(kept):
                start = chunk_idx * stride
                end = min(start + content_size, len(tokens))
                values = token_features[start:end]
                if values.shape[0]:
                    mean = values.mean(axis=0)
                    maximum = values.max(axis=0)
                    chunk_features[row_idx, chunk_idx] = torch.from_numpy(np.concatenate([mean, maximum]))
                chunk_mask[row_idx, chunk_idx] = True
            contract_id = str(item.get("id") or item.get("address") or f"{split}_{line_no}")
            ids.append(contract_id)
            multi_labels.append(item["multi_labels"])
            binary_labels.append(float(item["binary_label"]))
            metadata.append({"id": contract_id, "source_split": split, "token_count": len(tokens), "num_chunks": kept, "analysis": analysis["report"]})
            dependency_total += analysis["report"]["dependency_count"]
        payload = {
            "schema": "main6_opcode_execution_aware_v1",
            "ids": ids, "chunk_features": chunk_features, "chunk_mask": chunk_mask,
            "binary_labels": torch.tensor(binary_labels, dtype=torch.float32),
            "multi_labels": torch.tensor(multi_labels, dtype=torch.float32),
            "metadata": metadata,
            "feature_names": [f"mean_{x}" for x in FEATURE_NAMES] + [f"max_{x}" for x in FEATURE_NAMES],
            "report": {"route": "DIVE Main6 Opcode-Execution-Aware", "split": split, "samples": n, "feature_dim": feature_dim * 2, "max_chunks": max_chunks, "dependency_count": dependency_total, "truncated_samples": truncated, "test_checked": split == "test"},
        }
        torch.save(payload, output_dir / f"{split}.pt")
        reports[split] = payload["report"]
        print(f"[OK] wrote {output_dir / (split + '.pt')}")
    (output_dir / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
