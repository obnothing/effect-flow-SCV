"""Extract train/valid-only token and sparse stack-relation caches."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_stack_relation import pack_contract_relations  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rows(path):
    with resolve(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line.strip():
                yield line_no, json.loads(line)


def extract_split(split, config, tokenizer, output):
    if split == "test" and not (os.environ.get("ALLOW_TEST") == "1" and config.get("allow_test_cache", False)):
        raise RuntimeError("test stack cache is locked; set ALLOW_TEST=1 and allow_test_cache only after validation selection")
    max_len = int(config.get("max_len", 512))
    max_chunks = int(config.get("max_chunks", 64))
    stride = int(config.get("chunk_stride", 256))
    ids, labels, binary = [], [], []
    input_ids, attention, states, boundary = [], [], [], []
    chunk_offsets = [0]
    edge_offsets = [0]
    edge_src, edge_dst, edge_type, edge_slot, edge_distance, edge_confidence = [], [], [], [], [], []
    total_relations = 0
    capped = 0
    metadata = []
    sample_rows = list(rows(config[f"{split}_path"]))
    for line_no, item in tqdm(sample_rows, desc=f"stack-relation:{split}"):
        contract_id = str(item.get("id") or item.get("address") or f"{split}_{line_no}")
        item_labels = item.get("multi_labels")
        if item_labels is None or len(item_labels) != len(LABELS):
            raise ValueError(f"{split}:{line_no + 1} requires exactly six Main-6 labels")
        chunks, boundaries, analysis = pack_contract_relations(
            str(item.get("opcode", "")), tokenizer, max_len=max_len, chunk_stride=stride, max_chunks=max_chunks
        )
        ids.append(contract_id)
        labels.append(item_labels)
        binary.append(float(item.get("binary_label", max(item_labels))))
        for chunk, boundary_row in zip(chunks, boundaries):
            input_ids.append(torch.tensor(chunk["input_ids"], dtype=torch.int32))
            attention.append(torch.tensor(chunk["attention_mask"], dtype=torch.bool))
            states.append(torch.tensor(chunk["stack_state"], dtype=torch.uint8))
            boundary.append(torch.tensor(boundary_row, dtype=torch.uint8))
            for src, dst, relation, slot, distance, confidence in chunk["edges"]:
                edge_src.append(src); edge_dst.append(dst); edge_type.append(relation)
                edge_slot.append(slot); edge_distance.append(distance); edge_confidence.append(confidence)
            edge_offsets.append(len(edge_src))
            total_relations += len(chunk["edges"])
        chunk_offsets.append(len(input_ids))
        if analysis["report"]["stack_analysis_capped"]:
            capped += 1
        metadata.append({"id": contract_id, "source_split": split, **analysis["report"]})

    payload = {
        "schema": "main6_opcode_stack_relational_v1",
        "ids": ids,
        "chunk_offsets": torch.tensor(chunk_offsets, dtype=torch.long),
        "input_ids": torch.stack(input_ids) if input_ids else torch.empty((0, max_len), dtype=torch.int32),
        "attention_mask": torch.stack(attention) if attention else torch.empty((0, max_len), dtype=torch.bool),
        "stack_state": torch.stack(states) if states else torch.empty((0, max_len, 5), dtype=torch.uint8),
        "boundary_state": torch.stack(boundary) if boundary else torch.empty((0, 4), dtype=torch.uint8),
        "edge_offsets": torch.tensor(edge_offsets, dtype=torch.long),
        "edge_src": torch.tensor(edge_src, dtype=torch.int16),
        "edge_dst": torch.tensor(edge_dst, dtype=torch.int16),
        "edge_type": torch.tensor(edge_type, dtype=torch.uint8),
        "edge_slot": torch.tensor(edge_slot, dtype=torch.uint8),
        "edge_distance": torch.tensor(edge_distance, dtype=torch.uint8),
        "edge_confidence": torch.tensor(edge_confidence, dtype=torch.float16),
        "binary_labels": torch.tensor(binary, dtype=torch.float32),
        "multi_labels": torch.tensor(labels, dtype=torch.float32),
        "label_names": LABELS,
        "metadata": metadata,
        "report": {
            "route": "DIVE Main6 Opcode-Stack-Relational",
            "split": split,
            "samples": len(ids),
            "chunks": len(input_ids),
            "relations": total_relations,
            "stack_analysis_capped_samples": capped,
            "max_len": max_len,
            "max_chunks": max_chunks,
            "test_checked": split == "test",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output)
    print(f"[OK] wrote {output}")
    return payload["report"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_relational.yaml")
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train", "valid"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    out_dir = resolve(config["feature_dir"])
    reports = {}
    for split in args.splits:
        output = out_dir / f"{split}.pt"
        if output.exists() and not args.overwrite:
            print(f"[SKIP] {output} exists; use --overwrite to rebuild")
            continue
        reports[split] = extract_split(split, config, tokenizer, output)
    (out_dir / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

