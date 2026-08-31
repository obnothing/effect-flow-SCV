"""Validate alignment and safety of the isolated stack relation cache."""

import argparse
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def check(path):
    payload = torch.load(resolve(path), map_location="cpu")
    if payload.get("schema") != "main6_opcode_stack_relational_v2":
        raise ValueError(f"{path}: stale stack relation cache; rebuild with --overwrite")
    ids = payload["ids"]
    chunks = payload["input_ids"]
    n_chunks = chunks.shape[0]
    if payload["chunk_offsets"].numel() != len(ids) + 1 or int(payload["chunk_offsets"][-1]) != n_chunks:
        raise ValueError(f"{path}: sample/chunk offsets mismatch")
    if payload["attention_mask"].shape != chunks.shape:
        raise ValueError(f"{path}: attention mask mismatch")
    if payload["stack_state"].shape[:2] != chunks.shape or payload["stack_state"].shape[-1] != 5:
        raise ValueError(f"{path}: stack state shape mismatch")
    if payload["boundary_state"].shape != (n_chunks, 4):
        raise ValueError(f"{path}: boundary state shape mismatch")
    edge_offsets = payload["edge_offsets"]
    if edge_offsets.numel() != n_chunks + 1:
        raise ValueError(f"{path}: edge offsets mismatch")
    edge_count = int(edge_offsets[-1])
    for name in ("edge_src", "edge_dst", "edge_type", "edge_slot", "edge_distance", "edge_confidence"):
        if payload[name].numel() != edge_count:
            raise ValueError(f"{path}: {name} length mismatch")
    if edge_count:
        if int(payload["edge_src"].max()) >= chunks.shape[1] or int(payload["edge_dst"].max()) >= chunks.shape[1]:
            raise ValueError(f"{path}: edge index out of range")
    if payload["multi_labels"].shape != (len(ids), 6):
        raise ValueError(f"{path}: labels must be [N,6]")
    for name, value in payload.items():
        if torch.is_tensor(value) and not torch.isfinite(value.float()).all():
            raise ValueError(f"{path}: NaN/Inf in {name}")
    report = payload.get("report", {})
    print(f"[OK] {path}: samples={len(ids)} chunks={n_chunks} relations={edge_count} capped={report.get('stack_analysis_capped_samples', 0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_relational.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    for split in ("train", "valid"):
        check(resolve(config["feature_dir"]) / f"{split}.pt")


if __name__ == "__main__":
    main()
