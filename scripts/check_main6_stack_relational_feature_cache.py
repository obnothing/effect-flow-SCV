"""Validate fixed Stack-Aware BERT feature caches."""

import argparse
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def check(path):
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") not in {
        "main6_opcode_stack_relational_encoder_features_v1",
        "main6_opcode_stack_adapter_encoder_features_v1",
    }:
        raise ValueError(f"{path}: unsupported feature cache schema")
    ids = payload["ids"]
    offsets = payload["chunk_offsets"]
    features = payload["chunk_features"]
    labels = payload["multi_labels"]
    if offsets.numel() != len(ids) + 1 or int(offsets[-1]) != features.shape[0]:
        raise ValueError(f"{path}: sample/chunk offsets mismatch")
    if features.ndim != 3 or features.shape[1:] != (8, 768):
        raise ValueError(f"{path}: expected chunk features [chunks,8,768]")
    if labels.shape != (len(ids), 6):
        raise ValueError(f"{path}: labels must be [N,6]")
    if not torch.isfinite(features.float()).all() or not torch.isfinite(labels.float()).all():
        raise ValueError(f"{path}: NaN/Inf")
    if (offsets[1:] < offsets[:-1]).any():
        raise ValueError(f"{path}: non-monotonic offsets")
    print(f"[OK] {path}: samples={len(ids)} chunks={features.shape[0]} dtype={features.dtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_relational.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    for split in ("train", "valid"):
        check(resolve(config["encoder_feature_dir"]) / f"{split}.pt")


if __name__ == "__main__":
    main()
