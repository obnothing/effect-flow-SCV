"""Extract fixed Stack-Adapter train/valid features after train-only MLM."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stack_adapter_mil_model import StackAdapterMLM8MIL  # noqa: E402
from stack_relation_dataset import StackRelationDataset, collate_stack_relation  # noqa: E402

LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_adapter.yaml")
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train", "valid"])
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    if "test" in args.splits and not (os.environ.get("ALLOW_TEST") == "1" and config.get("allow_test_cache", False)):
        raise RuntimeError("test Stack-Adapter feature cache is locked")
    checkpoint = resolve(config["adapter_mlm_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackAdapterMLM8MIL(config).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("schema") != "main6_opcode_stack_adapter_mlm_v1":
        raise ValueError("unsupported Stack-Adapter MLM checkpoint schema")
    model.encoder.load_state_dict(payload["model_state_dict"], strict=True)
    model.encoder.freeze_base()
    output_dir = resolve(config["encoder_feature_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        dataset = StackRelationDataset(resolve(config["relation_cache_dir"]) / f"{split}.pt", num_labels=6, label_names=LABELS)
        loader = DataLoader(dataset, batch_size=int(config.get("feature_extract_batch_size", 1)), shuffle=False, num_workers=int(config.get("num_workers", 0)), collate_fn=collate_stack_relation, pin_memory=device.type == "cuda")
        ids, offsets, features, labels = [], [0], [], []
        with torch.no_grad():
            for raw_batch in tqdm(loader, desc=f"stack-adapter-features:{split}"):
                batch = move_batch(raw_batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and bool(config.get("feature_bf16", True)) and torch.cuda.is_bf16_supported()):
                    encoded = model._encode(batch).float().cpu()
                sample_offsets = batch["sample_chunk_offsets"].cpu().tolist()
                for index in range(len(sample_offsets) - 1):
                    left, right = sample_offsets[index], sample_offsets[index + 1]
                    features.append(encoded[left:right].half())
                    offsets.append(offsets[-1] + right - left)
                    ids.append(batch["ids"][index])
                labels.append(batch["multi_labels"].cpu().float())
        out = output_dir / f"{split}.pt"
        temporary = out.with_suffix(out.suffix + ".tmp")
        torch.save({
            "schema": "main6_opcode_stack_adapter_encoder_features_v1",
            "route": config["route_name"], "split": split, "ids": ids,
            "chunk_offsets": torch.tensor(offsets, dtype=torch.long),
            "chunk_features": torch.cat(features, dim=0) if features else torch.empty((0, 8, 768), dtype=torch.float16),
            "multi_labels": torch.cat(labels, dim=0), "label_names": LABELS,
            "source_cache": str(resolve(config["relation_cache_dir"]) / f"{split}.pt"),
            "adapter_mlm_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "train_only_encoder": True, "test_checked": split == "test",
        }, temporary)
        temporary.replace(out)
        print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
