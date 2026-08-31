"""Extract fixed train/valid Stack-Aware BERT representations once."""

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
from stack_aware_mil_model import StackAwareMLM8MIL  # noqa: E402
from stack_relation_dataset import StackRelationDataset, collate_stack_relation  # noqa: E402


LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def extract_split(split, config, model, device, output):
    if split == "test" and not (os.environ.get("ALLOW_TEST") == "1" and config.get("allow_test_cache", False)):
        raise RuntimeError("test Stack-Aware feature cache is locked")
    dataset = StackRelationDataset(resolve(config["feature_dir"]) / f"{split}.pt", num_labels=6, label_names=LABELS)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("feature_extract_batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_stack_relation,
        pin_memory=device.type == "cuda",
    )
    ids, offsets, features, labels = [], [0], [], []
    model.eval()
    with torch.no_grad():
        for raw_batch in tqdm(loader, desc=f"stack-relational-features:{split}"):
            batch = move_batch(raw_batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and bool(config.get("fp16", True))):
                encoded = model._encode(batch).detach().float().cpu()
            sample_offsets = batch["sample_chunk_offsets"].cpu().tolist()
            for index in range(len(sample_offsets) - 1):
                left, right = sample_offsets[index], sample_offsets[index + 1]
                features.append(encoded[left:right].half())
                offsets.append(offsets[-1] + right - left)
                ids.append(batch["ids"][index])
            labels.append(batch["multi_labels"].cpu().float())
    payload = {
        "schema": "main6_opcode_stack_relational_encoder_features_v1",
        "route": "DIVE Main6 Opcode-Stack-Relational",
        "split": split,
        "ids": ids,
        "chunk_offsets": torch.tensor(offsets, dtype=torch.long),
        "chunk_features": torch.cat(features, dim=0) if features else torch.empty((0, 8, 768), dtype=torch.float16),
        "multi_labels": torch.cat(labels, dim=0),
        "label_names": LABELS,
        "source_cache": str(resolve(config["feature_dir"]) / f"{split}.pt"),
        "stack_aware_mlm_checkpoint": str(resolve(config["stack_aware_mlm_checkpoint"])),
        "stack_aware_mlm_checkpoint_sha256": hashlib.sha256(resolve(config["stack_aware_mlm_checkpoint"]).read_bytes()).hexdigest(),
        "train_only_encoder": True,
        "test_checked": split == "test",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"[OK] wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_relational.yaml")
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train", "valid"])
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    checkpoint = resolve(config["stack_aware_mlm_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing train-only Stack-Aware MLM checkpoint: {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackAwareMLM8MIL(config).to(device)
    model.freeze_encoder()
    output_dir = resolve(config["encoder_feature_dir"])
    for split in args.splits:
        extract_split(split, config, model, device, output_dir / f"{split}.pt")
    (output_dir / "report.json").write_text(json.dumps({"route": config["route_name"], "splits": args.splits, "test_checked": "test" in args.splits}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
