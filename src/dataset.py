import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class OpcodeJsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, num_labels, debug_num_samples=None, seed=42):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_labels = num_labels
        self.debug_num_samples = debug_num_samples
        self.seed = seed
        self.samples = self._load_samples()

    def _load_samples(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self._validate_item(item, line_no)
                samples.append(item)

        if self.debug_num_samples is not None and self.debug_num_samples < len(samples):
            rng = random.Random(self.seed)
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            selected = sorted(indices[: self.debug_num_samples])
            samples = [samples[idx] for idx in selected]
        return samples

    def _validate_item(self, item, line_no):
        required_keys = {"opcode", "binary_label", "multi_labels"}
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(
                f"{self.path}:{line_no} missing required keys: {sorted(missing)}"
            )
        if len(item["multi_labels"]) != self.num_labels:
            raise ValueError(
                f"{self.path}:{line_no} expected {self.num_labels} multi-labels, "
                f"got {len(item['multi_labels'])}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoded = self.tokenizer(
            item["opcode"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "binary_label": torch.tensor(item["binary_label"], dtype=torch.float),
            "multi_labels": torch.tensor(item["multi_labels"], dtype=torch.float),
        }


def build_datasets(
    data_dir,
    tokenizer,
    max_len,
    num_labels,
    debug_num_train_samples=None,
    debug_num_valid_samples=None,
    seed=42,
):
    data_dir = Path(data_dir)
    return {
        "train": OpcodeJsonlDataset(
            data_dir / "train.jsonl",
            tokenizer,
            max_len,
            num_labels,
            debug_num_samples=debug_num_train_samples,
            seed=seed,
        ),
        "valid": OpcodeJsonlDataset(
            data_dir / "valid.jsonl",
            tokenizer,
            max_len,
            num_labels,
            debug_num_samples=debug_num_valid_samples,
            seed=seed,
        ),
        "test": OpcodeJsonlDataset(
            data_dir / "test.jsonl", tokenizer, max_len, num_labels, seed=seed
        ),
    }
