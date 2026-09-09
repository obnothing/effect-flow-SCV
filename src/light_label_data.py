"""Cached full-sequence opcode data for the local label-guidance experiment."""

import json
from pathlib import Path

import random

import torch
from torch.utils.data import Dataset

from evm_tokenizer import EVMOpcodeTokenizer


def build_sequence_cache(jsonl_path, vocab_path, output_path, max_len, num_labels):
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
    sequences, offsets, labels, ids, original_lengths = [], [0], [], [], []
    with Path(jsonl_path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            target = item.get("multi_labels", [])
            if len(target) != int(num_labels):
                raise ValueError(f"{jsonl_path}:{line_no} expected {num_labels} labels")
            token_ids = tokenizer.encode(item.get("opcode", ""), add_special_tokens=False)
            original_lengths.append(len(token_ids))
            token_ids = token_ids[:int(max_len)]
            if not token_ids:
                token_ids = [tokenizer.unk_token_id]
            sequences.append(torch.tensor(token_ids, dtype=torch.int32))
            offsets.append(offsets[-1] + len(token_ids))
            labels.append(target)
            ids.append(str(item.get("id", line_no)))
    payload = {
        "token_ids": torch.cat(sequences),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "original_lengths": torch.tensor(original_lengths, dtype=torch.int32),
        "ids": ids,
        "max_len": int(max_len),
        "vocab_path": str(vocab_path),
        "test_checked": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


class LightLabelDataset(Dataset):
    def __init__(self, cache_path, runtime_max_len=None, indices=None):
        payload = torch.load(cache_path, map_location="cpu")
        self.token_ids = payload["token_ids"]
        self.offsets = payload["offsets"]
        self.labels = payload["labels"]
        self.original_lengths = payload["original_lengths"]
        self.ids = payload["ids"]
        self.runtime_max_len = int(runtime_max_len or payload["max_len"])
        self.indices = list(range(len(self.ids))) if indices is None else [int(x) for x in indices]

    def sequence_length(self, row):
        index = self.indices[row]
        return min(int(self.offsets[index + 1] - self.offsets[index]), self.runtime_max_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        index = self.indices[row]
        left, right = int(self.offsets[index]), int(self.offsets[index + 1])
        values = self.token_ids[left:min(right, left + self.runtime_max_len)].long()
        return {
            "id": self.ids[index], "input_ids": values,
            "length": len(values), "original_length": int(self.original_lengths[index]),
            "labels": self.labels[index],
        }


def collate_light_label(items, pad_id):
    max_length = max(item["length"] for item in items)
    input_ids = torch.full((len(items), max_length), int(pad_id), dtype=torch.long)
    mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    for row, item in enumerate(items):
        input_ids[row, :item["length"]] = item["input_ids"]
        mask[row, :item["length"]] = True
    return {
        "ids": [item["id"] for item in items], "input_ids": input_ids, "mask": mask,
        "lengths": torch.tensor([item["length"] for item in items], dtype=torch.long),
        "original_lengths": torch.tensor([item["original_length"] for item in items], dtype=torch.long),
        "labels": torch.stack([item["labels"] for item in items]),
    }


class LengthBucketBatchSampler:
    """Shuffles batches while keeping similarly long contracts together."""

    def __init__(self, dataset, batch_size, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.order = sorted(range(len(dataset)), key=dataset.sequence_length)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        batches = [self.order[left:left + self.batch_size] for left in range(0, len(self.order), self.batch_size)]
        random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size
