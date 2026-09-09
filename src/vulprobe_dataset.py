"""Raw-opcode contract dataset for token-level VulProbe experiments."""

import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from evm_tokenizer import EVMOpcodeTokenizer


class VulProbeContractDataset(Dataset):
    def __init__(self, path, vocab_path, max_len=512, stride=256, max_chunks=64,
                 num_labels=6, sample_limit=None, seed=42):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
        self.max_len = int(max_len)
        self.content_size = self.max_len - 2
        self.stride = int(stride)
        self.max_chunks = int(max_chunks)
        self.num_labels = int(num_labels)
        self.offsets = self._offsets(sample_limit, seed)
        self._file = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def close(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def _offsets(self, sample_limit, seed):
        offsets = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        if sample_limit is not None and len(offsets) > int(sample_limit):
            offsets = sorted(random.Random(int(seed)).sample(offsets, int(sample_limit)))
        return offsets

    def __len__(self):
        return len(self.offsets)

    def _read(self, index):
        if self._file is None:
            self._file = self.path.open("rb")
        self._file.seek(self.offsets[index])
        item = json.loads(self._file.readline().decode("utf-8"))
        if len(item.get("multi_labels", [])) != self.num_labels:
            raise ValueError(f"{self.path}: row {index} does not contain {self.num_labels} labels")
        return item

    def _chunk(self, opcode):
        tokens = self.tokenizer.tokenize(opcode, add_special_tokens=False)
        starts = list(range(0, len(tokens), self.stride))[:self.max_chunks] or [0]
        chunks = []
        for start in starts:
            content = tokens[start:start + self.content_size]
            sequence = [self.tokenizer.cls_token] + content + [self.tokenizer.sep_token]
            ids = self.tokenizer.convert_tokens_to_ids(sequence)
            attention = [1] * len(ids)
            content_mask = [0] + [1] * len(content) + [0]
            padding = self.max_len - len(ids)
            ids += [self.tokenizer.pad_token_id] * padding
            attention += [0] * padding
            content_mask += [0] * padding
            chunks.append((ids, attention, content_mask, start))
        return chunks, len(tokens)

    def __getitem__(self, index):
        item = self._read(index)
        chunks, token_count = self._chunk(item.get("opcode", ""))
        return {
            "id": str(item.get("id", index)),
            "input_ids": torch.tensor([x[0] for x in chunks], dtype=torch.long),
            "attention_mask": torch.tensor([x[1] for x in chunks], dtype=torch.bool),
            "content_mask": torch.tensor([x[2] for x in chunks], dtype=torch.bool),
            "chunk_starts": torch.tensor([x[3] for x in chunks], dtype=torch.long),
            "multi_labels": torch.tensor(item["multi_labels"], dtype=torch.float32),
            "binary_label": torch.tensor(float(item.get("binary_label", any(item["multi_labels"]))), dtype=torch.float32),
            "original_token_count": token_count,
            "truncated": math.ceil(max(token_count, 1) / self.stride) > self.max_chunks,
        }


def collate_contracts(items):
    batch = len(items)
    max_chunks = max(x["input_ids"].shape[0] for x in items)
    max_len = items[0]["input_ids"].shape[1]
    ids = torch.zeros(batch, max_chunks, max_len, dtype=torch.long)
    attention = torch.zeros(batch, max_chunks, max_len, dtype=torch.bool)
    content = torch.zeros_like(attention)
    chunk_mask = torch.zeros(batch, max_chunks, dtype=torch.bool)
    starts = torch.full((batch, max_chunks), -1, dtype=torch.long)
    for row, item in enumerate(items):
        count = item["input_ids"].shape[0]
        ids[row, :count] = item["input_ids"]
        attention[row, :count] = item["attention_mask"]
        content[row, :count] = item["content_mask"]
        chunk_mask[row, :count] = True
        starts[row, :count] = item["chunk_starts"]
    return {
        "ids": [x["id"] for x in items],
        "input_ids": ids,
        "attention_mask": attention,
        "content_mask": content,
        "chunk_mask": chunk_mask,
        "chunk_starts": starts,
        "multi_labels": torch.stack([x["multi_labels"] for x in items]),
        "binary_label": torch.stack([x["binary_label"] for x in items]),
        "original_token_count": torch.tensor([x["original_token_count"] for x in items]),
        "truncated": torch.tensor([x["truncated"] for x in items]),
    }
