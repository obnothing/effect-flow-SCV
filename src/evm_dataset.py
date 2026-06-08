import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from evm_tokenizer import EVMOpcodeTokenizer


class EVMChunkDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        chunk_size,
        chunk_stride,
        max_chunks,
        num_labels,
        debug_num_samples=None,
        seed=42,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.chunk_stride = int(chunk_stride)
        self.max_chunks = int(max_chunks)
        self.num_labels = int(num_labels)
        self.debug_num_samples = debug_num_samples
        self.seed = seed
        self.offsets = self._build_offsets()
        self._file = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def _build_offsets(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        offsets = []
        rng = random.Random(self.seed)
        with self.path.open("rb") as f:
            line_no = 0
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                line_no += 1
                if self.debug_num_samples is None:
                    offsets.append(offset)
                elif len(offsets) < self.debug_num_samples:
                    offsets.append(offset)
                else:
                    replace_idx = rng.randint(0, line_no - 1)
                    if replace_idx < self.debug_num_samples:
                        offsets[replace_idx] = offset
        return sorted(offsets)

    def _get_file(self):
        if self._file is None:
            self._file = self.path.open("rb")
        return self._file

    def _read_item(self, idx):
        f = self._get_file()
        f.seek(self.offsets[idx])
        line = f.readline().decode("utf-8")
        item = json.loads(line)
        self._validate_item(item, idx)
        return item

    def _validate_item(self, item, idx):
        required_keys = {"opcode", "binary_label", "multi_labels"}
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(
                f"{self.path}:item {idx} missing required keys: {sorted(missing)}"
            )
        if len(item["multi_labels"]) != self.num_labels:
            raise ValueError(
                f"{self.path}:item {idx} expected {self.num_labels} multi-labels, "
                f"got {len(item['multi_labels'])}"
            )

    def _chunk_input_ids(self, input_ids):
        max_tokens = self.chunk_size + self.chunk_stride * (self.max_chunks - 1)
        input_ids = input_ids[:max_tokens]
        chunks = []
        start = 0
        while start < len(input_ids) and len(chunks) < self.max_chunks:
            chunks.append(input_ids[start : start + self.chunk_size])
            start += self.chunk_stride

        input_tensor = torch.full(
            (self.max_chunks, self.chunk_size),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (self.max_chunks, self.chunk_size), dtype=torch.long
        )
        chunk_mask = torch.zeros(self.max_chunks, dtype=torch.long)

        for chunk_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            length = min(len(chunk), self.chunk_size)
            input_tensor[chunk_idx, :length] = torch.tensor(
                chunk[:length], dtype=torch.long
            )
            attention_mask[chunk_idx, :length] = 1
            chunk_mask[chunk_idx] = 1

        return input_tensor, attention_mask, chunk_mask

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        item = self._read_item(idx)
        input_ids = self.tokenizer.encode(item["opcode"], add_special_tokens=True)
        input_tensor, attention_mask, chunk_mask = self._chunk_input_ids(input_ids)
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "chunk_mask": chunk_mask,
            "binary_label": torch.tensor(item["binary_label"], dtype=torch.float),
            "multi_labels": torch.tensor(item["multi_labels"], dtype=torch.float),
        }


def build_evm_chunk_datasets(config):
    data_dir = Path(config["data_dir"])
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(config["vocab_path"])
    train_file = "train_mlsmote.jsonl" if config.get("use_mlsmote_train") else "train.jsonl"
    train_path = data_dir / train_file
    if config.get("use_mlsmote_train") and not train_path.exists():
        raise FileNotFoundError(
            "train_mlsmote.jsonl not found. "
            "Please run scripts/apply_mlsmote.py first."
        )

    common_kwargs = {
        "tokenizer": tokenizer,
        "chunk_size": config["chunk_size"],
        "chunk_stride": config["chunk_stride"],
        "max_chunks": config["max_chunks"],
        "num_labels": config["num_labels"],
        "seed": config.get("seed", 42),
    }
    datasets = {
        "train": EVMChunkDataset(
            train_path,
            debug_num_samples=config.get("debug_num_train_samples"),
            **common_kwargs,
        ),
        "valid": EVMChunkDataset(
            data_dir / "valid.jsonl",
            debug_num_samples=config.get("debug_num_valid_samples"),
            **common_kwargs,
        ),
        "test": EVMChunkDataset(data_dir / "test.jsonl", **common_kwargs),
    }
    return datasets, tokenizer
