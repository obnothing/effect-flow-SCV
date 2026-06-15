import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from evm_tokenizer import EVMOpcodeTokenizer


REQUIRED_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def validate_evm_vocab(tokenizer):
    missing = [token for token in REQUIRED_SPECIAL_TOKENS if token not in tokenizer.vocab]
    if missing:
        raise ValueError(f"EVM vocab missing required special tokens: {missing}")


class EVMBertClassificationDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        max_len,
        num_labels,
        debug_num_samples=None,
        seed=42,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        validate_evm_vocab(self.tokenizer)
        self.max_len = int(max_len)
        self.content_len = self.max_len - 2
        if self.content_len <= 0:
            raise ValueError("max_len must be at least 3 for [CLS]/[SEP].")
        self.num_labels = int(num_labels)
        self.debug_num_samples = debug_num_samples
        self.seed = int(seed)
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
        limit = (
            int(self.debug_num_samples)
            if self.debug_num_samples is not None
            else None
        )
        with self.path.open("rb") as f:
            item_no = 0
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                item_no += 1
                if limit is None:
                    offsets.append(offset)
                elif len(offsets) < limit:
                    offsets.append(offset)
                else:
                    replace_idx = rng.randint(0, item_no - 1)
                    if replace_idx < limit:
                        offsets[replace_idx] = offset
        return sorted(offsets)

    def _get_file(self):
        if self._file is None:
            self._file = self.path.open("rb")
        return self._file

    def _read_item(self, idx):
        f = self._get_file()
        f.seek(self.offsets[idx])
        item = json.loads(f.readline().decode("utf-8"))
        self._validate_item(item, idx)
        return item

    def _validate_item(self, item, idx):
        required = {"opcode", "binary_label", "multi_labels"}
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"{self.path}:item {idx} missing required keys: {sorted(missing)}"
            )
        if len(item["multi_labels"]) != self.num_labels:
            raise ValueError(
                f"{self.path}:item {idx} expected {self.num_labels} labels, "
                f"got {len(item['multi_labels'])}"
            )

    def _encode_opcode(self, opcode):
        content_tokens = self.tokenizer.tokenize(
            opcode,
            add_special_tokens=False,
        )[: self.content_len]
        tokens = [self.tokenizer.cls_token] + content_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding = self.max_len - len(input_ids)
        if padding < 0:
            raise ValueError(
                f"Encoded length {len(input_ids)} exceeds max_len={self.max_len}."
            )
        if padding:
            input_ids.extend([self.tokenizer.pad_token_id] * padding)
            attention_mask.extend([0] * padding)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.zeros(self.max_len, dtype=torch.long)

        if int(input_ids.max().item()) >= len(self.tokenizer):
            raise ValueError(
                f"input_ids.max()={int(input_ids.max().item())} >= vocab_size={len(self.tokenizer)}"
            )
        if int(input_ids.min().item()) < 0:
            raise ValueError(f"input_ids.min()={int(input_ids.min().item())} < 0")
        return input_ids, attention_mask, token_type_ids

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        item = self._read_item(idx)
        input_ids, attention_mask, token_type_ids = self._encode_opcode(item["opcode"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "binary_label": torch.tensor(item["binary_label"], dtype=torch.float),
            "multi_labels": torch.tensor(item["multi_labels"], dtype=torch.float),
        }


def build_evm_bert_datasets(config):
    data_dir = Path(config["data_dir"])
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(config["vocab_path"])
    validate_evm_vocab(tokenizer)
    train_file = "train_mlsmote.jsonl" if config.get("use_mlsmote_train") else "train.jsonl"
    train_path = data_dir / train_file
    if config.get("use_mlsmote_train") and not train_path.exists():
        raise FileNotFoundError(
            "train_mlsmote.jsonl not found. Please run scripts/apply_mlsmote.py first."
        )

    common_kwargs = {
        "tokenizer": tokenizer,
        "max_len": config["max_len"],
        "num_labels": config["num_labels"],
        "seed": config.get("seed", 42),
    }
    return {
        "train": EVMBertClassificationDataset(
            train_path,
            debug_num_samples=config.get("debug_num_train_samples"),
            **common_kwargs,
        ),
        "valid": EVMBertClassificationDataset(
            data_dir / "valid.jsonl",
            debug_num_samples=config.get("debug_num_valid_samples"),
            **common_kwargs,
        ),
        "test": EVMBertClassificationDataset(
            data_dir / "test.jsonl",
            **common_kwargs,
        ),
    }, tokenizer
