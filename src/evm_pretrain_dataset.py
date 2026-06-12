import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from evm_tokenizer import EVMOpcodeTokenizer


REQUIRED_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def validate_evm_tokenizer(tokenizer):
    missing = [token for token in REQUIRED_SPECIAL_TOKENS if token not in tokenizer.vocab]
    if missing:
        raise ValueError(f"EVM vocab missing required special tokens: {missing}")


class EVMMLMDataset(Dataset):
    def __init__(
        self,
        corpus_path,
        tokenizer,
        max_len=512,
        mlm_probability=0.15,
        debug_num_samples=None,
        seed=42,
    ):
        self.corpus_path = Path(corpus_path)
        self.tokenizer = tokenizer
        validate_evm_tokenizer(self.tokenizer)
        self.max_len = int(max_len)
        self.mlm_probability = float(mlm_probability)
        self.debug_num_samples = debug_num_samples
        self.seed = int(seed)
        self.offsets = self._build_offsets()
        self._file = None
        self.special_token_ids = {
            self.tokenizer.vocab[token] for token in REQUIRED_SPECIAL_TOKENS
        }
        self.mask_token_id = self.tokenizer.vocab["[MASK]"]
        self.pad_token_id = self.tokenizer.vocab["[PAD]"]
        self.random_token_ids = [
            idx
            for token, idx in self.tokenizer.vocab.items()
            if idx not in self.special_token_ids
        ]
        if not self.random_token_ids:
            raise ValueError("EVM vocab has no non-special tokens for random MLM.")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def _build_offsets(self):
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Pretrain corpus not found: {self.corpus_path}")
        offsets = []
        rng = random.Random(self.seed)
        limit = (
            int(self.debug_num_samples)
            if self.debug_num_samples is not None
            else None
        )
        with self.corpus_path.open("rb") as f:
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
            self._file = self.corpus_path.open("rb")
        return self._file

    def _read_item(self, idx):
        f = self._get_file()
        f.seek(self.offsets[idx])
        return json.loads(f.readline().decode("utf-8"))

    def _normalize_token_ids(self, item):
        if "token_ids" in item:
            input_ids = [int(x) for x in item["token_ids"]]
        elif "opcode" in item:
            input_ids = self.tokenizer.encode(item["opcode"], add_special_tokens=True)
        else:
            raise ValueError(
                f"Corpus item missing token_ids or opcode: {item.get('id', '<no-id>')}"
            )

        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
        if len(input_ids) < self.max_len:
            input_ids = input_ids + [self.pad_token_id] * (self.max_len - len(input_ids))
        return torch.tensor(input_ids, dtype=torch.long)

    def _make_attention_mask(self, item, input_ids):
        if "attention_mask" in item:
            mask = [int(x) for x in item["attention_mask"]]
            if len(mask) > self.max_len:
                mask = mask[: self.max_len]
            if len(mask) < self.max_len:
                mask = mask + [0] * (self.max_len - len(mask))
            return torch.tensor(mask, dtype=torch.long)
        return (input_ids != self.pad_token_id).long()

    def _mask_tokens(self, input_ids, idx):
        labels = input_ids.clone()
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_mask |= input_ids == token_id

        generator = torch.Generator()
        generator.manual_seed(self.seed + int(idx))
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix, generator=generator).bool()
        if not masked_indices.any():
            candidate_positions = (~special_mask).nonzero(as_tuple=False).view(-1)
            if candidate_positions.numel() > 0:
                forced_idx = candidate_positions[
                    torch.randint(
                        low=0,
                        high=candidate_positions.numel(),
                        size=(1,),
                        generator=generator,
                    ).item()
                ]
                masked_indices[forced_idx] = True
        labels[~masked_indices] = -100

        replace_probs = torch.rand(input_ids.shape, generator=generator)
        mask_replace = masked_indices & (replace_probs < 0.8)
        random_replace = masked_indices & (replace_probs >= 0.8) & (replace_probs < 0.9)

        input_ids = input_ids.clone()
        input_ids[mask_replace] = self.mask_token_id

        random_token_tensor = torch.tensor(
            self.random_token_ids, dtype=torch.long
        )
        random_positions = random_replace.nonzero(as_tuple=False).view(-1)
        if random_positions.numel() > 0:
            random_indices = torch.randint(
                low=0,
                high=len(self.random_token_ids),
                size=(random_positions.numel(),),
                generator=generator,
            )
            input_ids[random_positions] = random_token_tensor[random_indices]
        return input_ids, labels

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        item = self._read_item(idx)
        original_input_ids = self._normalize_token_ids(item)
        attention_mask = self._make_attention_mask(item, original_input_ids)
        input_ids, labels = self._mask_tokens(original_input_ids, idx)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros(self.max_len, dtype=torch.long),
            "labels": labels,
        }


def build_evm_mlm_dataset(config):
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(config["vocab_path"])
    validate_evm_tokenizer(tokenizer)
    dataset = EVMMLMDataset(
        corpus_path=config["pretrain_corpus_path"],
        tokenizer=tokenizer,
        max_len=config.get("max_len", 512),
        mlm_probability=config.get("mlm_probability", 0.15),
        debug_num_samples=config.get("debug_num_samples"),
        seed=config.get("seed", 42),
    )
    return dataset, tokenizer
