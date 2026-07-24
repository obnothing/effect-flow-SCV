"""Minimal train-only MLM + multi-role ETP pretraining components."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertForMaskedLM


FORBIDDEN_FIELDS = {"binary_label", "multi_labels", "labels", "vulnerability_labels"}
CORPUS_FIELDS = {
    "id",
    "source_dataset",
    "source_split",
    "input_ids",
    "attention_mask",
    "effect_type_multihot",
    "etp_loss_mask",
}


def build_index(path):
    path = Path(path)
    offsets = []
    role_counts = None
    ids = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line.decode("utf-8"))
            forbidden = FORBIDDEN_FIELDS.intersection(row)
            if forbidden or set(row) != CORPUS_FIELDS or row.get("source_split") != "train":
                raise ValueError("ETP corpus must be label-free and train-only")
            width = len(row["effect_type_multihot"][0])
            if role_counts is None:
                role_counts = [0] * width
            if width != len(role_counts):
                raise ValueError("Inconsistent ETP role width")
            for roles, active in zip(row["effect_type_multihot"], row["etp_loss_mask"]):
                if int(active):
                    if sum(int(value) for value in roles) < 1 or sum(int(value) for value in roles) > 2:
                        raise ValueError("ETP roles must have cardinality one or two")
                    for role_id, value in enumerate(roles):
                        role_counts[role_id] += int(value)
            offsets.append(offset)
            ids.append(str(row["id"]))
    if not offsets:
        raise ValueError(f"ETP corpus is empty: {path}")
    return np.asarray(offsets, dtype=np.int64), ids, role_counts


def split_by_contract(offsets, ids, ratio, seed):
    unique_ids = sorted(set(ids))
    count = max(1, int(round(len(unique_ids) * float(ratio))))
    held_out = set(random.Random(int(seed)).sample(unique_ids, count))
    train = np.asarray([offset for offset, item_id in zip(offsets, ids) if item_id not in held_out], dtype=np.int64)
    valid = np.asarray([offset for offset, item_id in zip(offsets, ids) if item_id in held_out], dtype=np.int64)
    return train, valid


def role_pos_weights(role_counts, max_weight=10.0):
    total = max(1, sum(role_counts))
    return [min(float(max_weight), math.sqrt((total - value) / max(value, 1))) for value in role_counts]


class MultiRoleETPDataset(Dataset):
    def __init__(self, sources, tokenizer, mlm_probability=0.15, seed=42):
        self.sources = sources
        self.tokenizer = tokenizer
        self.mlm_probability = float(mlm_probability)
        self.seed = int(seed)
        self.epoch = 0
        self.handles = {}
        self.cumulative = []
        total = 0
        for source in sources:
            total += len(source["offsets"])
            self.cumulative.append(total)
        self.special_ids = {tokenizer.vocab[token] for token in ("[PAD]", "[CLS]", "[SEP]", "[MASK]")}
        self.random_ids = [value for value in tokenizer.vocab.values() if value not in self.special_ids]

    def __len__(self):
        return self.cumulative[-1]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _resolve(self, key):
        draw = 0
        if isinstance(key, tuple):
            return int(key[0]), int(key[1]), int(key[2])
        previous = 0
        for source_id, end in enumerate(self.cumulative):
            if int(key) < end:
                return source_id, int(key) - previous, draw
            previous = end
        raise IndexError(key)

    def _handle(self, source_id):
        if source_id not in self.handles:
            self.handles[source_id] = Path(self.sources[source_id]["path"]).open("rb")
        return self.handles[source_id]

    def __getitem__(self, key):
        source_id, local_id, draw = self._resolve(key)
        handle = self._handle(source_id)
        handle.seek(int(self.sources[source_id]["offsets"][local_id]))
        row = json.loads(handle.readline().decode("utf-8"))
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        attention = torch.tensor(row["attention_mask"], dtype=torch.long)
        role_targets = torch.tensor(row["effect_type_multihot"], dtype=torch.float32)
        role_mask = torch.tensor(row["etp_loss_mask"], dtype=torch.bool)
        content = attention.bool() & ~torch.isin(input_ids, torch.tensor(list(self.special_ids)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + source_id * 10007 + local_id * 17 + draw)
        selected = torch.bernoulli(torch.full(input_ids.shape, self.mlm_probability), generator=generator).bool() & content
        if not selected.any() and content.any():
            selected[content.nonzero()[0]] = True
        mom_labels = input_ids.clone()
        mom_labels[~selected] = -100
        masked = input_ids.clone()
        masked[selected] = self.tokenizer.vocab["[MASK]"]
        return {"input_ids": masked, "attention_mask": attention, "mom_labels": mom_labels, "etp_targets": role_targets, "etp_mask": role_mask}


class MultiRoleETPModel(nn.Module):
    def __init__(self, base_hf_model_path, vocab_size, num_effect_types, etp_pos_weights):
        super().__init__()
        base = BertForMaskedLM.from_pretrained(str(base_hf_model_path), local_files_only=True)
        if int(base.config.vocab_size) != int(vocab_size):
            raise ValueError("Vocabulary/model mismatch")
        self.config = base.config
        self.bert = base.bert
        self.mom_head = base.cls
        self.etp_head = nn.Linear(int(base.config.hidden_size), int(num_effect_types))
        self.register_buffer("etp_pos_weights", torch.tensor(etp_pos_weights, dtype=torch.float32))

    def forward(self, input_ids, attention_mask, mom_labels=None, etp_targets=None, etp_mask=None):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        mom_logits = self.mom_head(hidden)
        etp_logits = self.etp_head(hidden)
        mom_loss = None if mom_labels is None else nn.functional.cross_entropy(mom_logits.flatten(0, 1), mom_labels.flatten(), ignore_index=-100)
        etp_loss = None
        if etp_targets is not None:
            raw = nn.functional.binary_cross_entropy_with_logits(etp_logits, etp_targets, pos_weight=self.etp_pos_weights, reduction="none")
            weights = etp_mask.unsqueeze(-1).to(raw.dtype)
            etp_loss = (raw * weights).sum() / (weights.sum() * raw.shape[-1]).clamp_min(1.0)
        loss = None if mom_loss is None or etp_loss is None else mom_loss + 0.3 * etp_loss
        return {"loss": loss, "mom_loss": mom_loss, "etp_loss": etp_loss, "mom_logits": mom_logits, "etp_logits": etp_logits}

    def save(self, output_dir, metadata):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export = BertForMaskedLM(self.config)
        export.bert = self.bert
        export.cls = self.mom_head
        export.tie_weights()
        export.save_pretrained(output_dir / "hf_model")
        torch.save({"etp_head_state_dict": self.etp_head.state_dict(), "etp_pos_weights": self.etp_pos_weights.cpu(), "num_effect_types": int(self.etp_head.out_features), **metadata}, output_dir / "etp_head.pt")
