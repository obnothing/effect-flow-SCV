"""Ragged cache dataset for the opcode stack-relational route."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class StackRelationDataset(Dataset):
    def __init__(self, path, num_labels=6, label_names=None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        payload = torch.load(self.path, map_location="cpu")
        if payload.get("schema") != "main6_opcode_stack_relational_v2":
            raise ValueError("stack relation cache is stale; rebuild it with the v2 extractor")
        self.ids = [str(x) for x in payload["ids"]]
        self.chunk_offsets = payload["chunk_offsets"].long()
        self.input_ids = payload["input_ids"]
        self.attention_mask = payload["attention_mask"]
        self.stack_state = payload["stack_state"]
        self.boundary_state = payload["boundary_state"]
        self.edge_offsets = payload["edge_offsets"].long()
        self.edge_src = payload["edge_src"]
        self.edge_dst = payload["edge_dst"]
        self.edge_type = payload["edge_type"]
        self.edge_slot = payload["edge_slot"]
        self.edge_distance = payload["edge_distance"]
        self.edge_confidence = payload["edge_confidence"]
        self.multi_labels = payload["multi_labels"].float()
        self.binary_labels = payload["binary_labels"].float()
        self.report = payload.get("report", {})
        self.label_names = list(label_names or payload.get("label_names", []))
        if len(self.multi_labels.shape) != 2 or self.multi_labels.shape[1] != int(num_labels):
            raise ValueError("stack relation cache has wrong label width")
        self._validate()

    def _validate(self):
        n = len(self.ids)
        if self.chunk_offsets.numel() != n + 1:
            raise ValueError("chunk_offsets must have N+1 entries")
        chunks = int(self.chunk_offsets[-1])
        if self.input_ids.shape[0] != chunks:
            raise ValueError("chunk/input_ids alignment mismatch")
        if self.edge_offsets.numel() != chunks + 1:
            raise ValueError("edge_offsets must have chunks+1 entries")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask shape mismatch")
        if self.stack_state.shape[:2] != self.input_ids.shape:
            raise ValueError("stack_state shape mismatch")
        if self.boundary_state.shape[0] != chunks:
            raise ValueError("boundary_state shape mismatch")
        if int(self.edge_offsets[-1]) != self.edge_src.numel():
            raise ValueError("edge field lengths mismatch")
        if self.edge_src.numel():
            length = self.input_ids.shape[1]
            if int(self.edge_src.max()) >= length or int(self.edge_dst.max()) >= length:
                raise ValueError("relation edge token index out of range")
        for tensor_name in ("input_ids", "attention_mask", "stack_state", "boundary_state", "edge_src", "edge_dst", "edge_type", "edge_slot", "edge_distance", "edge_confidence", "multi_labels"):
            tensor = getattr(self, tensor_name)
            if not torch.isfinite(tensor.float()).all():
                raise ValueError(f"NaN/Inf in {tensor_name}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        left, right = int(self.chunk_offsets[index]), int(self.chunk_offsets[index + 1])
        return {
            "id": self.ids[index],
            "input_ids": self.input_ids[left:right].long(),
            "attention_mask": self.attention_mask[left:right].bool(),
            "stack_state": self.stack_state[left:right].long(),
            "boundary_state": self.boundary_state[left:right].long(),
            "multi_labels": self.multi_labels[index],
            "binary_label": self.binary_labels[index],
            "chunk_count": right - left,
            "edge_offsets": self.edge_offsets[left:right + 1] - int(self.edge_offsets[left]),
            "edge_src": self.edge_src[int(self.edge_offsets[left]):int(self.edge_offsets[right])].long(),
            "edge_dst": self.edge_dst[int(self.edge_offsets[left]):int(self.edge_offsets[right])].long(),
            "edge_type": self.edge_type[int(self.edge_offsets[left]):int(self.edge_offsets[right])].long(),
            "edge_slot": self.edge_slot[int(self.edge_offsets[left]):int(self.edge_offsets[right])].long(),
            "edge_distance": self.edge_distance[int(self.edge_offsets[left]):int(self.edge_offsets[right])].long(),
            "edge_confidence": self.edge_confidence[int(self.edge_offsets[left]):int(self.edge_offsets[right])].float(),
        }


def collate_stack_relation(batch):
    if not batch:
        raise ValueError("empty stack relation batch")
    chunk_counts = [item["chunk_count"] for item in batch]
    chunk_offsets = [0]
    for count in chunk_counts:
        chunk_offsets.append(chunk_offsets[-1] + count)
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
    stack_state = torch.cat([item["stack_state"] for item in batch], dim=0)
    boundary_state = torch.cat([item["boundary_state"] for item in batch], dim=0)
    edge_offsets = [0]
    edge_src, edge_dst, edge_type, edge_slot, edge_distance, edge_confidence = [], [], [], [], [], []
    for item in batch:
        base = edge_offsets[-1]
        local = item["edge_offsets"]
        for value in local[1:]:
            edge_offsets.append(base + int(value))
        edge_src.append(item["edge_src"])
        edge_dst.append(item["edge_dst"])
        edge_type.append(item["edge_type"])
        edge_slot.append(item["edge_slot"])
        edge_distance.append(item["edge_distance"])
        edge_confidence.append(item["edge_confidence"])
    empty_long = torch.empty(0, dtype=torch.long)
    empty_float = torch.empty(0, dtype=torch.float32)
    return {
        "ids": [item["id"] for item in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "stack_state": stack_state,
        "boundary_state": boundary_state,
        "sample_chunk_offsets": torch.tensor(chunk_offsets, dtype=torch.long),
        "edge_offsets": torch.tensor(edge_offsets, dtype=torch.long),
        "edge_src": torch.cat(edge_src) if edge_src else empty_long,
        "edge_dst": torch.cat(edge_dst) if edge_dst else empty_long,
        "edge_type": torch.cat(edge_type) if edge_type else empty_long,
        "edge_slot": torch.cat(edge_slot) if edge_slot else empty_long,
        "edge_distance": torch.cat(edge_distance) if edge_distance else empty_long,
        "edge_confidence": torch.cat(edge_confidence) if edge_confidence else empty_float,
        "multi_labels": torch.stack([item["multi_labels"] for item in batch]),
        "binary_labels": torch.stack([item["binary_label"] for item in batch]),
    }
