"""Dataset and collation for fixed Stack-Aware BERT chunk representations."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class StackRelationFeatureDataset(Dataset):
    def __init__(self, path, num_labels=6, label_names=None):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu")
        if payload.get("schema") not in {
            "main6_opcode_stack_relational_encoder_features_v1",
            "main6_opcode_stack_adapter_encoder_features_v1",
        }:
            raise ValueError("Stack-Aware encoder feature cache has an unsupported schema")
        self.ids = [str(value) for value in payload["ids"]]
        self.chunk_offsets = payload["chunk_offsets"].long()
        # Keep fp16 on CPU and let the training autocast handle the GPU copy;
        # converting a large ragged cache to fp32 would double its footprint.
        self.chunk_features = payload["chunk_features"]
        self.multi_labels = payload["multi_labels"].float()
        self.label_names = list(label_names or payload.get("label_names", []))
        if self.chunk_offsets.numel() != len(self.ids) + 1:
            raise ValueError("feature chunk_offsets must have N+1 entries")
        if int(self.chunk_offsets[-1]) != self.chunk_features.shape[0]:
            raise ValueError("feature chunk offsets do not match features")
        if self.chunk_features.ndim != 3 or self.chunk_features.shape[1] != 8:
            raise ValueError("chunk_features must be [chunks, 8, hidden]")
        if self.multi_labels.shape != (len(self.ids), int(num_labels)):
            raise ValueError("feature cache has wrong label width")
        if not torch.isfinite(self.chunk_features.float()).all():
            raise ValueError("feature cache contains NaN/Inf")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        left = int(self.chunk_offsets[index])
        right = int(self.chunk_offsets[index + 1])
        return {
            "id": self.ids[index],
            "chunk_features": self.chunk_features[left:right],
            "multi_labels": self.multi_labels[index],
        }


def collate_stack_relation_features(batch):
    if not batch:
        raise ValueError("empty feature batch")
    max_chunks = max(item["chunk_features"].shape[0] for item in batch)
    hidden = batch[0]["chunk_features"].shape[-1]
    features = batch[0]["chunk_features"].new_zeros((len(batch), max_chunks, 8, hidden))
    mask = torch.zeros((len(batch), max_chunks), dtype=torch.bool)
    for index, item in enumerate(batch):
        count = item["chunk_features"].shape[0]
        features[index, :count] = item["chunk_features"]
        mask[index, :count] = True
    return {
        "ids": [item["id"] for item in batch],
        "chunk_features": features,
        "chunk_mask": mask,
        "multi_labels": torch.stack([item["multi_labels"] for item in batch]),
    }
