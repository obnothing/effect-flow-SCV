"""Ragged opcode-graph cache loading and batching for Opcode-CSDG."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset


SCHEMAS = {"main6_opcode_csdg_v1", "main6_opcode_csdg_v2"}


class OpcodeGraphSequenceDataset(Dataset):
    def __init__(self, graph_path, sequence_path, label_names=None):
        graph_path = Path(graph_path)
        sequence_path = Path(sequence_path)
        graph_payload = torch.load(graph_path, map_location="cpu")
        sequence_payload = torch.load(sequence_path, map_location="cpu")
        if graph_payload.get("schema") not in SCHEMAS:
            raise ValueError(f"Unsupported graph cache schema: {graph_payload.get('schema')}")
        self.ids = list(graph_payload["ids"])
        sequence_ids = list(sequence_payload["ids"])
        if self.ids != sequence_ids:
            mismatch = next((idx for idx, pair in enumerate(zip(self.ids, sequence_ids)) if pair[0] != pair[1]), None)
            raise ValueError(f"Graph/sequence ID order mismatch at index {mismatch}")
        self.records = graph_payload["records"]
        self.sequence_features = sequence_payload["features"].float()
        self.sequence_mask = sequence_payload["chunk_mask"].bool()
        self.multi_labels = graph_payload["multi_labels"].float()
        if len(self.records) != len(self.ids) or self.sequence_features.shape[0] != len(self.ids):
            raise ValueError("Graph, sequence and ID counts do not match")
        if self.multi_labels.ndim != 2:
            raise ValueError("multi_labels must be [N,6]")
        if label_names is not None and len(label_names) != self.multi_labels.shape[1]:
            raise ValueError("label_names width does not match graph labels")
        if torch.isnan(self.sequence_features).any() or torch.isinf(self.sequence_features).any():
            raise ValueError("Sequence cache contains NaN/Inf")
        self.manifest = graph_payload.get("manifest", {})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        record = self.records[index]
        node_features = record["node_features"].float()
        edge_index = record["edge_index"].long()
        edge_type = record["edge_type"].long()
        node_mask = record.get("node_mask", torch.ones(node_features.shape[0], dtype=torch.bool)).bool()
        node_local_features = record.get("node_local_features")
        node_local_offsets = record.get("node_local_offsets")
        node_type = record.get("node_type", torch.zeros(node_features.shape[0], dtype=torch.long)).long()
        if node_features.ndim != 2 or node_features.shape[1] != 768:
            raise ValueError(f"Invalid node_features shape for {self.ids[index]}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError(f"Invalid edge tensors for {self.ids[index]}")
        if edge_index.numel() and int(edge_index.max()) >= node_features.shape[0]:
            raise ValueError(f"Out-of-range graph edge for {self.ids[index]}")
        if node_local_features is not None:
            node_local_features = node_local_features.float()
            node_local_offsets = node_local_offsets.long()
            if node_local_offsets.numel() != node_features.shape[0] + 1 or int(node_local_offsets[-1]) != node_local_features.shape[0]:
                raise ValueError(f"Invalid node-local feature offsets for {self.ids[index]}")
        return {
            "id": self.ids[index],
            "sequence_features": self.sequence_features[index],
            "sequence_mask": self.sequence_mask[index],
            "node_features": node_features,
            "node_mask": node_mask,
            "node_local_features": node_local_features,
            "node_local_offsets": node_local_offsets,
            "node_type": node_type,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "block_ranges": record.get("block_ranges", []),
            "multi_labels": self.multi_labels[index],
        }


def collate_opcode_graph(batch: Sequence[Dict]):
    if not batch:
        raise ValueError("Cannot collate an empty graph batch")
    return {
        "ids": [item["id"] for item in batch],
        "sequence_features": torch.stack([item["sequence_features"] for item in batch]),
        "sequence_mask": torch.stack([item["sequence_mask"] for item in batch]),
        "node_features": [item["node_features"] for item in batch],
        "node_mask": [item["node_mask"] for item in batch],
        "node_local_features": [item["node_local_features"] for item in batch],
        "node_local_offsets": [item["node_local_offsets"] for item in batch],
        "node_type": [item["node_type"] for item in batch],
        "edge_index": [item["edge_index"] for item in batch],
        "edge_type": [item["edge_type"] for item in batch],
        "block_ranges": [item["block_ranges"] for item in batch],
        "multi_labels": torch.stack([item["multi_labels"] for item in batch]),
    }


def move_graph_batch(batch, device):
    moved = dict(batch)
    moved["sequence_features"] = batch["sequence_features"].to(device, non_blocking=True)
    moved["sequence_mask"] = batch["sequence_mask"].to(device, non_blocking=True)
    moved["multi_labels"] = batch["multi_labels"].to(device, non_blocking=True)
    moved["node_features"] = [value.to(device, non_blocking=True) for value in batch["node_features"]]
    moved["node_mask"] = [value.to(device, non_blocking=True) for value in batch["node_mask"]]
    moved["node_local_features"] = [None if value is None else value.to(device, non_blocking=True) for value in batch["node_local_features"]]
    moved["node_local_offsets"] = [None if value is None else value.to(device, non_blocking=True) for value in batch["node_local_offsets"]]
    moved["node_type"] = [value.to(device, non_blocking=True) for value in batch["node_type"]]
    moved["edge_index"] = [value.to(device, non_blocking=True) for value in batch["edge_index"]]
    moved["edge_type"] = [value.to(device, non_blocking=True) for value in batch["edge_type"]]
    return moved
