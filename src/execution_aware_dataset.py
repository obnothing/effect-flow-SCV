from pathlib import Path

import torch
from torch.utils.data import Dataset

from chunk_feature_dataset import ChunkFeatureDataset


class ExecutionAwareDataset(Dataset):
    def __init__(self, semantic_path, execution_path, **kwargs):
        self.semantic = ChunkFeatureDataset(semantic_path, **kwargs)
        payload = torch.load(Path(execution_path), map_location="cpu")
        if [str(x) for x in payload["ids"]] != [str(x) for x in self.semantic.ids]:
            raise ValueError("Execution cache IDs are not aligned with semantic cache")
        self.execution = payload["chunk_features"].float()
        self.execution_mask = payload["chunk_mask"].bool()
        if self.execution.shape[:2] != self.semantic.features.shape[:2]:
            raise ValueError("Execution and semantic cache chunk shapes differ")
        if not torch.equal(self.execution_mask, self.semantic.chunk_mask):
            raise ValueError("Execution and semantic chunk masks differ")
        if not torch.isfinite(self.execution).all():
            raise ValueError("Execution cache contains NaN/Inf")

    def __len__(self):
        return len(self.semantic)

    def __getitem__(self, index):
        item = self.semantic[index]
        real = self.semantic.indices[index]
        item["execution_features"] = self.execution[real]
        return item
