import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ChunkFeatureDataset(Dataset):
    def __init__(self, path, debug_num_samples=None, seed=42, num_labels=None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Feature cache not found: {self.path}")
        payload = torch.load(self.path, map_location="cpu")
        self.ids = payload["ids"]
        self.features = payload["features"]
        self.chunk_mask = payload["chunk_mask"].bool()
        self.binary_labels = payload["binary_labels"].float()
        self.multi_labels = payload["multi_labels"].float()
        self.num_labels = int(num_labels) if num_labels is not None else int(self.multi_labels.shape[1])
        self.metadata = payload.get("metadata", [{} for _ in self.ids])
        self.report = payload.get("report", {})
        self.indices = list(range(len(self.ids)))
        if debug_num_samples is not None:
            rng = random.Random(int(seed))
            limit = min(int(debug_num_samples), len(self.indices))
            self.indices = sorted(rng.sample(self.indices, limit))
        self._validate()

    def _validate(self):
        n = len(self.ids)
        if self.features.ndim != 3:
            raise ValueError(f"features must be [N, C, H], got {self.features.shape}")
        if self.features.shape[0] != n:
            raise ValueError("features/id count mismatch")
        if self.chunk_mask.shape != self.features.shape[:2]:
            raise ValueError("chunk_mask shape mismatch")
        if self.binary_labels.shape[0] != n:
            raise ValueError("binary_labels shape mismatch")
        if self.multi_labels.shape != (n, self.num_labels):
            raise ValueError(
                f"multi_labels must be [N, {self.num_labels}], "
                f"got {tuple(self.multi_labels.shape)}"
            )
        if torch.isnan(self.features.float()).any() or torch.isinf(self.features.float()).any():
            raise ValueError(f"{self.path} contains NaN/Inf features")
        if self.chunk_mask.sum(dim=1).min().item() < 1:
            raise ValueError(f"{self.path} has at least one sample with no real chunks")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {
            "id": self.ids[real_idx],
            "chunk_features": self.features[real_idx].float(),
            "chunk_mask": self.chunk_mask[real_idx],
            "binary_label": self.binary_labels[real_idx],
            "multi_labels": self.multi_labels[real_idx],
            "metadata": self.metadata[real_idx],
        }


def build_chunk_feature_datasets(config):
    feature_dir = Path(config["feature_dir"])
    seed = config.get("seed", 42)
    return {
        "train": ChunkFeatureDataset(
            feature_dir / "train.pt",
            debug_num_samples=config.get("debug_num_train_samples"),
            seed=seed,
            num_labels=config.get("num_labels"),
        ),
        "valid": ChunkFeatureDataset(
            feature_dir / "valid.pt",
            debug_num_samples=config.get("debug_num_valid_samples"),
            seed=seed,
            num_labels=config.get("num_labels"),
        ),
        "test": ChunkFeatureDataset(
            feature_dir / "test.pt",
            seed=seed,
            num_labels=config.get("num_labels"),
        ),
    }
