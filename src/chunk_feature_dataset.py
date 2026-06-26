import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ChunkFeatureDataset(Dataset):
    def __init__(
        self,
        path,
        debug_num_samples=None,
        seed=42,
        num_labels=None,
        semantic_path=None,
    ):
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
        self.semantic_path = Path(semantic_path) if semantic_path else None
        self.efpp_probs = None
        self.etp_distribution = None
        self.semantic_report = {}
        if self.semantic_path is not None:
            self._load_semantic_cache()
        self.indices = list(range(len(self.ids)))
        if debug_num_samples is not None:
            rng = random.Random(int(seed))
            limit = min(int(debug_num_samples), len(self.indices))
            self.indices = sorted(rng.sample(self.indices, limit))
        self._validate()

    def _load_semantic_cache(self):
        if not self.semantic_path.exists():
            raise FileNotFoundError(f"Effect-flow semantic cache not found: {self.semantic_path}")
        payload = torch.load(self.semantic_path, map_location="cpu")
        semantic_ids = payload["ids"]
        if len(semantic_ids) != len(self.ids):
            raise ValueError(
                f"Semantic cache/id count mismatch: {self.semantic_path} has "
                f"{len(semantic_ids)}, feature cache has {len(self.ids)}"
            )
        mismatch = [
            idx for idx, (left, right) in enumerate(zip(self.ids, semantic_ids))
            if str(left) != str(right)
        ]
        if mismatch:
            first = mismatch[0]
            raise ValueError(
                f"Semantic cache ids are not aligned at index {first}: "
                f"{self.ids[first]} != {semantic_ids[first]}"
            )
        semantic_mask = payload["chunk_mask"].bool()
        if semantic_mask.shape != self.chunk_mask.shape:
            raise ValueError(
                f"Semantic chunk_mask shape {tuple(semantic_mask.shape)} does not "
                f"match feature chunk_mask {tuple(self.chunk_mask.shape)}"
            )
        if not torch.equal(semantic_mask, self.chunk_mask):
            raise ValueError(f"Semantic chunk_mask does not match feature cache: {self.semantic_path}")
        if not torch.equal(payload["binary_labels"].float(), self.binary_labels):
            raise ValueError(f"Semantic binary labels do not match feature cache: {self.semantic_path}")
        if not torch.equal(payload["multi_labels"].float(), self.multi_labels):
            raise ValueError(f"Semantic multi-labels do not match feature cache: {self.semantic_path}")
        self.efpp_probs = payload["efpp_probs"].float()
        self.etp_distribution = payload["etp_distribution"].float()
        self.semantic_report = payload.get("report", {})

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
        if self.efpp_probs is not None:
            if self.efpp_probs.shape[:2] != self.features.shape[:2]:
                raise ValueError("efpp_probs must have the same [N, C] prefix as features")
            if self.etp_distribution.shape[:2] != self.features.shape[:2]:
                raise ValueError("etp_distribution must have the same [N, C] prefix as features")
            if torch.isnan(self.efpp_probs).any() or torch.isinf(self.efpp_probs).any():
                raise ValueError(f"{self.semantic_path} contains NaN/Inf efpp_probs")
            if torch.isnan(self.etp_distribution).any() or torch.isinf(self.etp_distribution).any():
                raise ValueError(f"{self.semantic_path} contains NaN/Inf etp_distribution")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = {
            "id": self.ids[real_idx],
            "chunk_features": self.features[real_idx].float(),
            "chunk_mask": self.chunk_mask[real_idx],
            "binary_label": self.binary_labels[real_idx],
            "multi_labels": self.multi_labels[real_idx],
            "metadata": self.metadata[real_idx],
        }
        if self.efpp_probs is not None:
            item["efpp_probs"] = self.efpp_probs[real_idx]
            item["etp_distribution"] = self.etp_distribution[real_idx]
        return item


def build_chunk_feature_datasets(config):
    feature_dir = Path(config["feature_dir"])
    semantic_dir = Path(config["semantic_feature_dir"]) if config.get("semantic_feature_dir") else None
    seed = config.get("seed", 42)
    def semantic_path(split):
        return semantic_dir / f"{split}.pt" if semantic_dir is not None else None

    return {
        "train": ChunkFeatureDataset(
            feature_dir / "train.pt",
            debug_num_samples=config.get("debug_num_train_samples"),
            seed=seed,
            num_labels=config.get("num_labels"),
            semantic_path=semantic_path("train"),
        ),
        "valid": ChunkFeatureDataset(
            feature_dir / "valid.pt",
            debug_num_samples=config.get("debug_num_valid_samples"),
            seed=seed,
            num_labels=config.get("num_labels"),
            semantic_path=semantic_path("valid"),
        ),
        "test": ChunkFeatureDataset(
            feature_dir / "test.pt",
            seed=seed,
            num_labels=config.get("num_labels"),
            semantic_path=semantic_path("test"),
        ),
    }
