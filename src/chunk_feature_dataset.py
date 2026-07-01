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
        front_special_path=None,
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
        self.front_special_path = Path(front_special_path) if front_special_path else None
        self.efpp_probs = None
        self.etp_distribution = None
        self.relation_distribution = None
        self.vulnerability_evidence_probs = None
        self.template_match_scores = None
        self.chunk_vulnerability_evidence = None
        self.vulnerability_template_matches = None
        self.active_vulnerability_label_mask = None
        self.semantic_report = {}
        if self.semantic_path is not None:
            self._load_semantic_cache()
        self.front_special_features = None
        self.front_special_feature_names = []
        self.front_special_report = {}
        if self.front_special_path is not None:
            self._load_front_special_cache()
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
        required = [
            "efpp_probs",
            "etp_distribution",
            "relation_distribution",
            "vulnerability_evidence_probs",
            "template_match_scores",
            "chunk_vulnerability_evidence",
            "vulnerability_template_matches",
            "active_vulnerability_label_mask",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                f"{self.semantic_path} is not a semantic cache v2 payload. Missing: {missing}"
            )
        self.efpp_probs = payload["efpp_probs"].float()
        self.etp_distribution = payload["etp_distribution"].float()
        self.relation_distribution = payload["relation_distribution"].float()
        self.vulnerability_evidence_probs = payload["vulnerability_evidence_probs"].float()
        self.template_match_scores = payload["template_match_scores"].float()
        self.chunk_vulnerability_evidence = payload["chunk_vulnerability_evidence"].float()
        self.vulnerability_template_matches = payload["vulnerability_template_matches"].float()
        self.active_vulnerability_label_mask = payload["active_vulnerability_label_mask"].float()
        self.semantic_report = payload.get("report", {})

    def _load_front_special_cache(self):
        if not self.front_special_path.exists():
            raise FileNotFoundError(
                f"Front Running special cache not found: {self.front_special_path}"
            )
        payload = torch.load(self.front_special_path, map_location="cpu")
        special_ids = payload["ids"]
        if len(special_ids) != len(self.ids):
            raise ValueError(
                f"Front special cache/id count mismatch: {self.front_special_path} has "
                f"{len(special_ids)}, feature cache has {len(self.ids)}"
            )
        mismatch = [
            idx for idx, (left, right) in enumerate(zip(self.ids, special_ids))
            if str(left) != str(right)
        ]
        if mismatch:
            first = mismatch[0]
            raise ValueError(
                f"Front special cache ids are not aligned at index {first}: "
                f"{self.ids[first]} != {special_ids[first]}"
            )
        special_mask = payload["chunk_mask"].bool()
        if special_mask.shape != self.chunk_mask.shape:
            raise ValueError(
                f"Front special chunk_mask shape {tuple(special_mask.shape)} does not "
                f"match feature chunk_mask {tuple(self.chunk_mask.shape)}"
            )
        if not torch.equal(special_mask, self.chunk_mask):
            raise ValueError(
                f"Front special chunk_mask does not match feature cache: {self.front_special_path}"
            )
        if "front_special_features" not in payload:
            raise ValueError(f"{self.front_special_path} missing front_special_features")
        self.front_special_features = payload["front_special_features"].float()
        self.front_special_feature_names = list(payload.get("feature_names", []))
        self.front_special_report = payload.get("report", {})

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
            if self.relation_distribution.shape[:2] != self.features.shape[:2]:
                raise ValueError("relation_distribution must have the same [N, C] prefix as features")
            if self.vulnerability_evidence_probs.shape[:2] != self.features.shape[:2]:
                raise ValueError("vulnerability_evidence_probs must have the same [N, C] prefix as features")
            if self.template_match_scores.shape[:2] != self.features.shape[:2]:
                raise ValueError("template_match_scores must have the same [N, C] prefix as features")
            if self.chunk_vulnerability_evidence.shape[:2] != self.features.shape[:2]:
                raise ValueError("chunk_vulnerability_evidence must have the same [N, C] prefix as features")
            if self.vulnerability_template_matches.shape[:2] != self.features.shape[:2]:
                raise ValueError("vulnerability_template_matches must have the same [N, C] prefix as features")
            if self.active_vulnerability_label_mask.shape[0] != self.features.shape[0]:
                raise ValueError("active_vulnerability_label_mask sample count mismatch")
            if torch.isnan(self.efpp_probs).any() or torch.isinf(self.efpp_probs).any():
                raise ValueError(f"{self.semantic_path} contains NaN/Inf efpp_probs")
            if torch.isnan(self.etp_distribution).any() or torch.isinf(self.etp_distribution).any():
                raise ValueError(f"{self.semantic_path} contains NaN/Inf etp_distribution")
            if torch.isnan(self.relation_distribution).any() or torch.isinf(self.relation_distribution).any():
                raise ValueError(f"{self.semantic_path} contains NaN/Inf relation_distribution")
            if torch.isnan(self.vulnerability_evidence_probs).any() or torch.isinf(self.vulnerability_evidence_probs).any():
                raise ValueError(f"{self.semantic_path} contains NaN/Inf vulnerability_evidence_probs")
            if torch.isnan(self.template_match_scores).any() or torch.isinf(self.template_match_scores).any():
                raise ValueError(f"{self.semantic_path} contains NaN/Inf template_match_scores")
        if self.front_special_features is not None:
            if self.front_special_features.shape[:2] != self.features.shape[:2]:
                raise ValueError(
                    "front_special_features must have the same [N, C] prefix as features"
                )
            if self.front_special_features.ndim != 3:
                raise ValueError("front_special_features must be [N, C, K]")
            if torch.isnan(self.front_special_features).any() or torch.isinf(self.front_special_features).any():
                raise ValueError(
                    f"{self.front_special_path} contains NaN/Inf front_special_features"
                )
            if (
                self.front_special_feature_names
                and len(self.front_special_feature_names) != self.front_special_features.shape[-1]
            ):
                raise ValueError(
                    "front_special feature_names length does not match feature width"
                )

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
            item["relation_distribution"] = self.relation_distribution[real_idx]
            item["vulnerability_evidence_probs"] = self.vulnerability_evidence_probs[real_idx]
            item["template_match_scores"] = self.template_match_scores[real_idx]
            item["chunk_vulnerability_evidence"] = self.chunk_vulnerability_evidence[real_idx]
            item["vulnerability_template_matches"] = self.vulnerability_template_matches[real_idx]
            item["active_vulnerability_label_mask"] = self.active_vulnerability_label_mask[real_idx]
        if self.front_special_features is not None:
            item["front_special_features"] = self.front_special_features[real_idx]
        return item


def build_chunk_feature_datasets(config):
    feature_dir = Path(config["feature_dir"])
    semantic_dir = Path(config["semantic_feature_dir"]) if config.get("semantic_feature_dir") else None
    front_special_dir = (
        Path(config["front_special_feature_dir"])
        if config.get("front_special_feature_dir")
        else None
    )
    seed = config.get("seed", 42)
    def semantic_path(split):
        return semantic_dir / f"{split}.pt" if semantic_dir is not None else None

    def front_special_path(split):
        return front_special_dir / f"{split}.pt" if front_special_dir is not None else None

    return {
        "train": ChunkFeatureDataset(
            feature_dir / "train.pt",
            debug_num_samples=config.get("debug_num_train_samples"),
            seed=seed,
            num_labels=config.get("num_labels"),
            semantic_path=semantic_path("train"),
            front_special_path=front_special_path("train"),
        ),
        "valid": ChunkFeatureDataset(
            feature_dir / "valid.pt",
            debug_num_samples=config.get("debug_num_valid_samples"),
            seed=seed,
            num_labels=config.get("num_labels"),
            semantic_path=semantic_path("valid"),
            front_special_path=front_special_path("valid"),
        ),
        "test": ChunkFeatureDataset(
            feature_dir / "test.pt",
            seed=seed,
            num_labels=config.get("num_labels"),
            semantic_path=semantic_path("test"),
            front_special_path=front_special_path("test"),
        ),
    }
