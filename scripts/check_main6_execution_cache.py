import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/features/main6_opcode_execution_aware/execution")
    parser.add_argument("--semantic-dir", default="data/features/main6_opcode_csdg/sequence")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    args = parser.parse_args()
    root = Path(args.cache_dir)
    reference_ids = None
    for split in args.splits:
        payload = torch.load(root / f"{split}.pt", map_location="cpu")
        semantic = torch.load(Path(args.semantic_dir) / f"{split}.pt", map_location="cpu")
        ids = [str(x) for x in payload["ids"]]
        if ids != [str(x) for x in semantic["ids"]]:
            raise ValueError(f"{split}: execution IDs do not match semantic cache")
        features, mask = payload["chunk_features"], payload["chunk_mask"].bool()
        labels = payload["multi_labels"]
        if reference_ids is not None and split == "train" and ids != reference_ids:
            raise ValueError("unexpected ID comparison")
        if features.ndim != 3 or mask.shape != features.shape[:2]:
            raise ValueError(f"{split}: invalid feature/mask shape")
        if labels.shape[0] != len(ids) or labels.shape[1] != 6:
            raise ValueError(f"{split}: expected six aligned labels")
        if not torch.equal(mask, semantic["chunk_mask"].bool()):
            raise ValueError(f"{split}: execution mask does not match semantic cache")
        if not torch.equal(labels.float(), semantic["multi_labels"].float()):
            raise ValueError(f"{split}: execution labels do not match semantic cache")
        # Contracts can have fewer than max_chunks; only require one real
        # chunk per sample. Padding rows are expected to be zero-filled.
        if not torch.isfinite(features).all() or not mask.any(dim=1).all():
            raise ValueError(f"{split}: NaN/Inf or empty sample")
        print(f"[OK] {split}: samples={len(ids)} shape={tuple(features.shape)} active_chunks={int(mask.sum())}")


if __name__ == "__main__":
    main()
