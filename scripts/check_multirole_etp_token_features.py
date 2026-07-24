"""Validate alignment and invariants for multi-role ETP token caches."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from effect_flow_schema import EFFECT_TYPES


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--token_semantic_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    args = parser.parse_args()
    report = {"status": "ok", "splits": {}}
    for split in args.splits:
        feature_path = Path(args.feature_dir) / f"{split}.pt"
        semantic_path = Path(args.token_semantic_dir) / f"{split}.pt"
        manifest_path = Path(args.token_semantic_dir) / f"{split}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{split}: missing token-cache manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("effect_type_names") != EFFECT_TYPES or manifest.get("sample_ids_in_order") is None:
            raise ValueError(f"{split}: manifest ontology or ID order is invalid")
        if manifest["feature"].get("sha256") != sha256_file(feature_path):
            raise ValueError(f"{split}: feature cache hash does not match manifest")
        if manifest["token_semantic"].get("sha256") != sha256_file(semantic_path):
            raise ValueError(f"{split}: token cache hash does not match manifest")
        feature = torch.load(feature_path, map_location="cpu")
        semantic = torch.load(semantic_path, map_location="cpu")
        if feature["ids"] != semantic["ids"] or not torch.equal(feature["chunk_mask"].bool(), semantic["chunk_mask"].bool()):
            raise ValueError(f"{split}: feature/token cache alignment failure")
        ids = semantic["etp_top2_ids"]
        confidence = semantic["etp_top2_confidence"]
        if ids.dtype != torch.uint8 or confidence.dtype != torch.uint8 or ids.shape[-1] != 2 or ids.shape[:2] != feature["features"].shape[:2]:
            raise ValueError(f"{split}: invalid Top-2 cache schema")
        valid = (ids < len(EFFECT_TYPES)) | (ids == 255)
        if not bool(valid.all()) or torch.any((ids[..., 0] == ids[..., 1]) & (ids[..., 0] != 255)):
            raise ValueError(f"{split}: invalid role ID or duplicate Top-2 role")
        special = ids == 255
        if torch.any(confidence[special] != 0):
            raise ValueError(f"{split}: sentinel roles must have zero confidence")
        inactive = ~feature["chunk_mask"].bool()
        if torch.any(ids[inactive] != 255) or torch.any(confidence[inactive] != 0):
            raise ValueError(f"{split}: padded chunks must contain only sentinel roles")
        if manifest["sample_ids_in_order"] != [str(value) for value in feature["ids"]]:
            raise ValueError(f"{split}: manifest sample ID order does not match cache")
        report["splits"][split] = {"samples": len(feature["ids"]), "feature_shape": list(feature["features"].shape), "token_shape": list(ids.shape), "effect_type_count": len(EFFECT_TYPES)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
