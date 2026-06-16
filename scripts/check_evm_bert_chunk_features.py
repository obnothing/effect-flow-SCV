import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Check cached EVM-BERT chunk features.")
    parser.add_argument("--feature_dir", default="data/features/evm_bert_chunks")
    parser.add_argument("--data_dir", default="data/processed/BJUT_SC01")
    parser.add_argument("--report_dir", default="data/reports")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional txt report path. JSON is written beside it.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def count_jsonl(path):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def check_split(split, feature_dir, data_dir):
    path = feature_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature cache: {path}")
    payload = torch.load(path, map_location="cpu")
    features = payload["features"]
    chunk_mask = payload["chunk_mask"]
    binary_labels = payload["binary_labels"]
    multi_labels = payload["multi_labels"]
    ids = payload["ids"]
    expected_file = "train_mlsmote.jsonl" if split == "train" else f"{split}.jsonl"
    expected_samples = count_jsonl(data_dir / expected_file)
    checks = {
        "split": split,
        "path": project_relative(path),
        "exists": True,
        "samples": int(features.shape[0]),
        "expected_samples": expected_samples,
        "feature_shape": list(features.shape),
        "chunk_mask_shape": list(chunk_mask.shape),
        "binary_label_shape": list(binary_labels.shape),
        "multi_label_shape": list(multi_labels.shape),
        "id_count": len(ids),
        "dtype": str(features.dtype),
        "has_nan": bool(torch.isnan(features.float()).any().item()),
        "has_inf": bool(torch.isinf(features.float()).any().item()),
        "min_real_chunks": int(chunk_mask.sum(dim=1).min().item()),
        "max_real_chunks": int(chunk_mask.sum(dim=1).max().item()),
        "sample_count_matches_jsonl": int(features.shape[0]) == expected_samples,
        "disk_size_bytes": path.stat().st_size,
    }
    if features.ndim != 3:
        raise ValueError(f"{path} features must be [N, C, H], got {features.shape}")
    if chunk_mask.shape != features.shape[:2]:
        raise ValueError(f"{path} chunk_mask shape mismatch")
    if binary_labels.shape[0] != features.shape[0]:
        raise ValueError(f"{path} binary_labels shape mismatch")
    if multi_labels.shape != (features.shape[0], 10):
        raise ValueError(f"{path} multi_labels shape mismatch")
    if checks["has_nan"] or checks["has_inf"]:
        raise ValueError(f"{path} contains NaN or Inf features")
    if checks["min_real_chunks"] < 1:
        raise ValueError(f"{path} has a sample with no real chunks")
    if not checks["sample_count_matches_jsonl"]:
        raise ValueError(f"{path} sample count does not match {expected_file}")
    return checks


def main():
    args = parse_args()
    feature_dir = resolve_path(args.feature_dir)
    data_dir = resolve_path(args.data_dir)
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "feature_dir": project_relative(feature_dir),
        "splits": {
            split: check_split(split, feature_dir, data_dir)
            for split in ["train", "valid", "test"]
        },
    }
    if args.output:
        txt_path = resolve_path(args.output)
        json_path = txt_path.with_suffix(".json")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        txt_path = report_dir / "check_evm_bert_chunk_features_report.txt"
        json_path = report_dir / "check_evm_bert_chunk_features_report.json"
    lines = ["EVM-BERT chunk feature cache check", ""]
    lines.append(f"status: {report['status']}")
    lines.append(f"feature_dir: {report['feature_dir']}")
    for split, row in report["splits"].items():
        lines.append(f"{split}:")
        for key, value in row.items():
            lines.append(f"  {key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {project_relative(txt_path)}")
    print(f"[OK] wrote {project_relative(json_path)}")


if __name__ == "__main__":
    main()
