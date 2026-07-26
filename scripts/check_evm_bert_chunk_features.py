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
    parser.add_argument("--expected_max_chunks", type=int, default=None)
    parser.add_argument("--expected_feature_dim", type=int, default=None)
    parser.add_argument("--expected_num_views", type=int, default=None)
    parser.add_argument("--expected_num_labels", type=int, default=None)
    parser.add_argument(
        "--coverage_baseline_dir",
        default=None,
        help="Optional feature dir to compare coverage ratios against.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional txt report path. JSON is written beside it.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test"],
        default=["train", "valid", "test"],
        help="Cache splits to validate. Defaults to all splits.",
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


def check_split(
    split,
    feature_dir,
    data_dir,
    expected_max_chunks=None,
    expected_feature_dim=None,
    expected_num_views=None,
    expected_num_labels=None,
):
    path = feature_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature cache: {path}")
    payload = torch.load(path, map_location="cpu")
    features = payload["features"]
    chunk_mask = payload["chunk_mask"]
    binary_labels = payload["binary_labels"]
    multi_labels = payload["multi_labels"]
    ids = payload["ids"]
    source_report = payload.get("report", {})
    # Main-6 caches are built from the original split only.  MLSMOTE is not
    # part of this route and must never be required for cache validation.
    expected_file = f"{split}.jsonl"
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
        "mean_real_chunks": float(chunk_mask.sum(dim=1).float().mean().item()),
        "sample_count_matches_jsonl": int(features.shape[0]) == expected_samples,
        "disk_size_bytes": path.stat().st_size,
        "source_report": {
            "pooling": source_report.get("pooling"),
            "max_chunks_per_contract": source_report.get("max_chunks_per_contract"),
            "generated_chunks": source_report.get("generated_chunks"),
            "mean_chunks_per_contract": source_report.get("mean_chunks_per_contract"),
            "truncated_contract_count": source_report.get("truncated_contract_count"),
            "mean_coverage_ratio": source_report.get("mean_coverage_ratio"),
            "p50_coverage_ratio": source_report.get("p50_coverage_ratio"),
            "p90_coverage_ratio": source_report.get("p90_coverage_ratio"),
            "p95_coverage_ratio": source_report.get("p95_coverage_ratio"),
            "hf_model_path": source_report.get("hf_model_path"),
            "is_transductive_pretraining": source_report.get("is_transductive_pretraining"),
        },
    }
    if features.ndim not in {3, 4}:
        raise ValueError(f"{path} features must be [N, C, H] or [N, C, V, H], got {features.shape}")
    if expected_num_views is not None:
        if features.ndim != 4 or features.shape[2] != expected_num_views:
            raise ValueError(
                f"{path} expected num_views={expected_num_views}, got {list(features.shape)}"
            )
    elif features.ndim == 4:
        raise ValueError(f"{path} is a multi-view cache; pass --expected_num_views")
    if chunk_mask.shape != features.shape[:2]:
        raise ValueError(f"{path} chunk_mask shape mismatch")
    if expected_max_chunks is not None and features.shape[1] != expected_max_chunks:
        raise ValueError(
            f"{path} expected max_chunks={expected_max_chunks}, got {features.shape[1]}"
        )
    feature_dim = features.shape[-1]
    if expected_feature_dim is not None and feature_dim != expected_feature_dim:
        raise ValueError(
            f"{path} expected feature_dim={expected_feature_dim}, got {feature_dim}"
        )
    if binary_labels.shape[0] != features.shape[0]:
        raise ValueError(f"{path} binary_labels shape mismatch")
    inferred_num_labels = int(multi_labels.shape[1]) if multi_labels.ndim == 2 else None
    if multi_labels.ndim != 2 or multi_labels.shape[0] != features.shape[0]:
        raise ValueError(f"{path} multi_labels shape mismatch")
    if expected_num_labels is not None and inferred_num_labels != expected_num_labels:
        raise ValueError(
            f"{path} expected num_labels={expected_num_labels}, got {inferred_num_labels}"
        )
    checks["num_labels"] = inferred_num_labels
    if checks["has_nan"] or checks["has_inf"]:
        raise ValueError(f"{path} contains NaN or Inf features")
    if checks["min_real_chunks"] < 1:
        raise ValueError(f"{path} has a sample with no real chunks")
    if not checks["sample_count_matches_jsonl"]:
        raise ValueError(f"{path} sample count does not match {expected_file}")
    return checks


def coverage_from_payload(path):
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    report = payload.get("report", {})
    value = report.get("mean_coverage_ratio")
    return float(value) if value is not None else None


def add_coverage_comparison(report, baseline_dir):
    if baseline_dir is None:
        return
    baseline_dir = resolve_path(baseline_dir)
    report["coverage_baseline_dir"] = project_relative(baseline_dir)
    warnings = []
    for split, row in report["splits"].items():
        current = row.get("source_report", {}).get("mean_coverage_ratio")
        baseline = coverage_from_payload(baseline_dir / f"{split}.pt")
        row["baseline_mean_coverage_ratio"] = baseline
        row["coverage_delta_vs_baseline"] = (
            float(current) - float(baseline)
            if current is not None and baseline is not None
            else None
        )
        if row["coverage_delta_vs_baseline"] is not None and row["coverage_delta_vs_baseline"] < -0.02:
            warnings.append(
                f"{split}: mean coverage ratio is more than 0.02 below baseline"
            )
    report["coverage_warnings"] = warnings


def main():
    args = parse_args()
    feature_dir = resolve_path(args.feature_dir)
    data_dir = resolve_path(args.data_dir)
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "feature_dir": project_relative(feature_dir),
        "expected_max_chunks": args.expected_max_chunks,
        "expected_feature_dim": args.expected_feature_dim,
        "expected_num_views": args.expected_num_views,
        "expected_num_labels": args.expected_num_labels,
        "splits": {
            split: check_split(
                split,
                feature_dir,
                data_dir,
                expected_max_chunks=args.expected_max_chunks,
                expected_feature_dim=args.expected_feature_dim,
                expected_num_views=args.expected_num_views,
                expected_num_labels=args.expected_num_labels,
            )
            for split in args.splits
        },
    }
    add_coverage_comparison(report, args.coverage_baseline_dir)
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
    lines.append(f"expected_max_chunks: {report.get('expected_max_chunks')}")
    lines.append(f"expected_feature_dim: {report.get('expected_feature_dim')}")
    lines.append(f"expected_num_views: {report.get('expected_num_views')}")
    lines.append(f"expected_num_labels: {report.get('expected_num_labels')}")
    if report.get("coverage_baseline_dir"):
        lines.append(f"coverage_baseline_dir: {report['coverage_baseline_dir']}")
    if report.get("coverage_warnings"):
        lines.append("coverage_warnings:")
        for warning in report["coverage_warnings"]:
            lines.append(f"  - {warning}")
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
