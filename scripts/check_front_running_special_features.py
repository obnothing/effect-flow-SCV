import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check Front Running special feature cache alignment."
    )
    parser.add_argument("--special_dir", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--expected_max_chunks", type=int, default=64)
    parser.add_argument("--expected_feature_dim", type=int, default=15)
    parser.add_argument("--expected_num_labels", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def count_jsonl(path):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def rel(path):
    try:
        return str(Path(path).relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def check_split(split, args):
    special_path = resolve(args.special_dir) / f"{split}.pt"
    feature_path = resolve(args.feature_dir) / f"{split}.pt"
    expected_file = "train_mlsmote.jsonl" if split == "train" else f"{split}.jsonl"
    expected_samples = count_jsonl(resolve(args.data_dir) / expected_file)
    if not special_path.exists():
        raise FileNotFoundError(f"missing front special cache: {special_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"missing feature cache: {feature_path}")
    special = torch.load(special_path, map_location="cpu")
    feature = torch.load(feature_path, map_location="cpu")
    ids_match = [str(v) for v in special["ids"]] == [str(v) for v in feature["ids"]]
    mask_match = torch.equal(
        special["chunk_mask"].bool(),
        feature["chunk_mask"].bool(),
    )
    labels_match = torch.equal(
        special["multi_labels"].float(),
        feature["multi_labels"].float(),
    )
    tensor = special["front_special_features"].float()
    expected_shape = (
        expected_samples,
        int(args.expected_max_chunks),
        int(args.expected_feature_dim),
    )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{special_path} shape mismatch: {tuple(tensor.shape)}")
    if special["multi_labels"].shape[1] != int(args.expected_num_labels):
        raise ValueError(f"{special_path} label width mismatch")
    if not ids_match:
        raise ValueError(f"{split}: ids do not match base feature cache")
    if not mask_match:
        raise ValueError(f"{split}: chunk_mask does not match base feature cache")
    if not labels_match:
        raise ValueError(f"{split}: multi_labels do not match base feature cache")
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"{special_path} contains NaN/Inf")
    feature_names = list(special.get("feature_names", []))
    if len(feature_names) != int(args.expected_feature_dim):
        raise ValueError(f"{special_path} feature_names width mismatch")
    active = special["chunk_mask"].bool()
    active_values = tensor[active]
    return {
        "split": split,
        "special_path": rel(special_path),
        "feature_path": rel(feature_path),
        "expected_samples": expected_samples,
        "shape": list(tensor.shape),
        "ids_match": ids_match,
        "chunk_mask_match": mask_match,
        "labels_match": labels_match,
        "feature_names": feature_names,
        "positive_chunk_counts": {
            name: int(tensor[:, :, idx].sum().item())
            for idx, name in enumerate(feature_names)
        },
        "positive_chunk_rates_active": {
            name: float(active_values[:, idx].mean().item()) if active_values.numel() else 0.0
            for idx, name in enumerate(feature_names)
        },
    }


def write_report(output, report):
    output = resolve(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    txt_path = output.with_suffix(".txt")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "Front Running special feature cache check",
        "",
        f"status: {report['status']}",
    ]
    for split, row in report["splits"].items():
        lines.extend(
            [
                "",
                f"[{split}]",
                f"path: {row['special_path']}",
                f"shape: {row['shape']}",
                f"ids_match: {row['ids_match']}",
                f"chunk_mask_match: {row['chunk_mask_match']}",
                f"labels_match: {row['labels_match']}",
                "positive_chunk_counts:",
            ]
        )
        for name, count in row["positive_chunk_counts"].items():
            rate = row["positive_chunk_rates_active"][name]
            lines.append(f"  {name}: {count} active_rate={rate:.6f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {rel(txt_path)}")
    print(f"[OK] wrote {rel(json_path)}")


def main():
    args = parse_args()
    report = {
        "status": "ok",
        "special_dir": rel(resolve(args.special_dir)),
        "feature_dir": rel(resolve(args.feature_dir)),
        "splits": {
            split: check_split(split, args)
            for split in ("train", "valid", "test")
        },
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
