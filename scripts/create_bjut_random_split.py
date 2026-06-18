import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a paper-style random BJUT SC01 split from processed samples."
    )
    parser.add_argument(
        "--source_data_dir",
        default="data/processed/BJUT_SC01",
        help="Existing processed directory with train/valid/test jsonl files.",
    )
    parser.add_argument(
        "--output_data_dir",
        default="data/processed/BJUT_SC01_random_split",
        help="Output directory for random train/valid/test jsonl files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--report_path",
        default="data/reports/bjut_sc01_random_split_report.txt",
    )
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path):
    try:
        return Path(path).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_source_split"] = path.stem
            item["_source_line"] = line_no
            rows.append(item)
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            clean = dict(item)
            clean.pop("_source_split", None)
            clean.pop("_source_line", None)
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def opcode_hash(row):
    normalized = " ".join(str(row.get("opcode", "")).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def label_counts(rows, num_labels=10):
    counts = [0] * num_labels
    for row in rows:
        labels = row.get("multi_labels", [])
        if len(labels) != num_labels:
            raise ValueError(f"Expected {num_labels} labels, got {len(labels)}")
        for idx, value in enumerate(labels):
            counts[idx] += int(value)
    return counts


def split_overlap(left, right):
    return len({opcode_hash(row) for row in left} & {opcode_hash(row) for row in right})


def duplicate_opcode_stats(rows):
    counts = Counter(opcode_hash(row) for row in rows)
    duplicate_groups = sum(1 for value in counts.values() if value > 1)
    duplicate_samples = sum(value for value in counts.values() if value > 1)
    return {
        "unique_opcode_hashes": len(counts),
        "duplicate_opcode_hash_groups": duplicate_groups,
        "duplicate_opcode_hash_samples": duplicate_samples,
    }


def main():
    args = parse_args()
    source_dir = resolve(args.source_data_dir)
    output_dir = resolve(args.output_data_dir)
    ratios = [args.train_ratio, args.valid_ratio, args.test_ratio]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.")

    rows = []
    for split in ["train", "valid", "test"]:
        path = source_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing source split: {path}")
        rows.extend(load_jsonl(path))
    if not rows:
        raise RuntimeError("No samples loaded.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    total = len(rows)
    train_count = int(total * args.train_ratio)
    valid_count = int(total * args.valid_ratio)
    train_rows = rows[:train_count]
    valid_rows = rows[train_count : train_count + valid_count]
    test_rows = rows[train_count + valid_count :]
    splits = {"train": train_rows, "valid": valid_rows, "test": test_rows}

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_rows)

    mapping_src = source_dir / "label_mapping.json"
    if mapping_src.exists():
        shutil.copy2(mapping_src, output_dir / "label_mapping.json")

    label_names = []
    if mapping_src.exists():
        label_names = json.loads(mapping_src.read_text(encoding="utf-8")).get(
            "label_names", []
        )
    if not label_names:
        label_names = [f"label_{idx}" for idx in range(10)]

    report = {
        "status": "ok",
        "warning": (
            "This random split intentionally does not group identical opcode hashes. "
            "It is a non-strict paper-style comparison split and may contain leakage."
        ),
        "source_data_dir": rel(source_dir),
        "output_data_dir": rel(output_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "valid": args.valid_ratio,
            "test": args.test_ratio,
        },
        "total_samples": total,
        "split_samples": {name: len(split_rows) for name, split_rows in splits.items()},
        "label_names": label_names,
        "label_counts": {
            name: {
                label_names[idx]: count
                for idx, count in enumerate(label_counts(split_rows))
            }
            for name, split_rows in splits.items()
        },
        "opcode_duplicate_stats_all_samples": duplicate_opcode_stats(rows),
        "opcode_hash_overlap": {
            "train_valid": split_overlap(train_rows, valid_rows),
            "train_test": split_overlap(train_rows, test_rows),
            "valid_test": split_overlap(valid_rows, test_rows),
        },
    }
    report["leakage_status"] = (
        "warning"
        if report["opcode_hash_overlap"]["train_test"] > 0
        else "ok"
    )

    report_path = resolve(args.report_path)
    report_json_path = report_path.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["BJUT SC01 random split report", ""]
    for key, value in report.items():
        if key == "label_counts":
            lines.append("label_counts:")
            for split, counts in value.items():
                lines.append(f"- {split}:")
                for label, count in counts.items():
                    lines.append(f"  {label}: {count}")
        else:
            lines.append(f"{key}: {value}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] wrote {rel(output_dir / 'train.jsonl')}")
    print(f"[OK] wrote {rel(output_dir / 'valid.jsonl')}")
    print(f"[OK] wrote {rel(output_dir / 'test.jsonl')}")
    print(f"[OK] wrote {rel(report_path)}")
    print(f"[OK] wrote {rel(report_json_path)}")


if __name__ == "__main__":
    main()

