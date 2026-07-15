"""Create deterministic train-only Access Control subsampling allowlists."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
]
ACCESS_INDEX = LABEL_NAMES.index("Access Control")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build deterministic Main-6 train-only Access Control subsets."
    )
    parser.add_argument("--data-dir", default="data/processed/DIVE_main6_random_split")
    parser.add_argument(
        "--output-dir", default="data/processed/DIVE_main6_access_train_subsets"
    )
    parser.add_argument("--removals", type=int, nargs="+", default=[0, 1000, 2000, 3000, 4000, 5000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-path", default="data/reports/dive_main6_access_train_subsets.txt"
    )
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def summarize(rows):
    count = len(rows)
    positives = [sum(int(row["multi_labels"][index]) for row in rows) for index in range(len(LABEL_NAMES))]
    return {
        "samples": count,
        "positive_counts": dict(zip(LABEL_NAMES, positives)),
        "negative_counts": {name: count - value for name, value in zip(LABEL_NAMES, positives)},
        "positive_ratios": {name: value / max(1, count) for name, value in zip(LABEL_NAMES, positives)},
        "label_cardinality": dict(
            sorted(Counter(sum(int(value) for value in row["multi_labels"]) for row in rows).items())
        ),
    }


def main():
    args = parse_args()
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.report_path)
    train_path = data_dir / "train.jsonl"
    mapping_path = data_dir / "label_mapping.json"
    if not train_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("Main-6 train.jsonl and label_mapping.json must exist")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping.get("label_names") != LABEL_NAMES:
        raise ValueError("Input data_dir is not the expected DIVE Main-6 projection")
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Main-6 train IDs must be unique for ID-based subsampling")
    access_rows = [index for index, row in enumerate(rows) if int(row["multi_labels"][ACCESS_INDEX])]
    if max(args.removals) > len(access_rows) or min(args.removals) < 0:
        raise ValueError(f"removals must be within [0, {len(access_rows)}]")
    order = list(access_rows)
    random.Random(args.seed).shuffle(order)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for removal in sorted(set(args.removals)):
        removed = set(order[:removal])
        kept_rows = [row for index, row in enumerate(rows) if index not in removed]
        path = output_dir / f"train_ids_remove{removal:04d}.txt"
        path.write_text("\n".join(str(row["id"]) for row in kept_rows) + "\n", encoding="utf-8")
        variants.append(
            {
                "removed_access_positive_rows": removal,
                "id_subset_path": relative(path),
                "distribution": summarize(kept_rows),
            }
        )
    report = {
        "status": "ok",
        "policy": "train_only_random_access_positive_removal",
        "data_dir": relative(data_dir),
        "seed": args.seed,
        "access_positive_rows_available": len(access_rows),
        "valid_test_policy": "unchanged",
        "variants": variants,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(
        "DIVE Main-6 train-only Access Control subset report\n\n"
        + "\n".join(f"{key}: {json.dumps(value)}" for key, value in report.items())
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {relative(output_dir)}")
    print(f"[OK] wrote {relative(report_path)}")


if __name__ == "__main__":
    main()
