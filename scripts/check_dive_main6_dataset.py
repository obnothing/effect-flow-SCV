"""Verify that a DIVE Main-6 view is an exact declared projection of DIVE."""

import argparse
import json
from collections import Counter
from pathlib import Path

from build_dive_main6_dataset import (
    MAIN6_LABEL_NAMES,
    MAIN6_SOURCE_INDICES,
    RARE_SOURCE_INDICES,
    SOURCE_LABEL_NAMES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Check a derived DIVE Main-6 dataset.")
    parser.add_argument("--source-dir", default="data/processed/DIVE_random_split")
    parser.add_argument("--main6-dir", default="data/processed/DIVE_main6_random_split")
    parser.add_argument(
        "--output", default="data/reports/check_dive_main6_dataset.txt"
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


def next_json_row(handle, path):
    for line_no, line in enumerate(handle, start=1):
        if line.strip():
            return line_no, json.loads(line)
    return None, None


def validate_mapping(path):
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("label_names") != MAIN6_LABEL_NAMES:
        raise ValueError(f"{path} label_names do not match the Main-6 contract")
    if mapping.get("source_label_names") != SOURCE_LABEL_NAMES:
        raise ValueError(f"{path} source_label_names do not match DIVE")
    if mapping.get("source_label_indices") != MAIN6_SOURCE_INDICES:
        raise ValueError(f"{path} source_label_indices do not match Main-6")
    return bool(mapping.get("exclude_rare_positive_contracts", False))


def check_split(source_path, main6_path, exclude_rare):
    expected_rows = 0
    observed_rows = 0
    positives = Counter()
    source_rare_rows = 0
    source_rare_rows_with_main6 = 0
    with source_path.open("r", encoding="utf-8") as source, main6_path.open(
        "r", encoding="utf-8"
    ) as main6:
        for source_line_no, source_line in enumerate(source, start=1):
            if not source_line.strip():
                continue
            source_row = json.loads(source_line)
            source_labels = [int(value) for value in source_row.get("multi_labels", [])]
            if len(source_labels) != len(SOURCE_LABEL_NAMES):
                raise ValueError(
                    f"{source_path}:{source_line_no} has invalid source label width"
                )
            projected = [source_labels[index] for index in MAIN6_SOURCE_INDICES]
            has_rare = any(source_labels[index] for index in RARE_SOURCE_INDICES)
            source_rare_rows += int(has_rare)
            source_rare_rows_with_main6 += int(has_rare and any(projected))
            if exclude_rare and has_rare:
                continue
            expected_rows += 1
            main6_line_no, main6_row = next_json_row(main6, main6_path)
            if main6_row is None:
                raise ValueError(
                    f"{main6_path} ended early; expected row {expected_rows} from {source_path}"
                )
            observed_rows += 1
            if str(main6_row.get("id")) != str(source_row.get("id")):
                raise ValueError(
                    f"{main6_path}:{main6_line_no} id does not align with "
                    f"{source_path}:{source_line_no}"
                )
            if main6_row.get("opcode") != source_row.get("opcode"):
                raise ValueError(f"{main6_path}:{main6_line_no} opcode does not align")
            if [int(value) for value in main6_row.get("multi_labels", [])] != projected:
                raise ValueError(f"{main6_path}:{main6_line_no} labels are not the declared projection")
            if int(main6_row.get("binary_label", -1)) != int(any(projected)):
                raise ValueError(f"{main6_path}:{main6_line_no} binary_label is invalid")
            for name, value in zip(MAIN6_LABEL_NAMES, projected):
                positives[name] += int(value)
        extra_line_no, extra = next_json_row(main6, main6_path)
        if extra is not None:
            raise ValueError(f"{main6_path}:{extra_line_no} contains an unexpected extra row")
    return {
        "expected_rows": expected_rows,
        "observed_rows": observed_rows,
        "main6_positive_counts": {name: int(positives[name]) for name in MAIN6_LABEL_NAMES},
        "source_rare_positive_rows": source_rare_rows,
        "source_rare_positive_rows_with_main6": source_rare_rows_with_main6,
    }


def main():
    args = parse_args()
    source_dir = resolve(args.source_dir)
    main6_dir = resolve(args.main6_dir)
    output = resolve(args.output)
    mapping_path = main6_dir / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing Main-6 mapping: {mapping_path}")
    exclude_rare = validate_mapping(mapping_path)
    split_rows = {}
    for split in ("train", "valid", "test"):
        source_path = source_dir / f"{split}.jsonl"
        main6_path = main6_dir / f"{split}.jsonl"
        if not source_path.exists() or not main6_path.exists():
            raise FileNotFoundError(f"Missing {split} source or Main-6 JSONL")
        split_rows[split] = check_split(source_path, main6_path, exclude_rare)
    report = {
        "status": "ok",
        "source_dir": relative(source_dir),
        "main6_dir": relative(main6_dir),
        "exclude_rare_positive_contracts": exclude_rare,
        "main6_label_names": MAIN6_LABEL_NAMES,
        "splits": split_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["DIVE Main-6 dataset check", ""]
    for key, value in report.items():
        lines.append(f"{key}: {json.dumps(value)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {relative(output)}")


if __name__ == "__main__":
    main()
