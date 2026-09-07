"""Create a deduplicated Main-6 data copy using exact normalized opcode hashes.

This is an offline data-quality transformation. It never overwrites the
official random split. Hash groups with conflicting six-label annotations are
removed entirely; consistent groups contribute one record, assigned to the
highest-priority split in train/valid/test order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


LABEL_NAMES = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
]
SPLITS = ("train", "valid", "test")


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(__file__).resolve().parents[1] / value


def opcode_hash(opcode: object) -> str:
    normalized = " ".join(str(opcode or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_rows(data_dir: Path):
    groups = defaultdict(list)
    split_rows = {}
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                labels = tuple(int(value) for value in row.get("multi_labels", []))
                if len(labels) != len(LABEL_NAMES):
                    raise ValueError(f"{path}:{line_number}: expected six labels")
                item = {
                    "split": split,
                    "line": line_number,
                    "id": str(row["id"]),
                    "labels": labels,
                    "row": row,
                }
                rows.append(item)
                groups[opcode_hash(row.get("opcode", ""))].append(item)
        split_rows[split] = rows
    return groups, split_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed/DIVE_main6_random_split")
    parser.add_argument("--output-dir", default="data/processed/DIVE_main6_opcode_deduplicated")
    parser.add_argument("--report", default="data/reports/main6_opcode_deduplication_report.json")
    args = parser.parse_args()

    data_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.report)
    groups, split_rows = read_rows(data_dir)

    conflicting = {}
    consistent = {}
    for digest, members in groups.items():
        label_vectors = sorted({item["labels"] for item in members})
        target = conflicting if len(label_vectors) > 1 else consistent
        target[digest] = members

    kept = {split: [] for split in SPLITS}
    removed_conflict = {split: 0 for split in SPLITS}
    removed_duplicate = {split: 0 for split in SPLITS}
    kept_group = {}

    # A consistent hash group is represented once in its earliest split. This
    # removes both within-split duplicates and cross-split exact duplicates.
    for digest, members in groups.items():
        if digest in conflicting:
            for item in members:
                removed_conflict[item["split"]] += 1
            continue
        winner = min(members, key=lambda item: (SPLITS.index(item["split"]), item["line"]))
        kept[winner["split"]].append(winner["row"])
        kept_group[digest] = {"kept_split": winner["split"], "kept_id": winner["id"]}
        for item in members:
            if item is not winner:
                removed_duplicate[item["split"]] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in kept[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def hashes(rows):
        return {opcode_hash(item.get("opcode", "")) for item in rows}

    output_hashes = {split: hashes(rows) for split, rows in kept.items()}
    input_counts = {split: len(split_rows[split]) for split in SPLITS}
    output_counts = {split: len(kept[split]) for split in SPLITS}
    overlap = {
        f"{left}_{right}": len(output_hashes[left] & output_hashes[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    conflict_rows = {
        digest: {
            "labels": [list(labels) for labels in sorted({item["labels"] for item in members})],
            "members": [
                {"split": item["split"], "id": item["id"], "line": item["line"]}
                for item in members
            ],
        }
        for digest, members in sorted(conflicting.items())
    }
    report = {
        "route": "DIVE Main6 opcode deduplication audit",
        "input_dir": str(data_dir),
        "output_dir": str(output_dir),
        "official_data_overwritten": False,
        "test_used_for_cleaning": True,
        "test_used_for_model_selection": False,
        "rule": "remove every hash group with conflicting labels; keep one row for each consistent hash, preferring train then valid then test",
        "label_names": LABEL_NAMES,
        "input_samples": input_counts,
        "output_samples": output_counts,
        "removed_conflicting_hash_groups": len(conflicting),
        "removed_conflicting_rows": removed_conflict,
        "removed_consistent_duplicate_rows": removed_duplicate,
        "input_unique_hashes": {split: len(hashes([x["row"] for x in split_rows[split]])) for split in SPLITS},
        "output_unique_hashes": {split: len(output_hashes[split]) for split in SPLITS},
        "output_opcode_hash_overlap": overlap,
        "conflicting_hash_groups": conflict_rows,
        "warnings": [
            "This is a cleaned benchmark copy, not the official random split.",
            "Because test labels were used to remove conflicting groups, the cleaned test split must not be used as an unbiased final test set.",
            "All hash matches are exact after whitespace normalization; this does not prove source-code identity.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "conflicting_hash_groups"}, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {output_dir}")
    print(f"[OK] wrote {report_path}")


if __name__ == "__main__":
    main()
