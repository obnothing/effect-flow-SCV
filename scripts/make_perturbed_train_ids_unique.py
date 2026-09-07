"""Make the intentionally duplicated train IDs unique in a derived copy.

Only the train-side copies listed in the perturbation report are renamed.
Opcode, labels, and all other fields remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SPLITS = ("train", "valid", "test")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed/DIVE_main6_opcode_deduplicated_leak_500")
    parser.add_argument("--report", default="data/reports/main6_opcode_leak_500_report.json")
    parser.add_argument("--output-dir", default="data/processed/DIVE_main6_opcode_deduplicated_leak_500_unique_ids")
    parser.add_argument("--output-report", default="data/reports/main6_opcode_leak_500_unique_ids_report.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir, report_path = resolve(args.input_dir), resolve(args.report)
    output_dir, output_report = resolve(args.output_dir), resolve(args.output_report)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; use --overwrite: {output_dir}")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    target_ids = [
        str(value)
        for value in source_report.get(
            "moved_valid_ids",
            [item["id"] for item in source_report.get("mutations", [])],
        )
    ]
    if len(target_ids) != 500 or len(set(target_ids)) != 500:
        raise ValueError("source report must contain exactly 500 unique moved_valid_ids")

    rows = {split: read_rows(input_dir / f"{split}.jsonl") for split in SPLITS}
    target_set = set(target_ids)
    matches = [row for row in rows["train"] if str(row["id"]) in target_set]
    if len(matches) != 500 or {str(row["id"]) for row in matches} != target_set:
        raise ValueError(f"expected exactly 500 train-side matches, found {len(matches)}")

    valid_ids = {str(row["id"]) for row in rows["valid"]}
    existing_ids = {str(row["id"]) for split in SPLITS for row in rows[split]}
    renamed = {}
    output = {split: [copy.deepcopy(row) for row in rows[split]] for split in SPLITS}
    for row in output["train"]:
        old_id = str(row["id"])
        if old_id not in target_set:
            continue
        new_id = f"perturbed_train:{old_id}"
        if new_id in existing_ids or new_id in renamed.values():
            raise ValueError(f"generated ID already exists: {new_id}")
        row["id"] = new_id
        renamed[old_id] = new_id

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_rows(output_dir / f"{split}.jsonl", output[split])

    output_ids = {split: {str(row["id"]) for row in output[split]} for split in SPLITS}
    id_overlap = {
        f"{left}_{right}": len(output_ids[left] & output_ids[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    source_by_id = {str(row["id"]): row for row in rows["train"]}
    checks = []
    for old_id, new_id in renamed.items():
        original = source_by_id[old_id]
        changed = next(row for row in output["train"] if row["id"] == new_id)
        checks.append(
            changed["opcode"] == original["opcode"]
            and changed["multi_labels"] == original["multi_labels"]
            and changed.get("binary_label") == original.get("binary_label")
        )
    result = {
        "route": "DIVE Main6 perturbed-valid train ID de-duplication",
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "source_report": str(report_path), "renamed_count": len(renamed),
        "renaming_rule": "perturbed train copy: <old_id> -> perturbed_train:<old_id>",
        "output_samples": {split: len(output[split]) for split in SPLITS},
        "train_valid_id_overlap": id_overlap["train_valid"],
        "id_overlap": id_overlap,
        "all_500_train_ids_unique_globally": len(set(renamed.values()) & (output_ids["valid"] | output_ids["test"])) == 0,
        "opcode_labels_unchanged_for_renamed_rows": all(checks),
        "warnings": [
            "This only changes train IDs; it does not remove the underlying content relationship created by one-character opcode perturbation.",
            "This remains a deliberately contaminated experiment, not a clean benchmark split.",
            "The input directory was not overwritten.",
        ],
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {output_dir}")
    print(f"[OK] wrote {output_report}")


if __name__ == "__main__":
    main()
