"""Build the audited, source-aligned DIVE Source-Main6 protocol dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidity_graph_utils import extract_source_units, source_file_text, source_sha256  # noqa: E402


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def normalized_opcode(value: str) -> str:
    return " ".join(str(value or "").split())


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_address_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["Address"].strip().lower(): str(index)
            for index, row in enumerate(csv.DictReader(handle), start=1)
        }


def identifier_to_contract_id(identifier: str, address_map: dict[str, str]) -> str | None:
    raw = str(identifier).rsplit(":", 1)[-1].lower()
    return raw if raw.isdigit() else address_map.get(raw)


def grouped_split(rows, seed: int, ratios: list[float]):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_sha256"]].append(row)
    group_rows = list(groups.values())
    labels = np.asarray([group[0]["multi_labels"] for group in group_rows], dtype=np.int64)
    group_ids = np.arange(len(group_rows))
    first = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=ratios[1] + ratios[2], random_state=seed
    )
    train_indices, holdout_indices = next(first.split(group_ids, labels))
    holdout_labels = labels[holdout_indices]
    relative_test = ratios[2] / (ratios[1] + ratios[2])
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
    valid_relative, test_relative = next(second.split(holdout_indices, holdout_labels))
    split_indices = {
        "train": train_indices,
        "valid": holdout_indices[valid_relative],
        "test": holdout_indices[test_relative],
    }
    output = {}
    for split, indices in split_indices.items():
        output[split] = [row for index in indices for row in group_rows[int(index)]]
    return output


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build_dive_source_main6.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    raw_dir = resolve(config["raw_dir"])
    main6_dir = resolve(config["main6_data_dir"])
    output_dir = resolve(config["output_dir"])
    report_dir = resolve(config["report_dir"])
    source_dir = raw_dir / "PRE" / "Source codes"
    address_map = load_address_map(raw_dir / "DIVE_Samples.csv")
    runtime = {
        str(row["contractID"]): normalized_opcode(row["Opcodes"])
        for row in load_jsonl(raw_dir / "POST" / "Runtime_Opcode.jsonl")
    }
    source_files = {path.stem: path for path in source_dir.glob("*.sol")}
    source_cache = {}
    audit_rows, exclusions = [], []
    for old_split in ("train", "valid", "test"):
        for row in load_jsonl(main6_dir / f"{old_split}.jsonl"):
            contract_id = identifier_to_contract_id(row["id"], address_map)
            reason = None
            source_path = source_files.get(str(contract_id)) if contract_id else None
            if source_path is None:
                reason = "missing_source_mapping"
            elif normalized_opcode(row["opcode"]) != runtime.get(str(contract_id), ""):
                reason = "runtime_opcode_mismatch"
            if reason:
                exclusions.append({"id": row["id"], "reason": reason, "contract_id": contract_id})
                continue
            text = source_cache.setdefault(str(contract_id), source_file_text(source_path))
            digest = source_sha256(text)
            audit_rows.append({
                "original_id": row["id"], "contract_id": str(contract_id),
                "source_path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": digest, "multi_labels": [int(v) for v in row["multi_labels"]],
                "binary_label": int(row.get("binary_label", any(row["multi_labels"]))),
                "runtime_opcode_match": True, "old_split": old_split,
            })
    by_hash = defaultdict(list)
    for row in audit_rows:
        by_hash[row["source_sha256"]].append(row)
    accepted = []
    for digest, group in by_hash.items():
        labels = {tuple(row["multi_labels"]) for row in group}
        if len(labels) != 1:
            exclusions.extend({"id": row["original_id"], "reason": "conflicting_duplicate_source_label", "source_sha256": digest} for row in group)
            continue
        try:
            extract_source_units(source_cache[group[0]["contract_id"]])
        except Exception as exc:
            exclusions.extend({"id": row["original_id"], "reason": "solidity_parse_or_unit_failure", "detail": str(exc), "source_sha256": digest} for row in group)
            continue
        accepted.extend(group)
    if not accepted:
        raise RuntimeError("No source-aligned rows survived the audit")
    splits = grouped_split(accepted, int(config["seed"]), list(config["split_ratios"]))
    random.Random(int(config["seed"])).shuffle(splits["train"])
    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    manifest = {
        "dataset": "DIVE Source-Main6", "seed": int(config["seed"]),
        "label_names": config["label_names"], "source_only": True,
        "test_labels_used_for_selection": False, "source_grouping": "normalized_sha256",
        "rows_before_source_audit": sum(1 for split in ("train", "valid", "test") for _ in load_jsonl(main6_dir / f"{split}.jsonl")),
        "accepted_rows": len(accepted), "accepted_source_groups": len({row["source_sha256"] for row in accepted}),
        "split_rows": {split: len(rows) for split, rows in splits.items()},
        "split_label_support": {split: np.asarray([row["multi_labels"] for row in rows]).sum(axis=0).tolist() for split, rows in splits.items()},
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "source_main6_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_jsonl(report_dir / "source_main6_exclusions.jsonl", exclusions)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
