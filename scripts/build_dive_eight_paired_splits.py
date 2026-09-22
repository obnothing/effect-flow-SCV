"""Build paired 8:1:1 opcode/source DIVE splits from the official label file."""

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "DIVE_Raw_Data/Raw"
LABELS_PATH = ROOT / "data/external/dive_labels/unpacked/Labels/DIVE_Labels.csv"
OPCODE_PATH = RAW / "POST/Runtime_Opcode.jsonl"
ADDRESS_PATH = RAW / "DIVE_Samples.csv"
SOURCE_DIR = RAW / "PRE/Source codes"
OPCODE_OUT = ROOT / "data/processed/DIVE_8_opcode_random_split"
SOURCE_OUT = ROOT / "data/processed/DIVE_8_source_random_split"
REPORT_OUT = ROOT / "reports/dive_8_paired_split"
SEED = 42
LABEL_NAMES = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
               "DoS", "Bad Randomness", "Front Running", "Time manipulation"]


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def raw_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_split(rows):
    combinations = np.asarray(["".join(str(value) for value in row["multi_labels"]) for row in rows])
    counts = Counter(combinations.tolist())
    # Exact multilabel combinations are used as strata; very rare combinations
    # are pooled only to satisfy StratifiedShuffleSplit's minimum support rule.
    strata = np.asarray([value if counts[value] >= 2 else "__rare__" for value in combinations])
    indices = list(range(len(rows)))
    first = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, holdout_idx = next(first.split(indices, strata))
    holdout_indices = np.asarray(holdout_idx)
    holdout_strata = strata[holdout_indices].copy()
    holdout_counts = Counter(holdout_strata.tolist())
    holdout_strata = np.asarray([value if holdout_counts[value] >= 2 else "__rare_holdout__" for value in holdout_strata])
    second = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    valid_relative, test_relative = next(second.split(holdout_indices, holdout_strata))
    return {
        "train": [rows[int(index)] for index in train_idx],
        "valid": [rows[int(holdout_indices[index])] for index in valid_relative],
        "test": [rows[int(holdout_indices[index])] for index in test_relative],
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def manifest(split_rows, dataset_kind):
    return {
        "dataset": f"DIVE Main-8 {dataset_kind}", "seed": SEED,
        "split_ratios": [0.8, 0.1, 0.1], "label_names": LABEL_NAMES,
        "split_sizes": {name: len(rows) for name, rows in split_rows.items()},
        "split_label_support": {
            name: [sum(row["multi_labels"][index] for row in rows) for index in range(8)]
            for name, rows in split_rows.items()
        },
        "split_ids_sha256": {
            name: hashlib.sha256("\n".join(sorted(row["contract_id"] for row in rows)).encode()).hexdigest()
            for name, rows in split_rows.items()
        },
        "test_checked": False,
    }


def main():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Download the official DIVE_Labels.zip first: {LABELS_PATH}")
    labels = {}
    with LABELS_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["contractID", *LABEL_NAMES]:
            raise ValueError(f"Unexpected label columns: {reader.fieldnames}")
        for row in reader:
            contract_id = str(row["contractID"]).strip()
            values = [int(row[name]) for name in LABEL_NAMES]
            if any(value not in (0, 1) for value in values):
                raise ValueError(f"Non-binary label for contract {contract_id}")
            labels[contract_id] = values
    addresses = {}
    with ADDRESS_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            addresses[str(index)] = row["Address"].strip()
    rows = []
    for row in read_jsonl(OPCODE_PATH):
        contract_id = str(row["contractID"])
        if contract_id not in labels:
            raise ValueError(f"Missing official label for contractID={contract_id}")
        source_path = SOURCE_DIR / f"{contract_id}.sol"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        values = labels[contract_id]
        rows.append({
            "id": f"dive:{contract_id}", "contract_id": contract_id,
            "address": addresses.get(contract_id), "opcode": str(row["Opcodes"]),
            "source_path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": raw_sha256(source_path), "multi_labels": values,
            "binary_label": int(any(values)),
        })
    if len(rows) != 22330 or len({row["contract_id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected 22330 unique paired contracts, got {len(rows)}")
    split_rows = paired_split(rows)
    for output_dir, kind in ((OPCODE_OUT, "opcode"), (SOURCE_OUT, "source")):
        for split, values in split_rows.items():
            if kind == "opcode":
                output = [{key: row[key] for key in ("id", "contract_id", "address", "opcode", "multi_labels", "binary_label")} for row in values]
            else:
                output = [{key: row[key] for key in ("id", "contract_id", "address", "source_path", "source_sha256", "multi_labels", "binary_label")} for row in values]
            write_jsonl(output_dir / f"{split}.jsonl", output)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest(split_rows, kind), indent=2) + "\n", encoding="utf-8")
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "label_source": str(LABELS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "opcode_source": str(OPCODE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_code_dir": str(SOURCE_DIR.relative_to(ROOT)).replace("\\", "/"),
        "shared_split": True, "seed": SEED, "split_sizes": {key: len(value) for key, value in split_rows.items()},
        "label_names": LABEL_NAMES, "test_checked": False,
    }
    (REPORT_OUT / "paired_split_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
