"""Build a deterministic grouped development split from train+valid only."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/DIVE_main6_opcode_process01"
OUTPUT = ROOT / "data/splits/e5_grouped_dev/seed42"


def group_key(item):
    normalized = " ".join(str(item.get("opcode", "")).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_vector(row):
    return row.get("multi_labels", [0] * 6)


def main():
    train_rows = read(SOURCE / "train.jsonl")
    valid_rows = read(SOURCE / "valid.jsonl")
    rows = train_rows + valid_rows
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    rng = random.Random(42)
    keys = list(groups)
    rng.shuffle(keys)
    total = len(rows)
    target_samples = round(total * 0.20)
    target_labels = [sum(int(label_vector(row)[index]) for row in rows) * 0.20 for index in range(6)]
    valid_keys, valid_count, valid_labels = set(), 0, [0] * 6

    def score(count, labels):
        sample_error = abs(count - target_samples) / max(target_samples, 1)
        label_error = sum(abs(labels[index] - target_labels[index]) / max(target_labels[index], 1) for index in range(6)) / 6
        return sample_error + label_error

    for key in sorted(keys, key=lambda item: (-len(groups[item]), item)):
        candidate_rows = groups[key]
        candidate_count = valid_count + len(candidate_rows)
        candidate_labels = [valid_labels[index] + sum(int(label_vector(row)[index]) for row in candidate_rows) for index in range(6)]
        if valid_count < target_samples or score(candidate_count, candidate_labels) <= score(valid_count, valid_labels):
            valid_keys.add(key)
            valid_count = candidate_count
            valid_labels = candidate_labels
    grouped_valid = [row for row in rows if group_key(row) in valid_keys]
    grouped_train = [row for row in rows if group_key(row) not in valid_keys]
    if not grouped_train or not grouped_valid:
        raise RuntimeError("grouped split produced an empty partition")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, values in (("train.jsonl", grouped_train), ("valid.jsonl", grouped_valid)):
        with (OUTPUT / name).open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    train_groups = {group_key(row) for row in grouped_train}
    valid_groups = {group_key(row) for row in grouped_valid}
    manifest = {
        "dataset": "DIVE_main6_opcode_process01 train+valid development pool",
        "seed": 42,
        "source_train_sha256": file_hash(SOURCE / "train.jsonl"),
        "source_valid_sha256": file_hash(SOURCE / "valid.jsonl"),
        "group_rule": "sha256 of whitespace-normalized opcode",
        "train_samples": len(grouped_train), "valid_samples": len(grouped_valid),
        "train_groups": len(train_groups), "valid_groups": len(valid_groups),
        "group_overlap": len(train_groups & valid_groups),
        "train_label_prevalence": [sum(int(label_vector(row)[index]) for row in grouped_train) / len(grouped_train) for index in range(6)],
        "valid_label_prevalence": [sum(int(label_vector(row)[index]) for row in grouped_valid) / len(grouped_valid) for index in range(6)],
        "test_checked": False,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report_dir = ROOT / "reports/light_label_model/e5_final"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "grouped_split_audit.md").write_text("\n".join([
        "# E5 Grouped Development Split Audit", "", "The split uses train+valid development data only; test is locked.", "",
        f"- Train samples: `{manifest['train_samples']}`; groups: `{manifest['train_groups']}`.",
        f"- Grouped-valid samples: `{manifest['valid_samples']}`; groups: `{manifest['valid_groups']}`.",
        f"- Group overlap: `{manifest['group_overlap']}`.",
        "- Group key: SHA-256 of whitespace-normalized opcode.",
        "- This is a secondary development protocol and does not replace the official random split.",
    ]) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()

