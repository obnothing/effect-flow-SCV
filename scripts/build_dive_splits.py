import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from dive_common import (
    DIVE_LABEL_NAMES,
    PROJECT_ROOT,
    REPORT_DIR,
    default_label_path,
    default_opcode_path,
    iter_opcodes,
    load_labels,
    opcode_hash,
    resolve,
    write_json_txt,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build strict opcode-hash grouped DIVE splits.")
    parser.add_argument("--opcode_jsonl", default=None)
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--output_dir", default="data/processed/DIVE")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_groups(group_ids, group_labels, train_ratio, valid_ratio, test_ratio, seed):
    warnings = []
    method = None
    total = train_ratio + valid_ratio + test_ratio
    train_ratio, valid_ratio, test_ratio = train_ratio / total, valid_ratio / total, test_ratio / total
    try:
        import numpy as np
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        x = np.arange(len(group_ids))
        y = np.asarray(group_labels, dtype=int)
        first = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=valid_ratio + test_ratio,
            random_state=seed,
        )
        train_idx, temp_idx = next(first.split(x, y))
        temp_ratio = test_ratio / (valid_ratio + test_ratio)
        second = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=temp_ratio,
            random_state=seed,
        )
        valid_rel, test_rel = next(second.split(x[temp_idx], y[temp_idx]))
        split = {
            "train": [group_ids[i] for i in train_idx],
            "valid": [group_ids[temp_idx[i]] for i in valid_rel],
            "test": [group_ids[temp_idx[i]] for i in test_rel],
        }
        method = "iterative_stratification_on_opcode_hash_groups"
        return split, method, warnings
    except Exception as exc:
        warnings.append(f"iterative-stratification unavailable or failed: {exc}")

    try:
        from sklearn.model_selection import train_test_split

        train_ids, temp_ids = train_test_split(
            group_ids,
            test_size=valid_ratio + test_ratio,
            random_state=seed,
            shuffle=True,
        )
        valid_size = valid_ratio / (valid_ratio + test_ratio)
        valid_ids, test_ids = train_test_split(
            temp_ids,
            train_size=valid_size,
            random_state=seed,
            shuffle=True,
        )
        method = "sklearn_train_test_split_on_opcode_hash_groups"
        warnings.append("sklearn fallback is group-safe but not multi-label stratified.")
        return {"train": train_ids, "valid": valid_ids, "test": test_ids}, method, warnings
    except Exception as exc:
        warnings.append(f"sklearn fallback unavailable or failed: {exc}")

    rng = random.Random(seed)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    valid_end = train_end + int(len(shuffled) * valid_ratio)
    method = "random_shuffle_on_opcode_hash_groups"
    warnings.append("random fallback is group-safe but not stratified.")
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }, method, warnings


def label_distribution(records):
    counts = [0] * len(DIVE_LABEL_NAMES)
    for record in records:
        for idx, value in enumerate(record["multi_labels"]):
            counts[idx] += int(value)
    return dict(zip(DIVE_LABEL_NAMES, counts))


def overlap(left, right):
    return len(set(left) & set(right))


def main():
    args = parse_args()
    opcode_path = resolve(args.opcode_jsonl) if args.opcode_jsonl else default_opcode_path()
    label_path = resolve(args.label_csv) if args.label_csv else default_label_path()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(label_path)

    records = []
    groups = defaultdict(list)
    group_label_or = {}
    for contract_id, opcode in iter_opcodes(opcode_path):
        if contract_id not in labels:
            continue
        multi = labels[contract_id]
        h = opcode_hash(opcode)
        record = {
            "id": contract_id,
            "opcode": opcode,
            "binary_label": int(any(multi)),
            "multi_labels": multi,
            "opcode_hash": h,
        }
        groups[h].append(record)
        if h not in group_label_or:
            group_label_or[h] = list(multi)
        else:
            group_label_or[h] = [
                int(left or right) for left, right in zip(group_label_or[h], multi)
            ]
        records.append(record)

    group_ids = list(groups)
    group_labels = [group_label_or[group_id] for group_id in group_ids]
    split_group_ids, split_method, warnings = split_groups(
        group_ids,
        group_labels,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        args.seed,
    )
    split_records = {}
    for split, ids in split_group_ids.items():
        split_records[split] = []
        for group_id in ids:
            split_records[split].extend(groups[group_id])
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in split_records[split]:
                out = {
                    "id": record["id"],
                    "opcode": record["opcode"],
                    "binary_label": record["binary_label"],
                    "multi_labels": record["multi_labels"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    split_hashes = {
        split: {record["opcode_hash"] for record in rows}
        for split, rows in split_records.items()
    }
    overlap_counts = {
        "train_valid_opcode_hash_overlap": overlap(split_hashes["train"], split_hashes["valid"]),
        "train_test_opcode_hash_overlap": overlap(split_hashes["train"], split_hashes["test"]),
        "valid_test_opcode_hash_overlap": overlap(split_hashes["valid"], split_hashes["test"]),
    }
    leakage_status = "ok" if all(value == 0 for value in overlap_counts.values()) else "warning"
    report = {
        "status": "ok" if leakage_status == "ok" else "warning",
        "opcode_jsonl": opcode_path.relative_to(PROJECT_ROOT).as_posix(),
        "label_csv": label_path.relative_to(PROJECT_ROOT).as_posix(),
        "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        "split_seed": args.seed,
        "ratios": {"train": args.train_ratio, "valid": args.valid_ratio, "test": args.test_ratio},
        "split_method": split_method,
        "fallback_warning": warnings,
        "total_aligned_samples": len(records),
        "unique_opcode_hash_groups": len(groups),
        "train_samples": len(split_records["train"]),
        "valid_samples": len(split_records["valid"]),
        "test_samples": len(split_records["test"]),
        "label_names": DIVE_LABEL_NAMES,
        "label_distribution": {
            split: label_distribution(rows) for split, rows in split_records.items()
        },
        **overlap_counts,
        "leakage_status": leakage_status,
    }
    write_json_txt(
        report,
        REPORT_DIR / "dive_split_report.json",
        REPORT_DIR / "dive_split_report.txt",
        "DIVE split report",
    )


if __name__ == "__main__":
    main()
