import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIVE_MAIN6_LABEL_NAMES = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create non-grouped random DIVE Main-6 split.")
    parser.add_argument("--source_data_dir", default="data/processed/DIVE_main6_access4000_clean3952")
    parser.add_argument("--output_data_dir", default="data/processed/DIVE_main6_random_split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--report_path", default="data/reports/dive_main6_random_split_report.txt")
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path):
    try:
        return Path(path).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def digest_opcode(opcode):
    normalized = " ".join(str(opcode).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def index_jsonl(path, source_split):
    refs = []
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                refs.append((source_split, offset))
    return refs


def read_at(handle, offset):
    handle.seek(offset)
    return json.loads(handle.readline().decode("utf-8"))


def write_report(report, txt_path):
    json_path = txt_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["DIVE Main-6 random split report", ""]
    for key, value in report.items():
        lines.append(f"{key}: {json.dumps(value) if isinstance(value, (dict, list)) else value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {rel(txt_path)}")
    print(f"[OK] wrote {rel(json_path)}")


def main():
    args = parse_args()
    if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-9:
        raise ValueError("train/valid/test ratios must sum to 1.")
    source_dir = resolve(args.source_data_dir)
    output_dir = resolve(args.output_data_dir)
    paths = {split: source_dir / f"{split}.jsonl" for split in ["train", "valid", "test"]}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing DIVE Main-6 split: {path}")

    refs = []
    for split, path in paths.items():
        refs.extend(index_jsonl(path, split))
    random.Random(args.seed).shuffle(refs)
    total = len(refs)
    train_count = int(total * args.train_ratio)
    valid_count = int(total * args.valid_ratio)
    assignments = {
        "train": refs[:train_count],
        "valid": refs[train_count : train_count + valid_count],
        "test": refs[train_count + valid_count :],
    }

    handles = {split: path.open("rb") for split, path in paths.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    label_counts = {}
    safe_counts = {}
    label_cardinality = {}
    opcode_lengths = {}
    hash_sets = {}
    ids_by_output = {}
    old_train_hashes = set()
    old_train_ids = set()

    try:
        for source_split, offset in index_jsonl(paths["train"], "train"):
            row = read_at(handles[source_split], offset)
            old_train_ids.add(str(row["id"]))
            old_train_hashes.add(digest_opcode(row.get("opcode", "")))

        for output_split, split_refs in assignments.items():
            counts = [0] * len(DIVE_MAIN6_LABEL_NAMES)
            safe = 0
            cardinality_sum = 0
            lengths = []
            hashes = set()
            ids = set()
            with (output_dir / f"{output_split}.jsonl").open("w", encoding="utf-8") as out:
                for source_split, offset in split_refs:
                    row = read_at(handles[source_split], offset)
                    labels = [int(value) for value in row.get("multi_labels", [])]
                    if len(labels) != len(DIVE_MAIN6_LABEL_NAMES):
                        raise ValueError(f"Expected 6 DIVE Main-6 labels, got {len(labels)}")
                    clean = {
                        "id": str(row["id"]),
                        "opcode": str(row.get("opcode", "")),
                        "binary_label": int(any(labels)),
                        "multi_labels": labels,
                    }
                    out.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    for idx, value in enumerate(labels):
                        counts[idx] += value
                    safe += int(not any(labels))
                    cardinality_sum += sum(labels)
                    lengths.append(len(clean["opcode"].split()))
                    hashes.add(digest_opcode(clean["opcode"]))
                    ids.add(clean["id"])
            label_counts[output_split] = dict(zip(DIVE_MAIN6_LABEL_NAMES, counts))
            safe_counts[output_split] = safe
            label_cardinality[output_split] = cardinality_sum / max(1, len(split_refs))
            opcode_lengths[output_split] = {
                "mean": sum(lengths) / max(1, len(lengths)),
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
            }
            hash_sets[output_split] = hashes
            ids_by_output[output_split] = ids
    finally:
        for handle in handles.values():
            handle.close()

    mapping = {
        "label_names": DIVE_MAIN6_LABEL_NAMES,
        "label_to_id": {name: idx for idx, name in enumerate(DIVE_MAIN6_LABEL_NAMES)},
        "id_to_label": {str(idx): name for idx, name in enumerate(DIVE_MAIN6_LABEL_NAMES)},
        "source": "DIVE_main6_access4000_clean3952",
    }
    (output_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    overlaps = {
        "train_valid": len(hash_sets["train"] & hash_sets["valid"]),
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "valid_test": len(hash_sets["valid"] & hash_sets["test"]),
    }
    pretrain_exposure = {}
    for split in ["train", "valid", "test"]:
        sample_count = len(assignments[split])
        id_seen = len(ids_by_output[split] & old_train_ids)
        hash_seen = len(hash_sets[split] & old_train_hashes)
        pretrain_exposure[split] = {
            "sample_count": sample_count,
            "ids_seen_in_main6_strict_train": id_seen,
            "id_seen_ratio": id_seen / max(1, sample_count),
            "opcode_hashes_seen_in_main6_strict_train": hash_seen,
            "unique_opcode_hash_count": len(hash_sets[split]),
            "opcode_hash_seen_ratio": hash_seen / max(1, len(hash_sets[split])),
        }

    report = {
        "status": "ok",
        "source_data_dir": rel(source_dir),
        "output_data_dir": rel(output_dir),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "valid": args.valid_ratio, "test": args.test_ratio},
        "total_samples": total,
        "split_samples": {split: len(rows) for split, rows in assignments.items()},
        "schema": ["id", "opcode", "binary_label", "multi_labels"],
        "num_labels": len(DIVE_MAIN6_LABEL_NAMES),
        "label_names": DIVE_MAIN6_LABEL_NAMES,
        "label_counts": label_counts,
        "safe_counts": safe_counts,
        "safe_ratios": {
            split: safe_counts[split] / max(1, len(assignments[split]))
            for split in assignments
        },
        "average_labels_per_sample": label_cardinality,
        "raw_opcode_length_statistics": opcode_lengths,
        "opcode_hash_overlap": overlaps,
        "leakage_status": "ok" if all(v == 0 for v in overlaps.values()) else "warning",
        "continued_mlm_exposure": pretrain_exposure,
    }
    write_report(report, resolve(args.report_path))
    for split in assignments:
        print(f"[OK] wrote {rel(output_dir / f'{split}.jsonl')}")


if __name__ == "__main__":
    main()
