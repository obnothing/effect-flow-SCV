"""Derive an auditable six-label DIVE view without changing raw DIVE splits."""

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LABEL_NAMES = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation",
]
MAIN6_LABEL_NAMES = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
]
MAIN6_SOURCE_INDICES = [SOURCE_LABEL_NAMES.index(name) for name in MAIN6_LABEL_NAMES]
RARE_SOURCE_INDICES = [
    SOURCE_LABEL_NAMES.index("Bad Randomness"),
    SOURCE_LABEL_NAMES.index("Front Running"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a six-label DIVE view from the fixed DIVE random split."
    )
    parser.add_argument(
        "--source-dir", default="data/processed/DIVE_random_split"
    )
    parser.add_argument(
        "--output-dir", default="data/processed/DIVE_main6_random_split"
    )
    parser.add_argument(
        "--report-path", default="data/reports/dive_main6_dataset_report.txt"
    )
    parser.add_argument(
        "--exclude-rare-positive-contracts",
        action="store_true",
        help=(
            "Drop contracts containing Bad Randomness or Front Running. This is off "
            "by default because almost all such contracts also supervise Main-6 labels."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def empty_stats():
    return {
        "samples": 0,
        "main6_positive_counts": Counter(),
        "main6_cardinality": Counter(),
        "all_zero_main6": 0,
        "rare_positive_rows_seen": 0,
        "rare_positive_rows_removed": 0,
        "rare_positive_rows_kept": 0,
        "rare_positive_rows_with_main6": 0,
        "opcode_token_counts": [],
        "ids": set(),
    }


def update_stats(stats, row, labels, has_rare, removed):
    stats["rare_positive_rows_seen"] += int(has_rare)
    stats["rare_positive_rows_with_main6"] += int(has_rare and any(labels))
    stats["rare_positive_rows_removed"] += int(has_rare and removed)
    stats["rare_positive_rows_kept"] += int(has_rare and not removed)
    if removed:
        return
    stats["samples"] += 1
    stats["ids"].add(str(row["id"]))
    stats["all_zero_main6"] += int(not any(labels))
    stats["main6_cardinality"][sum(labels)] += 1
    stats["opcode_token_counts"].append(len(str(row.get("opcode", "")).split()))
    for name, value in zip(MAIN6_LABEL_NAMES, labels):
        stats["main6_positive_counts"][name] += int(value)


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return float(ordered[index])


def serializable_stats(stats):
    samples = stats["samples"]
    positives = {
        label: int(stats["main6_positive_counts"][label]) for label in MAIN6_LABEL_NAMES
    }
    return {
        "samples": samples,
        "unique_ids": len(stats["ids"]),
        "main6_positive_counts": positives,
        "main6_negative_counts": {label: samples - count for label, count in positives.items()},
        "main6_positive_ratios": {
            label: count / max(1, samples) for label, count in positives.items()
        },
        "main6_negative_ratios": {
            label: (samples - count) / max(1, samples)
            for label, count in positives.items()
        },
        "all_zero_main6": int(stats["all_zero_main6"]),
        "main6_label_cardinality": {
            str(key): int(value)
            for key, value in sorted(stats["main6_cardinality"].items())
        },
        "rare_positive_rows_seen": int(stats["rare_positive_rows_seen"]),
        "rare_positive_rows_removed": int(stats["rare_positive_rows_removed"]),
        "rare_positive_rows_kept": int(stats["rare_positive_rows_kept"]),
        "rare_positive_rows_with_main6": int(stats["rare_positive_rows_with_main6"]),
        "opcode_token_length": {
            "mean": sum(stats["opcode_token_counts"]) / max(1, samples),
            "median": percentile(stats["opcode_token_counts"], 0.5),
            "p90": percentile(stats["opcode_token_counts"], 0.9),
        },
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["DIVE Main-6 derivation report", ""]
    for key, value in report.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    source_dir = resolve(args.source_dir)
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.report_path)
    splits = ("train", "valid", "test")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
        )
    for split in splits:
        path = source_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing source split: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for split in splits:
        stats = empty_stats()
        source_path = source_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"
        with source_path.open("r", encoding="utf-8") as source, output_path.open(
            "w", encoding="utf-8"
        ) as output:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source_labels = [int(value) for value in row.get("multi_labels", [])]
                if len(source_labels) != len(SOURCE_LABEL_NAMES):
                    raise ValueError(
                        f"{source_path}:{line_no} has {len(source_labels)} labels; expected 8"
                    )
                labels = [source_labels[index] for index in MAIN6_SOURCE_INDICES]
                has_rare = any(source_labels[index] for index in RARE_SOURCE_INDICES)
                remove = bool(args.exclude_rare_positive_contracts and has_rare)
                update_stats(stats, row, labels, has_rare, remove)
                if remove:
                    continue
                output.write(
                    json.dumps(
                        {
                            "id": str(row["id"]),
                            "opcode": str(row.get("opcode", "")),
                            "binary_label": int(any(labels)),
                            "multi_labels": labels,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        split_stats[split] = serializable_stats(stats)

    mapping = {
        "label_names": MAIN6_LABEL_NAMES,
        "label_to_id": {name: index for index, name in enumerate(MAIN6_LABEL_NAMES)},
        "id_to_label": {str(index): name for index, name in enumerate(MAIN6_LABEL_NAMES)},
        "source": "DIVE_random_split label projection",
        "source_label_names": SOURCE_LABEL_NAMES,
        "source_label_indices": MAIN6_SOURCE_INDICES,
        "dropped_label_names": ["Bad Randomness", "Front Running"],
        "exclude_rare_positive_contracts": bool(args.exclude_rare_positive_contracts),
    }
    (output_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    report = {
        "status": "ok",
        "source_dir": relative(source_dir),
        "output_dir": relative(output_dir),
        "schema": ["id", "opcode", "binary_label", "multi_labels"],
        "source_label_names": SOURCE_LABEL_NAMES,
        "main6_label_names": MAIN6_LABEL_NAMES,
        "dropped_label_names": ["Bad Randomness", "Front Running"],
        "exclude_rare_positive_contracts": bool(args.exclude_rare_positive_contracts),
        "split_statistics": split_stats,
        "cache_compatibility": {
            "existing_evm_bert_feature_cache": "8-label cache can be reused by label-name slicing",
            "existing_semantic_cache": "8-label semantic cache can be reused by label-name slicing",
            "required_config_source_label_names": SOURCE_LABEL_NAMES,
            "required_config_label_names": MAIN6_LABEL_NAMES,
        },
        "caveat": (
            "This derives from DIVE_random_split and retains its opcode-hash overlap "
            "and continued-MLM transductive caveat."
        ),
    }
    write_report(report_path, report)
    print(f"[OK] wrote {relative(output_dir)}")
    print(f"[OK] wrote {relative(report_path)}")


if __name__ == "__main__":
    main()
