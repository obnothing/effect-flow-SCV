import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare BJUT and DIVE random-split structures.")
    parser.add_argument("--bjut_dir", default="data/processed/BJUT_SC01_random_split")
    parser.add_argument("--dive_dir", default="data/processed/DIVE_random_split")
    parser.add_argument(
        "--dive_random_report", default="data/reports/dive_random_split_report.json"
    )
    parser.add_argument(
        "--output", default="data/reports/bjut_vs_dive_random_structure_report.txt"
    )
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path):
    try:
        return Path(path).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * p
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (pos - low))


def opcode_hash(opcode):
    normalized = " ".join(str(opcode).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_label_names(data_dir):
    mapping = data_dir / "label_mapping.json"
    if mapping.exists():
        return json.loads(mapping.read_text(encoding="utf-8")).get("label_names", [])
    return []


def scan_dataset(data_dir):
    label_names = load_label_names(data_dir)
    report = {
        "data_dir": rel(data_dir),
        "label_names": label_names,
        "num_labels": len(label_names),
        "splits": {},
    }
    hash_sets = {}
    all_hashes = Counter()
    for split in ["train", "valid", "test"]:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing random split: {path}")
        count = 0
        safe = 0
        cardinality = 0
        label_counts = [0] * len(label_names)
        opcode_lengths = []
        schema = None
        hashes = set()
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                schema = schema or list(row.keys())
                labels = [int(v) for v in row.get("multi_labels", [])]
                if len(labels) != len(label_names):
                    raise ValueError(
                        f"{path}:{line_no} expected {len(label_names)} labels, got {len(labels)}"
                    )
                count += 1
                safe += int(not any(labels))
                cardinality += sum(labels)
                for idx, value in enumerate(labels):
                    label_counts[idx] += value
                opcode = str(row.get("opcode", ""))
                opcode_lengths.append(len(opcode.split()))
                digest = opcode_hash(opcode)
                hashes.add(digest)
                all_hashes[digest] += 1
        hash_sets[split] = hashes
        report["splits"][split] = {
            "samples": count,
            "schema": schema,
            "safe_samples": safe,
            "safe_ratio": safe / max(1, count),
            "vulnerable_samples": count - safe,
            "average_labels_per_sample": cardinality / max(1, count),
            "label_counts": dict(zip(label_names, label_counts)),
            "raw_opcode_length": {
                "mean": sum(opcode_lengths) / max(1, len(opcode_lengths)),
                "p50": percentile(opcode_lengths, 0.50),
                "p90": percentile(opcode_lengths, 0.90),
                "max": max(opcode_lengths) if opcode_lengths else 0,
            },
            "unique_opcode_hashes": len(hashes),
        }
    report["total_samples"] = sum(row["samples"] for row in report["splits"].values())
    report["opcode_hash_overlap"] = {
        "train_valid": len(hash_sets["train"] & hash_sets["valid"]),
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "valid_test": len(hash_sets["valid"] & hash_sets["test"]),
    }
    report["unique_opcode_hashes"] = len(all_hashes)
    report["duplicate_opcode_hash_groups"] = sum(v > 1 for v in all_hashes.values())
    report["duplicate_opcode_samples"] = sum(v for v in all_hashes.values() if v > 1)
    return report


def write_report(report, txt_path):
    json_path = txt_path.with_suffix(".json")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["BJUT SC01 random split vs DIVE random split structure report", ""]
    lines.append(json.dumps(report, indent=2))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {rel(txt_path)}")
    print(f"[OK] wrote {rel(json_path)}")


def main():
    args = parse_args()
    bjut = scan_dataset(resolve(args.bjut_dir))
    dive = scan_dataset(resolve(args.dive_dir))
    dive_random_report_path = resolve(args.dive_random_report)
    dive_random_report = (
        json.loads(dive_random_report_path.read_text(encoding="utf-8"))
        if dive_random_report_path.exists()
        else {}
    )
    comparison = {
        "same_jsonl_schema": all(
            bjut["splits"][split]["schema"] == dive["splits"][split]["schema"]
            for split in ["train", "valid", "test"]
        ),
        "bjut_num_labels": bjut["num_labels"],
        "dive_num_labels": dive["num_labels"],
        "sample_count_ratio_bjut_over_dive": bjut["total_samples"] / max(1, dive["total_samples"]),
        "semantic_difference": (
            "BJUT uses 10 fine-grained vulnerability labels; DIVE uses 8 broader "
            "categories. Label indices and classifier heads are not interchangeable."
        ),
        "class_balance_difference": (
            "DIVE has far fewer all-zero safe contracts and substantially higher "
            "multi-label cardinality than BJUT, so raw F1 values are not directly comparable."
        ),
        "split_warning": (
            "Both requested experiments use non-grouped random splits. BJUT also uses "
            "BJUT-full-corpus MLM; DIVE random valid/test overlap the old strict-train "
            "continued-MLM corpus. Both are non-strict/transductive results."
        ),
    }
    report = {
        "status": "warning",
        "bjut": bjut,
        "dive": dive,
        "comparison": comparison,
        "dive_continued_mlm_exposure": dive_random_report.get("continued_mlm_exposure"),
        "reporting_rule": (
            "Report results separately by dataset. Do not merge label spaces or claim "
            "strict cross-dataset generalization from these random-split experiments."
        ),
    }
    write_report(report, resolve(args.output))


if __name__ == "__main__":
    main()
