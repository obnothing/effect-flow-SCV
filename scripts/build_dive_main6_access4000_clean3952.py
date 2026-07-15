"""Build a grouped, fresh-split DIVE Main-6 benchmark with screened clean contracts."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LABELS = (
    "Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
    "DoS", "Bad Randomness", "Front Running", "Time manipulation",
)
MAIN6_LABELS = (
    "Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
    "DoS", "Time manipulation",
)
MAIN6_INDICES = tuple(SOURCE_LABELS.index(label) for label in MAIN6_LABELS)
SPLITS = ("train", "valid", "test")
SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dive-dir", default="data/processed/DIVE_random_split")
    parser.add_argument(
        "--clean-candidates",
        default="data/raw/smartbugs_wild/tool_screened_main6_clean_runtime_deduplicated.jsonl",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/DIVE_main6_access4000_clean3952"
    )
    parser.add_argument(
        "--report", default="data/reports/dive_main6_access4000_clean3952_report.txt"
    )
    parser.add_argument("--access-removals", type=int, default=4000)
    parser.add_argument("--expected-clean-count", type=int, default=3952)
    parser.add_argument("--seed", type=int, default=42)
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


def opcode_hash(opcode):
    opcode = " ".join(str(opcode).split())
    return hashlib.sha256(opcode.encode("utf-8")).hexdigest()


def length_bin(tokens):
    if tokens < 256:
        return "lt256"
    if tokens < 1024:
        return "256_1023"
    if tokens < 4096:
        return "1024_4095"
    return "ge4096"


def compiler_era(value):
    text = str(value or "").lower().lstrip("v")
    parts = text.split(".")
    if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
        return f"{parts[0]}.{parts[1]}"
    return "unknown"


def is_proxy(opcode):
    tokens = opcode.split()
    return int("DELEGATECALL" in tokens and len(tokens) <= 300)


def load_dive_rows(dive_dir):
    rows = []
    for split in SPLITS:
        path = dive_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                source_labels = [int(value) for value in raw["multi_labels"]]
                if len(source_labels) != len(SOURCE_LABELS):
                    raise ValueError(f"Invalid DIVE label width for {raw.get('id')}")
                labels = [source_labels[index] for index in MAIN6_INDICES]
                opcode = " ".join(str(raw.get("opcode", "")).split())
                if not opcode:
                    continue
                rows.append({
                    "id": f"dive:{raw['id']}",
                    "opcode": opcode,
                    "multi_labels": labels,
                    "binary_label": int(any(labels)),
                    "source": "DIVE",
                    "source_original_split": split,
                    "compiler_era": "unknown",
                    "proxy": is_proxy(opcode),
                    "opcode_hash": opcode_hash(opcode),
                })
    return rows


def load_clean_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            opcode = " ".join(str(raw.get("opcode", "")).split())
            if not opcode:
                continue
            rows.append({
                "id": raw["id"],
                "opcode": opcode,
                "multi_labels": [0] * len(MAIN6_LABELS),
                "binary_label": 0,
                "source": "SmartBugs Wild tool-screened clean",
                "address": raw.get("address"),
                "compiler_version": raw.get("compiler_version"),
                "compiler_era": compiler_era(raw.get("compiler_version")),
                "proxy": is_proxy(opcode),
                "opcode_hash": raw.get("opcode_hash") or opcode_hash(opcode),
            })
    return rows


def stratified_remove_access(rows, removal_count, seed):
    access_index = MAIN6_LABELS.index("Access Control")
    candidates = [index for index, row in enumerate(rows) if row["multi_labels"][access_index]]
    if len(candidates) < removal_count:
        raise ValueError(f"Only {len(candidates)} Access-positive rows available")
    strata = defaultdict(list)
    for index in candidates:
        row = rows[index]
        key = (tuple(row["multi_labels"]), length_bin(len(row["opcode"].split())))
        strata[key].append(index)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)
    exact = []
    remainders = []
    for key, values in strata.items():
        raw_quota = removal_count * len(values) / len(candidates)
        base = min(len(values), int(raw_quota))
        exact.extend((key, index) for index in values[:base])
        remainders.append((raw_quota - base, key))
    remaining = removal_count - len(exact)
    selected = {index for _, index in exact}
    for _, key in sorted(remainders, reverse=True):
        values = list(strata[key])
        rng.shuffle(values)
        for index in values:
            if remaining == 0:
                break
            if index not in selected:
                selected.add(index)
                remaining -= 1
        if remaining == 0:
            break
    if len(selected) != removal_count:
        raise RuntimeError("Unable to allocate exactly the requested Access removals")
    return selected


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["opcode_hash"]].append(row)
    return list(groups.values())


def group_stratum(group):
    first = group[0]
    labels = [max(row["multi_labels"][index] for row in group) for index in range(len(MAIN6_LABELS))]
    tokens = max(len(row["opcode"].split()) for row in group)
    return (
        tuple(labels),
        length_bin(tokens),
        int(any(row["proxy"] for row in group)),
        first["source"],
        first["compiler_era"],
    )


def grouped_stratified_split(rows, seed):
    groups = group_rows(rows)
    total_rows = len(rows)
    strata = defaultdict(list)
    for group in groups:
        strata[group_stratum(group)].append(group)
    target_sizes = {split: total_rows * SPLIT_RATIOS[split] for split in SPLITS}
    split_sizes = Counter()
    assigned = {split: [] for split in SPLITS}
    rng = random.Random(seed)
    for stratum, stratum_groups in sorted(strata.items(), key=lambda item: (len(item[1]), str(item[0]))):
        rng.shuffle(stratum_groups)
        stratum_sizes = Counter()
        stratum_total = sum(len(group) for group in stratum_groups)
        stratum_targets = {
            split: stratum_total * SPLIT_RATIOS[split] for split in SPLITS
        }
        for group in sorted(stratum_groups, key=len, reverse=True):
            size = len(group)
            chosen = min(
                SPLITS,
                key=lambda split: (
                    (stratum_sizes[split] + size) / max(1.0, stratum_targets[split]),
                    (split_sizes[split] + size) / max(1.0, target_sizes[split]),
                    split,
                ),
            )
            assigned[chosen].extend(group)
            split_sizes[chosen] += size
            stratum_sizes[chosen] += size
    return assigned


def split_stats(rows):
    result = {"samples": len(rows), "positives": {}, "negatives": {}, "positive_ratio": {}}
    for index, label in enumerate(MAIN6_LABELS):
        count = sum(row["multi_labels"][index] for row in rows)
        result["positives"][label] = count
        result["negatives"][label] = len(rows) - count
        result["positive_ratio"][label] = count / max(1, len(rows))
    result["all_zero"] = sum(not any(row["multi_labels"]) for row in rows)
    result["sources"] = dict(Counter(row["source"] for row in rows))
    result["compiler_eras"] = dict(Counter(row["compiler_era"] for row in rows))
    result["proxy_count"] = sum(row["proxy"] for row in rows)
    result["opcode_token_length"] = {
        "mean": sum(len(row["opcode"].split()) for row in rows) / max(1, len(rows)),
        "median": sorted(len(row["opcode"].split()) for row in rows)[len(rows) // 2] if rows else 0,
    }
    return result


def main():
    args = parse_args()
    dive_dir, clean_path = resolve(args.dive_dir), resolve(args.clean_candidates)
    output_dir, report_path = resolve(args.output_dir), resolve(args.report)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_dir}; use --overwrite")
    dive_rows = load_dive_rows(dive_dir)
    clean_rows = load_clean_rows(clean_path)
    if len(clean_rows) != args.expected_clean_count:
        raise ValueError(f"Expected {args.expected_clean_count} clean rows, got {len(clean_rows)}")
    remove_indices = stratified_remove_access(dive_rows, args.access_removals, args.seed)
    retained_dive = [row for index, row in enumerate(dive_rows) if index not in remove_indices]
    all_rows = retained_dive + clean_rows
    splits = grouped_stratified_split(all_rows, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        rows.sort(key=lambda row: row["id"])
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                output = {key: row[key] for key in ("id", "opcode", "binary_label", "multi_labels", "source", "opcode_hash")}
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    mapping = {
        "label_names": list(MAIN6_LABELS),
        "label_to_id": {label: index for index, label in enumerate(MAIN6_LABELS)},
        "source": "DIVE Main-6 + SmartBugs Wild multi-tool-negative runtime-opcode pool",
        "access_positive_rows_removed": args.access_removals,
        "clean_rows_added": len(clean_rows),
        "split_seed": args.seed,
        "split_policy": "opcode-hash grouped greedy multilabel stratification",
    }
    (output_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "output_dir": relative(output_dir),
        "dive_rows_before_access_removal": len(dive_rows),
        "dive_access_positive_rows_removed": len(remove_indices),
        "dive_rows_retained": len(retained_dive),
        "clean_rows_added": len(clean_rows),
        "total_rows": len(all_rows),
        "unique_opcode_groups": len(group_rows(all_rows)),
        "split_policy": mapping["split_policy"],
        "split_statistics": {split: split_stats(rows) for split, rows in splits.items()},
        "caveats": [
            "The clean pool is multi-tool-negative, not a formal proof of safety.",
            "DIVE records do not carry compiler metadata, so exact DIVE-clean compiler-era matching is unavailable.",
            "This is a new random/grouped benchmark and is not directly comparable to legacy DIVE_random_split results.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
