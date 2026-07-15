"""Remove runtime-opcode overlap between a clean pool and all original DIVE splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dive-dir", default="data/processed/DIVE_random_split")
    parser.add_argument(
        "--candidates",
        default="data/raw/smartbugs_wild/tool_screened_main6_clean_runtime.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/raw/smartbugs_wild/tool_screened_main6_clean_runtime_deduplicated.jsonl",
    )
    parser.add_argument("--report", default="data/reports/dive_main6_clean_candidate_dedup.txt")
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def opcode_hash(opcode):
    normalized = " ".join(str(opcode).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_dive_hashes(directory):
    hashes = set()
    rows = 0
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows += 1
                    hashes.add(opcode_hash(json.loads(line).get("opcode", "")))
    return hashes, rows


def main():
    args = parse_args()
    dive_dir, candidates_path = resolve(args.dive_dir), resolve(args.candidates)
    output_path, report_path = resolve(args.output), resolve(args.report)
    dive_hashes, dive_rows = load_dive_hashes(dive_dir)
    counters = Counter()
    kept, seen = [], set()
    with candidates_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["candidates_input"] += 1
            row = json.loads(line)
            digest = row.get("opcode_hash") or opcode_hash(row.get("opcode", ""))
            if digest in seen:
                counters["candidate_internal_opcode_duplicates"] += 1
                continue
            seen.add(digest)
            if digest in dive_hashes:
                counters["overlap_with_any_dive_split"] += 1
                continue
            kept.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    lines = [
        "DIVE Main-6 clean candidate runtime-opcode deduplication",
        f"dive_rows_checked: {dive_rows}",
        f"unique_dive_opcode_hashes: {len(dive_hashes)}",
        f"candidates_input: {counters['candidates_input']}",
        f"candidate_internal_opcode_duplicates: {counters['candidate_internal_opcode_duplicates']}",
        f"overlap_with_any_dive_split: {counters['overlap_with_any_dive_split']}",
        f"candidates_kept: {len(kept)}",
        f"output: {output_path.relative_to(PROJECT_ROOT).as_posix()}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
