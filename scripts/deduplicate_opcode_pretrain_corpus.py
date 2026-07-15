"""Write a first-occurrence runtime-opcode-deduplicated pretraining corpus."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/processed/ethereum_public_pretrain_100k/runtime_opcode.jsonl"
    )
    parser.add_argument(
        "--output",
        default="data/processed/ethereum_public_pretrain_19143_unique_runtime/runtime_opcode.jsonl",
    )
    parser.add_argument(
        "--report", default="data/reports/ethereum_public_pretrain_19143_unique_report.txt"
    )
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    input_path, output_path, report_path = map(resolve, (args.input, args.output, args.report))
    seen, input_count, output_count = set(), 0, 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for line in source:
            if not line.strip():
                continue
            input_count += 1
            row = json.loads(line)
            digest = row["opcode_hash"]
            if digest in seen:
                continue
            seen.add(digest)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_count += 1
    report = {
        "status": "ok",
        "input": display_path(input_path),
        "output": display_path(output_path),
        "input_address_level_rows": input_count,
        "unique_runtime_opcode_rows": output_count,
        "exact_duplicate_rows_removed": input_count - output_count,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
