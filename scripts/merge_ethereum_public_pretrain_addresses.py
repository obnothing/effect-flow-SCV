"""Merge independently crawled Blockscout address ranges into one unique pool."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/raw/ethereum_public_pretrain_100k/blockscout_verified_addresses.jsonl",
            "data/raw/ethereum_public_pretrain_100k/blockscout_verified_addresses_part2.jsonl",
            "data/raw/ethereum_public_pretrain_100k/blockscout_verified_addresses_part3.jsonl",
        ],
    )
    parser.add_argument(
        "--output",
        default="data/raw/ethereum_public_pretrain_100k/blockscout_verified_addresses_merged.jsonl",
    )
    parser.add_argument("--target-addresses", type=int, default=102000)
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    seen, rows = set(), []
    for value in args.inputs:
        path = resolve(value)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                address = str(row.get("address", "")).lower()
                if not address or address in seen:
                    continue
                seen.add(address)
                rows.append(row)
                if len(rows) >= args.target_addresses:
                    break
        if len(rows) >= args.target_addresses:
            break
    if len(rows) < args.target_addresses:
        raise ValueError(f"Need {args.target_addresses} addresses, found {len(rows)}")
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[OK] wrote {len(rows)} unique addresses to {output_path}")


if __name__ == "__main__":
    main()
