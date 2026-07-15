"""Build an external, runtime-opcode-only public Ethereum pretraining corpus."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-inputs",
        nargs="+",
        default=[
            "data/raw/ethereum_public_pretrain_100k/runtime_part1.jsonl",
            "data/raw/ethereum_public_pretrain_100k/runtime_part2.jsonl",
            "data/raw/ethereum_public_pretrain_100k/runtime_part3.jsonl",
            "data/raw/ethereum_public_pretrain_100k/runtime_part4.jsonl",
        ],
    )
    parser.add_argument(
        "--downstream-dir", default="data/processed/DIVE_main6_access4000_clean3952"
    )
    parser.add_argument(
        "--output", default="data/processed/ethereum_public_pretrain_100k/runtime_opcode.jsonl"
    )
    parser.add_argument(
        "--report", default="data/reports/ethereum_public_pretrain_100k_report.txt"
    )
    parser.add_argument("--target-samples", type=int, default=100000)
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def opcode_hash(opcode):
    normalized = " ".join(str(opcode).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_downstream_hashes(directory):
    hashes = set()
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    hashes.add(row.get("opcode_hash") or opcode_hash(row.get("opcode", "")))
    return hashes


def main():
    args = parse_args()
    runtime_paths = [resolve(value) for value in args.runtime_inputs]
    downstream_hashes = load_downstream_hashes(resolve(args.downstream_dir))
    output_path, report_path = resolve(args.output), resolve(args.report)
    counters, seen_addresses, seen_opcodes, output_addresses = Counter(), set(), set(), set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_count = 0
    with output_path.open("w", encoding="utf-8") as output_handle:
        for path in runtime_paths:
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    counters["runtime_rows_input"] += 1
                    row = json.loads(line)
                    address = str(row.get("address", "")).lower()
                    digest = row.get("opcode_hash") or opcode_hash(row.get("opcode", ""))
                    if not address or address in seen_addresses:
                        counters["duplicate_address"] += 1
                        continue
                    seen_addresses.add(address)
                    if digest in downstream_hashes:
                        counters["excluded_downstream_opcode_overlap"] += 1
                        continue
                    opcode = " ".join(str(row.get("opcode", "")).split())
                    if not opcode:
                        counters["empty_opcode"] += 1
                        continue
                    seen_opcodes.add(digest)
                    output_addresses.add(address)
                    output_handle.write(json.dumps({
                        "id": row["id"], "opcode": opcode, "opcode_hash": digest,
                        "address": address, "source": row.get("source"),
                        "compiler_version": row.get("compiler_version"),
                        "proxy_type": row.get("proxy_type"),
                    }, ensure_ascii=False) + "\n")
                    output_count += 1
                    if output_count >= args.target_samples:
                        break
            if output_count >= args.target_samples:
                break
    if output_count < args.target_samples:
        raise ValueError(
            f"Only {output_count} eligible rows; need {args.target_samples}."
        )
    report = {
        "status": "ok",
        "target_samples": args.target_samples,
        "output_samples": output_count,
        "unique_output_addresses": len(output_addresses),
        "unique_output_opcode_hashes": len(seen_opcodes),
        "exact_opcode_duplicate_rows_retained": output_count - len(seen_opcodes),
        "downstream_opcode_hashes_excluded_against": len(downstream_hashes),
        "counters": dict(counters),
        "runtime_inputs": [display_path(path) for path in runtime_paths],
        "output": display_path(output_path),
        "caveats": [
            "The corpus contains public Ethereum runtime opcode only and no vulnerability labels.",
            "Exact runtime-opcode overlap with the new downstream benchmark is excluded.",
            "Address-level contracts are retained; exact opcode duplicate prevalence is reported for optional deduplicated training ablation.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
