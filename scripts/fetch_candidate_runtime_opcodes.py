"""Fetch reproducible Ethereum runtime opcodes for screened clean candidates."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evm_opcode import bytecode_to_opcode_sequence, normalize_opcode_sequence  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default="data/raw/smartbugs_wild/tool_screened_main6_clean_candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/raw/smartbugs_wild/tool_screened_main6_clean_runtime.jsonl",
    )
    parser.add_argument(
        "--rejected-output",
        default="data/raw/smartbugs_wild/tool_screened_main6_clean_runtime_rejected.jsonl",
    )
    parser.add_argument("--rpc-url", default="https://ethereum-rpc.publicnode.com")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def rpc_batch(session, url, candidates, timeout, retries):
    payload = [
        {
            "jsonrpc": "2.0",
            "id": index,
            "method": "eth_getCode",
            "params": [candidate["address"], "latest"],
        }
        for index, candidate in enumerate(candidates)
    ]
    last_error = None
    for attempt in range(retries):
        try:
            response = session.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("RPC batch response is not a list")
            values = {item.get("id"): item for item in data if isinstance(item, dict)}
            if len(values) != len(candidates):
                raise ValueError("RPC batch response has missing ids")
            return values
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(min(20.0, 1.5 * (2**attempt)))
    raise RuntimeError(last_error or "RPC request failed")


def runtime_row(candidate, bytecode, rpc_url):
    opcode = normalize_opcode_sequence(bytecode_to_opcode_sequence(bytecode))
    if not opcode:
        raise ValueError("runtime bytecode did not produce an opcode sequence")
    bytecode_hex = str(bytecode).lower().removeprefix("0x")
    row = dict(candidate)
    row.update(
        {
            "runtime_rpc_url": rpc_url,
            "runtime_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_bytecode_sha256": hashlib.sha256(bytecode_hex.encode("ascii")).hexdigest(),
            "runtime_byte_length": len(bytecode_hex) // 2,
            "opcode": opcode,
            "opcode_hash": hashlib.sha256(opcode.encode("utf-8")).hexdigest(),
            "opcode_token_count": len(opcode.split()),
        }
    )
    return row


def main():
    args = parse_args()
    candidates_path = resolve(args.candidates)
    output_path = resolve(args.output)
    rejected_path = resolve(args.rejected_output)
    candidates = load_jsonl(candidates_path)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    completed = {
        row.get("id") for row in load_jsonl(output_path) + load_jsonl(rejected_path)
    }
    pending = [row for row in candidates if row.get("id") not in completed]
    if not pending:
        print("[OK] no pending candidates")
        return

    session = requests.Session()
    total_ok = total_rejected = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        successful, rejected = [], []
        try:
            responses = rpc_batch(
                session, args.rpc_url, batch, args.timeout_seconds, args.retries
            )
            for index, candidate in enumerate(batch):
                response = responses[index]
                bytecode = response.get("result")
                if response.get("error") or not isinstance(bytecode, str) or bytecode in ("", "0x"):
                    rejected.append(
                        {
                            "id": candidate.get("id"),
                            "address": candidate.get("address"),
                            "reason": "rpc_error_or_empty_runtime",
                            "rpc_error": response.get("error"),
                        }
                    )
                    continue
                try:
                    successful.append(runtime_row(candidate, bytecode, args.rpc_url))
                except ValueError as exc:
                    rejected.append(
                        {
                            "id": candidate.get("id"),
                            "address": candidate.get("address"),
                            "reason": str(exc),
                        }
                    )
        except RuntimeError as exc:
            rejected.extend(
                {
                    "id": candidate.get("id"),
                    "address": candidate.get("address"),
                    "reason": "rpc_batch_failure",
                    "detail": str(exc),
                }
                for candidate in batch
            )
        append_jsonl(output_path, successful)
        append_jsonl(rejected_path, rejected)
        total_ok += len(successful)
        total_rejected += len(rejected)
        print(
            f"[INFO] {min(start + len(batch), len(pending))}/{len(pending)} "
            f"ok={total_ok} rejected={total_rejected}"
        )
        time.sleep(args.sleep_seconds)
    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
