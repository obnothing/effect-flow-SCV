"""Audit SmartBugs Wild contracts as tool-screened Main-6 clean candidates.

This is a negative pre-filter only.  A contract is emitted only when every
required SmartBugs tool has a result entry and none of the six DIVE Main-6
categories was reported by any available tool.  It does not establish that a
contract is vulnerability-free and it does not produce runtime opcode.
"""

import argparse
import csv
import json
import tarfile
from io import TextIOWrapper
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN6_CATEGORIES = (
    "reentrancy",
    "access_control",
    "arithmetic",
    "unchecked_low_calls",
    "denial_service",
    "time_manipulation",
)
DEFAULT_REQUIRED_TOOLS = (
    "mythril",
    "slither",
    "oyente",
    "osiris",
    "smartcheck",
    "securify",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a strict multi-tool-negative SmartBugs Wild candidate manifest."
    )
    parser.add_argument("--raw-dir", default="data/raw/smartbugs_wild")
    parser.add_argument("--results", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--contracts-dir", default=None)
    parser.add_argument(
        "--required-tools",
        nargs="+",
        default=list(DEFAULT_REQUIRED_TOOLS),
        help="Tools that must have a result entry for a candidate.",
    )
    parser.add_argument(
        "--candidate-output", default=None, help="JSONL candidate manifest output."
    )
    parser.add_argument("--report", default="data/reports/smartbugs_wild_clean_candidate_audit.txt")
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_contract_metadata(path):
    metadata = {}
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember("contracts.csv")
        raw_handle = archive.extractfile(member)
        if raw_handle is None:
            raise ValueError(f"{path} does not contain contracts.csv")
        with TextIOWrapper(raw_handle, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                address = (row.get("address") or "").lower()
                if address:
                    metadata[address] = row
    return metadata


def category_hits(tool_results):
    hits = set()
    for result in tool_results.values():
        if not isinstance(result, dict):
            continue
        categories = result.get("categories") or {}
        hits.update(category for category, count in categories.items() if count)
    return hits.intersection(MAIN6_CATEGORIES)


def main():
    args = parse_args()
    raw_dir = resolve(args.raw_dir)
    results_path = resolve(args.results) if args.results else raw_dir / "results_wild.json"
    manifest_path = (
        resolve(args.manifest) if args.manifest else raw_dir / "contracts.csv.tar.gz"
    )
    contracts_dir = (
        resolve(args.contracts_dir) if args.contracts_dir else raw_dir / "contracts"
    )
    candidate_output = (
        resolve(args.candidate_output)
        if args.candidate_output
        else raw_dir / "tool_screened_main6_clean_candidates.jsonl"
    )
    report_path = resolve(args.report)
    required_tools = tuple(args.required_tools)

    for path in (results_path, manifest_path, contracts_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    metadata = load_contract_metadata(manifest_path)

    candidates = []
    rejected = Counter()
    hit_counts = Counter()
    compiler_counts = Counter()
    source_missing = 0
    for address, record in results.items():
        address = address.lower()
        tool_results = record.get("tools") or {}
        missing_tools = sorted(set(required_tools).difference(tool_results))
        if missing_tools:
            rejected["missing_required_tool"] += 1
            continue
        hits = sorted(category_hits(tool_results))
        if hits:
            rejected["main6_category_hit"] += 1
            hit_counts.update(hits)
            continue
        source_path = contracts_dir / f"{address}.sol"
        if not source_path.is_file():
            source_missing += 1
            rejected["missing_source_file"] += 1
            continue
        meta = metadata.get(address, {})
        compiler = meta.get("compiler_version") or "unknown"
        compiler_counts[compiler] += 1
        candidates.append(
            {
                "id": f"smartbugs_wild:{address}",
                "address": address,
                "source": "SmartBugs Wild (smartbugs/smartbugs-wild, commit 91ec1757)",
                "source_path": str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "compiler_version": compiler,
                "creation_date": meta.get("creation_date"),
                "last_transaction_date": meta.get("last_transaction_date"),
                "nb_transaction": meta.get("nb_transaction"),
                "source_md5": meta.get("md5"),
                "required_tools": list(required_tools),
                "main6_tool_screen": "no_reported_hits",
                "main6_categories_checked": list(MAIN6_CATEGORIES),
                "reported_main6_categories": [],
            }
        )

    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with candidate_output.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "SmartBugs Wild Main-6 tool-screened clean candidate audit",
        f"results: {display_path(results_path)}",
        f"manifest: {display_path(manifest_path)}",
        f"contracts_with_results: {len(results)}",
        f"required_tools: {', '.join(required_tools)}",
        f"main6_categories: {', '.join(MAIN6_CATEGORIES)}",
        f"candidate_count: {len(candidates)}",
        f"candidate_manifest: {display_path(candidate_output)}",
        f"missing_source_files: {source_missing}",
        "",
        "Rejection counts:",
    ]
    lines.extend(f"{key}: {value}" for key, value in sorted(rejected.items()))
    lines.append("")
    lines.append("Main-6 category hits among rejected contracts:")
    lines.extend(f"{key}: {hit_counts[key]}" for key in MAIN6_CATEGORIES)
    lines.append("")
    lines.append("Candidate compiler-version distribution:")
    lines.extend(
        f"{compiler}: {count}"
        for compiler, count in compiler_counts.most_common()
    )
    lines.append("")
    lines.append(
        "Caveat: this manifest is a multi-tool negative pre-filter, not a proof of safety."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
