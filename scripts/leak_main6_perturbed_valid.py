"""Create a controlled train/valid contamination experiment.

The source split is not overwritten. A sampled set of valid rows is copied to
train after a one-character opcode mutation, while valid remains unchanged.
The report explicitly records the resulting ID overlap: opcode-hash overlap
can be zero even though the same labeled records occur in both splits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path


SPLITS = ("train", "valid", "test")
HEX_OPERAND = re.compile(r"0x([0-9a-fA-F])")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def opcode_hash(opcode: object) -> str:
    normalized = " ".join(str(opcode or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mutate_at(opcode: str, position: int):
    old = opcode[position]
    if old in "0123456789abcdefABCDEF":
        digits = "0123456789ABCDEF" if old.isupper() else "0123456789abcdef"
        new = digits[(digits.index(old) + 1) % len(digits)]
    else:
        new = "X" if old != "X" else "Y"
    return opcode[:position] + new + opcode[position + 1 :], old, new


def mutate_unique(opcode: str, forbidden_hashes: set[str]):
    candidates = []
    for match in HEX_OPERAND.finditer(opcode):
        candidates.append(match.start(1))
    candidates.extend(index for index, char in enumerate(opcode) if char not in " \t\r\n")
    seen = set()
    for position in candidates:
        if position in seen:
            continue
        seen.add(position)
        mutated, old, new = mutate_at(opcode, position)
        if mutated != opcode and opcode_hash(mutated) not in forbidden_hashes:
            return mutated, {"position": position, "old": old, "new": new}
    raise ValueError("could not construct a hash-disjoint one-character mutation")


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed/DIVE_main6_opcode_deduplicated")
    parser.add_argument("--output-dir", default="data/processed/DIVE_main6_opcode_deduplicated_leak_500")
    parser.add_argument("--report", default="data/reports/main6_opcode_leak_500_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir, output_dir, report_path = resolve(args.input_dir), resolve(args.output_dir), resolve(args.report)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {split: read_rows(input_dir / f"{split}.jsonl") for split in SPLITS}
    if len(rows["train"]) < args.count or len(rows["valid"]) < args.count:
        raise ValueError("train and valid must each contain at least --count rows")

    rng = random.Random(args.seed)
    remove_train = set(rng.sample(range(len(rows["train"])), args.count))
    move_valid = set(rng.sample(range(len(rows["valid"])), args.count))
    input_hashes = {opcode_hash(row.get("opcode", "")) for split in SPLITS for row in rows[split]}
    forbidden = set(input_hashes)
    moved, mutations = [], []
    for index in sorted(move_valid):
        source = rows["valid"][index]
        item = copy.deepcopy(source)
        original_opcode = str(source.get("opcode", ""))
        item["opcode"], change = mutate_unique(original_opcode, forbidden)
        if item.get("id") != source.get("id") or item.get("multi_labels") != source.get("multi_labels"):
            raise AssertionError("id or labels changed")
        forbidden.add(opcode_hash(item["opcode"]))
        moved.append(item)
        mutations.append({"id": str(source["id"]), "valid_index": index, "original_hash": opcode_hash(original_opcode), "mutated_hash": opcode_hash(item["opcode"]), **change})

    output = {
        "train": [row for index, row in enumerate(rows["train"]) if index not in remove_train] + moved,
        "valid": rows["valid"],
        "test": rows["test"],
    }
    for split in SPLITS:
        write_rows(output_dir / f"{split}.jsonl", output[split])

    hashes = {split: {opcode_hash(row.get("opcode", "")) for row in output[split]} for split in SPLITS}
    ids = {split: {str(row["id"]) for row in output[split]} for split in SPLITS}
    report = {
        "route": "DIVE Main6 controlled perturbed-valid leakage experiment",
        "input_dir": str(input_dir), "output_dir": str(output_dir), "seed": args.seed,
        "count": args.count,
        "rule": "remove count train rows; copy count valid rows into train after exactly one opcode-character mutation; keep valid unchanged",
        "input_samples": {split: len(rows[split]) for split in SPLITS},
        "output_samples": {split: len(output[split]) for split in SPLITS},
        "train_removed_count": len(remove_train), "valid_copied_count": len(move_valid),
        "valid_unchanged": output["valid"] == rows["valid"], "test_unchanged": output["test"] == rows["test"],
        "output_id_overlap": {f"{left}_{right}": len(ids[left] & ids[right]) for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))},
        "output_opcode_hash_overlap": {f"{left}_{right}": len(hashes[left] & hashes[right]) for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))},
        "all_mutated_hashes_disjoint_from_input": all(item["mutated_hash"] not in input_hashes for item in mutations),
        "mutations": mutations,
        "warnings": [
            "This is intentionally contaminated data for a controlled experiment, not a valid benchmark split.",
            "The same 500 IDs and labels occur in train and valid; zero opcode-hash overlap does not remove this leakage.",
            "The source dataset was not overwritten.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "mutations"}, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {output_dir}")
    print(f"[OK] wrote {report_path}")


if __name__ == "__main__":
    main()
