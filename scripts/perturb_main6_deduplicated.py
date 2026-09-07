"""Build a controlled Main-6 split with 300 perturbed valid rows moved to train.

The source directory is never overwritten. Each moved validation record keeps
its id and labels, while exactly one opcode character is changed. The default
mutation changes the first hexadecimal digit following an ``0x`` operand so
the opcode tokenization and sequence length remain unchanged.
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


def digest_opcode(opcode: object) -> str:
    normalized = " ".join(str(opcode or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mutate_one_opcode_character(opcode: str):
    match = HEX_OPERAND.search(opcode)
    if match:
        position = match.start(1)
        old = opcode[position]
        digits = "0123456789ABCDEF" if old.isupper() else "0123456789abcdef"
        new = digits[(digits.index(old) + 1) % len(digits)]
    elif opcode:
        position = 0
        old = opcode[position]
        new = "X" if old != "X" else "Y"
    else:
        raise ValueError("cannot mutate an empty opcode")
    mutated = opcode[:position] + new + opcode[position + 1 :]
    if len(mutated) != len(opcode) or sum(left != right for left, right in zip(opcode, mutated)) != 1:
        raise AssertionError("opcode mutation must change exactly one character")
    return mutated, {"position": position, "old": old, "new": new}


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed/DIVE_main6_opcode_deduplicated")
    parser.add_argument("--output-dir", default="data/processed/DIVE_main6_opcode_deduplicated_perturbed")
    parser.add_argument("--report", default="data/reports/main6_opcode_perturbation_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.report)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; use --overwrite to replace it: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = {split: read_rows(input_dir / f"{split}.jsonl") for split in SPLITS}
    if len(rows["train"]) < args.count or len(rows["valid"]) < args.count:
        raise ValueError("train and valid must each contain at least --count rows")

    rng = random.Random(args.seed)
    train_remove = set(rng.sample(range(len(rows["train"])), args.count))
    valid_move = set(rng.sample(range(len(rows["valid"])), args.count))

    moved = []
    mutations = []
    for index in sorted(valid_move):
        original = rows["valid"][index]
        item = copy.deepcopy(original)
        old_opcode = str(item.get("opcode", ""))
        new_opcode, change = mutate_one_opcode_character(old_opcode)
        item["opcode"] = new_opcode
        if item.get("id") != original.get("id") or item.get("multi_labels") != original.get("multi_labels"):
            raise AssertionError("id or labels changed during perturbation")
        moved.append(item)
        mutations.append({
            "id": str(original["id"]),
            "valid_original_index": index,
            "opcode_char": change,
            "original_hash": digest_opcode(old_opcode),
            "mutated_hash": digest_opcode(new_opcode),
        })

    output_rows = {
        "train": [row for index, row in enumerate(rows["train"]) if index not in train_remove] + moved,
        "valid": [row for index, row in enumerate(rows["valid"]) if index not in valid_move],
        "test": rows["test"],
    }
    for split in SPLITS:
        write_rows(output_dir / f"{split}.jsonl", output_rows[split])

    def hash_sets(split):
        return {digest_opcode(row.get("opcode", "")) for row in output_rows[split]}

    output_hashes = {split: hash_sets(split) for split in SPLITS}
    overlap = {
        f"{left}_{right}": len(output_hashes[left] & output_hashes[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    report = {
        "route": "DIVE Main6 opcode perturbation experiment",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "moved_count": args.count,
        "train_removed_count": len(train_remove),
        "valid_removed_count": len(valid_move),
        "mutation_rule": "change exactly one character: first hexadecimal digit after the first 0x operand; fallback to first character only if no operand exists",
        "input_samples": {split: len(rows[split]) for split in SPLITS},
        "output_samples": {split: len(output_rows[split]) for split in SPLITS},
        "train_removed_ids": [str(rows["train"][index]["id"]) for index in sorted(train_remove)],
        "moved_valid_ids": [item["id"] for item in mutations],
        "moved_rows_preserve_id_and_labels": all(
            output_rows["train"][-args.count + offset].get("id") == item["id"]
            for offset, item in enumerate(moved)
        ),
        "test_copied_without_modification": output_rows["test"] == rows["test"],
        "output_unique_hashes": {split: len(output_hashes[split]) for split in SPLITS},
        "output_opcode_hash_overlap": overlap,
        "mutations": mutations,
        "warnings": [
            "This is a new experimental copy; the deduplicated source directory was not overwritten.",
            "The moved validation records retain their original ids and labels but have one-character opcode perturbations.",
            "The copied test split is unchanged, but this transformed dataset must not be treated as the original official benchmark without a new protocol audit.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"mutations", "train_removed_ids"}}, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {output_dir}")
    print(f"[OK] wrote {report_path}")


if __name__ == "__main__":
    main()
