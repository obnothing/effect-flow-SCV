"""Label, compiler, and length-stratified availability from Phase 2.5 caches."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer

DATA = ROOT / "data/processed/DIVE_8_opcode_random_split"
OUT = ROOT / "results/spor_phase25"
OPCODES = ("JUMPI", "CALL", "SLOAD", "SSTORE")


def bucket(length):
    return "1-2048" if length <= 2048 else "2049-4096" if length <= 4096 else "4097-8192" if length <= 8192 else "8193+"


def main():
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / "data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json")
    target_ids = {op: tokenizer.vocab[op] for op in OPCODES}
    with (ROOT / "DIVE_Raw_Data/Raw/PRE/Code-based.csv").open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        compilers = {str(row["contractID"]): {"compiler": row.get("CompilerVersion") or "unknown",
                                               "evm_version": row.get("EVMVersion") or "unknown"} for row in csv.DictReader(handle)}
    labels = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["label_names"]
    report = {"test_checked": False, "feature_names": ["valid_ratio", "unknown_ratio"] + [f"{op}_known_ratio" for op in OPCODES], "groups": {}}
    for split in ("train", "valid"):
        cache = torch.load(ROOT / "data/features/spor_dive8_phase25" / f"{split}.pt", map_location="cpu", mmap=True, weights_only=False)
        rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").open(encoding="utf-8") if line.strip()]
        if [row["id"] for row in rows] != cache["ids"]:
            raise ValueError(f"ID order mismatch: {split}")
        groups = defaultdict(lambda: {"count": 0, "sum": [0.0] * 6})
        for index, row in enumerate(rows):
            left, right = int(cache["offsets"][index]), int(cache["offsets"][index + 1])
            valid = cache["valid_mask"][left:right]
            unknown = cache["provenance"][left:right, -1]
            tokens = cache["token_ids"][left:right]
            values = [float(valid.float().mean()), float(unknown.float().mean())]
            for op in OPCODES:
                mask = tokens == target_ids[op]
                values.append(float(valid[mask].float().mean()) if bool(mask.any()) else 0.0)
            version = compilers.get(str(row["contract_id"]), {"compiler": "unknown", "evm_version": "unknown"})
            length = bucket(right - left)
            for label_index, label in enumerate(labels):
                polarity = "positive" if row["multi_labels"][label_index] else "negative"
                for dimension, group in (("all", "all"), ("compiler", version["compiler"]),
                    ("evm_version", version["evm_version"]), ("length", length),
                    ("compiler_length", f"{version['compiler']}|{length}"),
                    ("evm_version_length", f"{version['evm_version']}|{length}")):
                    record = groups[(dimension, group, label, polarity)]
                    record["count"] += 1
                    for n, value in enumerate(values):
                        record["sum"][n] += value
            if (index + 1) % 1000 == 0:
                print(f"[groups:{split}] {index+1}", flush=True)
        report["groups"][split] = [dict(dimension=dimension, group=group, label=label, polarity=polarity,
            count=value["count"], mean=[number/value["count"] for number in value["sum"]])
            for (dimension, group, label, polarity), value in groups.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage_by_label_compiler_length.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": {split: len(group) for split, group in report["groups"].items()}, "test_checked": False}))


if __name__ == "__main__":
    main()
