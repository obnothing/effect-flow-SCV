"""Freeze and audit the Main-6 random-split train-only pretraining protocol."""

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def digest_bytes(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opcode_hash(opcode):
    normalized = " ".join(str(opcode).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_split(path):
    ids = []
    hashes = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row["id"]))
            hashes.add(opcode_hash(row.get("opcode", "")))
    return {"file_sha256": digest_bytes(path), "ids": ids, "hashes": hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/DIVE_main6_random_split")
    parser.add_argument(
        "--public_corpus",
        default="data/processed/ethereum_public_pretrain_19143_unique_runtime/runtime_opcode.jsonl",
    )
    parser.add_argument("--template_path", default="configs/vulnerability_templates_dive_main6_new.yaml")
    parser.add_argument("--output", default="data/reports/main6_random_train_pretrain_protocol.json")
    args = parser.parse_args()

    data_dir = resolve(args.data_dir)
    splits = {name: read_split(data_dir / f"{name}.jsonl") for name in ("train", "valid", "test")}
    overlap = {
        "train_valid": len(splits["train"]["hashes"] & splits["valid"]["hashes"]),
        "train_test": len(splits["train"]["hashes"] & splits["test"]["hashes"]),
        "valid_test": len(splits["valid"]["hashes"] & splits["test"]["hashes"]),
    }
    public_hashes = set()
    with resolve(args.public_corpus).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                public_hashes.add(opcode_hash(row.get("opcode", "")))
    public_overlap = {
        split: len(public_hashes & payload["hashes"])
        for split, payload in splits.items()
    }
    report = {
        "status": "ok" if not any(public_overlap.values()) else "error",
        "policy": "Main-6 random train only; no train-label retrieval at inference",
        "data_dir": str(data_dir.relative_to(PROJECT_ROOT)),
        "split_files": {
            name: {
                "samples": len(payload["ids"]),
                "unique_ids": len(set(payload["ids"])),
                "unique_opcode_hashes": len(payload["hashes"]),
                "file_sha256": payload["file_sha256"],
            }
            for name, payload in splits.items()
        },
        "random_split_opcode_hash_overlap": overlap,
        "public_corpus_exact_hash_overlap": public_overlap,
        "template_path": str(resolve(args.template_path).relative_to(PROJECT_ROOT)),
        "template_sha256": digest_bytes(resolve(args.template_path)),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit("Public pretraining corpus overlaps a Main-6 random split")


if __name__ == "__main__":
    main()
