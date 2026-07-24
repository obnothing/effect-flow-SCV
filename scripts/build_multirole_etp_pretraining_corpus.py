"""Build train-only MLM + multi-role ETP pretraining chunks.

The output deliberately contains no downstream vulnerability labels, templates,
patterns, or relations.  It is shared by the public Ethereum and Main-6 train
sources used by the ETP-only pretraining route.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from effect_flow_schema import EFFECT_TYPES, annotate_effect_types, build_token_units
from evm_tokenizer import EVMOpcodeTokenizer


FORBIDDEN_FIELDS = {
    "binary_label",
    "multi_labels",
    "labels",
    "vulnerability_labels",
    "efpp_pattern_labels",
    "effect_relations",
    "vulnerability_template_matches",
    "chunk_vulnerability_evidence",
}
CORPUS_FIELDS = {
    "id",
    "source_dataset",
    "source_split",
    "input_ids",
    "attention_mask",
    "effect_type_multihot",
    "etp_loss_mask",
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Build label-free multi-role ETP chunks.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=256)
    parser.add_argument("--max_chunks_per_contract", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def opcode_from_row(row):
    return str(
        row.get("opcode", row.get("opcodes", row.get("opcode_sequence", ""))) or ""
    )


def pad(values, size, value):
    return values + [value] * (size - len(values))


def make_chunk(tokenizer, dataset, contract_id, units, max_len):
    effects = annotate_effect_types(units)
    if any(sum(effect["multihot"]) > 2 for effect in effects):
        raise ValueError("Multi-role ETP invariant violated: more than two roles")
    zero_roles = [0] * len(EFFECT_TYPES)
    tokens = [tokenizer.cls_token] + [unit.token for unit in units] + [tokenizer.sep_token]
    payload = {
        "id": str(contract_id),
        "source_dataset": dataset,
        "source_split": "train",
        "input_ids": tokenizer.convert_tokens_to_ids(tokens),
        "attention_mask": [1] * len(tokens),
        "effect_type_multihot": [list(zero_roles)]
        + [effect["multihot"] for effect in effects]
        + [list(zero_roles)],
        "etp_loss_mask": [0] + [effect["loss_mask"] for effect in effects] + [0],
    }
    payload["input_ids"] = pad(payload["input_ids"], max_len, tokenizer.pad_token_id)
    payload["attention_mask"] = pad(payload["attention_mask"], max_len, 0)
    payload["effect_type_multihot"] = pad(
        payload["effect_type_multihot"], max_len, list(zero_roles)
    )
    payload["etp_loss_mask"] = pad(payload["etp_loss_mask"], max_len, 0)
    forbidden = FORBIDDEN_FIELDS.intersection(payload)
    if forbidden:
        raise AssertionError(f"Forbidden downstream fields in ETP corpus: {forbidden}")
    if set(payload) != CORPUS_FIELDS:
        raise AssertionError("Multi-role ETP corpus must contain only its approved fields")
    return payload


def main():
    args = parse_args()
    if args.max_len < 3 or args.chunk_stride <= 0 or args.max_chunks_per_contract <= 0:
        raise ValueError("max_len, chunk_stride, and max_chunks_per_contract must be valid")
    train_path = resolve(args.train_path)
    output_dir = resolve(args.output_dir)
    output_path = output_dir / "train_multirole_etp_chunks.jsonl"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; pass --force to rebuild")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(args.vocab_path))
    content_size = args.max_len - 2
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".jsonl.tmp")
    role_counts = Counter()
    contracts = chunks = empty = 0
    source_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    with train_path.open("r", encoding="utf-8") as source, temp_path.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(tqdm(source, desc=f"etp:{args.dataset}"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            contract_id = row.get("id", row.get("address", f"train_{line_number}"))
            units = build_token_units(opcode_from_row(row), tokenizer)
            contracts += 1
            if not units:
                empty += 1
                continue
            starts = list(range(0, len(units), args.chunk_stride))[: args.max_chunks_per_contract]
            for start in starts:
                payload = make_chunk(
                    tokenizer,
                    args.dataset,
                    contract_id,
                    units[start : start + content_size],
                    args.max_len,
                )
                for roles, active in zip(payload["effect_type_multihot"], payload["etp_loss_mask"]):
                    if active:
                        for role_id, value in enumerate(roles):
                            role_counts[role_id] += int(value)
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                chunks += 1
    temp_path.replace(output_path)
    report = {
        "dataset": args.dataset,
        "source_split": "train",
        "source_path": str(train_path),
        "source_sha256": source_hash,
        "output_path": str(output_path),
        "contracts": contracts,
        "chunks": chunks,
        "empty_contracts": empty,
        "max_len": args.max_len,
        "chunk_stride": args.chunk_stride,
        "max_chunks_per_contract": args.max_chunks_per_contract,
        "effect_type_names": EFFECT_TYPES,
        "effect_type_count": len(EFFECT_TYPES),
        "effect_role_counts": [role_counts[index] for index in range(len(EFFECT_TYPES))],
        "contains_downstream_labels": False,
    }
    report_path = output_dir / "multirole_etp_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {output_path}")
    print(f"[OK] wrote {report_path}")


if __name__ == "__main__":
    main()
