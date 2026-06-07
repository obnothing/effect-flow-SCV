import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evm_tokenizer import (  # noqa: E402
    COMMON_OPERANDS,
    MNEMONIC_TOKENS,
    NORMALIZED_OPERAND_TOKENS,
    PUSH_PATTERN,
    SPECIAL_TOKENS,
    hex_byte_length,
    normalize_operand,
    split_opcode_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build EVM opcode-aware vocab.")
    parser.add_argument(
        "--config",
        default="configs/evm_tokenizer.yaml",
        help="Path to EVM tokenizer YAML config.",
    )
    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path):
    with resolve_project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            opcode = item.get("opcode")
            if opcode is None:
                raise ValueError(f"{path}:{line_no} missing opcode")
            yield opcode


def update_counts(opcode_sequence, mnemonic_counter, operand_counter, operand_lengths):
    tokens = split_opcode_sequence(opcode_sequence)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        upper_token = token.upper()
        if upper_token in MNEMONIC_TOKENS or PUSH_PATTERN.match(upper_token):
            mnemonic_counter[upper_token] += 1
        if PUSH_PATTERN.match(upper_token) and index + 1 < len(tokens):
            operand = tokens[index + 1].strip().lower()
            if operand.startswith("0x"):
                operand_counter[operand] += 1
                byte_len = hex_byte_length(operand)
                operand_lengths[str(byte_len) if byte_len is not None else "invalid"] += 1
                index += 2
                continue
        index += 1


def select_preserved_operands(operand_counter, config):
    top_k = int(config.get("top_k_operands", 1000))
    min_freq = int(config.get("min_operand_freq", 5))
    preserved = set()
    if config.get("preserve_common_operands", True):
        preserved.update(COMMON_OPERANDS)
    for operand, count in operand_counter.most_common(top_k):
        if count >= min_freq:
            preserved.add(operand)
    return sorted(preserved)


def build_vocab(preserved_operands):
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    tokens.extend(MNEMONIC_TOKENS)
    tokens.extend(sorted(preserved_operands))
    tokens.extend(NORMALIZED_OPERAND_TOKENS)
    token_to_id = {}
    for token in tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    return token_to_id


def classify_operand_types(operand_counter, preserved_operands, config):
    preserved_set = set(preserved_operands)
    type_counts = Counter()
    examples = {}
    for operand, count in operand_counter.items():
        normalized = normalize_operand(
            operand,
            preserved_operands=preserved_set,
            preserve_common_operands=config.get("preserve_common_operands", True),
            normalize_rare_long_operands=config.get(
                "normalize_rare_long_operands", True
            ),
        )
        if normalized != operand:
            type_counts[normalized] += count
            examples.setdefault(normalized, operand)
    return type_counts, examples


def write_vocab(path, token_to_id, preserved_operands, config):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_to_id": token_to_id,
        "id_to_token": {str(idx): token for token, idx in token_to_id.items()},
        "special_tokens": SPECIAL_TOKENS,
        "preserved_operands": preserved_operands,
        "preserve_common_operands": config.get("preserve_common_operands", True),
        "normalize_rare_long_operands": config.get(
            "normalize_rare_long_operands", True
        ),
        "add_jump_aware_tokens": False,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_reports(report, txt_path, json_path):
    txt_path = resolve_project_path(txt_path)
    json_path = resolve_project_path(json_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "EVM opcode-aware vocab report",
        "",
        f"vocab_size: {report['vocab_size']}",
        f"mnemonic_count: {report['mnemonic_count']}",
        f"operand_count: {report['operand_count']}",
        f"preserved_operand_count: {report['preserved_operand_count']}",
        f"normalized_operand_count: {report['normalized_operand_count']}",
        "add_jump_aware_tokens: false",
        "",
        "Preserved common operands:",
    ]
    lines.extend(f"- {operand}" for operand in report["preserved_common_operands"])
    lines.append("")
    lines.append("Top 50 operands:")
    for operand, count in report["top_50_operands"]:
        lines.append(f"- {operand}: {count}")
    lines.append("")
    lines.append("Examples of normalization:")
    for token, example in report["examples_of_normalization"].items():
        lines.append(f"- {example} -> {token}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = [
        resolve_project_path(config["train_path"]),
        resolve_project_path(config["valid_path"]),
        resolve_project_path(config["test_path"]),
    ]
    mnemonic_counter = Counter()
    operand_counter = Counter()
    operand_lengths = Counter()
    for path in paths:
        print(f"[SCAN] {path}")
        for opcode in iter_jsonl(path):
            update_counts(opcode, mnemonic_counter, operand_counter, operand_lengths)

    preserved_operands = select_preserved_operands(operand_counter, config)
    token_to_id = build_vocab(preserved_operands)
    type_counts, normalization_examples = classify_operand_types(
        operand_counter, preserved_operands, config
    )
    write_vocab(config["vocab_path"], token_to_id, preserved_operands, config)

    report = {
        "vocab_size": len(token_to_id),
        "mnemonic_count": len(mnemonic_counter),
        "operand_count": len(operand_counter),
        "preserved_operand_count": len(preserved_operands),
        "normalized_operand_count": int(sum(type_counts.values())),
        "operand_length_distribution": dict(operand_lengths),
        "normalized_operand_type_counts": dict(type_counts),
        "top_50_operands": operand_counter.most_common(50),
        "preserved_common_operands": sorted(COMMON_OPERANDS),
        "preserved_operands": preserved_operands,
        "examples_of_normalization": normalization_examples,
        "add_jump_aware_tokens": False,
    }
    write_reports(
        report,
        "data/reports/evm_vocab_report.txt",
        "data/reports/evm_vocab_report.json",
    )
    print(f"[OK] wrote {config['vocab_path']}")
    print("[OK] wrote data/reports/evm_vocab_report.txt")
    print("[OK] wrote data/reports/evm_vocab_report.json")


if __name__ == "__main__":
    main()
