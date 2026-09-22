"""Audit process01 opcode/source alignment before any SPOR implementation."""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/DIVE_main6_opcode_process01"
RAW = ROOT / "DIVE_Raw_Data/Raw"
VOCAB = ROOT / "data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json"
REPORT = ROOT / "reports/spor_audit"
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
PUSH_RE = re.compile(r"^PUSH([0-9]+)$")
HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def normalize(value):
    return " ".join(str(value or "").split())


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def parse_runtime_tokens(opcode_text, known_mnemonics):
    tokens = normalize(opcode_text).split()
    instructions = []
    errors = []
    pc = 0
    index = 0
    while index < len(tokens):
        opcode = tokens[index].upper()
        match = PUSH_RE.match(opcode)
        immediate = None
        immediate_bytes = 0
        if match:
            width = int(match.group(1))
            if width == 0:
                if index + 1 < len(tokens) and HEX_RE.match(tokens[index + 1]):
                    errors.append({"kind": "PUSH0_unexpected_immediate", "token_index": index})
            elif 1 <= width <= 32:
                if index + 1 >= len(tokens) or not HEX_RE.match(tokens[index + 1]):
                    errors.append({"kind": "missing_push_immediate", "token_index": index, "opcode": opcode})
                else:
                    immediate = tokens[index + 1]
                    immediate_bytes = (len(immediate) - 2) // 2
                    if immediate_bytes != width:
                        errors.append({"kind": "push_immediate_width_mismatch", "token_index": index,
                                       "opcode": opcode, "expected": width, "actual": immediate_bytes})
                    index += 1
            else:
                errors.append({"kind": "invalid_push_width", "token_index": index, "opcode": opcode})
        elif opcode.startswith("0X") or HEX_RE.match(tokens[index]):
            errors.append({"kind": "orphan_immediate_or_hex_token", "token_index": index, "token": tokens[index]})
        elif opcode not in known_mnemonics:
            errors.append({"kind": "unknown_opcode_token", "token_index": index, "opcode": opcode})
        instructions.append({"instruction_index": len(instructions), "program_counter": pc,
                             "opcode": opcode, "immediate_operand": immediate,
                             "source_token_index": index})
        pc += 1 + immediate_bytes
        index += 1
    return instructions, errors, len(tokens)


def tokenizer_alignment(opcode_text, tokenizer):
    raw = normalize(opcode_text).split()
    model_tokens = tokenizer.tokenize(opcode_text, add_special_tokens=True)
    positions = []
    raw_index = 0
    model_index = 1  # [CLS] occupies position zero.
    while raw_index < len(raw):
        opcode = raw[raw_index].upper()
        item = {"opcode": opcode, "instruction_model_token_index": model_index}
        model_index += 1
        raw_index += 1
        if re.match(r"^PUSH(?:0|[1-9]|[12][0-9]|3[0-2])$", opcode) and raw_index < len(raw) and HEX_RE.match(raw[raw_index]):
            item["immediate_model_token_index"] = model_index
            model_index += 1
            raw_index += 1
        positions.append(item)
    expected_without_special = model_tokens[1:-1]
    return model_tokens, positions, len(expected_without_special) == model_index - 1


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from evm_opcode import OPCODES
    from evm_tokenizer import EVMOpcodeTokenizer

    tokenizer = EVMOpcodeTokenizer.from_vocab_file(VOCAB)
    raw_runtime = {str(row["contractID"]): normalize(row.get("Opcodes", "")) for _, row in read_jsonl(RAW / "POST/Runtime_Opcode.jsonl")}
    compiler = {}
    with (RAW / "PRE/Code-based.csv").open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            compiler[str(row["contractID"])] = {"CompilerVersion": row.get("CompilerVersion", ""), "EVMVersion": row.get("EVMVersion", "")}
    source_ids = {path.stem for path in (RAW / "PRE/Source codes").glob("*.sol")}
    split_rows = {}
    label_widths = Counter()
    id_alignment = Counter()
    opcode_mismatches = []
    parser_error_counts = Counter()
    parser_error_examples = []
    tokenizer_alignment_failures = []
    instruction_counts = []
    for split in ("train", "valid", "test"):
        rows = []
        for line_no, row in read_jsonl(DATA / f"{split}.jsonl"):
            rows.append(row)
            item_id = str(row["id"]).split(":")[-1]
            label_widths[len(row.get("multi_labels", []))] += 1
            if item_id not in raw_runtime:
                id_alignment["missing_runtime"] += 1
                continue
            id_alignment["runtime_match"] += 1
            if normalize(row.get("opcode", "")) != raw_runtime[item_id]:
                opcode_mismatches.append({"split": split, "line": line_no, "id": row["id"]})
            instructions, errors, token_count = parse_runtime_tokens(raw_runtime[item_id], set(OPCODES.values()))
            instruction_counts.append(len(instructions))
            for error in errors:
                parser_error_counts[error["kind"]] += 1
                if len(parser_error_examples) < 20:
                    parser_error_examples.append({"id": row["id"], **error})
            _, _, aligned = tokenizer_alignment(raw_runtime[item_id], tokenizer)
            if not aligned and len(tokenizer_alignment_failures) < 20:
                tokenizer_alignment_failures.append({"id": row["id"], "split": split})
        split_rows[split] = len(rows)
    report = {
        "dataset": str(DATA.relative_to(ROOT)), "raw_runtime": str((RAW / "POST/Runtime_Opcode.jsonl").relative_to(ROOT)),
        "raw_source_dir": str((RAW / "PRE/Source codes").relative_to(ROOT)),
        "raw_runtime_rows": len(raw_runtime), "source_solidity_files": len(source_ids),
        "code_based_rows_with_compiler_metadata": len(compiler), "compiler_versions": Counter(item["CompilerVersion"] for item in compiler.values()).most_common(20),
        "split_rows": split_rows, "label_width_counts": dict(label_widths), "expected_project_labels": LABELS,
        "runtime_id_alignment": dict(id_alignment), "opcode_exact_normalized_mismatch_count": len(opcode_mismatches),
        "opcode_mismatch_examples": opcode_mismatches[:20], "parser_error_counts": dict(parser_error_counts),
        "parser_error_examples": parser_error_examples, "tokenizer_alignment_failure_examples": tokenizer_alignment_failures,
        "instruction_count_summary": {"min": min(instruction_counts), "max": max(instruction_counts), "mean": sum(instruction_counts) / len(instruction_counts)},
        "runtime_bytecode_available": False,
        "runtime_opcode_mnemonic_sequence_available": True,
        "program_counter_independently_verifiable": False,
        "push_immediate_recoverable_from_raw_opcode_text": not any(key.startswith("missing_push") or key.startswith("push_immediate_width") for key in parser_error_counts),
        "tokenizer_push_operand_behavior": "PUSH mnemonic and a normalized operand token are emitted; operand is not treated as an independent EVM instruction.",
        "source_and_opcode_same_contract_id": id_alignment["runtime_match"] == sum(split_rows.values()),
        "complete_eight_label_training_data": set(label_widths) == {8},
        "test_checked": False,
        "conclusion": "The process01 data is six-label and the Raw directory contains disassembled runtime opcode text rather than original runtime bytecode. Local instruction boundaries and PUSH widths are recoverable from the trusted mnemonic sequence, but program counters cannot be independently verified against bytes. A full SPOR experiment requires an explicit decision to use this bounded text-level reconstruction and a separate eight-label dataset if eight labels are required.",
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "spor_input_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# SPOR Phase 1 Input Audit", "",
        f"- Dataset: `{report['dataset']}`; splits: `{report['split_rows']}`.",
        f"- Runtime opcode rows: `{report['raw_runtime_rows']}`; source files: `{report['source_solidity_files']}`; compiler metadata rows: `{report['code_based_rows_with_compiler_metadata']}`.",
        f"- Runtime/source ID alignment across process01: `{report['source_and_opcode_same_contract_id']}`.",
        f"- Normalized opcode mismatches: `{report['opcode_exact_normalized_mismatch_count']}`.",
        f"- Label widths in process01: `{report['label_width_counts']}`; complete eight-label data: `{report['complete_eight_label_training_data']}`.",
        f"- Original runtime bytecode available: `{report['runtime_bytecode_available']}`.",
        f"- Trusted disassembled opcode sequence available: `{report['runtime_opcode_mnemonic_sequence_available']}`.",
        f"- Program counter independently verifiable against bytecode: `{report['program_counter_independently_verifiable']}`.",
        f"- Tokenizer alignment failures in audit sample: `{len(report['tokenizer_alignment_failure_examples'])}` examples.",
        "",
        "## Tokenizer finding",
        "The tokenizer recognizes PUSH0 and PUSH1-PUSH32, consumes the following hex operand as the PUSH immediate, and emits a normalized operand token. That operand token is a model token but is not an independently executed EVM instruction.",
        "",
        "## SPOR boundary",
        "Basic-block-local instruction order, PUSH width, instruction index and derived local program counters can be reconstructed from the trusted mnemonic/operand text. The original runtime bytes and independently verifiable program counters are absent, so the result would be a bounded text-level reconstruction rather than byte-verified or complete EVM execution semantics.",
        "",
        "## Label boundary",
        "process01 contains six labels, not the eight-label DIVE target. The eight-label DIVE labels exist separately in DIVE_Labels.csv, but they are not the labels currently attached to process01. The SPOR model experiment must keep the chosen dataset label definition explicit.",
        "",
        "Test data was audited only for schema/alignment and remains locked for model selection: `test_checked=false`.",
    ]
    (REPORT / "spor_input_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
