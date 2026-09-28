"""Generate validation/train SPOR token-aligned feature caches for DIVE_8."""

import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/DIVE_8_opcode_random_split"
RAW_RUNTIME = ROOT / "DIVE_Raw_Data/Raw/POST/Runtime_Opcode.jsonl"
VOCAB = ROOT / "data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json"
OUT = ROOT / "data/features/spor_dive8"
REPORT = ROOT / "reports/spor_phase2"
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from spor.evm_parser import parse_disassembled_opcode  # noqa: E402
from spor.stack_provenance import SOURCE_CATEGORIES, analyze_basic_blocks  # noqa: E402


ROLE_NAMES = ["OTHER", "CONSTANT", "STACK_POP", "STACK_DUP", "STACK_SWAP", "ARITHMETIC", "COMPARISON",
              "CALLER", "CALLVALUE", "CALLDATA", "BLOCK_ENV", "MEMORY", "STORAGE", "CALL", "CONTROL_FLOW", "PUSH_IMMEDIATE"]
STATUS_NAMES = ["known", "unknown_entry", "analysis_failure", "analysis_failure_parse_status", "operand_alias", "not_applicable"]
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}
STATUS_TO_ID = {name: index for index, name in enumerate(STATUS_NAMES)}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_runtime():
    result = {}
    with RAW_RUNTIME.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line); result[str(row["contractID"])] = row["Opcodes"]
    return result


def load_compilers():
    path = ROOT / "DIVE_Raw_Data/Raw/PRE/Code-based.csv"
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return {str(row["contractID"]): row.get("CompilerVersion", "unknown") or "unknown"
                for row in csv.DictReader(handle)}


def token_features(instructions, token_count):
    provenance = np.zeros((token_count, len(SOURCE_CATEGORIES)), dtype=np.uint8)
    roles = np.zeros(token_count, dtype=np.int16)
    valid = np.zeros(token_count, dtype=np.bool_)
    statuses = np.full(token_count, STATUS_TO_ID["not_applicable"], dtype=np.int8)
    instruction_index = np.full(token_count, -1, dtype=np.int32)
    derived_pc = np.full(token_count, -1, dtype=np.int64)
    for instruction in instructions:
        bits = int(instruction.provenance_bits)
        status = instruction.analysis_status if instruction.analysis_status in STATUS_TO_ID else "analysis_failure"
        if status == "known" and bits == 0:
            status = "not_applicable"
        role_id = ROLE_TO_ID.get(instruction.operation_role, ROLE_TO_ID["OTHER"])
        for offset, token_index in enumerate(instruction.model_token_indices):
            if not 0 <= token_index < token_count:
                raise ValueError(f"invalid model token index: {token_index}/{token_count}")
            value_bits = bits
            for bit_index in range(len(SOURCE_CATEGORIES)):
                provenance[token_index, bit_index] = (value_bits >> bit_index) & 1
            roles[token_index] = role_id if offset == 0 else ROLE_TO_ID["PUSH_IMMEDIATE"]
            valid[token_index] = status == "known" and offset == 0 and bits != 0
            statuses[token_index] = STATUS_TO_ID[status] if offset == 0 else STATUS_TO_ID["operand_alias"]
            instruction_index[token_index] = instruction.instruction_index
            derived_pc[token_index] = instruction.derived_pc if instruction.derived_pc is not None else -1
    return provenance, roles, valid, statuses, instruction_index, derived_pc


def build_split(split, runtime, compilers, tokenizer, report):
    started = time.perf_counter()
    parts = {"provenance": [], "roles": [], "valid": [], "status": [], "instruction_index": [], "derived_pc": [], "token_ids": [], "offsets": [0], "ids": [], "original_lengths": []}
    counters = Counter(); instruction_counts = []; token_counts = []; block_counts = []; durations = []
    by_compiler = {}; failures = []; target_status = {op: Counter() for op in ("JUMPI", "CALL", "SLOAD", "SSTORE")}
    for row_index, row in enumerate(read_jsonl(DATA / f"{split}.jsonl"), 1):
        contract_id = str(row["contract_id"])
        raw = runtime.get(contract_id)
        if raw is None:
            raise ValueError(f"runtime opcode missing for {contract_id}")
        if " ".join(str(row["opcode"]).split()) != " ".join(str(raw).split()):
            raise ValueError(f"runtime opcode mismatch for {contract_id}")
        t0 = time.perf_counter()
        instructions, blocks, meta = parse_disassembled_opcode(raw, tokenizer)
        analyze_basic_blocks(instructions, blocks)
        model_tokens = tokenizer.tokenize(raw, add_special_tokens=False)
        feature = token_features(instructions, len(model_tokens))
        covered = np.zeros(len(model_tokens), dtype=np.bool_)
        for item in instructions:
            covered[item.model_token_indices] = True
        if not covered.all():
            raise ValueError(f"unmapped model tokens for {contract_id}: {np.where(~covered)[0][:10]}")
        for key, value in zip(("provenance", "roles", "valid", "status", "instruction_index", "derived_pc"), feature):
            parts[key].append(value)
        parts["token_ids"].append(torch.tensor(tokenizer.convert_tokens_to_ids(model_tokens), dtype=torch.int32))
        parts["offsets"].append(parts["offsets"][-1] + len(model_tokens)); parts["ids"].append(row["id"])
        parts["original_lengths"].append(len(model_tokens))
        counters.update(meta["errors"]); counters.update(item.analysis_status for item in instructions)
        counters.update(item.operation_role for item in instructions)
        counters["push_instructions"] += sum(item.opcode.startswith("PUSH") for item in instructions)
        counters["push_operand_tokens"] += sum(len(item.model_token_indices) - 1 for item in instructions if item.opcode.startswith("PUSH"))
        counters["feature_valid_tokens"] += int(feature[2].sum())
        counters["unknown_source_tokens"] += int(feature[0][:, -1].sum())
        counters["not_applicable_tokens"] += int((feature[3] == STATUS_TO_ID["not_applicable"]).sum())
        if meta["errors"] and len(failures) < 50:
            failures.append({"contract_id": contract_id, "errors": dict(Counter(meta["errors"]))})
        version = compilers.get(contract_id, "unknown")
        group = by_compiler.setdefault(version, Counter())
        group["contracts"] += 1; group["instructions"] += len(instructions)
        group["known_instructions"] += sum(item.analysis_status == "known" for item in instructions)
        group["unknown_instructions"] += sum(item.analysis_status == "unknown_entry" for item in instructions)
        group["failed_instructions"] += sum(item.analysis_status.startswith("analysis_failure") for item in instructions)
        for item in instructions:
            if item.opcode in target_status:
                target_status[item.opcode][item.analysis_status] += 1
        instruction_counts.append(len(instructions)); token_counts.append(len(model_tokens)); block_counts.append(len(blocks)); durations.append(time.perf_counter() - t0)
        if row_index % 500 == 0:
            print(f"[{split}] processed={row_index}", flush=True)
    output = {"provenance": torch.from_numpy(np.concatenate(parts["provenance"])),
              "operation_roles": torch.from_numpy(np.concatenate(parts["roles"])),
              "valid_mask": torch.from_numpy(np.concatenate(parts["valid"])),
              "analysis_status": torch.from_numpy(np.concatenate(parts["status"])),
              "instruction_index": torch.from_numpy(np.concatenate(parts["instruction_index"])),
              "derived_pc": torch.from_numpy(np.concatenate(parts["derived_pc"])),
              "token_ids": torch.cat(parts["token_ids"]), "offsets": torch.tensor(parts["offsets"], dtype=torch.int64),
              "ids": parts["ids"],
              "original_lengths": torch.tensor(parts["original_lengths"], dtype=torch.int32),
              "source_split": split, "test_checked": False}
    OUT.mkdir(parents=True, exist_ok=True)
    temp_path = OUT / f"{split}.tmp.pt"
    torch.save(output, temp_path)
    temp_path.replace(OUT / f"{split}.pt")
    total_instructions = sum(instruction_counts)
    report[split] = {"contracts": len(parts["ids"]), "instruction_count": {"min": min(instruction_counts), "mean": statistics.mean(instruction_counts), "max": max(instruction_counts)},
                     "model_token_count": {"min": min(token_counts), "mean": statistics.mean(token_counts), "max": max(token_counts)},
                     "basic_block_count": {"min": min(block_counts), "mean": statistics.mean(block_counts), "max": max(block_counts)},
                     "counters": dict(counters), "target_status": {name: dict(counts) for name, counts in target_status.items()},
                     "instruction_status_ratio": {name: counters[name] / total_instructions for name in ("known", "unknown_entry", "analysis_failure", "analysis_failure_unsupported_stack_effect", "analysis_failure_parse_status")},
                     "token_valid_ratio": counters["feature_valid_tokens"] / parts["offsets"][-1],
                     "token_unknown_ratio": counters["unknown_source_tokens"] / parts["offsets"][-1],
                     "push_operand_alignment": {"mapped_operand_tokens": counters["push_operand_tokens"], "mapped_not_applicable_tokens": counters["not_applicable_tokens"], "unmapped_model_tokens": 0},
                     "compiler_coverage": {version: dict(counts) for version, counts in by_compiler.items()},
                     "failure_examples": failures,
                     "processing_seconds": {"mean": statistics.mean(durations), "p50": float(np.quantile(durations, .5)), "p95": float(np.quantile(durations, .95))},
                     "cache_bytes": (OUT / f"{split}.pt").stat().st_size,
                     "total_seconds": time.perf_counter() - started}


def main():
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(VOCAB); runtime = load_runtime(); compilers = load_compilers()
    quality = {"dataset": str(DATA.relative_to(ROOT)), "label_names": list(json.loads((DATA / "manifest.json").read_text())["label_names"]),
               "provenance_categories": SOURCE_CATEGORIES, "role_names": ROLE_NAMES, "status_names": STATUS_NAMES,
               "splits": {}, "runtime_bytecode_available": False, "test_checked": False,
               "scope": "disassembled-opcode-based basic-block-local reconstruction; no cross-block execution claim"}
    build_split("train", runtime, compilers, tokenizer, quality["splits"]); build_split("valid", runtime, compilers, tokenizer, quality["splits"])
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "spor_feature_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    lines = ["# SPOR Phase 2 Feature Quality", "", "DIVE_8 train/valid only; test_checked=false.", "", json.dumps(quality, indent=2)]
    (REPORT / "spor_feature_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(quality, indent=2), flush=True)


if __name__ == "__main__": main()
