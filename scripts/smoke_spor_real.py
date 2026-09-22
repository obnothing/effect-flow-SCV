"""Run readable SPOR traces on a representative DIVE_8 train subset."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer
from spor.evm_parser import parse_disassembled_opcode
from spor.stack_provenance import SOURCE_CATEGORIES, analyze_basic_blocks


def main():
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / "data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json")
    raw = {str(row["contractID"]): row["Opcodes"] for line in (ROOT / "DIVE_Raw_Data/Raw/POST/Runtime_Opcode.jsonl").open(encoding="utf-8") if line.strip() for row in [json.loads(line)]}
    rows = [json.loads(line) for line in (ROOT / "data/processed/DIVE_8_opcode_random_split/train.jsonl").open(encoding="utf-8") if line.strip()]
    predicates = {
        "PUSH0": lambda text: "PUSH0" in text,
        "CALL": lambda text: " CALL" in f" {text}",
        "JUMPI": lambda text: "JUMPI" in text,
        "SLOAD": lambda text: "SLOAD" in text,
        "SSTORE": lambda text: "SSTORE" in text,
        "long": lambda text: len(text.split()) > 4000,
    }
    selected = []
    used = set()
    for name, predicate in predicates.items():
        candidates = [row for row in rows if str(row["contract_id"]) not in used and predicate(raw[str(row["contract_id"])])]
        if candidates:
            candidates.sort(key=lambda row: len(raw[str(row["contract_id"])].split()))
            selected.append((name, candidates[len(candidates) // 2])); used.add(str(selected[-1][1]["contract_id"]))
    def serialize(value):
        return {"provenance": [SOURCE_CATEGORIES[i] for i in range(len(SOURCE_CATEGORIES)) if value.provenance_bits & (1 << i)],
                "status": value.analysis_status}
    traces = []
    for reason, row in selected:
        contract_id = str(row["contract_id"]); instructions, blocks, meta = parse_disassembled_opcode(raw[contract_id], tokenizer)
        analyze_basic_blocks(instructions, blocks)
        targets = [item for item in instructions if item.opcode in {"CALL", "JUMPI", "SLOAD", "SSTORE"}]
        trace = {"selection_reason": reason, "contract_id": contract_id, "instruction_count": len(instructions),
                 "block_count": len(blocks), "parser_meta": meta, "targets": []}
        for item in targets[:12]:
            trace["targets"].append({"instruction_index": item.instruction_index, "derived_pc": item.derived_pc,
                "opcode": item.opcode, "basic_block_id": item.basic_block_id, "analysis_status": item.analysis_status,
                "operation_role": item.operation_role, "model_token_indices": item.model_token_indices,
                "operand_sources": {key: serialize(value) if not isinstance(value, list) else [serialize(item) for item in value]
                    for key, value in item.operand_sources.items()}})
        traces.append(trace)
    output = ROOT / "reports/spor_phase2"; output.mkdir(parents=True, exist_ok=True)
    (output / "real_data_smoke_traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    lines = ["# SPOR Real-Data Smoke Test", "", "DIVE_8 train subset only; labels were not used to construct provenance features.", ""]
    for trace in traces:
        lines.append(f"## {trace['contract_id']} ({trace['selection_reason']})")
        lines.append(f"instructions={trace['instruction_count']}; blocks={trace['block_count']}; targets={len(trace['targets'])}")
        for item in trace["targets"][:5]:
            lines.append(f"- {item['opcode']} instruction={item['instruction_index']} pc={item['derived_pc']} block={item['basic_block_id']} status={item['analysis_status']} tokens={item['model_token_indices']}")
        lines.append("")
    (output / "real_data_smoke_traces.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected": len(traces), "output": str(output), "test_checked": False}, indent=2))


if __name__ == "__main__": main()
