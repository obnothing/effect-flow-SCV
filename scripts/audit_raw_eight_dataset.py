"""Audit raw opcode/source alignment before building a supervised 8-label dataset."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "DIVE_Raw_Data/Raw"
OUT = ROOT / "reports/raw_eight_dataset_audit"


def iter_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def main():
    runtime_path = RAW / "POST/Runtime_Opcode.jsonl"
    source_dir = RAW / "PRE/Source codes"
    sample_path = RAW / "DIVE_Samples.csv"
    runtime_ids, runtime_rows = set(), 0
    runtime_label_fields = set()
    empty_opcode = 0
    for _, row in iter_jsonl(runtime_path):
        runtime_rows += 1
        contract_id = str(row.get("contractID", ""))
        runtime_ids.add(contract_id)
        runtime_label_fields.update(key for key in row if "label" in key.lower() or "vuln" in key.lower())
        if not str(row.get("Opcodes", "")).strip():
            empty_opcode += 1
    source_ids = {path.stem for path in source_dir.glob("*.sol")}
    source_non_sol = len([path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() != ".sol"])
    with sample_path.open(encoding="utf-8-sig", newline="") as handle:
        sample_reader = csv.DictReader(handle)
        sample_fields = sample_reader.fieldnames or []
        sample_rows = list(sample_reader)
    sample_label_fields = {key for key in sample_fields if "label" in key.lower() or "vuln" in key.lower()}
    overlap = runtime_ids & source_ids
    missing_source = runtime_ids - source_ids
    report = {
        "runtime_path": str(runtime_path.relative_to(ROOT)),
        "source_dir": str(source_dir.relative_to(ROOT)),
        "runtime_rows": runtime_rows,
        "runtime_unique_contract_ids": len(runtime_ids),
        "runtime_empty_opcode_rows": empty_opcode,
        "source_solidity_files": len(source_ids),
        "source_non_solidity_files": source_non_sol,
        "runtime_source_id_overlap": len(overlap),
        "runtime_without_source": len(missing_source),
        "runtime_fields_containing_label_or_vulnerability": sorted(runtime_label_fields),
        "sample_rows": len(sample_rows),
        "sample_fields": sample_fields,
        "sample_fields_containing_label_or_vulnerability": sorted(sample_label_fields),
        "supervised_eight_label_dataset_ready": bool(runtime_label_fields or sample_label_fields),
        "test_checked": False,
        "conclusion": "Raw opcode/source files expose no vulnerability labels; an external 8-label mapping is required before writing supervised train/valid/test JSONL.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Raw DIVE 8-label Dataset Audit", "",
        f"- Runtime rows: `{runtime_rows}`; unique contract IDs: `{len(runtime_ids)}`.",
        f"- Solidity source files: `{len(source_ids)}`; runtime/source ID overlap: `{len(overlap)}`.",
        f"- Runtime rows without source file: `{len(missing_source)}`; empty opcode rows: `{empty_opcode}`.",
        f"- Runtime label-like fields: `{sorted(runtime_label_fields)}`.",
        f"- DIVE_Samples fields: `{sample_fields}`; label-like fields: `{sorted(sample_label_fields)}`.", "",
        "The raw files contain opcode, source and metadata, but no 8-label vulnerability mapping.",
        "A label mapping keyed by contractID/address is required before creating a supervised opcode dataset and a supervised source dataset.",
        "Any 8:1:1 split produced before that mapping would be only an unlabeled membership split.",
        "test_checked=false.", "",
        "Potential mapping keys are `contractID` or the address mapping in `DIVE_Samples.csv`; the label names and per-contract assignments must come from the authoritative annotation source.",
    ]
    (OUT / "raw_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
