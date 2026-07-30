"""Freeze the train-only 384-BPE Source-Main6 v2 unit budget."""

import json
import math
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidity_contract_graph_v2 import extract_audit_units, source_file_text  # noqa: E402

MAX_CODE_TOKENS = 384
STRIDE = 192


def windows_for_tokens(token_count):
    return 1 if token_count <= MAX_CODE_TOKENS else 1 + math.ceil((token_count - MAX_CODE_TOKENS) / STRIDE)


def main():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/graphcodebert-base", use_fast=True, local_files_only=True)
    rows = [json.loads(line) for line in (ROOT / "data/processed/DIVE_source_main6_v2/train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    counts, mandatory, token_counts = [], [], []
    for index, row in enumerate(rows, 1):
        units = extract_audit_units(source_file_text(ROOT / row["source_path"]))
        per_unit_tokens = [tokenizer(unit.source, add_special_tokens=False, truncation=False, verbose=False)["input_ids"] for unit in units]
        per_unit = [windows_for_tokens(len(tokens)) for tokens in per_unit_tokens]
        counts.append(sum(per_unit)); mandatory.append(sum(value for unit, value in zip(units, per_unit) if unit.mandatory)); token_counts.append(sum(len(tokens) for tokens in per_unit_tokens))
        if index % 250 == 0:
            print(f"budget: {index}/{len(rows)}", flush=True)
    p95 = int(np.percentile(counts, 95, method="higher"))
    budget = int(math.ceil(p95 / 8.0) * 8)
    report = {"source_split": "train_only", "max_code_tokens": MAX_CODE_TOKENS, "window_stride": STRIDE, "contracts": len(rows), "unit_windows": {"p50": int(np.percentile(counts, 50)), "p90": int(np.percentile(counts, 90)), "p95": p95, "max": int(max(counts))}, "mandatory_windows": {"p50": int(np.percentile(mandatory, 50)), "p95": int(np.percentile(mandatory, 95)), "max": int(max(mandatory))}, "token_count": {"p50": int(np.percentile(token_counts, 50)), "p95": int(np.percentile(token_counts, 95))}, "frozen_unit_budget": budget, "coverage_at_budget": float(np.mean(np.asarray(counts) <= budget))}
    output = ROOT / "data/reports/dive_source_main6_v2/source_v2_train_budget.json"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
