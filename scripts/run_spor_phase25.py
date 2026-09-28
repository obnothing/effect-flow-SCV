"""Train/valid-only Phase 2.5 coverage, cache, and missingness audit."""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer
from spor.evm_parser import parse_disassembled_opcode
from spor.phase25 import TARGETS, analyze_partial_cfg
from spor.stack_effects import stack_effect
from spor.stack_provenance import SOURCE_CATEGORIES
from build_spor_cache import token_features, ROLE_NAMES, STATUS_NAMES

DATA = ROOT / "data/processed/DIVE_8_opcode_random_split"
RESULTS = ROOT / "results/spor_phase25"
REPORTS = ROOT / "reports/spor_phase25"
CACHE = ROOT / "data/features/spor_dive8_phase25"
REASONS = ["basic_block_entry_unknown", "unsupported_opcode_stack_effect",
           "known_stack_effect_but_unknown_provenance", "dynamic_jump_or_unresolved_cfg",
           "predecessor_stack_height_conflict", "abstract_stack_underflow",
           "parser_failure", "token_alignment_failure", "previous_analysis_failure",
           "operand_alias", "not_applicable", "other"]


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def runtime_map():
    return {str(row["contractID"]): row["Opcodes"] for row in rows(ROOT / "DIVE_Raw_Data/Raw/POST/Runtime_Opcode.jsonl")}


def compiler_map():
    with (ROOT / "DIVE_Raw_Data/Raw/PRE/Code-based.csv").open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return {str(row["contractID"]): row.get("CompilerVersion") or "unknown" for row in csv.DictReader(handle)}


def length_bin(length):
    if length <= 2048:
        return "1-2048"
    if length <= 4096:
        return "2049-4096"
    if length <= 8192:
        return "4097-8192"
    return "8193+"


def contract_features(ins, valid, provenance, length):
    values = [float(valid.mean()), float(provenance[:, -1].mean())]
    for op in ("JUMPI", "CALL", "SLOAD", "SSTORE"):
        relevant = [item for item in ins if item.opcode == op]
        values.append(sum(item.analysis_status == "known" for item in relevant) / len(relevant) if relevant else 0.0)
    return values + [float(length)]


def build_split(split, runtime, compilers, tokenizer, limit=None):
    counts = Counter(); reasons = Counter(); target_reasons = {op: Counter() for op in TARGETS}
    opcodes = Counter(); compiler_groups = defaultdict(Counter); length_groups = defaultdict(Counter)
    totals = Counter(); metadata = []; targets = []
    arrays = {name: [] for name in ("provenance", "operation_roles", "valid_mask", "analysis_status", "instruction_index", "derived_pc", "token_ids")}
    offsets = [0]; ids = []; original_lengths = []
    started = time.perf_counter()
    for row_index, row in enumerate(rows(DATA / f"{split}.jsonl"), 1):
        if limit and row_index > limit:
            break
        cid = str(row["contract_id"])
        raw = runtime.get(cid)
        if raw is None or " ".join(raw.split()) != " ".join(row["opcode"].split()):
            raise ValueError(f"raw opcode mismatch: {split}/{cid}")
        ins, blocks, parse_info = parse_disassembled_opcode(raw, tokenizer)
        stats = analyze_partial_cfg(ins, blocks)
        token_ids = tokenizer.encode(raw, add_special_tokens=False)
        feature = token_features(ins, len(token_ids))
        mapped = np.zeros(len(token_ids), dtype=np.bool_)
        for item in ins:
            mapped[item.model_token_indices] = True
            opcodes[item.opcode] += 1
            if item.opcode in target_reasons:
                target_reasons[item.opcode]["known" if item.analysis_status == "known" else item.failure_reason or "other"] += 1
            token_reason = item.failure_reason or ("not_applicable" if item.provenance_bits == 0 else "other")
            for position, token_index in enumerate(item.model_token_indices):
                if not feature[2][token_index]:
                    reasons["operand_alias" if position else token_reason] += 1
        if not mapped.all():
            raise ValueError(f"token alignment failed: {split}/{cid}")
        for name, values in zip(("provenance", "operation_roles", "valid_mask", "analysis_status", "instruction_index", "derived_pc"), feature):
            arrays[name].append(torch.from_numpy(values))
        arrays["token_ids"].append(torch.tensor(token_ids, dtype=torch.int32))
        ids.append(row["id"]); offsets.append(offsets[-1] + len(token_ids)); original_lengths.append(len(token_ids))
        label = row["multi_labels"]
        if len(label) != 8:
            raise ValueError(f"expected 8 labels for {cid}")
        values = contract_features(ins, feature[2], feature[0], len(token_ids))
        metadata.append(values); targets.append(label)
        known_targets = {op: sum(item.opcode == op and item.analysis_status == "known" for item in ins) for op in TARGETS}
        present_targets = {op: sum(item.opcode == op for item in ins) for op in TARGETS}
        version = compilers.get(cid, "unknown")
        for group in (compiler_groups[version], length_groups[length_bin(len(token_ids))]):
            group["contracts"] += 1; group["valid_tokens"] += int(feature[2].sum()); group["tokens"] += len(token_ids)
            for op in TARGETS:
                group[f"{op}_known"] += known_targets[op]; group[f"{op}_total"] += present_targets[op]
        counts["contracts"] += 1; counts["instructions"] += len(ins); counts["tokens"] += len(token_ids)
        counts["strict_valid_tokens"] += int(feature[2].sum()); counts["unknown_source_tokens"] += int(feature[0][:, -1].sum())
        counts["parse_issues"] += len(parse_info["errors"])
        counts["blocks"] += len(blocks)
        counts["static_cfg_edges"] += stats["edges"].get("static_jump", 0) + stats["edges"].get("static_jumpi", 0)
        counts["fallthrough_edges"] += stats["edges"].get("jumpi_fallthrough", 0) + stats["edges"].get("sequential_fallthrough", 0)
        counts["unresolved_jumps"] += stats["unresolved_jumps"]
        counts["stack_height_conflicts"] += stats["stack_height_conflicts"]
        counts["fixpoint_visits"] += stats["fixpoint_visits"]
        counts["nonconverged_blocks"] += stats["nonconverged_blocks"]
        for item in ins:
            if item.failure_reason:
                totals[item.failure_reason] += 1
            else:
                totals["known"] += 1
        if row_index % 500 == 0:
            print(f"[{split}] {row_index}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (f"{split}_smoke.pt" if limit else f"{split}.pt")
    payload = {name: torch.cat(values) for name, values in arrays.items()}
    payload.update({"ids": ids, "offsets": torch.tensor(offsets), "original_lengths": torch.tensor(original_lengths),
                    "source_split": split, "test_checked": False, "status_names": STATUS_NAMES, "role_names": ROLE_NAMES})
    temp = path.with_suffix(".tmp.pt")
    torch.save(payload, temp); temp.replace(path)
    return {"counts": dict(counts), "failure_reasons_tokens": dict(reasons), "failure_reasons_instructions": dict(totals),
            "target_reasons": {op: dict(value) for op, value in target_reasons.items()}, "opcodes": dict(opcodes),
            "by_compiler": {key: dict(value) for key, value in compiler_groups.items()},
            "by_length": {key: dict(value) for key, value in length_groups.items()},
            "cache_bytes": path.stat().st_size, "elapsed_seconds": time.perf_counter() - started,
            "metadata": metadata, "labels": targets}


def audit_missingness(splits, label_names):
    train = np.asarray(splits["train"]["metadata"], dtype=float)
    valid = np.asarray(splits["valid"]["metadata"], dtype=float)
    y_train = np.asarray(splits["train"]["labels"], dtype=int)
    y_valid = np.asarray(splits["valid"]["labels"], dtype=int)
    result = {"test_checked": False, "features": ["valid_ratio", "unknown_ratio", "JUMPI_known", "CALL_known", "SLOAD_known", "SSTORE_known", "sequence_length"], "labels": {}}
    for index, name in enumerate(label_names):
        item = {"train_prevalence": float(y_train[:, index].mean()), "valid_prevalence": float(y_valid[:, index].mean())}
        for width, key in ((6, "without_length"), (7, "with_length")):
            if len(np.unique(y_train[:, index])) < 2 or len(np.unique(y_valid[:, index])) < 2:
                item[key] = {"roc_auc": None, "pr_auc": None}
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
            model.fit(train[:, :width], y_train[:, index])
            prediction = model.predict_proba(valid[:, :width])[:, 1]
            item[key] = {"roc_auc": float(roc_auc_score(y_valid[:, index], prediction)),
                         "pr_auc": float(average_precision_score(y_valid[:, index], prediction))}
        item["dummy"] = {"roc_auc": 0.5, "pr_auc": item["valid_prevalence"]}
        for split, matrix, labels in (("train", train, y_train), ("valid", valid, y_valid)):
            item[split] = {"positive_count": int(labels[:, index].sum()), "negative_count": int((1-labels[:, index]).sum()),
                "coverage_positive": matrix[labels[:, index] == 1, :6].mean(0).tolist() if (labels[:, index] == 1).any() else None,
                "coverage_negative": matrix[labels[:, index] == 0, :6].mean(0).tolist() if (labels[:, index] == 0).any() else None}
        result["labels"][name] = item
    return result


def write_reports(splits, labels, limit):
    RESULTS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    previous = json.loads((ROOT / "reports/spor_phase2/spor_feature_quality.json").read_text(encoding="utf-8"))
    breakdown = {"dataset": str(DATA.relative_to(ROOT)), "scope": "train_valid_smoke" if limit else "train_valid_full",
                 "test_checked": False, "reason_names": REASONS, "splits": {}}
    comparison = {"test_checked": False, "splits": {}}
    for split, result in splits.items():
        counts = result["counts"]
        n = counts["tokens"] - counts["strict_valid_tokens"]
        reasons = {key: {"count": int(result["failure_reasons_tokens"].get(key, 0)),
                         "pct_of_invalid_tokens": 100 * result["failure_reasons_tokens"].get(key, 0) / max(n, 1)} for key in REASONS}
        breakdown["splits"][split] = {"contracts": counts["contracts"], "total_tokens": counts["tokens"], "invalid_tokens": n,
            "token_reasons": reasons, "target_reasons": result["target_reasons"], "instruction_reasons": result["failure_reasons_instructions"]}
        old = previous["splits"][split]
        targets = {}
        for op in TARGETS:
            old_counts = old["target_status"][op]; new_counts = result["target_reasons"][op]
            targets[op] = {"phase2": {"known": old_counts.get("known", 0), "total": sum(old_counts.values())},
                           "phase25": {"known": new_counts.get("known", 0), "total": sum(new_counts.values())}}
        comparison["splits"][split] = {"phase2_strict_valid_tokens": old["counters"]["feature_valid_tokens"],
            "phase2_total_tokens": int(round(old["counters"]["feature_valid_tokens"] / old["token_valid_ratio"])),
            "phase25_strict_valid_tokens": counts["strict_valid_tokens"], "phase25_total_tokens": counts["tokens"],
            "targets": targets, "static_cfg_edges": counts["static_cfg_edges"], "fallthrough_edges": counts["fallthrough_edges"],
            "unresolved_jumps": counts["unresolved_jumps"], "stack_height_conflicts": counts["stack_height_conflicts"],
            "mean_fixpoint_visits_per_block": counts["fixpoint_visits"] / max(counts["blocks"], 1),
            "nonconverged_blocks": counts["nonconverged_blocks"]}
    for name, value in (("failure_breakdown", breakdown), ("coverage_comparison", comparison)):
        (RESULTS / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    opcode_rows = []
    for op in sorted(set().union(*(value["opcodes"] for value in splits.values()))):
        effect = stack_effect(op)
        opcode_rows.append({"opcode": op, "train_count": splits["train"]["opcodes"].get(op, 0),
                            "valid_count": splits["valid"]["opcodes"].get(op, 0),
                            "pop_count": effect.pop_count if effect else None, "push_count": effect.push_count if effect else None,
                            "min_stack": effect.min_stack if effect else None})
    (RESULTS / "opcode_coverage.json").write_text(json.dumps(opcode_rows, indent=2), encoding="utf-8")
    if not limit:
        missingness = audit_missingness(splits, labels)
        (RESULTS / "missingness_audit.json").write_text(json.dumps(missingness, indent=2), encoding="utf-8")
    else:
        missingness = None
    for split, value in splits.items():
        (RESULTS / f"{split}_groups.json").write_text(json.dumps({"compiler": value["by_compiler"], "length": value["by_length"]}, indent=2), encoding="utf-8")
    summary = {"comparison": comparison, "missingness": missingness, "unsupported_opcodes": [row for row in opcode_rows if row["pop_count"] is None]}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"scope": breakdown["scope"], "contracts": {s: v["counts"]["contracts"] for s,v in splits.items()},
                      "strict_token_coverage": {s: v["counts"]["strict_valid_tokens"] / v["counts"]["tokens"] for s,v in splits.items()},
                      "unsupported_opcode_types": len(summary["unsupported_opcodes"])}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / "data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json")
    runtime = runtime_map(); compilers = compiler_map()
    labels = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["label_names"]
    splits = {name: build_split(name, runtime, compilers, tokenizer, args.limit) for name in ("train", "valid")}
    write_reports(splits, labels, args.limit)


if __name__ == "__main__":
    main()
