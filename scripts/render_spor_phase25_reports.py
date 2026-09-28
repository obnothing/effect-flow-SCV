"""Render Phase 2.5 JSON audit artifacts as readable Markdown reports."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/spor_phase25"
P = ROOT / "reports/spor_phase25"


def read(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def pct(value, total):
    return 100.0 * value / total if total else 0.0


def main():
    P.mkdir(parents=True, exist_ok=True)
    failure = read("failure_breakdown.json")
    comparison = read("coverage_comparison.json")
    opcodes = read("opcode_coverage.json")
    missing = read("missingness_audit.json")
    group = read("coverage_by_label_compiler_length.json")
    summary = read("summary.json")

    lines = ["# SPOR Phase 2.5 Failure Breakdown", "", "Dataset: `DIVE_8_opcode_random_split`; train/valid only. `test_checked=false`.",
             "Percentages below are shown both against all model tokens and against tokens failing the strict known-source criterion.", ""]
    for split, data in failure["splits"].items():
        lines += [f"## {split}", "", f"Total tokens: {data['total_tokens']:,}; invalid tokens: {data['invalid_tokens']:,} ({pct(data['invalid_tokens'], data['total_tokens']):.2f}%).", "",
                  "| Cause | Absolute count | % all tokens | % invalid tokens |", "|---|---:|---:|---:|"]
        for name, item in sorted(data["token_reasons"].items(), key=lambda pair: -pair[1]["count"]):
            lines.append(f"| {name} | {item['count']:,} | {pct(item['count'], data['total_tokens']):.2f}% | {item['pct_of_invalid_tokens']:.2f}% |")
        lines += ["", "Target instruction causes:", "", "| Opcode | Cause | Count | % of this opcode |", "|---|---|---:|---:|"]
        for opcode, reasons in data["target_reasons"].items():
            total = sum(reasons.values())
            for reason, count in sorted(reasons.items(), key=lambda pair: -pair[1]):
                lines.append(f"| {opcode} | {reason} | {count:,} | {pct(count, total):.2f}% |")
        lines.append("")
    (P / "failure_breakdown.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# SPOR Phase 2.5 Stack-Effect Extension", "", "The registry records only verified pop/push counts. An opcode with known stack effect but unmodeled value semantics consumes its inputs and pushes an explicit unknown value. Unknown byte annotations without a verified effect remain unsupported.",
             "", "Stack effects were checked against the [Ethereum opcode reference](https://ethereum.org/developers/docs/evm/opcodes), [go-ethereum jump table](https://github.com/ethereum/go-ethereum/blob/master/core/vm/jump_table.go), and [Yellow Paper](https://ethereum.github.io/yellowpaper/paper.pdf).", "",
             "| Opcode | Train count | Valid count | Pop | Push | Minimum stack |", "|---|---:|---:|---:|---:|---:|"]
    for row in opcodes:
        lines.append(f"| {row['opcode']} | {row['train_count']:,} | {row['valid_count']:,} | {row['pop_count'] if row['pop_count'] is not None else 'unsupported'} | {row['push_count'] if row['push_count'] is not None else 'unsupported'} | {row['min_stack'] if row['min_stack'] is not None else 'unknown'} |")
    unsupported = [row for row in opcodes if row["pop_count"] is None]
    known_count = sum(row["train_count"] + row["valid_count"] for row in opcodes if row["pop_count"] is not None)
    all_count = sum(row["train_count"] + row["valid_count"] for row in opcodes)
    lines += ["", f"Known stack effects: {known_count:,}/{all_count:,} instructions ({pct(known_count, all_count):.3f}%). Unsupported opcode types: {len(unsupported)}.", "",
              "Unsupported observed opcodes (with total count):", ""]
    lines += [f"- `{row['opcode']}`: {row['train_count'] + row['valid_count']:,}" for row in sorted(unsupported, key=lambda row: -(row["train_count"] + row["valid_count"]))]
    lines += ["", "Version-sensitive opcodes in the registry: `BASEFEE` (London), `PREVRANDAO` (Paris transition), `PUSH0` (Shanghai), and `BLOBHASH`, `BLOBBASEFEE`, `TLOAD`, `TSTORE`, `MCOPY` (Cancun). A mnemonic's documented stack effect is stable when that opcode is valid for the executing fork; unknown byte annotations are not reinterpreted through a compiler-version guess.", "",
              "The parser's `UNKNOWN_0xFE` annotation is treated as `INVALID` for stack effect/control termination because FE is the documented INVALID opcode. Other unknown bytes are not assigned guessed stack effects.", ""]
    (P / "stack_effect_extension.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# SPOR Phase 2 vs Phase 2.5", "", "Same strict criterion: a token is valid only when its instruction has a supported and known provenance state; PUSH operand aliases are excluded. Train and valid only; test locked.", "",
             "| Split | Phase | Strict valid tokens | Total tokens | Coverage |", "|---|---|---:|---:|---:|"]
    for split, data in comparison["splits"].items():
        for phase, good, total in (("Phase 2", data["phase2_strict_valid_tokens"], data["phase2_total_tokens"]), ("Phase 2.5", data["phase25_strict_valid_tokens"], data["phase25_total_tokens"])):
            lines.append(f"| {split} | {phase} | {good:,} | {total:,} | {pct(good,total):.2f}% |")
    lines += ["", "| Split | Opcode | Phase 2 known/total | Phase 2.5 known/total | Phase 2 known % | Phase 2.5 known % |", "|---|---|---:|---:|---:|---:|"]
    for split, data in comparison["splits"].items():
        for opcode, values in data["targets"].items():
            a, b = values["phase2"], values["phase25"]
            lines.append(f"| {split} | {opcode} | {a['known']:,}/{a['total']:,} | {b['known']:,}/{b['total']:,} | {pct(a['known'],a['total']):.2f}% | {pct(b['known'],b['total']):.2f}% |")
    lines += ["", "## CFG and convergence", "", "| Split | Static jump edges | Fall-through edges | Unresolved jumps / branch ops | Height-conflict blocks | Mean worklist visits per block | Nonconverged blocks |", "|---|---:|---:|---:|---:|---:|---:|"]
    opmap = {row["opcode"]: row for row in opcodes}
    for split, data in comparison["splits"].items():
        branches = opmap.get("JUMP", {}).get(f"{split}_count", 0) + opmap.get("JUMPI", {}).get(f"{split}_count", 0)
        lines.append(f"| {split} | {data['static_cfg_edges']:,} | {data['fallthrough_edges']:,} | {data['unresolved_jumps']:,}/{branches:,} ({pct(data['unresolved_jumps'],branches):.2f}%) | {data['stack_height_conflicts']:,} | {data['mean_fixpoint_visits_per_block']:.4f} | {data['nonconverged_blocks']:,} |")
    failures = read("failure_breakdown.json")
    lines += ["", "| Split | Instructions | Block-entry unknown count / rate | Analysis-failure count / rate | Parser failures | Token-alignment failures |", "|---|---:|---:|---:|---:|---:|"]
    for split, data in comparison["splits"].items():
        reasons = failures["splits"][split]["instruction_reasons"]
        total = sum(reasons.values())
        unknown_entry = reasons.get("basic_block_entry_unknown", 0)
        fail_names = ("unsupported_opcode_stack_effect", "known_stack_effect_but_unknown_provenance", "predecessor_stack_height_conflict", "abstract_stack_underflow", "parser_failure", "previous_analysis_failure", "other")
        fail_count = sum(reasons.get(name, 0) for name in fail_names)
        parser_fail = reasons.get("parser_failure", 0)
        alignment_fail = failures["splits"][split]["token_reasons"].get("token_alignment_failure", {}).get("count", 0)
        lines.append(f"| {split} | {total:,} | {unknown_entry:,} ({pct(unknown_entry,total):.2f}%) | {fail_count:,} ({pct(fail_count,total):.2f}%) | {parser_fail:,} | {alignment_fail:,} |")
    (P / "phase2_vs_phase25.md").write_text("\n".join(lines), encoding="utf-8")

    names = missing["features"][:6]
    lines = ["# SPOR Missingness Bias Audit", "", "Classifier inputs contain only per-contract availability ratios and optional sequence length. No opcode identity, provenance category, source code, or model embedding is used. ROC-AUC and PR-AUC are measured on valid; the comparison dummy has ROC-AUC 0.5 and PR-AUC equal to label prevalence.", "",
             "| Label | Valid prevalence | ROC-AUC no length | PR-AUC no length | ROC-AUC + length | PR-AUC + length | Dummy PR-AUC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, item in missing["labels"].items():
        lines.append(f"| {label} | {item['valid_prevalence']:.4f} | {item['without_length']['roc_auc']:.4f} | {item['without_length']['pr_auc']:.4f} | {item['with_length']['roc_auc']:.4f} | {item['with_length']['pr_auc']:.4f} | {item['dummy']['pr_auc']:.4f} |")
    lines += ["", "## Positive vs negative availability", "", "Coverage vectors use the feature order shown in the JSON: overall valid ratio, UNKNOWN ratio, and known ratios for JUMPI/CALL/SLOAD/SSTORE.", "",
              "| Split | Label | Positive n | Negative n | Positive availability vector | Negative availability vector |", "|---|---|---:|---:|---|---|"]
    for label, item in missing["labels"].items():
        for split in ("train", "valid"):
            entry = item[split]
            pos = json.dumps([round(v,4) for v in entry["coverage_positive"]])
            neg = json.dumps([round(v,4) for v in entry["coverage_negative"]])
            lines.append(f"| {split} | {label} | {entry['positive_count']:,} | {entry['negative_count']:,} | `{pos}` | `{neg}` |")
    version_groups = {split: {dimension: len({row['group'] for row in rows if row['dimension'] == dimension})
                              for dimension in ("compiler", "evm_version", "length")}
                      for split, rows in group["groups"].items()}
    lines += ["", f"The group audit includes compiler release and target EVM version separately, as well as length bins; train group counts: {version_groups['train']}; valid group counts: {version_groups['valid']}.",
              "Full compiler/EVM-version × length × split × label × polarity results are in `results/spor_phase25/coverage_by_label_compiler_length.json`.", ""]
    (P / "missingness_bias_audit.md").write_text("\n".join(lines), encoding="utf-8")

    smoke = json.loads((R / "E3_P11_8_A0/smoke.json").read_text(encoding="utf-8"))
    valid_auc = [item["without_length"]["roc_auc"] for item in missing["labels"].values()]
    strong = [name for name, item in missing["labels"].items() if item["without_length"]["roc_auc"] >= .75]
    decision = "NO-GO"
    summary_lines = ["# SPOR Phase 2.5 Summary and Decision", "", "Dataset: `DIVE_8_opcode_random_split`; seed 42 baseline preflight; train/valid only; test locked.", "",
        "## Findings", "",
        f"Strict coverage rose from {pct(comparison['splits']['train']['phase2_strict_valid_tokens'],comparison['splits']['train']['phase2_total_tokens']):.2f}% to {pct(comparison['splits']['train']['phase25_strict_valid_tokens'],comparison['splits']['train']['phase25_total_tokens']):.2f}% on train, and from {pct(comparison['splits']['valid']['phase2_strict_valid_tokens'],comparison['splits']['valid']['phase2_total_tokens']):.2f}% to {pct(comparison['splits']['valid']['phase25_strict_valid_tokens'],comparison['splits']['valid']['phase25_total_tokens']):.2f}% on valid.",
        "", "Known-source rates improved for all four tracked opcodes, but CALL and SSTORE remain limited. Missingness alone is predictive for several labels, so a downstream SPOR model could use analysis availability as a label shortcut.", "",
        f"The metadata-only classifier has ROC-AUC >= 0.75 for: {', '.join(strong) if strong else 'none'}.", "",
        f"E3_P11_8_A0 forward/backward smoke passed; parameters={smoke['parameter_count']:,}, eight labels, 16 polarity queries, query/evidence 512, BiGRU output 1024, K/V 1024->512, 8x64 attention. No training or test evaluation was run.", "",
        "## Decision: NO-GO for Phase 3", "",
        f"Do not integrate the full SPOR feature stream into P11 yet. {100-pct(comparison['splits']['train']['phase25_strict_valid_tokens'],comparison['splits']['train']['phase25_total_tokens']):.2f}% of train tokens still fail the strict criterion, CALL known-source coverage is {pct(comparison['splits']['train']['targets']['CALL']['phase25']['known'],comparison['splits']['train']['targets']['CALL']['phase25']['total']):.2f}%, SSTORE is {pct(comparison['splits']['train']['targets']['SSTORE']['phase25']['known'],comparison['splits']['train']['targets']['SSTORE']['phase25']['total']):.2f}%, and missingness-only prediction exposes a measurable shortcut. A later controlled phase may evaluate an explicitly selected reliable subset after adding missingness controls; current evidence does not support a clean full-SPOR causal test.", "",
        "Safe interpretation today: retain the improved provenance cache for diagnostics and use the per-instruction coverage flags in future controlled experiments. Do not treat coverage availability as evidence of vulnerability semantics. Phase 3 remains pending human decision.", ""]
    (P / "phase25_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
