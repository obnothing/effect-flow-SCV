import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from effect_flow_utils import REPORT_DIR, relative, write_json, write_text  # noqa: E402


def read(name):
    path = REPORT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 16A prerequisite: {relative(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Stage 16A reports.")
    parser.add_argument("--report_suffix", default="")
    return parser.parse_args()


def top_related(relevance):
    result = {}
    for label, summary in relevance["label_summaries"].items():
        result[label] = [
            {
                "pattern": row["pattern_name"],
                "coverage": row["coverage"],
                "lift": row["lift"],
                "balanced_score": row["balanced_score"],
                "interpretation": row["interpretation"],
            }
            for row in summary["top_by_balanced_score"][:5]
        ]
    return result


def main():
    args = parse_args()
    suffix = f"_{args.report_suffix}" if args.report_suffix else ""
    inventory = read("stage16a_input_inventory.json")
    audits = {
        dataset: read(f"effect_flow_annotation_audit_{dataset}{suffix}.json")
        for dataset in ("BJUT", "DIVE")
    }
    relevance = {
        dataset: read(f"label_pattern_relevance_{dataset}_train{suffix}.json")
        for dataset in ("BJUT", "DIVE")
    }
    recommendation = read(f"usable_efpp_patterns_recommendation{suffix}.json")

    etp_rare = {}
    pattern_usable = {}
    pattern_sparse = {}
    pattern_frequent = {}
    for dataset, audit in audits.items():
        etp_rare[dataset] = [
            row["effect_type"]
            for row in audit["etp_primary_distribution"]
            if row["token_ratio"] < 0.001
        ]
        pattern_usable[dataset] = [
            row["pattern_name"]
            for row in audit["efpp_pattern_distribution"]
            if row["usable_for_pretraining"]
        ]
        pattern_sparse[dataset] = [
            row["pattern_name"]
            for row in audit["efpp_pattern_distribution"]
            if "too_sparse" in row["warnings"]
        ]
        pattern_frequent[dataset] = [
            row["pattern_name"]
            for row in audit["efpp_pattern_distribution"]
            if "too_frequent_low_discrimination" in row["warnings"]
        ]

    recommended = recommendation["recommended_pattern_names"]
    proceed = (
        inventory["status"] == "ok"
        and all(audit["usable_for_pretraining"] for audit in audits.values())
        and len(recommended) >= 5
    )
    report = {
        "stage": "16A",
        "input_files_confirmed": inventory["status"] == "ok",
        "selected_inputs": inventory["selected_inputs"],
        "effect_flow_chunks": {
            dataset: {
                "contracts": audit["total_contracts"],
                "chunks": audit["total_chunks"],
                "mean_chunks_per_contract": audit["mean_chunks_per_contract"],
                "coverage_ratio_mean": audit["token_coverage_ratio_mean"],
            }
            for dataset, audit in audits.items()
        },
        "etp_distribution_reasonable": {
            dataset: audit["active_etp_tokens"] > 0 for dataset, audit in audits.items()
        },
        "rare_etp_effect_types": etp_rare,
        "usable_efpp_patterns_by_distribution": pattern_usable,
        "too_sparse_efpp_patterns": pattern_sparse,
        "too_frequent_efpp_patterns": pattern_frequent,
        "recommended_efpp_patterns": recommended,
        "BJUT_label_top_patterns": top_related(relevance["BJUT"]),
        "DIVE_label_top_patterns": top_related(relevance["DIVE"]),
        "effect_flow_expected_strong_signal_labels": [
            "Reentrancy",
            "Unchecked Return / Unchecked call return value",
            "Time manipulation / Timestamp dependence",
            "Bad Randomness",
            "Access Control",
            "Contract contains unknown address",
            "Arithmetic / Multiplication after division",
        ],
        "effect_flow_expected_weak_signal_labels": [
            "DoS",
            "Assert violation",
            "Extra gas consumption",
            "Front Running",
        ],
        "recommend_stage16b": proceed,
        "stage16b_recommended_tasks": ["MOM", "ETP", "EFPP", "ERR", "VEP", "VTM"] if proceed else [],
        "stage16b_not_recommended_yet": ["EEP", "complex EOP"],
        "external_unlabeled_dataset_schema_reusable": True,
        "external_dataset_minimum_format": {
            "required": ["contract id", "runtime opcode sequence"],
            "optional": ["source dataset name"],
            "vulnerability_labels_required": False,
        },
        "tokenizer_retraining_required": False,
        "tokenizer_retraining_condition": "only if external-data OOV/UNK ratio is abnormally high",
        "warnings": [
            "EFPP rules are weak semantic annotations, not vulnerability labels.",
            "Only train-split relevance may inform Stage 16B pattern selection; valid/test remain diagnostic.",
            "Loop-candidate patterns are noisy because Stage 16A does not build a CFG or perform stack-precise jump analysis.",
        ],
    }
    json_path = REPORT_DIR / f"stage16a_effect_flow_summary{suffix}.json"
    txt_path = REPORT_DIR / f"stage16a_effect_flow_summary{suffix}.txt"
    write_json(json_path, report)
    lines = [
        "Stage 16A Effect-Flow Annotation + EFPP Relevance Summary",
        "",
        f"input_files_confirmed: {report['input_files_confirmed']}",
        f"recommend_stage16b: {proceed}",
        f"stage16b_recommended_tasks: {report['stage16b_recommended_tasks']}",
        f"stage16b_not_recommended_yet: {report['stage16b_not_recommended_yet']}",
        "",
        "Effect-flow chunks:",
    ]
    for dataset, row in report["effect_flow_chunks"].items():
        lines.append(f"- {dataset}: {row}")
    lines.extend(["", "Rare ETP effect types:"])
    for dataset, names in etp_rare.items():
        lines.append(f"- {dataset}: {names}")
    lines.extend(["", "Recommended EFPP patterns:", *[f"- {name}" for name in recommended]])
    lines.extend(["", "Too sparse EFPP patterns:"])
    for dataset, names in pattern_sparse.items():
        lines.append(f"- {dataset}: {names}")
    lines.extend(["", "Too frequent EFPP patterns:"])
    for dataset, names in pattern_frequent.items():
        lines.append(f"- {dataset}: {names}")
    for dataset, key in (("BJUT", "BJUT_label_top_patterns"), ("DIVE", "DIVE_label_top_patterns")):
        lines.extend(["", f"{dataset} label-pattern coverage/lift (train only):"])
        for label, rows in report[key].items():
            lines.append(f"[{label}]")
            for row in rows:
                lines.append(
                    f"- {row['pattern']}: coverage={row['coverage']:.4f}, "
                    f"lift={row['lift']:.4f}, score={row['balanced_score']:.4f}, "
                    f"{row['interpretation']}"
                )
    lines.extend(
        [
            "",
            "Effect-flow expected strong-signal labels:",
            *[f"- {name}" for name in report["effect_flow_expected_strong_signal_labels"]],
            "",
            "Effect-flow expected weak-signal labels:",
            *[f"- {name}" for name in report["effect_flow_expected_weak_signal_labels"]],
            "",
            "External unlabeled dataset reuse:",
            "- required: contract id and runtime opcode sequence",
            "- optional: source dataset name",
            "- vulnerability labels are not required",
            "- tokenizer retraining is not required unless OOV/UNK ratio is abnormally high",
            "",
            "Warnings:",
            *[f"- {warning}" for warning in report["warnings"]],
        ]
    )
    write_text(txt_path, lines)
    print(f"[OK] wrote {relative(txt_path)}")
    print(f"[OK] wrote {relative(json_path)}")


if __name__ == "__main__":
    main()
