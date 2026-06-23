import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from effect_flow_schema import EFPP_PATTERNS, NOISY_PATTERNS  # noqa: E402
from effect_flow_utils import REPORT_DIR, relative, write_json, write_text  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Select usable EFPP patterns.")
    parser.add_argument("--report_suffix", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = f"_{args.report_suffix}" if args.report_suffix else ""
    annotation = {
        dataset: read(REPORT_DIR / f"effect_flow_annotation_audit_{dataset}{suffix}.json")
        for dataset in ("BJUT", "DIVE")
    }
    relevance = {
        dataset: read(REPORT_DIR / f"label_pattern_relevance_{dataset}_train{suffix}.json")
        for dataset in ("BJUT", "DIVE")
    }
    annotation_rows = {
        dataset: {
            row["pattern_name"]: row
            for row in annotation[dataset]["efpp_pattern_distribution"]
        }
        for dataset in annotation
    }
    rows = []
    for pattern in EFPP_PATTERNS:
        ratios = {
            dataset: annotation_rows[dataset][pattern]["positive_chunk_ratio"]
            for dataset in annotation
        }
        distribution_ok = any(0.001 <= ratio <= 0.80 for ratio in ratios.values())
        related = {}
        for dataset in relevance:
            candidates = [
                row
                for row in relevance[dataset]["rows"]
                if row["pattern_name"] == pattern
                and row["interpretation"]
                in {"strong_candidate", "high_precision_low_coverage"}
            ]
            related[dataset] = sorted(
                {row["label_name"] for row in candidates}
            )
        high_relevance = bool(related["BJUT"] or related["DIVE"])
        use_pretraining = distribution_ok or high_relevance
        distribution_warning = []
        for dataset, ratio in ratios.items():
            if ratio < 0.001:
                distribution_warning.append(f"{dataset}:too_sparse")
            elif ratio > 0.80:
                distribution_warning.append(f"{dataset}:too_frequent")
        noise_warning = (
            "noisy_but_candidate: weak loop heuristic without CFG/backward-jump proof"
            if pattern in NOISY_PATTERNS
            else None
        )
        reasons = []
        if distribution_ok:
            reasons.append("prevalence falls in [0.1%, 80%] for at least one dataset")
        if high_relevance:
            reasons.append("train-only relevance audit finds a strong/high-precision candidate")
        if not reasons:
            reasons.append("too sparse/frequent and no strong train-only relevance")
        rows.append(
            {
                "pattern_name": pattern,
                "use_in_pretraining": use_pretraining,
                "use_in_label_prior_BJUT": bool(related["BJUT"]),
                "use_in_label_prior_DIVE": bool(related["DIVE"]),
                "reason": "; ".join(reasons),
                "related_BJUT_labels": related["BJUT"],
                "related_DIVE_labels": related["DIVE"],
                "positive_chunk_ratio": ratios,
                "distribution_warning": distribution_warning,
                "noise_warning": noise_warning,
            }
        )
    report = {
        "selection_source": "annotation prevalence plus train-only label relevance audit",
        "valid_test_used_for_prior_selection": False,
        "warning": "EFPP patterns are weak semantic annotations, not vulnerability labels.",
        "patterns": rows,
        "recommended_pattern_names": [
            row["pattern_name"] for row in rows if row["use_in_pretraining"]
        ],
        "excluded_pattern_names": [
            row["pattern_name"] for row in rows if not row["use_in_pretraining"]
        ],
    }
    json_path = REPORT_DIR / f"usable_efpp_patterns_recommendation{suffix}.json"
    txt_path = REPORT_DIR / f"usable_efpp_patterns_recommendation{suffix}.txt"
    write_json(json_path, report)
    lines = [
        "Usable EFPP pattern recommendation",
        "",
        "Selection uses annotation prevalence and train-only relevance statistics.",
        "EFPP patterns are weak semantic annotations, not vulnerability labels.",
        "valid_test_used_for_prior_selection: false",
        "",
        "pattern | pretrain | BJUT prior | DIVE prior | ratios | warnings | reason",
    ]
    for row in rows:
        lines.append(
            f"{row['pattern_name']} | {row['use_in_pretraining']} | "
            f"{row['use_in_label_prior_BJUT']} | {row['use_in_label_prior_DIVE']} | "
            f"{row['positive_chunk_ratio']} | {row['distribution_warning']} | "
            f"{row['noise_warning']} | {row['reason']}"
        )
        lines.append(f"  related_BJUT_labels: {row['related_BJUT_labels']}")
        lines.append(f"  related_DIVE_labels: {row['related_DIVE_labels']}")
    write_text(txt_path, lines)
    print(f"[OK] wrote {relative(txt_path)}")
    print(f"[OK] wrote {relative(json_path)}")


if __name__ == "__main__":
    main()
