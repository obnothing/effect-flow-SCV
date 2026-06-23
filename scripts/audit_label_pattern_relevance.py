import argparse
import json
import math
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from effect_flow_schema import EFPP_PATTERNS  # noqa: E402
from effect_flow_utils import (  # noqa: E402
    DATASET_SPECS,
    REPORT_DIR,
    corpus_split_paths,
    iter_jsonl,
    relative,
    resolve,
    strict_split_paths,
    write_json,
    write_text,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit vulnerability-label/pattern relevance.")
    parser.add_argument("--dataset", choices=("BJUT", "DIVE"), required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default=None)
    parser.add_argument("--all_splits_diagnostic", action="store_true")
    parser.add_argument("--corpus_dir", default=None)
    parser.add_argument("--report_suffix", default="")
    return parser.parse_args()


def load_contract_patterns(path):
    patterns = {}
    for _, item in tqdm(iter_jsonl(path), desc=f"patterns:{path.stem}", unit="chunk"):
        contract_id = str(item["id"])
        vector = item["efpp_pattern_labels"]
        if contract_id not in patterns:
            patterns[contract_id] = list(vector)
        else:
            patterns[contract_id] = [
                int(left or right) for left, right in zip(patterns[contract_id], vector)
            ]
    return patterns


def interpretation(coverage, lift):
    if coverage >= 0.3 and lift >= 1.5:
        return "strong_candidate"
    if coverage < 0.3 and lift >= 2.0:
        return "high_precision_low_coverage"
    if coverage >= 0.3 and lift < 1.2:
        return "common_but_weak"
    return "weak_or_unrelated"


def association_row(dataset, split, label_name, y, pattern_name, p):
    a = sum(1 for label, pattern in zip(y, p) if label and pattern)
    b = sum(1 for label, pattern in zip(y, p) if not label and pattern)
    c = sum(1 for label, pattern in zip(y, p) if label and not pattern)
    d = sum(1 for label, pattern in zip(y, p) if not label and not pattern)
    support = a + c
    negatives = b + d
    pattern_positive = a + b
    coverage = a / support if support else 0.0
    negative_rate = b / negatives if negatives else 0.0
    pattern_rate = pattern_positive / len(y) if y else 0.0
    lift = coverage / max(negative_rate, 1e-8)
    precision_like = a / pattern_positive if pattern_positive else 0.0
    odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / denominator if denominator else 0.0
    balanced_score = coverage * math.log(lift + 1.0)
    warnings = []
    if support < 30:
        warnings.append("low_label_support")
    if pattern_rate < 0.001:
        warnings.append("pattern_too_sparse")
    if lift >= 2.0 and coverage < 0.1:
        warnings.append("high_lift_but_very_low_coverage")
    if coverage >= 0.3 and lift < 1.2:
        warnings.append("high_coverage_but_low_discrimination")
    return {
        "dataset": dataset,
        "split": split,
        "label_name": label_name,
        "label_support": support,
        "pattern_name": pattern_name,
        "pattern_positive_rate": pattern_rate,
        "coverage": coverage,
        "negative_rate": negative_rate,
        "lift": lift,
        "precision_like": precision_like,
        "odds_ratio": odds_ratio,
        "phi_correlation": phi,
        "balanced_score": balanced_score,
        "interpretation": interpretation(coverage, lift),
        "warnings": warnings,
        "contingency": {"y1_p1": a, "y0_p1": b, "y1_p0": c, "y0_p0": d},
    }


def audit_split(dataset, split, corpus_dir=None):
    label_names = DATASET_SPECS[dataset]["label_names"]
    label_path = strict_split_paths(dataset)[split]
    pattern_path = corpus_split_paths(dataset)[split]
    if corpus_dir:
        pattern_path = resolve(corpus_dir) / f"{split}_effect_flow_chunks.jsonl"
    if not label_path.exists() or not pattern_path.exists():
        raise FileNotFoundError(
            f"Missing label/corpus pair: {relative(label_path)}, {relative(pattern_path)}"
        )
    contract_patterns = load_contract_patterns(pattern_path)
    aligned = []
    missing_patterns = 0
    for _, item in tqdm(iter_jsonl(label_path), desc=f"labels:{dataset}:{split}", unit="contract"):
        raw_id = item.get("id")
        if raw_id is None:
            raw_id = item.get("address")
        contract_id = str(raw_id)
        if contract_id not in contract_patterns:
            missing_patterns += 1
            continue
        labels = item.get("multi_labels")
        if labels is None:
            labels = item.get("labels")
        if labels is None or len(labels) != len(label_names):
            raise ValueError(f"Invalid labels for {dataset}/{split}/{contract_id}")
        aligned.append((labels, contract_patterns[contract_id]))
    rows = []
    for label_index, label_name in enumerate(label_names):
        y = [int(labels[label_index]) for labels, _ in aligned]
        for pattern_index, pattern_name in enumerate(EFPP_PATTERNS):
            p = [int(patterns[pattern_index]) for _, patterns in aligned]
            rows.append(
                association_row(dataset, split, label_name, y, pattern_name, p)
            )
    summaries = {}
    for label_name in label_names:
        label_rows = [row for row in rows if row["label_name"] == label_name]
        summaries[label_name] = {
            "label_support": label_rows[0]["label_support"] if label_rows else 0,
            "top_by_lift": sorted(label_rows, key=lambda row: row["lift"], reverse=True)[:5],
            "top_by_coverage": sorted(label_rows, key=lambda row: row["coverage"], reverse=True)[:5],
            "top_by_balanced_score": sorted(
                label_rows, key=lambda row: row["balanced_score"], reverse=True
            )[:5],
            "strong_or_high_precision_patterns": [
                row["pattern_name"]
                for row in label_rows
                if row["interpretation"]
                in {"strong_candidate", "high_precision_low_coverage"}
            ],
        }
    return {
        "dataset": dataset,
        "split": split,
        "purpose": "analysis only; no vulnerability labels enter the pretraining corpus",
        "is_training_prior_source": split == "train",
        "aligned_contracts": len(aligned),
        "contracts_missing_pattern_chunks": missing_patterns,
        "label_names": label_names,
        "pattern_names": EFPP_PATTERNS,
        "rows": rows,
        "label_summaries": summaries,
    }


def render(report):
    lines = [
        f"Label-pattern relevance audit: {report['dataset']} / {report['split']}",
        "",
        "IMPORTANT: label-pattern relevance is for analysis only.",
        "Vulnerability labels are not stored in the effect-flow pretraining corpus.",
        f"is_training_prior_source: {report['is_training_prior_source']}",
        f"aligned_contracts: {report['aligned_contracts']}",
        f"contracts_missing_pattern_chunks: {report['contracts_missing_pattern_chunks']}",
    ]
    for label, summary in report["label_summaries"].items():
        lines.extend(["", f"[{label}] support={summary['label_support']}"])
        for title, key in (
            ("top by lift", "top_by_lift"),
            ("top by coverage", "top_by_coverage"),
            ("top by balanced score", "top_by_balanced_score"),
        ):
            lines.append(title + ":")
            for row in summary[key]:
                lines.append(
                    f"- {row['pattern_name']}: coverage={row['coverage']:.4f}, "
                    f"negative_rate={row['negative_rate']:.4f}, lift={row['lift']:.4f}, "
                    f"precision_like={row['precision_like']:.4f}, "
                    f"balanced_score={row['balanced_score']:.4f}, "
                    f"interpretation={row['interpretation']}, warnings={row['warnings']}"
                )
    return lines


def main():
    args = parse_args()
    if not args.split and not args.all_splits_diagnostic:
        raise SystemExit("Specify --split or --all_splits_diagnostic.")
    if args.split:
        report = audit_split(args.dataset, args.split, args.corpus_dir)
        report_suffix = f"_{args.report_suffix}" if args.report_suffix else ""
        suffix = f"{args.dataset}_{args.split}{report_suffix}"
        json_path = REPORT_DIR / f"label_pattern_relevance_{suffix}.json"
        txt_path = REPORT_DIR / f"label_pattern_relevance_{suffix}.txt"
        write_json(json_path, report)
        write_text(txt_path, render(report))
        print(f"[OK] wrote {relative(txt_path)}")
    if args.all_splits_diagnostic:
        reports = [
            audit_split(args.dataset, split, args.corpus_dir)
            for split in ("train", "valid", "test")
        ]
        payload = {
            "dataset": args.dataset,
            "purpose": "diagnostic only; valid/test are not sources of training priors",
            "splits": reports,
        }
        report_suffix = f"_{args.report_suffix}" if args.report_suffix else ""
        txt_path = REPORT_DIR / f"label_pattern_relevance_{args.dataset}_all_splits_diagnostic{report_suffix}.txt"
        json_path = REPORT_DIR / f"label_pattern_relevance_{args.dataset}_all_splits_diagnostic{report_suffix}.json"
        lines = [
            f"All-splits diagnostic: {args.dataset}",
            "valid/test labels are diagnostic only and must not define training priors.",
        ]
        for report in reports:
            lines.extend(["", *render(report)])
        write_json(json_path, payload)
        write_text(txt_path, lines)
        print(f"[OK] wrote {relative(txt_path)}")


if __name__ == "__main__":
    main()
