import argparse
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOCUS_LABELS = {"Access Control", "Front Running"}


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def metric_block(test_report):
    if "per_label_threshold_result" in test_report:
        return "per_label_threshold_result", test_report["per_label_threshold_result"]
    if "best_global_threshold_result" in test_report:
        return "best_global_threshold_result", test_report["best_global_threshold_result"]
    if "threshold_0_5_baseline" in test_report:
        return "threshold_0_5_baseline", test_report["threshold_0_5_baseline"]
    raise ValueError("test report does not contain a known metric block")


def contribution_gap(contribution_report):
    rows = {}
    for row in contribution_report.get("per_label", []):
        label = row["label_name"]
        tp = row.get("tp_contribution", {}).get("final_evidence_mean")
        fp = row.get("fp_contribution", {}).get("final_evidence_mean")
        fn = row.get("fn_contribution", {}).get("final_evidence_mean")
        rows[label] = {
            "tp_final": tp,
            "fp_final": fp,
            "fn_final": fn,
            "tp_minus_fp": None if tp is None or fp is None else tp - fp,
            "tp_minus_fn": None if tp is None or fn is None else tp - fn,
            "risk_mean": row.get("predicted_positive_contribution", {}).get(
                "risk_mean"
            ),
            "protective_mean": row.get(
                "predicted_positive_contribution",
                {},
            ).get("protective_mean"),
            "missing_check_mean": row.get(
                "predicted_positive_contribution",
                {},
            ).get("missing_check_mean"),
            "final_evidence_mean": row.get(
                "predicted_positive_contribution",
                {},
            ).get("final_evidence_mean"),
        }
    return rows


def collect_variant(config_path):
    config_path = resolve_path(config_path)
    config = load_yaml(config_path)
    result_dir = resolve_path(config["result_dir"])
    test_path = result_dir / "test_threshold_calibration_metrics.json"
    contribution_path = result_dir / "behavior_contribution_summary.json"
    row = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "variant": config.get("behavior_search_variant")
        or config.get("ablation_variant")
        or config["experiment_name"],
        "experiment_name": config["experiment_name"],
        "result_dir": str(result_dir.relative_to(PROJECT_ROOT)),
        "behavior_weight_path": config.get("behavior_weight_path"),
        "status": "missing",
    }
    if not test_path.exists():
        row["missing_file"] = str(test_path.relative_to(PROJECT_ROOT))
        return row
    if not contribution_path.exists():
        row["missing_file"] = str(contribution_path.relative_to(PROJECT_ROOT))
        return row

    test_report = load_json(test_path)
    contribution_report = load_json(contribution_path)
    metric_name, metrics = metric_block(test_report)
    per_label_metrics = {
        item["label_name"]: item for item in test_report.get("per_label_metrics", [])
    }
    gaps = contribution_gap(contribution_report)
    baseline = config.get("baseline_comparison", {})
    row.update(
        {
            "status": "ok",
            "metric_mode": metric_name,
            "micro_f1": metrics["recognition_micro_f1"],
            "macro_f1": metrics["recognition_macro_f1"],
            "detection_f1": metrics["detection_f1"],
            "predicted_positive_total": metrics["predicted_positive_total"],
            "macro_f1_delta_vs_side_scale_200": metrics["recognition_macro_f1"]
            - baseline.get("dive_side_scale_200_macro_f1", 0.0),
            "micro_f1_delta_vs_side_scale_200": metrics["recognition_micro_f1"]
            - baseline.get("dive_side_scale_200_micro_f1", 0.0),
            "detection_f1_delta_vs_side_scale_200": metrics["detection_f1"]
            - baseline.get("dive_side_scale_200_detection_f1", 0.0),
            "per_label": per_label_metrics,
            "contribution_gaps": gaps,
        }
    )
    focus = {}
    for label in FOCUS_LABELS:
        focus[label] = {
            "metrics": per_label_metrics.get(label),
            "contribution_gap": gaps.get(label),
        }
    row["focus_labels"] = focus
    row["warnings"] = []
    for label, gap in gaps.items():
        tp_minus_fp = gap.get("tp_minus_fp")
        tp_minus_fn = gap.get("tp_minus_fn")
        if tp_minus_fp is not None and tp_minus_fp <= 0:
            row["warnings"].append(f"{label}: TP final <= FP final")
        if tp_minus_fn is not None and tp_minus_fn <= 0:
            row["warnings"].append(f"{label}: TP final <= FN final")
    return row


def format_float(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_report(output_prefix, report):
    output_prefix = resolve_path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    txt_path = output_prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = [row for row in report["variants"] if row.get("status") == "ok"]
    rows = sorted(rows, key=lambda item: item["macro_f1"], reverse=True)
    lines = [
        "DIVE behavior weight search summary",
        "",
        "variant | micro_f1 | macro_f1 | detection_f1 | macro_delta_vs_side_scale_200 | predicted_positive_total",
        "-" * 118,
    ]
    for row in rows:
        lines.append(
            f"{row['variant']} | {row['micro_f1']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['detection_f1']:.6f} | "
            f"{row['macro_f1_delta_vs_side_scale_200']:.6f} | "
            f"{row['predicted_positive_total']}"
        )

    lines.extend(["", "Focus label diagnostics:", ""])
    for row in rows:
        lines.append(f"[{row['variant']}]")
        for label in sorted(FOCUS_LABELS):
            metrics = row["focus_labels"].get(label, {}).get("metrics") or {}
            gap = row["focus_labels"].get(label, {}).get("contribution_gap") or {}
            lines.append(
                f"- {label}: f1={format_float(metrics.get('f1'))}, "
                f"precision={format_float(metrics.get('precision'))}, "
                f"recall={format_float(metrics.get('recall'))}, "
                f"tp_minus_fp={format_float(gap.get('tp_minus_fp'))}, "
                f"tp_minus_fn={format_float(gap.get('tp_minus_fn'))}, "
                f"tp_final={format_float(gap.get('tp_final'))}, "
                f"fp_final={format_float(gap.get('fp_final'))}, "
                f"fn_final={format_float(gap.get('fn_final'))}"
            )
        if row.get("warnings"):
            lines.append("  warnings:")
            lines.extend(f"  - {warning}" for warning in row["warnings"])
        lines.append("")

    missing = [row for row in report["variants"] if row.get("status") != "ok"]
    if missing:
        lines.append("Missing variants:")
        for row in missing:
            lines.append(f"- {row['variant']}: {row.get('missing_file')}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize DIVE behavior-weight search results."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument(
        "--output_prefix",
        default="results/dive_behavior_weight_search/summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    variants = [collect_variant(path) for path in args.configs]
    report = {
        "selection_metric": "test per-label-threshold macro-F1",
        "baseline": "DIVE side_scale_200 per-label-threshold macro-F1=0.7399",
        "variants": variants,
    }
    write_report(args.output_prefix, report)


if __name__ == "__main__":
    main()
