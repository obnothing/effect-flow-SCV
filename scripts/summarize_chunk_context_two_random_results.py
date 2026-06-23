import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "BJUT_SC01_random": {
        "path": Path(
            "results/train_continued_dive_evm_bert_bjut_random_stride256_context_mil/"
            "test_threshold_calibration_metrics.json"
        ),
        "baselines": {
            "threshold_0_5": {"micro_f1": 0.6522934452322384, "macro_f1": 0.5409677739563126},
            "validation_best_global": {"micro_f1": 0.6554483583875552, "macro_f1": 0.5461364844497139},
            "validation_per_label": {"micro_f1": 0.6515302600291477, "macro_f1": 0.5360547820854075},
        },
    },
    "DIVE_random": {
        "path": Path(
            "results/train_continued_dive_evm_bert_dive_random_stride256_context_mil/"
            "test_threshold_calibration_metrics.json"
        ),
        "baselines": {
            "threshold_0_5": {"micro_f1": 0.8103360054579566, "macro_f1": 0.7126141168953797},
            "validation_best_global": {"micro_f1": 0.810144172311968, "macro_f1": 0.7107865287628011},
            "validation_per_label": {"micro_f1": 0.812446862778439, "macro_f1": 0.709976621068362},
        },
    },
}


def metric_row(metrics):
    return {
        "detection_f1": metrics.get("detection_f1"),
        "recognition_micro_f1": metrics.get("recognition_micro_f1"),
        "recognition_macro_f1": metrics.get("recognition_macro_f1"),
        "predicted_positive_total": metrics.get("predicted_positive_total"),
        "thresholds": metrics.get("thresholds"),
    }


def main():
    rows = {}
    for dataset_name, spec in EXPERIMENTS.items():
        path = PROJECT_ROOT / spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing chunk-context result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        fixed = metric_row(result["threshold_0_5_baseline"])
        best_global = metric_row(result["best_global_threshold_result"])
        per_label = metric_row(result["per_label_threshold_result"])
        modes = {
            "threshold_0_5": fixed,
            "validation_best_global": best_global,
            "validation_per_label": per_label,
        }
        matched_comparisons = {}
        for mode, metrics in modes.items():
            baseline = spec["baselines"][mode]
            matched_comparisons[mode] = {
                "baseline_micro_f1": baseline["micro_f1"],
                "baseline_macro_f1": baseline["macro_f1"],
                "micro_f1_change": (
                    metrics["recognition_micro_f1"] - baseline["micro_f1"]
                ),
                "macro_f1_change": (
                    metrics["recognition_macro_f1"] - baseline["macro_f1"]
                ),
            }
        rows[dataset_name] = {
            "result_path": spec["path"].as_posix(),
            "evaluated_samples": result.get("evaluated_samples"),
            "checkpoint_epoch": result.get("checkpoint_epoch"),
            "modes": modes,
            "matched_baseline_comparisons": matched_comparisons,
            "warnings": result.get("warnings", []),
        }

    report = {
        "status": "warning",
        "model": (
            "DIVE-continued EVM-BERT stride256 masked-mean cache + two-layer "
            "chunk Transformer + label-wise gated-attention MIL + weighted BCE"
        ),
        "experiments": rows,
        "comparison_warning": (
            "Both experiments use prior random splits and are non-strict/transductive. "
            "Compare each dataset only with its own no-context random-split baseline."
        ),
        "reporting_rule": (
            "Do not select a threshold mode by test performance. Report fixed, global, "
            "and per-label modes separately against their matched no-context baselines."
        ),
    }
    report_dir = PROJECT_ROOT / "data/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "chunk_context_two_random_results_summary.json"
    txt_path = report_dir / "chunk_context_two_random_results_summary.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["Two-dataset random-split chunk-context MIL summary", "", report["model"], ""]
    lines.append(
        "dataset | mode | detection_f1 | micro_f1 | macro_f1 | predicted_positive_total"
    )
    for dataset_name, row in rows.items():
        for mode, metrics in row["modes"].items():
            lines.append(
                f"{dataset_name} | {mode} | {metrics['detection_f1']} | "
                f"{metrics['recognition_micro_f1']} | "
                f"{metrics['recognition_macro_f1']} | "
                f"{metrics['predicted_positive_total']}"
            )
            comparison = row["matched_baseline_comparisons"][mode]
            lines.append(
                f"{dataset_name} | {mode} | matched_delta | "
                f"micro={comparison['micro_f1_change']:+.6f} | "
                f"macro={comparison['macro_f1_change']:+.6f}"
            )
    lines.extend(
        [
            "",
            f"warning: {report['comparison_warning']}",
            f"reporting_rule: {report['reporting_rule']}",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
