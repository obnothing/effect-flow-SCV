import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "BJUT_SC01_random": {
        "path": Path(
            "results/train_continued_dive_evm_bert_bjut_random_stride256_context_mil/"
            "test_threshold_calibration_metrics.json"
        ),
        "baseline_micro_f1": 0.6554483583875552,
        "baseline_macro_f1": 0.5461364844497139,
    },
    "DIVE_random": {
        "path": Path(
            "results/train_continued_dive_evm_bert_dive_random_stride256_context_mil/"
            "test_threshold_calibration_metrics.json"
        ),
        "baseline_micro_f1": 0.8103360054579566,
        "baseline_macro_f1": 0.7126141168953797,
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
        best_mode, best_metrics = max(
            modes.items(),
            key=lambda item: item[1]["recognition_macro_f1"],
        )
        rows[dataset_name] = {
            "result_path": spec["path"].as_posix(),
            "evaluated_samples": result.get("evaluated_samples"),
            "checkpoint_epoch": result.get("checkpoint_epoch"),
            "modes": modes,
            "best_test_macro_mode": best_mode,
            "best_test_metrics": best_metrics,
            "baseline_micro_f1": spec["baseline_micro_f1"],
            "baseline_macro_f1": spec["baseline_macro_f1"],
            "micro_f1_change": (
                best_metrics["recognition_micro_f1"] - spec["baseline_micro_f1"]
            ),
            "macro_f1_change": (
                best_metrics["recognition_macro_f1"] - spec["baseline_macro_f1"]
            ),
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
        lines.append(
            f"{dataset_name} best macro mode={row['best_test_macro_mode']} "
            f"micro_change={row['micro_f1_change']:+.6f} "
            f"macro_change={row['macro_f1_change']:+.6f}"
        )
    lines.extend(["", f"warning: {report['comparison_warning']}"])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
