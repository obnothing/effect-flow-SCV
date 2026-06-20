import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "BJUT_SC01_random": Path(
        "results/train_continued_dive_evm_bert_bjut_random_stride256_mil/"
        "test_threshold_calibration_metrics.json"
    ),
    "DIVE_random": Path(
        "results/train_continued_dive_evm_bert_dive_random_stride256_mil/"
        "test_threshold_calibration_metrics.json"
    ),
}


def metric_row(metrics):
    return {
        "detection_precision": metrics.get("detection_precision"),
        "detection_recall": metrics.get("detection_recall"),
        "detection_f1": metrics.get("detection_f1"),
        "recognition_micro_precision": metrics.get("recognition_micro_precision"),
        "recognition_micro_recall": metrics.get("recognition_micro_recall"),
        "recognition_micro_f1": metrics.get("recognition_micro_f1"),
        "recognition_macro_precision": metrics.get("recognition_macro_precision"),
        "recognition_macro_recall": metrics.get("recognition_macro_recall"),
        "recognition_macro_f1": metrics.get("recognition_macro_f1"),
        "predicted_positive_total": metrics.get("predicted_positive_total"),
        "thresholds": metrics.get("thresholds"),
    }


def main():
    rows = {}
    for dataset_name, relative_path in EXPERIMENTS.items():
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 12C result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        rows[dataset_name] = {
            "result_path": relative_path.as_posix(),
            "evaluated_samples": result.get("evaluated_samples"),
            "checkpoint_epoch": result.get("checkpoint_epoch"),
            "is_transductive_pretraining": result.get("is_transductive_pretraining"),
            "threshold_0_5": metric_row(result["threshold_0_5_baseline"]),
            "validation_best_global": metric_row(result["best_global_threshold_result"]),
            "validation_per_label": metric_row(result["per_label_threshold_result"]),
            "per_label_metrics": result.get("per_label_metrics"),
            "warnings": result.get("warnings", []),
        }

    report = {
        "status": "warning",
        "model": (
            "DIVE-train-only continued EVM-BERT + stride256 masked-mean features + "
            "label-wise gated-attention MIL + weighted BCE"
        ),
        "experiments": rows,
        "comparison_warning": (
            "BJUT and DIVE have different label spaces, class balance, sample counts, "
            "and random-split pretraining exposure. Compare each dataset against its "
            "own baseline; do not interpret cross-dataset F1 differences as model superiority."
        ),
        "strictness": {
            "BJUT_SC01_random": (
                "non-strict: random opcode-hash overlap and BJUT full-corpus MLM"
            ),
            "DIVE_random": (
                "non-strict: random opcode-hash overlap and approximately 70% of random "
                "test contracts were present in DIVE strict-train continued MLM"
            ),
        },
    }
    report_dir = PROJECT_ROOT / "data/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "stage12c_two_random_full_results_summary.json"
    txt_path = report_dir / "stage12c_two_random_full_results_summary.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["Stage 12C two-dataset random-split full-model summary", ""]
    lines.append(report["model"])
    lines.append("")
    lines.append(
        "dataset | mode | detection_f1 | micro_f1 | macro_f1 | predicted_positive_total"
    )
    for dataset_name, row in rows.items():
        for mode in ["threshold_0_5", "validation_best_global", "validation_per_label"]:
            metrics = row[mode]
            lines.append(
                f"{dataset_name} | {mode} | {metrics['detection_f1']} | "
                f"{metrics['recognition_micro_f1']} | {metrics['recognition_macro_f1']} | "
                f"{metrics['predicted_positive_total']}"
            )
    lines.extend(["", f"warning: {report['comparison_warning']}"])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
