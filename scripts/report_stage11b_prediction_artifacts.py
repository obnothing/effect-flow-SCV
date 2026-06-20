import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

EXPERIMENTS = [
    {
        "name": "codebert_weighted",
        "result_dir": "results/train_mlsmote_codebert_weighted",
        "test_path": "data/processed/BJUT_SC01/test.jsonl",
        "prediction_file": "test_predictions.jsonl",
    },
    {
        "name": "evm_bert_first512_weighted",
        "result_dir": "results/train_evm_bert_weighted",
        "test_path": "data/processed/BJUT_SC01/test.jsonl",
        "prediction_file": "test_predictions.jsonl",
    },
    {
        "name": "strict_grouped_stride256_mil",
        "result_dir": "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted",
        "test_path": "data/processed/BJUT_SC01/test.jsonl",
        "prediction_file": "test_predictions.jsonl",
    },
    {
        "name": "random_split_stride256_mil",
        "result_dir": "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split",
        "test_path": "data/processed/BJUT_SC01_random_split/test.jsonl",
        "prediction_file": "test_predictions.jsonl",
    },
]


def count_jsonl(path):
    path = PROJECT_ROOT / path
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def scan_prediction_file(path):
    path = PROJECT_ROOT / path
    result = {
        "exists": path.exists(),
        "sample_count": 0,
        "multi_true_dim_ok": True,
        "multi_prob_dim_ok": True,
        "multi_pred_dim_ok": True,
        "has_nan_or_inf_prob": False,
        "threshold_modes": {},
        "threshold_examples": [],
        "samples_with_chunk_fields": 0,
        "warnings": [],
    }
    if not path.exists():
        result["warnings"].append("prediction file missing")
        return result
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result["sample_count"] += 1
            multi_true = item.get("multi_true", [])
            multi_prob = item.get("multi_prob", [])
            multi_pred = item.get("multi_pred", [])
            result["multi_true_dim_ok"] = result["multi_true_dim_ok"] and len(multi_true) == 10
            result["multi_prob_dim_ok"] = result["multi_prob_dim_ok"] and len(multi_prob) == 10
            result["multi_pred_dim_ok"] = result["multi_pred_dim_ok"] and len(multi_pred) == 10
            for value in multi_prob + [item.get("binary_prob", 0.0)]:
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    result["has_nan_or_inf_prob"] = True
            mode = item.get("threshold_mode", "unknown")
            result["threshold_modes"][mode] = result["threshold_modes"].get(mode, 0) + 1
            if len(result["threshold_examples"]) < 3:
                result["threshold_examples"].append(item.get("thresholds"))
            if "num_chunks_kept" in item:
                result["samples_with_chunk_fields"] += 1
    return result


def load_completed_metrics():
    path = REPORT_DIR / "metric_recomputation_summary_completed.csv"
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["experiment_name"]] = row
    return rows


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_by_name = load_completed_metrics()
    rows = []
    for spec in EXPERIMENTS:
        pred_rel = f"{spec['result_dir']}/{spec['prediction_file']}"
        test_count = count_jsonl(spec["test_path"])
        pred_scan = scan_prediction_file(pred_rel)
        metrics = metrics_by_name.get(spec["name"]) or metrics_by_name.get(
            {
                "codebert_weighted": "codebert_weighted",
                "evm_bert_first512_weighted": "evm_bert_first512_weighted",
                "strict_grouped_stride256_mil": "strict_grouped_stride256_mil_threshold_0_5",
                "random_split_stride256_mil": "random_split_stride256_mil_best_global_0_55",
            }[spec["name"]],
            {},
        )
        row = {
            "experiment_name": spec["name"],
            "prediction_path": pred_rel,
            "prediction_exists": pred_scan["exists"],
            "prediction_samples": pred_scan["sample_count"],
            "test_jsonl_samples": test_count,
            "sample_count_matches": (
                pred_scan["sample_count"] == test_count if test_count is not None else False
            ),
            "has_nan_or_inf_prob": pred_scan["has_nan_or_inf_prob"],
            "multi_true_dim_ok": pred_scan["multi_true_dim_ok"],
            "multi_prob_dim_ok": pred_scan["multi_prob_dim_ok"],
            "multi_pred_dim_ok": pred_scan["multi_pred_dim_ok"],
            "threshold_modes": pred_scan["threshold_modes"],
            "threshold_examples": pred_scan["threshold_examples"],
            "samples_with_chunk_fields": pred_scan["samples_with_chunk_fields"],
            "samples_f1": metrics.get("recognition_samples_f1"),
            "subset_accuracy": metrics.get("subset_accuracy"),
            "hamming_loss": metrics.get("hamming_loss"),
            "recognition_micro_f1": metrics.get("recognition_micro_f1"),
            "recognition_macro_f1": metrics.get("recognition_macro_f1"),
            "detection_f1": metrics.get("detection_f1"),
            "warnings": pred_scan["warnings"],
        }
        if not row["sample_count_matches"]:
            row["warnings"].append("prediction sample count does not match test jsonl")
        rows.append(row)

    report = {
        "status": "ok" if all(row["prediction_exists"] for row in rows) else "warning",
        "experiments": rows,
    }
    json_path = REPORT_DIR / "stage11b_prediction_artifact_report.json"
    txt_path = REPORT_DIR / "stage11b_prediction_artifact_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["Stage 11B prediction artifact report", ""]
    lines.append(
        "experiment | exists | pred_samples | test_samples | count_match | samples_f1 | subset_accuracy | hamming_loss | warnings"
    )
    lines.append("-" * 150)
    for row in rows:
        lines.append(
            f"{row['experiment_name']} | {row['prediction_exists']} | "
            f"{row['prediction_samples']} | {row['test_jsonl_samples']} | "
            f"{row['sample_count_matches']} | {row['samples_f1']} | "
            f"{row['subset_accuracy']} | {row['hamming_loss']} | "
            f"{'; '.join(row['warnings'])}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
