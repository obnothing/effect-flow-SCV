import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_NAMES = [
    "Reentrancy",
    "Contract contains unknown address",
    "Integer overflow or underflow",
    "Timestamp dependence",
    "DoS with failed call",
    "Assert violation",
    "Unchecked call return value",
    "Unsafe send",
    "Multiplication after division",
    "Extra gas consumption",
]


DEFAULT_EXPERIMENTS = [
    {
        "experiment_name": "strict_grouped_stride256_mil_threshold_0_5",
        "split_type": "opcode_hash_grouped",
        "leakage_status": "ok",
        "aggregate_path": "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/test_threshold_calibration_metrics.json",
        "aggregate_key": "threshold_0_5_baseline",
        "prediction_path": "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/test_predictions_threshold_0_5.jsonl",
    },
    {
        "experiment_name": "random_split_stride256_mil_best_global_0_55",
        "split_type": "random",
        "leakage_status": "warning",
        "aggregate_path": "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split/test_threshold_calibration_metrics.txt",
        "aggregate_key": "best_global_threshold_result",
        "prediction_path": "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split/test_predictions.jsonl",
    },
    {
        "experiment_name": "evm_bert_first512_weighted",
        "split_type": "opcode_hash_grouped",
        "leakage_status": "ok_transductive_pretraining",
        "aggregate_path": "results/train_evm_bert_weighted/test_best_macro_metrics.json",
        "aggregate_key": None,
        "prediction_path": "results/train_evm_bert_weighted/test_predictions.jsonl",
    },
    {
        "experiment_name": "codebert_weighted",
        "split_type": "opcode_hash_grouped",
        "leakage_status": "ok",
        "aggregate_path": "results/train_mlsmote_codebert_weighted/test_best_macro_metrics.json",
        "aggregate_key": None,
        "prediction_path": "results/train_mlsmote_codebert_weighted/test_predictions.jsonl",
    },
]


SUMMARY_FIELDS = [
    "experiment_name",
    "split_type",
    "leakage_status",
    "metric_source",
    "threshold_mode",
    "detection_f1",
    "recognition_micro_f1",
    "recognition_macro_f1",
    "recognition_weighted_f1",
    "recognition_samples_f1",
    "subset_accuracy",
    "hamming_loss",
    "per_label_mean_f1",
    "positive_label_only_mean_f1",
    "macro_f1_excluding_support_lt_20",
    "macro_f1_excluding_support_lt_50",
    "macro_f1_excluding_support_lt_100",
    "paper_like_avg_f1_label_mean",
    "paper_like_avg_f1_micro",
    "paper_like_avg_f1_weighted",
    "paper_like_detection_recognition_avg",
    "paper_like_detection_recognition_micro_avg",
    "paper_like_exclude_rare_macro",
    "paper_like_random_split_macro",
    "predicted_positive_total",
    "unavailable_fields",
    "notes",
]


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def precision_recall_f1(tp, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def multilabel_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    micro_p, micro_r, micro_f1 = precision_recall_f1(tp, fp, fn)

    per_label = []
    for idx in range(y_true.shape[1]):
        label_true = y_true[:, idx]
        label_pred = y_pred[:, idx]
        label_tp = int(((label_pred == 1) & (label_true == 1)).sum())
        label_fp = int(((label_pred == 1) & (label_true == 0)).sum())
        label_fn = int(((label_pred == 0) & (label_true == 1)).sum())
        label_tn = int(((label_pred == 0) & (label_true == 0)).sum())
        precision, recall, f1 = precision_recall_f1(label_tp, label_fp, label_fn)
        support = int(label_true.sum())
        per_label.append(
            {
                "label_id": idx,
                "label_name": LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else f"label_{idx}",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted_positive_count": int(label_pred.sum()),
                "true_positive_count": support,
                "tp": label_tp,
                "fp": label_fp,
                "fn": label_fn,
                "tn": label_tn,
            }
        )

    supports = np.asarray([row["support"] for row in per_label], dtype=float)
    f1s = np.asarray([row["f1"] for row in per_label], dtype=float)
    precisions = np.asarray([row["precision"] for row in per_label], dtype=float)
    recalls = np.asarray([row["recall"] for row in per_label], dtype=float)
    total_support = float(supports.sum())

    sample_scores = []
    for true_row, pred_row in zip(y_true, y_pred):
        intersection = int(((true_row == 1) & (pred_row == 1)).sum())
        pred_count = int(pred_row.sum())
        true_count = int(true_row.sum())
        sample_p = safe_divide(intersection, pred_count)
        sample_r = safe_divide(intersection, true_count)
        sample_f1 = safe_divide(2 * sample_p * sample_r, sample_p + sample_r)
        sample_scores.append((sample_p, sample_r, sample_f1))
    sample_scores = np.asarray(sample_scores, dtype=float)

    def macro_excluding(min_support):
        keep = supports >= min_support
        return float(f1s[keep].mean()) if keep.any() else None

    return {
        "recognition_micro_precision": micro_p,
        "recognition_micro_recall": micro_r,
        "recognition_micro_f1": micro_f1,
        "recognition_macro_precision": float(precisions.mean()),
        "recognition_macro_recall": float(recalls.mean()),
        "recognition_macro_f1": float(f1s.mean()),
        "recognition_weighted_f1": (
            float((f1s * supports).sum() / total_support) if total_support else 0.0
        ),
        "recognition_samples_precision": float(sample_scores[:, 0].mean()),
        "recognition_samples_recall": float(sample_scores[:, 1].mean()),
        "recognition_samples_f1": float(sample_scores[:, 2].mean()),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float((y_true != y_pred).mean()),
        "per_label": per_label,
        "per_label_mean_f1": float(f1s.mean()),
        "per_label_mean_precision": float(precisions.mean()),
        "per_label_mean_recall": float(recalls.mean()),
        "positive_label_only_mean_f1": float(f1s[supports > 0].mean())
        if (supports > 0).any()
        else None,
        "macro_f1_excluding_support_lt_20": macro_excluding(20),
        "macro_f1_excluding_support_lt_50": macro_excluding(50),
        "macro_f1_excluding_support_lt_100": macro_excluding(100),
        "predicted_positive_total": int(y_pred.sum()),
    }


def load_predictions(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No prediction rows found: {path}")
    return rows


def metrics_from_predictions(path):
    rows = load_predictions(path)
    y_binary_true = np.asarray([row["binary_true"] for row in rows], dtype=int)
    y_binary_pred = np.asarray([row["binary_pred"] for row in rows], dtype=int)
    y_true = np.asarray([row["multi_true"] for row in rows], dtype=int)
    y_pred = np.asarray([row["multi_pred"] for row in rows], dtype=int)
    detection = binary_metrics(y_binary_true, y_binary_pred)
    recognition = multilabel_metrics(y_true, y_pred)
    return {
        "metric_source": "raw_predictions",
        "evaluated_samples": len(rows),
        "threshold_mode": rows[0].get("threshold_mode"),
        "detection_f1": detection["f1"],
        **recognition,
        "unavailable_fields": [],
        "notes": [],
    }


def load_json_or_txt_report(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if value.startswith("{") or value.startswith("["):
            try:
                data[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                data[key] = value
        else:
            data[key] = value
    return data


def nested_report_value(report, key):
    if not key:
        return report
    value = report
    for part in key.split("."):
        value = value[part]
    return value


def per_label_from_aggregate(report):
    rows = report.get("per_label") or report.get("per_label_metrics")
    if not rows:
        return None
    normalized = []
    for idx, row in enumerate(rows):
        normalized.append(
            {
                "label_id": int(row.get("label_id", idx)),
                "label_name": row.get(
                    "label_name",
                    LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else f"label_{idx}",
                ),
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "f1": float(row.get("f1", 0.0)),
                "support": int(row.get("support", row.get("true_positive_count", 0))),
                "predicted_positive_count": int(
                    row.get("predicted_positive_count", 0)
                ),
                "true_positive_count": int(
                    row.get("true_positive_count", row.get("support", 0))
                ),
            }
        )
    return normalized


def metrics_from_aggregate(path, key=None):
    report = load_json_or_txt_report(path)
    selected = nested_report_value(report, key)
    per_label = per_label_from_aggregate(selected)
    unavailable = ["recognition_samples_f1", "subset_accuracy", "hamming_loss"]
    notes = ["Loaded from aggregate report; raw y_prob/y_pred were not available."]
    metric = {
        "metric_source": "aggregate_report",
        "threshold_mode": selected.get("threshold_mode"),
        "detection_f1": selected.get("detection_f1"),
        "recognition_micro_precision": selected.get("recognition_micro_precision"),
        "recognition_micro_recall": selected.get("recognition_micro_recall"),
        "recognition_micro_f1": selected.get("recognition_micro_f1"),
        "recognition_macro_precision": selected.get("recognition_macro_precision"),
        "recognition_macro_recall": selected.get("recognition_macro_recall"),
        "recognition_macro_f1": selected.get("recognition_macro_f1"),
        "predicted_positive_total": selected.get("predicted_positive_total"),
        "per_label": per_label,
        "recognition_samples_f1": None,
        "subset_accuracy": None,
        "hamming_loss": None,
        "unavailable_fields": unavailable,
        "notes": notes,
    }
    if per_label:
        supports = np.asarray([row["support"] for row in per_label], dtype=float)
        f1s = np.asarray([row["f1"] for row in per_label], dtype=float)
        total_support = float(supports.sum())
        metric["per_label_mean_f1"] = float(f1s.mean())
        metric["positive_label_only_mean_f1"] = (
            float(f1s[supports > 0].mean()) if (supports > 0).any() else None
        )
        metric["recognition_weighted_f1"] = (
            float((f1s * supports).sum() / total_support) if total_support else None
        )
        for cutoff in [20, 50, 100]:
            keep = supports >= cutoff
            metric[f"macro_f1_excluding_support_lt_{cutoff}"] = (
                float(f1s[keep].mean()) if keep.any() else None
            )
    else:
        metric["per_label_mean_f1"] = metric.get("recognition_macro_f1")
        metric["positive_label_only_mean_f1"] = None
        metric["recognition_weighted_f1"] = None
        metric["macro_f1_excluding_support_lt_20"] = None
        metric["macro_f1_excluding_support_lt_50"] = None
        metric["macro_f1_excluding_support_lt_100"] = None
        unavailable.extend(
            [
                "recognition_weighted_f1",
                "macro_f1_excluding_support_lt_20",
                "macro_f1_excluding_support_lt_50",
                "macro_f1_excluding_support_lt_100",
            ]
        )
        notes.append("Per-label F1 at this threshold is missing in the aggregate report.")
    return metric


def paper_like_fields(row):
    macro = row.get("recognition_macro_f1")
    micro = row.get("recognition_micro_f1")
    weighted = row.get("recognition_weighted_f1")
    detection = row.get("detection_f1")
    exclude_rare = row.get("macro_f1_excluding_support_lt_50")
    return {
        "paper_like_avg_f1_label_mean": row.get("per_label_mean_f1", macro),
        "paper_like_avg_f1_micro": micro,
        "paper_like_avg_f1_weighted": weighted,
        "paper_like_detection_recognition_avg": (
            (detection + macro) / 2
            if detection is not None and macro is not None
            else None
        ),
        "paper_like_detection_recognition_micro_avg": (
            (detection + micro) / 2
            if detection is not None and micro is not None
            else None
        ),
        "paper_like_exclude_rare_macro": exclude_rare,
    }


def summarize_experiment(spec):
    prediction_path = PROJECT_ROOT / spec.get("prediction_path", "")
    if prediction_path.exists():
        metrics = metrics_from_predictions(prediction_path)
    else:
        metrics = metrics_from_aggregate(
            PROJECT_ROOT / spec["aggregate_path"],
            key=spec.get("aggregate_key"),
        )
        metrics["notes"].append(
            f"Prediction file not found: {spec.get('prediction_path')}"
        )
    row = {
        "experiment_name": spec["experiment_name"],
        "split_type": spec["split_type"],
        "leakage_status": spec["leakage_status"],
        **metrics,
    }
    row.update(paper_like_fields(row))
    row["paper_like_random_split_macro"] = (
        row.get("recognition_macro_f1") if spec["split_type"] == "random" else None
    )
    row["unavailable_fields"] = sorted(set(row.get("unavailable_fields", [])))
    return row


def write_outputs(rows, output_dir, output_stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_stem}.json"
    txt_path = output_dir / f"{output_stem}.txt"
    csv_path = output_dir / f"{output_stem}.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["unavailable_fields"] = ";".join(row.get("unavailable_fields", []))
            csv_row["notes"] = " | ".join(row.get("notes", []))
            writer.writerow(csv_row)

    lines = ["Metric recomputation summary", ""]
    lines.append(
        "experiment_name | split_type | leakage_status | source | detection_f1 | micro_f1 | macro_f1 | weighted_f1 | samples_f1 | paper_label_mean | det_rec_avg | predicted_positive_total"
    )
    lines.append("-" * 180)
    for row in rows:
        lines.append(
            f"{row['experiment_name']} | {row['split_type']} | {row['leakage_status']} | "
            f"{row.get('metric_source')} | {fmt(row.get('detection_f1'))} | "
            f"{fmt(row.get('recognition_micro_f1'))} | "
            f"{fmt(row.get('recognition_macro_f1'))} | "
            f"{fmt(row.get('recognition_weighted_f1'))} | "
            f"{fmt(row.get('recognition_samples_f1'))} | "
            f"{fmt(row.get('paper_like_avg_f1_label_mean'))} | "
            f"{fmt(row.get('paper_like_detection_recognition_avg'))} | "
            f"{row.get('predicted_positive_total')}"
        )
        if row.get("unavailable_fields"):
            lines.append(
                f"  unavailable: {', '.join(row['unavailable_fields'])}"
            )
        if row.get("notes"):
            for note in row["notes"]:
                lines.append(f"  note: {note}")
    lines.extend(
        [
            "",
            "Why our results may differ from the paper",
            "",
            "- The paper defines Precision/Recall/F1 formulas and states that Pre, Rec and F1 are averages over 10 vulnerabilities, but it does not explicitly specify whether this is the mean of per-label F1 or F1 computed from averaged precision/recall in every table.",
            "- The paper does not explicitly specify the decision threshold used for multi-label recognition.",
            "- The paper does not explicitly specify the BJUT SC01 train/test split ratio, validation split, random seed, or opcode-hash duplicate control.",
            "- Random split contains opcode-hash overlap and can inflate test scores; our random split reaches higher F1 than the strict grouped split but remains non-strict.",
            "- Average F1 in the paper is probably not strict multi-label micro-F1; Table XI explicitly says Pre, Rec and F1 are averages in 10 vulnerabilities.",
            "- BJUT label mapping is uncertain, especially the Unchecked call return value / Overpowered role compatibility mapping.",
            "- Small labels have low support, so macro-F1 is very sensitive to false positives and false negatives.",
            "- Opcode-only representation may miss source-level semantics and data/control-flow evidence needed by some vulnerability types.",
            "- Several fields, including samples-F1, subset accuracy, and hamming loss, require raw per-sample predictions; rerun evaluation with --save_predictions to compute them exactly.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {csv_path}")


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute paper-like and standard metrics from saved predictions or aggregate reports."
    )
    parser.add_argument("--predictions", default=None, help="Optional test_predictions.jsonl.")
    parser.add_argument("--experiment_name", default="custom_predictions")
    parser.add_argument("--split_type", default="unknown")
    parser.add_argument("--leakage_status", default="unknown")
    parser.add_argument("--output_dir", default="data/reports")
    parser.add_argument("--output_stem", default="metric_recomputation_summary")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.predictions:
        spec = {
            "experiment_name": args.experiment_name,
            "split_type": args.split_type,
            "leakage_status": args.leakage_status,
            "prediction_path": args.predictions,
            "aggregate_path": args.predictions,
        }
        rows = [summarize_experiment(spec)]
    else:
        rows = [summarize_experiment(spec) for spec in DEFAULT_EXPERIMENTS]
    write_outputs(rows, PROJECT_ROOT / args.output_dir, args.output_stem)


if __name__ == "__main__":
    main()
