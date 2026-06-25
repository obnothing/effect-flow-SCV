import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data/reports"


COMPARISONS = [
    {
        "name": "BJUT first512",
        "dataset": "BJUT",
        "split_type": "opcode_hash_grouped",
        "downstream_model": "first512_BiGRU",
        "old_encoder": "BJUT full-corpus EVM-BERT",
        "new_encoder": "effect-flow continued EVM-BERT",
        "old_prediction": "results/train_evm_bert_weighted/test_predictions.jsonl",
        "new_prediction": "results/train_bjut_effect_flow_first512_weighted/test_predictions.jsonl",
    },
    {
        "name": "DIVE first512",
        "dataset": "DIVE",
        "split_type": "random_sample_split",
        "downstream_model": "first512_BiGRU",
        "old_encoder": "previous DIVE first512 EVM-BERT",
        "new_encoder": "effect-flow continued EVM-BERT",
        "old_prediction": "results/train_dive_evm_bert_weighted/test_predictions.jsonl",
        "new_prediction": "results/train_dive_effect_flow_first512_weighted/test_predictions.jsonl",
    },
    {
        "name": "BJUT stride256 chunk-context MIL",
        "dataset": "BJUT",
        "split_type": "random_sample_split",
        "downstream_model": "stride256_chunk_context_labelwise_gated_MIL",
        "old_encoder": "DIVE-continued EVM-BERT",
        "new_encoder": "effect-flow continued EVM-BERT",
        "old_prediction": "results/train_continued_dive_evm_bert_bjut_random_stride256_context_mil/test_predictions.jsonl",
        "new_prediction": "results/train_bjut_effect_flow_chunk_mil_stride256_labelattn_ctx_weighted/test_predictions.jsonl",
    },
    {
        "name": "DIVE stride256 chunk-context MIL",
        "dataset": "DIVE",
        "split_type": "random_sample_split",
        "downstream_model": "stride256_chunk_context_labelwise_gated_MIL",
        "old_encoder": "DIVE-continued EVM-BERT",
        "new_encoder": "effect-flow continued EVM-BERT",
        "old_prediction": "results/train_continued_dive_evm_bert_dive_random_stride256_context_mil/test_predictions.jsonl",
        "new_prediction": "results/train_dive_effect_flow_chunk_mil_stride256_labelattn_ctx_weighted/test_predictions.jsonl",
    },
]


VARIANT_SUFFIXES = [
    ("fixed_0_5", "test_predictions_threshold_0_5.jsonl"),
    ("valid_best_global", "test_predictions.jsonl"),
    ("valid_per_label", "test_predictions_per_label.jsonl"),
]


def resolve(path):
    return PROJECT_ROOT / path


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def precision_recall_f1(tp, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def load_predictions(path):
    path = resolve(path)
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def arrays_from_rows(rows):
    ids = [row.get("id", str(idx)) for idx, row in enumerate(rows)]
    y_binary = np.asarray([row["binary_true"] for row in rows], dtype=int)
    p_binary = np.asarray([row["binary_pred"] for row in rows], dtype=int)
    y = np.asarray([row["multi_true"] for row in rows], dtype=int)
    p = np.asarray([row["multi_pred"] for row in rows], dtype=int)
    return ids, y_binary, p_binary, y, p


def binary_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "detection_accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
    }


def multilabel_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    micro_p, micro_r, micro_f1 = precision_recall_f1(tp, fp, fn)
    per_label = []
    for label_id in range(y_true.shape[1]):
        true_label = y_true[:, label_id]
        pred_label = y_pred[:, label_id]
        ltp = int(((pred_label == 1) & (true_label == 1)).sum())
        lfp = int(((pred_label == 1) & (true_label == 0)).sum())
        lfn = int(((pred_label == 0) & (true_label == 1)).sum())
        precision, recall, f1 = precision_recall_f1(ltp, lfp, lfn)
        per_label.append(
            {
                "label_id": label_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(true_label.sum()),
                "predicted_positive_count": int(pred_label.sum()),
            }
        )
    sample_f1 = []
    for true_row, pred_row in zip(y_true, y_pred):
        intersection = int(((true_row == 1) & (pred_row == 1)).sum())
        pred_count = int(pred_row.sum())
        true_count = int(true_row.sum())
        sample_p = safe_divide(intersection, pred_count)
        sample_r = safe_divide(intersection, true_count)
        sample_f1.append(safe_divide(2 * sample_p * sample_r, sample_p + sample_r))
    return {
        "recognition_micro_precision": micro_p,
        "recognition_micro_recall": micro_r,
        "recognition_micro_f1": micro_f1,
        "recognition_macro_precision": float(np.mean([r["precision"] for r in per_label])),
        "recognition_macro_recall": float(np.mean([r["recall"] for r in per_label])),
        "recognition_macro_f1": float(np.mean([r["f1"] for r in per_label])),
        "recognition_samples_f1": float(np.mean(sample_f1)) if sample_f1 else 0.0,
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float((y_true != y_pred).mean()),
        "predicted_positive_total": int(y_pred.sum()),
        "per_label": per_label,
    }


def metrics_from_prediction_path(path):
    rows = load_predictions(path)
    if rows is None:
        return None
    _, y_binary, p_binary, y, p = arrays_from_rows(rows)
    metrics = binary_metrics(y_binary, p_binary)
    metrics.update(multilabel_metrics(y, p))
    metrics["evaluated_samples"] = len(rows)
    metrics["threshold_mode"] = rows[0].get("threshold_mode")
    return metrics


def summarize_pair(spec):
    old_metrics = metrics_from_prediction_path(spec["old_prediction"])
    new_metrics = metrics_from_prediction_path(spec["new_prediction"])
    row = {
        "comparison": spec["name"],
        "dataset": spec["dataset"],
        "split_type": spec["split_type"],
        "downstream_model": spec["downstream_model"],
        "old_encoder": spec["old_encoder"],
        "new_encoder": spec["new_encoder"],
        "old_prediction": spec["old_prediction"],
        "new_prediction": spec["new_prediction"],
        "old_status": "ok" if old_metrics else "missing",
        "new_status": "ok" if new_metrics else "missing",
    }
    for prefix, metrics in (("old", old_metrics), ("new", new_metrics)):
        if not metrics:
            continue
        for key in [
            "threshold_mode",
            "evaluated_samples",
            "detection_f1",
            "recognition_micro_f1",
            "recognition_macro_f1",
            "recognition_macro_precision",
            "recognition_macro_recall",
            "recognition_samples_f1",
            "subset_accuracy",
            "hamming_loss",
            "predicted_positive_total",
        ]:
            row[f"{prefix}_{key}"] = metrics.get(key)
        row[f"{prefix}_per_label_f1"] = [
            item["f1"] for item in metrics["per_label"]
        ]
    if old_metrics and new_metrics:
        row["delta_micro_f1"] = (
            new_metrics["recognition_micro_f1"] - old_metrics["recognition_micro_f1"]
        )
        row["delta_macro_f1"] = (
            new_metrics["recognition_macro_f1"] - old_metrics["recognition_macro_f1"]
        )
        row["delta_detection_f1"] = (
            new_metrics["detection_f1"] - old_metrics["detection_f1"]
        )
        row["per_label_f1_delta"] = [
            new - old
            for old, new in zip(row["old_per_label_f1"], row["new_per_label_f1"])
        ]
    return row


def metric_triplet(rows, indices):
    subset = [rows[i] for i in indices]
    _, yb, pb, y, p = arrays_from_rows(subset)
    det = binary_metrics(yb, pb)["detection_f1"]
    rec = multilabel_metrics(y, p)
    return rec["recognition_micro_f1"], rec["recognition_macro_f1"], det


def paired_bootstrap(spec, n_bootstrap=1000, seed=42):
    old_rows = load_predictions(spec["old_prediction"])
    new_rows = load_predictions(spec["new_prediction"])
    if old_rows is None or new_rows is None:
        return {
            "comparison": spec["name"],
            "status": "missing_predictions",
        }
    old_ids = [row.get("id", str(idx)) for idx, row in enumerate(old_rows)]
    new_ids = [row.get("id", str(idx)) for idx, row in enumerate(new_rows)]
    if old_ids != new_ids:
        return {
            "comparison": spec["name"],
            "status": "id_mismatch",
            "old_samples": len(old_rows),
            "new_samples": len(new_rows),
        }
    rng = np.random.default_rng(seed)
    n = len(old_rows)
    deltas = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        old_micro, old_macro, old_det = metric_triplet(old_rows, indices)
        new_micro, new_macro, new_det = metric_triplet(new_rows, indices)
        deltas.append([new_micro - old_micro, new_macro - old_macro, new_det - old_det])
    values = np.asarray(deltas, dtype=float)
    keys = ["micro_f1", "macro_f1", "detection_f1"]
    result = {
        "comparison": spec["name"],
        "status": "ok",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "samples": n,
    }
    for idx, key in enumerate(keys):
        lower, upper = np.percentile(values[:, idx], [2.5, 97.5])
        mean = float(values[:, idx].mean())
        result[f"delta_{key}_mean"] = mean
        result[f"delta_{key}_ci95"] = [float(lower), float(upper)]
        result[f"delta_{key}_ci_crosses_zero"] = bool(lower <= 0 <= upper)
    return result


def collect_variant_rows():
    rows = []
    for spec in COMPARISONS:
        old_dir = str(Path(spec["old_prediction"]).parent)
        new_dir = str(Path(spec["new_prediction"]).parent)
        for side, encoder, directory in (
            ("old", spec["old_encoder"], old_dir),
            ("new", spec["new_encoder"], new_dir),
        ):
            for protocol, suffix in VARIANT_SUFFIXES:
                rel = str(Path(directory) / suffix).replace("\\", "/")
                metrics = metrics_from_prediction_path(rel)
                if not metrics:
                    continue
                rows.append(
                    {
                        "comparison": spec["name"],
                        "dataset": spec["dataset"],
                        "split_type": spec["split_type"],
                        "encoder_side": side,
                        "encoder": encoder,
                        "downstream_model": spec["downstream_model"],
                        "threshold_protocol": protocol,
                        "prediction_path": rel,
                        "detection_f1": metrics["detection_f1"],
                        "micro_f1": metrics["recognition_micro_f1"],
                        "macro_f1": metrics["recognition_macro_f1"],
                        "macro_precision": metrics["recognition_macro_precision"],
                        "macro_recall": metrics["recognition_macro_recall"],
                        "samples_f1": metrics["recognition_samples_f1"],
                        "subset_accuracy": metrics["subset_accuracy"],
                        "hamming_loss": metrics["hamming_loss"],
                        "predicted_positive_total": metrics["predicted_positive_total"],
                    }
                )
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary_rows, variant_rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "stage16c_effect_flow_downstream_validation_summary.json"
    txt_path = REPORT_DIR / "stage16c_effect_flow_downstream_validation_summary.txt"
    csv_path = REPORT_DIR / "stage16c_effect_flow_downstream_validation_summary.csv"
    payload = {
        "stage": "16C",
        "purpose": "Effect-flow encoder downstream validation",
        "caveat": (
            "The effect-flow encoder was continued from BJUT full-corpus EVM-BERT; "
            "base_encoder_init_setting=transductive_initialization."
        ),
        "summary_pairs": summary_rows,
        "all_threshold_variants": variant_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, variant_rows if variant_rows else summary_rows)
    lines = ["Stage 16C effect-flow downstream validation summary", ""]
    lines.append(payload["caveat"])
    lines.append("")
    lines.append(
        "comparison | old_status | new_status | old micro/macro/det | "
        "new micro/macro/det | delta micro/macro/det"
    )
    lines.append("-" * 140)
    for row in summary_rows:
        lines.append(
            f"{row['comparison']} | {row['old_status']} | {row['new_status']} | "
            f"{fmt(row.get('old_recognition_micro_f1'))}/"
            f"{fmt(row.get('old_recognition_macro_f1'))}/"
            f"{fmt(row.get('old_detection_f1'))} | "
            f"{fmt(row.get('new_recognition_micro_f1'))}/"
            f"{fmt(row.get('new_recognition_macro_f1'))}/"
            f"{fmt(row.get('new_detection_f1'))} | "
            f"{fmt(row.get('delta_micro_f1'))}/"
            f"{fmt(row.get('delta_macro_f1'))}/"
            f"{fmt(row.get('delta_detection_f1'))}"
        )
        if row.get("per_label_f1_delta") is not None:
            lines.append(f"  per_label_f1_delta: {row['per_label_f1_delta']}")
    if variant_rows:
        lines.append("")
        lines.append("Available threshold variants:")
        for row in variant_rows:
            lines.append(
                f"{row['comparison']} | {row['encoder_side']} | "
                f"{row['threshold_protocol']} | micro={row['micro_f1']:.6f} "
                f"macro={row['macro_f1']:.6f} det={row['detection_f1']:.6f}"
            )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {csv_path}")


def write_bootstrap(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "stage16c_effect_flow_bootstrap_report.json"
    txt_path = REPORT_DIR / "stage16c_effect_flow_bootstrap_report.txt"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["Stage 16C paired bootstrap report", ""]
    for row in results:
        lines.append(f"comparison: {row['comparison']}")
        lines.append(f"status: {row['status']}")
        if row["status"] == "ok":
            for metric in ["micro_f1", "macro_f1", "detection_f1"]:
                lines.append(
                    f"delta_{metric}: mean={row[f'delta_{metric}_mean']:.6f}, "
                    f"ci95={row[f'delta_{metric}_ci95']}, "
                    f"crosses_zero={row[f'delta_{metric}_ci_crosses_zero']}"
                )
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {txt_path}")


def fmt(value):
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def main():
    summary_rows = [summarize_pair(spec) for spec in COMPARISONS]
    variant_rows = collect_variant_rows()
    bootstrap_results = [paired_bootstrap(spec) for spec in COMPARISONS]
    write_summary(summary_rows, variant_rows)
    write_bootstrap(bootstrap_results)


if __name__ == "__main__":
    main()

