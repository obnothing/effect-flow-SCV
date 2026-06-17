import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    path = PROJECT_ROOT / path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(
    model,
    feature_pooling,
    chunk_stride,
    max_chunks,
    aggregation,
    encoder_frozen,
    metrics_path,
    summary_path=None,
    notes="",
    threshold_mode="global",
    threshold_source="checkpoint",
):
    metrics = load_json(metrics_path)
    summary = load_json(summary_path) if summary_path else None
    base = {
        "model": model,
        "feature_pooling": feature_pooling,
        "chunk_stride": chunk_stride,
        "max_chunks": max_chunks,
        "aggregation": aggregation,
        "encoder_frozen": encoder_frozen,
        "threshold_mode": threshold_mode,
        "threshold_source": threshold_source,
        "notes": notes,
    }
    if metrics is None:
        base["status"] = "missing"
        return base
    base.update(
        {
            "status": "ok",
            "micro_f1": metrics.get("recognition_micro_f1"),
            "macro_f1": metrics.get("recognition_macro_f1"),
            "macro_precision": metrics.get("recognition_macro_precision"),
            "macro_recall": metrics.get("recognition_macro_recall"),
            "detection_f1": metrics.get("detection_f1"),
            "predicted_positives": metrics.get("predicted_positive_total"),
            "best_valid_macro_f1": (
                summary.get("best_macro_f1_value") if summary else None
            ),
            "best_epoch": metrics.get("checkpoint_epoch")
            or (summary.get("best_macro_f1_epoch") if summary else None),
            "threshold": metrics.get("threshold")
            or (summary.get("best_macro_f1_threshold") if summary else None),
        }
    )
    return base


def calibrated_metric_row(
    model,
    feature_pooling,
    chunk_stride,
    max_chunks,
    aggregation,
    threshold_mode,
    metrics_path,
    result_key,
    notes="",
):
    metrics = load_json(metrics_path)
    base = {
        "model": model,
        "feature_pooling": feature_pooling,
        "chunk_stride": chunk_stride,
        "max_chunks": max_chunks,
        "aggregation": aggregation,
        "encoder_frozen": True,
        "threshold_mode": threshold_mode,
        "threshold_source": "validation set",
        "notes": notes,
    }
    if metrics is None or result_key not in metrics:
        base["status"] = "missing"
        return base
    result = metrics[result_key]
    base.update(
        {
            "status": "ok",
            "threshold_mode": result.get("threshold_mode", threshold_mode),
            "threshold_source": result.get("threshold_source", "validation set"),
            "micro_f1": result.get("recognition_micro_f1"),
            "macro_f1": result.get("recognition_macro_f1"),
            "macro_precision": result.get("recognition_macro_precision"),
            "macro_recall": result.get("recognition_macro_recall"),
            "detection_f1": result.get("detection_f1"),
            "predicted_positives": result.get("predicted_positive_total"),
            "best_valid_macro_f1": None,
            "best_epoch": metrics.get("checkpoint_epoch"),
            "threshold": result.get("thresholds"),
        }
    )
    return base


def delta(value, baseline):
    if value is None or baseline is None:
        return None
    return value - baseline


def fmt_float(value):
    if value is None:
        return "missing"
    return f"{float(value):.4f}"


def find_macro(rows, model=None, threshold_mode=None, chunk_stride=None):
    for row in rows:
        if row.get("status") != "ok":
            continue
        if model is not None and row.get("model") != model:
            continue
        if threshold_mode is not None and row.get("threshold_mode") != threshold_mode:
            continue
        if chunk_stride is not None and row.get("chunk_stride") != chunk_stride:
            continue
        return row.get("macro_f1")
    return None


def build_rows():
    return [
        metric_row(
            "CodeBERT weighted",
            "codebert tokenizer",
            None,
            None,
            "BiGRU",
            False,
            "results/train_mlsmote_codebert_weighted/test_best_macro_metrics.json",
            "results/train_mlsmote_codebert_weighted/checkpoint_summary.json",
            "strict inductive baseline",
        ),
        metric_row(
            "EVM-BERT first-512 weighted",
            "first_512",
            None,
            None,
            "BiGRU",
            False,
            "results/train_evm_bert_weighted/test_best_macro_metrics.json",
            "results/train_evm_bert_weighted/checkpoint_summary.json",
            "transductive EVM-domain pretraining baseline",
        ),
        metric_row(
            "CLS top-k MIL",
            "cls",
            510,
            16,
            "topk_mean",
            True,
            "results/train_evm_bert_chunk_mil_weighted/test_best_macro_metrics.json",
            "results/train_evm_bert_chunk_mil_weighted/checkpoint_summary.json",
            "Stage 10A frozen chunk feature MIL",
        ),
        metric_row(
            "Masked-mean top-k MIL",
            "masked_mean",
            510,
            16,
            "topk_mean",
            True,
            "results/train_evm_bert_chunk_mil_mean_topk_weighted/test_best_macro_metrics.json",
            "results/train_evm_bert_chunk_mil_mean_topk_weighted/checkpoint_summary.json",
            "Stage 10B pooling ablation",
        ),
        metric_row(
            "Masked-mean label-attention MIL",
            "masked_mean",
            510,
            16,
            "label_gated_attention",
            True,
            "results/train_evm_bert_chunk_mil_mean_labelattn_weighted/test_best_macro_metrics.json",
            "results/train_evm_bert_chunk_mil_mean_labelattn_weighted/checkpoint_summary.json",
            "Stage 10B threshold=0.5",
        ),
        calibrated_metric_row(
            "Masked-mean label-attention MIL",
            "masked_mean",
            510,
            16,
            "label_gated_attention",
            "global_best_macro",
            "results/train_evm_bert_chunk_mil_mean_labelattn_weighted/test_threshold_calibration_metrics.json",
            "best_global_threshold_result",
            "Stage 10C validation-selected global threshold",
        ),
        calibrated_metric_row(
            "Masked-mean label-attention MIL",
            "masked_mean",
            510,
            16,
            "label_gated_attention",
            "per_label",
            "results/train_evm_bert_chunk_mil_mean_labelattn_weighted/test_threshold_calibration_metrics.json",
            "per_label_threshold_result",
            "Stage 10C validation-selected per-label thresholds",
        ),
        metric_row(
            "Stride256 masked-mean label-attention MIL",
            "masked_mean",
            256,
            32,
            "label_gated_attention",
            True,
            "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/test_best_macro_metrics.json",
            "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/checkpoint_summary.json",
            "Stage 10D threshold=0.5",
        ),
        calibrated_metric_row(
            "Stride256 masked-mean label-attention MIL",
            "masked_mean",
            256,
            32,
            "label_gated_attention",
            "global_best_macro",
            "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/test_threshold_calibration_metrics.json",
            "best_global_threshold_result",
            "Stage 10D validation-selected global threshold",
        ),
        calibrated_metric_row(
            "Stride256 masked-mean label-attention MIL",
            "masked_mean",
            256,
            32,
            "label_gated_attention",
            "per_label",
            "results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/test_threshold_calibration_metrics.json",
            "per_label_threshold_result",
            "Stage 10D validation-selected per-label thresholds",
        ),
    ]


def main():
    rows = build_rows()
    cls_macro = find_macro(rows, model="CLS top-k MIL")
    first_macro = find_macro(rows, model="EVM-BERT first-512 weighted")
    codebert_macro = find_macro(rows, model="CodeBERT weighted")
    nonoverlap_calibrated_macro = find_macro(
        rows,
        model="Masked-mean label-attention MIL",
        threshold_mode="per_label",
        chunk_stride=510,
    )
    for row in rows:
        row["macro_f1_delta_vs_cls_topk_mil"] = delta(row.get("macro_f1"), cls_macro)
        row["macro_f1_delta_vs_evm_bert_first512"] = delta(row.get("macro_f1"), first_macro)
        row["macro_f1_delta_vs_codebert_weighted"] = delta(row.get("macro_f1"), codebert_macro)
        row["macro_f1_delta_vs_nonoverlap_calibrated"] = delta(
            row.get("macro_f1"),
            nonoverlap_calibrated_macro,
        )

    report = {
        "status": "ok",
        "rows": rows,
        "success_criteria": {
            "masked_mean_topk_better_than_cls_topk": (
                (find_macro(rows, model="Masked-mean top-k MIL") or -1)
                > (cls_macro if cls_macro is not None else float("inf"))
            ),
            "label_attention_beats_first512": (
                (
                    find_macro(
                        rows,
                        model="Masked-mean label-attention MIL",
                        threshold_mode="global",
                        chunk_stride=510,
                    )
                    or -1
                )
                > (first_macro if first_macro is not None else float("inf"))
            ),
            "stride256_calibrated_beats_nonoverlap_calibrated": (
                (
                    find_macro(
                        rows,
                        model="Stride256 masked-mean label-attention MIL",
                        threshold_mode="per_label",
                        chunk_stride=256,
                    )
                    or -1
                )
                > (
                    nonoverlap_calibrated_macro
                    if nonoverlap_calibrated_macro is not None
                    else float("inf")
                )
            ),
        },
    }
    report_dir = PROJECT_ROOT / "data/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "stage10_ablation_summary.json"
    txt_path = report_dir / "stage10_ablation_summary.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["Stage 10 ablation summary", ""]
    header = (
        "model | feature_pooling | chunk_stride | max_chunks | aggregation | "
        "threshold_mode | threshold_source | encoder_frozen | micro-F1 | macro-F1 | "
        "macro precision | macro recall | detection F1 | predicted positives | "
        "best valid macro-F1 | best epoch | threshold | notes"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"{row['model']} | {row['feature_pooling']} | "
                f"{row.get('chunk_stride')} | {row.get('max_chunks')} | "
                f"{row['aggregation']} | {row.get('threshold_mode')} | "
                f"{row.get('threshold_source')} | {row['encoder_frozen']} | "
                f"missing | missing | missing | missing | missing | missing | "
                f"missing | missing | missing | {row['notes']}"
            )
            continue
        lines.append(
            f"{row['model']} | {row['feature_pooling']} | "
            f"{row.get('chunk_stride')} | {row.get('max_chunks')} | "
            f"{row['aggregation']} | {row.get('threshold_mode')} | "
            f"{row.get('threshold_source')} | {row['encoder_frozen']} | "
            f"{fmt_float(row.get('micro_f1'))} | {fmt_float(row.get('macro_f1'))} | "
            f"{fmt_float(row.get('macro_precision'))} | {fmt_float(row.get('macro_recall'))} | "
            f"{fmt_float(row.get('detection_f1'))} | {row.get('predicted_positives')} | "
            f"{row.get('best_valid_macro_f1')} | {row.get('best_epoch')} | "
            f"{row.get('threshold')} | {row['notes']}"
        )
    lines.append("")
    lines.append("Macro-F1 deltas:")
    for row in rows:
        lines.append(
            f"- {row['model']} [{row.get('threshold_mode')}, stride={row.get('chunk_stride')}]: "
            f"vs_cls={row.get('macro_f1_delta_vs_cls_topk_mil')}, "
            f"vs_first512={row.get('macro_f1_delta_vs_evm_bert_first512')}, "
            f"vs_codebert={row.get('macro_f1_delta_vs_codebert_weighted')}, "
            f"vs_nonoverlap_calibrated={row.get('macro_f1_delta_vs_nonoverlap_calibrated')}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {json_path}")


if __name__ == "__main__":
    main()

