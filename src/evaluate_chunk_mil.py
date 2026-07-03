import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from chunk_feature_dataset import ChunkFeatureDataset
from evm_chunk_mil_model import (
    EVEFMVDV2SideEvidenceMIL,
    EVMChunkMILClassifier,
    EffectFlowGuidedChunkMIL,
)
from metrics import (
    binary_detection_metrics,
    compute_metrics,
    compute_multilabel_metrics_from_probs,
    precision_recall_f1,
    sigmoid,
)
from train_chunk_mil import collate_batch, compute_pos_weight_from_feature_cache, label_table


EFFECT_FLOW_MODEL_TYPES = {
    "effect_flow_guided_chunk_mil",
    "evef_mvd_v2_side_evidence_mil",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EVM chunk MIL classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--threshold", default="auto")
    parser.add_argument(
        "--threshold_search",
        default=None,
        choices=[None, "global_and_per_label"],
    )
    parser.add_argument("--threshold_file", default=None)
    parser.add_argument("--global_threshold_file", default=None)
    parser.add_argument(
        "--result_dir_override",
        default=None,
        help="Write evaluation artifacts to this directory instead of config result_dir.",
    )
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save per-sample probabilities and predictions to test_predictions.jsonl.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def threshold_candidates():
    return [round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05)]


def feature_path(config, split):
    return Path(config["feature_dir"]) / f"{split}.pt"


def make_loader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def load_model(config, checkpoint, device):
    model_type = config.get("model_type", "evm_chunk_mil")
    if model_type == "effect_flow_guided_chunk_mil":
        model = EffectFlowGuidedChunkMIL(config).to(device)
    elif model_type == "evef_mvd_v2_side_evidence_mil":
        model = EVEFMVDV2SideEvidenceMIL(config).to(device)
    else:
        model = EVMChunkMILClassifier(config).to(device)
    if config.get("use_pos_weight", False):
        pos_weight, _ = compute_pos_weight_from_feature_cache(config)
        model.set_recognition_pos_weight(pos_weight.to(device))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    if hasattr(model, "set_training_epoch") and state.get("epoch") is not None:
        model.set_training_epoch(int(state["epoch"]))
    if config.get("use_data_parallel", False) and torch.cuda.device_count() >= 2:
        print(
            "[INFO] evaluation uses one GPU; DataParallel is limited to training "
            "for PyTorch 2.0.x Transformer stability"
        )
    model.eval()
    return model, state


@torch.no_grad()
def collect_predictions(model, loader, device):
    losses = []
    ids = []
    metadata = []
    detection_logits = []
    recognition_logits = []
    chunk_logits = []
    chunk_scores = []
    chunk_masks = []
    binary_labels = []
    multi_labels = []
    for batch in tqdm(loader, desc="eval", leave=False):
        inputs = {
            "chunk_features": batch["chunk_features"].to(device),
            "chunk_mask": batch["chunk_mask"].to(device),
            "binary_label": batch["binary_label"].to(device),
            "multi_labels": batch["multi_labels"].to(device),
        }
        if "efpp_probs" in batch:
            inputs["efpp_probs"] = batch["efpp_probs"].to(device)
            inputs["etp_distribution"] = batch["etp_distribution"].to(device)
            inputs["relation_distribution"] = batch["relation_distribution"].to(device)
            inputs["vulnerability_evidence_probs"] = batch["vulnerability_evidence_probs"].to(device)
            inputs["template_match_scores"] = batch["template_match_scores"].to(device)
            inputs["chunk_vulnerability_evidence"] = batch["chunk_vulnerability_evidence"].to(device)
            inputs["vulnerability_template_matches"] = batch["vulnerability_template_matches"].to(device)
            inputs["active_vulnerability_label_mask"] = batch["active_vulnerability_label_mask"].to(device)
        if "front_special_features" in batch:
            inputs["front_special_features"] = batch["front_special_features"].to(device)
        outputs = model(**inputs)
        losses.append(float(outputs["loss"].mean().detach().cpu().item()))
        ids.extend(batch["id"])
        metadata.extend(batch["metadata"])
        detection_logits.append(outputs["detection_logits"].detach().cpu())
        recognition_logits.append(outputs["recognition_logits"].detach().cpu())
        chunk_logits.append(outputs["chunk_logits"].detach().cpu())
        chunk_scores.append(outputs.get("chunk_scores", outputs["chunk_logits"]).detach().cpu())
        chunk_masks.append(inputs["chunk_mask"].detach().cpu())
        binary_labels.append(inputs["binary_label"].detach().cpu())
        multi_labels.append(inputs["multi_labels"].detach().cpu())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "ids": ids,
        "metadata": metadata,
        "detection_logits": torch.cat(detection_logits).numpy(),
        "recognition_logits": torch.cat(recognition_logits).numpy(),
        "chunk_logits": torch.cat(chunk_logits),
        "chunk_scores": torch.cat(chunk_scores),
        "chunk_mask": torch.cat(chunk_masks),
        "binary_labels": torch.cat(binary_labels).numpy(),
        "multi_labels": torch.cat(multi_labels).numpy(),
    }


def threshold_array(thresholds, num_labels):
    values = np.asarray(thresholds, dtype=float)
    if values.ndim == 0:
        values = np.full(num_labels, float(values))
    if values.shape[0] != num_labels:
        raise ValueError(
            f"threshold count {values.shape[0]} does not match num_labels {num_labels}"
        )
    return values


def recognition_metrics_from_predictions(predictions, thresholds):
    labels = predictions["multi_labels"].astype(int)
    probs = sigmoid(predictions["recognition_logits"])
    return compute_multilabel_metrics_from_probs(labels, probs, thresholds)


def full_metrics_from_predictions(predictions, thresholds):
    recognition = recognition_metrics_from_predictions(predictions, thresholds)
    detection = binary_detection_metrics(
        predictions["detection_logits"],
        predictions["binary_labels"],
        threshold=0.5,
    )
    metrics = {}
    metrics.update(detection)
    metrics.update(recognition)
    return metrics


def metric_summary(metrics, threshold_mode, threshold_source, thresholds):
    if isinstance(thresholds, np.ndarray):
        thresholds = [float(value) for value in thresholds.tolist()]
    return {
        "threshold_mode": threshold_mode,
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "detection_accuracy": metrics["detection_accuracy"],
        "detection_precision": metrics["detection_precision"],
        "detection_recall": metrics["detection_recall"],
        "detection_f1": metrics["detection_f1"],
        "recognition_micro_precision": metrics["recognition_micro_precision"],
        "recognition_micro_recall": metrics["recognition_micro_recall"],
        "recognition_micro_f1": metrics["recognition_micro_f1"],
        "recognition_macro_precision": metrics["recognition_macro_precision"],
        "recognition_macro_recall": metrics["recognition_macro_recall"],
        "recognition_macro_f1": metrics["recognition_macro_f1"],
        "predicted_positive_total": metrics["predicted_positive_total"],
        "per_label_predicted_positive_count": metrics[
            "per_label_predicted_positive_count"
        ],
        "per_label_support": metrics["per_label_support"],
        "per_label_true_positive_count": metrics["per_label_true_positive_count"],
        "per_label_false_positive_count": metrics.get(
            "per_label_false_positive_count", []
        ),
        "per_label_false_negative_count": metrics.get(
            "per_label_false_negative_count", []
        ),
        "per_label_true_negative_count": metrics.get(
            "per_label_true_negative_count", []
        ),
        "per_label_mean_pred_prob": metrics["per_label_mean_pred_prob"],
    }


def select_per_label_thresholds(
    y_true,
    y_prob,
    thresholds,
    label_names,
    global_threshold=0.5,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    selected = []
    rows = []
    warnings = []
    for label_id, label_name in enumerate(label_names):
        targets = y_true[:, label_id]
        probs = y_prob[:, label_id]
        support = int(targets.sum())
        if support == 0:
            threshold = float(global_threshold)
            preds = (probs >= threshold).astype(int)
            tp = int(((preds == 1) & (targets == 1)).sum())
            fp = int(((preds == 1) & (targets == 0)).sum())
            fn = int(((preds == 0) & (targets == 1)).sum())
            precision, recall, f1 = precision_recall_f1(tp, fp, fn)
            warnings.append(f"{label_name}: support=0 on validation, use 0.5")
        else:
            candidates = []
            for threshold in thresholds:
                threshold = float(threshold)
                preds = (probs >= threshold).astype(int)
                tp = int(((preds == 1) & (targets == 1)).sum())
                fp = int(((preds == 1) & (targets == 0)).sum())
                fn = int(((preds == 0) & (targets == 1)).sum())
                precision, recall, f1 = precision_recall_f1(tp, fp, fn)
                candidates.append(
                    {
                        "threshold": threshold,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "balance_gap": abs(precision - recall),
                        "distance_to_0_5": abs(threshold - 0.5),
                        "predicted_positive_count": int(preds.sum()),
                    }
                )
            best = max(
                candidates,
                key=lambda item: (
                    item["f1"],
                    -item["balance_gap"],
                    -item["distance_to_0_5"],
                ),
            )
            threshold = float(best["threshold"])
            precision = float(best["precision"])
            recall = float(best["recall"])
            f1 = float(best["f1"])
            preds = (probs >= threshold).astype(int)

        selected.append(float(threshold))
        rows.append(
            {
                "label_id": int(label_id),
                "label_name": label_name,
                "support": support,
                "best_threshold": float(threshold),
                "best_valid_precision": float(precision),
                "best_valid_recall": float(recall),
                "best_valid_f1": float(f1),
                "predicted_positive_count_at_best_threshold": int(preds.sum()),
            }
        )
    return {"thresholds": selected, "per_label": rows, "warnings": warnings}


def save_per_label_thresholds(result_dir, selection):
    ensure_dir(result_dir)
    json_path = result_dir / "per_label_thresholds_valid.json"
    txt_path = result_dir / "per_label_thresholds_valid.txt"
    json_path.write_text(
        json.dumps(selection, indent=2, default=json_default),
        encoding="utf-8",
    )
    lines = [
        "Validation-selected per-label thresholds",
        "",
        "label_name | support | best_threshold | valid_precision | valid_recall | valid_f1 | predicted_positive_count",
        "-" * 112,
    ]
    for row in selection["per_label"]:
        lines.append(
            f"{row['label_name']} | {row['support']} | "
            f"{row['best_threshold']:.2f} | {row['best_valid_precision']:.6f} | "
            f"{row['best_valid_recall']:.6f} | {row['best_valid_f1']:.6f} | "
            f"{row['predicted_positive_count_at_best_threshold']}"
        )
    if selection.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in selection["warnings"])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {json_path}")


def save_global_threshold_scan(result_dir, predictions, thresholds):
    ensure_dir(result_dir)
    rows = []
    for threshold in thresholds:
        metrics = recognition_metrics_from_predictions(predictions, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "micro_precision": metrics["recognition_micro_precision"],
                "micro_recall": metrics["recognition_micro_recall"],
                "micro_f1": metrics["recognition_micro_f1"],
                "macro_precision": metrics["recognition_macro_precision"],
                "macro_recall": metrics["recognition_macro_recall"],
                "macro_f1": metrics["recognition_macro_f1"],
                "predicted_positive_total": metrics["predicted_positive_total"],
            }
        )
    best_macro = max(rows, key=lambda row: (row["macro_f1"], -abs(row["threshold"] - 0.5)))
    best_micro = max(rows, key=lambda row: (row["micro_f1"], -abs(row["threshold"] - 0.5)))
    baseline = min(rows, key=lambda row: abs(row["threshold"] - 0.5))
    report = {
        "threshold_source": "validation set",
        "thresholds": thresholds,
        "rows": rows,
        "best_macro_f1_threshold": best_macro["threshold"],
        "best_macro_f1_value": best_macro["macro_f1"],
        "best_micro_f1_threshold": best_micro["threshold"],
        "best_micro_f1_value": best_micro["micro_f1"],
        "threshold_0_5_baseline": baseline,
    }
    json_path = result_dir / "valid_global_threshold_scan.json"
    txt_path = result_dir / "valid_global_threshold_scan.txt"
    csv_path = result_dir / "threshold_curve_data.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "Validation global threshold scan",
        "",
        f"best_macro_f1_threshold: {best_macro['threshold']}",
        f"best_macro_f1_value: {best_macro['macro_f1']:.6f}",
        f"best_micro_f1_threshold: {best_micro['threshold']}",
        f"best_micro_f1_value: {best_micro['micro_f1']:.6f}",
        "",
        "threshold | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 | predicted_positive_total",
        "-" * 126,
    ]
    for row in rows:
        lines.append(
            f"{row['threshold']:.2f} | {row['micro_precision']:.6f} | "
            f"{row['micro_recall']:.6f} | {row['micro_f1']:.6f} | "
            f"{row['macro_precision']:.6f} | {row['macro_recall']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['predicted_positive_total']}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    return report


def load_threshold_selection(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return threshold_array(data["thresholds"], len(data["thresholds"]))


def load_best_global_macro_threshold(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data["best_macro_f1_threshold"])


def model_description(config):
    if config.get("model_type") == "evef_mvd_v2_side_evidence_mil":
        return "evef_mvd_v2_strong_backbone_side_evidence_label_gated_MIL"
    if config.get("model_type") == "effect_flow_guided_chunk_mil":
        return "effect_flow_guided_chunk_context_transformer_label_gated_attention_MIL"
    if config.get("use_chunk_context", False):
        return "chunk_context_transformer_label_gated_attention_MIL"
    return "masked_mean_label_gated_attention_MIL"


def write_top_chunks(path, config, predictions, thresholds):
    label_names = config.get("label_names", [f"label_{idx}" for idx in range(10)])
    probs = sigmoid(predictions["recognition_logits"])
    thresholds = threshold_array(thresholds, len(label_names))
    pred_labels = (probs >= thresholds.reshape(1, -1)).astype(int)
    true_labels = predictions["multi_labels"].astype(int)
    chunk_scores = predictions.get("chunk_scores", predictions["chunk_logits"])
    chunk_mask = predictions["chunk_mask"].bool()
    top_k = int(config.get("top_k", 2))
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row_idx, contract_id in enumerate(predictions["ids"]):
            label_ids = sorted(
                set(np.where(pred_labels[row_idx] == 1)[0].tolist())
                | set(np.where(true_labels[row_idx] == 1)[0].tolist())
            )
            real_count = int(chunk_mask[row_idx].sum().item())
            if not label_ids:
                continue
            for label_id in label_ids:
                scores = chunk_scores[row_idx, :, label_id].clone()
                scores[~chunk_mask[row_idx]] = -1e9
                k = min(top_k, real_count)
                top_scores, top_indices = torch.topk(scores, k=k)
                record = {
                    "id": contract_id,
                    "label_name": label_names[label_id],
                    "label_id": int(label_id),
                    "true_label": int(true_labels[row_idx, label_id]),
                    "predicted_label": int(pred_labels[row_idx, label_id]),
                    "threshold_used": float(thresholds[label_id]),
                    "contract_probability": float(probs[row_idx, label_id]),
                    "top_chunk_indices": [int(v) for v in top_indices.tolist()],
                    "top_chunk_scores": [float(v) for v in top_scores.tolist()],
                    "num_chunks_kept": real_count,
                    "feature_pooling": config.get("feature_pooling"),
                    "recognition_aggregation": config.get("recognition_aggregation"),
                    "metadata": predictions["metadata"][row_idx],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(prefix, report):
    ensure_dir(prefix.parent)
    json_path = prefix.with_suffix(".json")
    txt_path = prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    lines = ["EVM chunk MIL evaluation report", ""]
    for key, value in report.items():
        if key == "per_label":
            lines.append("Per-label metrics:")
            for row in value:
                lines.append(
                    f"{row['label_id']} | {row['label_name']} | "
                    f"precision={row['precision']:.6f} recall={row['recall']:.6f} "
                    f"accuracy={row['accuracy']:.6f} f1={row['f1']:.6f} "
                    f"support={row['support']} "
                    f"predicted={row['predicted_positive_count']}"
                )
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {json_path}")


def threshold_array_for_export(thresholds, num_labels):
    values = np.asarray(thresholds, dtype=float)
    if values.ndim == 0:
        values = np.full(num_labels, float(values))
    if values.shape[0] != num_labels:
        raise ValueError(
            f"threshold count {values.shape[0]} does not match num_labels {num_labels}"
        )
    return values


def write_prediction_jsonl(path, predictions, thresholds, threshold_mode):
    num_labels = predictions["multi_labels"].shape[1]
    thresholds = threshold_array_for_export(thresholds, num_labels)
    multi_probs = sigmoid(predictions["recognition_logits"])
    multi_preds = (multi_probs >= thresholds.reshape(1, -1)).astype(int)
    binary_probs = sigmoid(predictions["detection_logits"]).reshape(-1)
    binary_preds = (binary_probs >= 0.5).astype(int)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for idx, contract_id in enumerate(predictions["ids"]):
            chunk_mask = predictions["chunk_mask"][idx].bool()
            real_count = int(chunk_mask.sum().item())
            chunk_scores = predictions.get("chunk_scores", predictions["chunk_logits"])
            top_chunk_indices = []
            top_chunk_scores = []
            if real_count > 0:
                sample_scores = chunk_scores[idx].detach().cpu()
                label_scores = sample_scores.max(dim=1).values
                label_scores[~chunk_mask] = -1e9
                k = min(2, real_count)
                scores, indices = torch.topk(label_scores, k=k)
                top_chunk_indices = [int(value) for value in indices.tolist()]
                top_chunk_scores = [float(value) for value in scores.tolist()]
            record = {
                "id": contract_id,
                "binary_true": int(predictions["binary_labels"][idx]),
                "binary_prob": float(binary_probs[idx]),
                "binary_pred": int(binary_preds[idx]),
                "multi_true": [
                    int(value) for value in predictions["multi_labels"][idx].tolist()
                ],
                "multi_prob": [float(value) for value in multi_probs[idx].tolist()],
                "multi_pred": [int(value) for value in multi_preds[idx].tolist()],
                "threshold_mode": threshold_mode,
                "thresholds": [float(value) for value in thresholds.tolist()],
                "num_chunks_kept": real_count,
                "top_chunk_indices": top_chunk_indices,
                "top_chunk_scores": top_chunk_scores,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def add_comparison_fields(result, baseline, config):
    comparisons = config.get("baseline_comparison", {})
    result["macro_f1_change_over_threshold_0_5"] = (
        result["recognition_macro_f1"] - baseline["recognition_macro_f1"]
    )
    result["micro_f1_change_over_threshold_0_5"] = (
        result["recognition_micro_f1"] - baseline["recognition_micro_f1"]
    )
    result["macro_f1_change_over_evm_bert_first512"] = (
        result["recognition_macro_f1"]
        - comparisons.get("evm_bert_first512_macro_f1", 0.0)
    )
    result["macro_f1_change_over_legacy_weighted_baseline"] = (
        result["recognition_macro_f1"]
        - comparisons.get("legacy_weighted_macro_f1", 0.0)
    )
    if "nonoverlap_labelattn_calibrated_macro_f1" in comparisons:
        result["macro_f1_change_over_nonoverlap_calibrated"] = (
            result["recognition_macro_f1"]
            - comparisons["nonoverlap_labelattn_calibrated_macro_f1"]
        )
    if "nonoverlap_labelattn_calibrated_micro_f1" in comparisons:
        result["micro_f1_change_over_nonoverlap_calibrated"] = (
            result["recognition_micro_f1"]
            - comparisons["nonoverlap_labelattn_calibrated_micro_f1"]
        )
    return result


def warning_rows(metrics, label_names, sample_count):
    warnings = []
    for idx, count in enumerate(metrics["per_label_predicted_positive_count"]):
        if count == 0:
            warnings.append(f"{label_names[idx]} predicts all-zero")
        elif sample_count and count / sample_count > 0.5:
            warnings.append(
                f"{label_names[idx]} predicts too many positives: {count}/{sample_count}"
            )
    return warnings


def write_threshold_calibration_report(
    result_dir,
    config,
    args,
    checkpoint,
    predictions,
    baseline_metrics,
    best_global_threshold,
    best_global_metrics,
    per_label_thresholds,
    per_label_metrics,
):
    label_names = config.get("label_names", [f"label_{idx}" for idx in range(10)])
    calibrated_top_chunks = result_dir / "top_chunks_test_calibrated.jsonl"
    write_top_chunks(calibrated_top_chunks, config, predictions, per_label_thresholds)
    baseline = metric_summary(
        baseline_metrics,
        "global",
        "fixed_0.5_baseline",
        0.5,
    )
    best_global = add_comparison_fields(
        metric_summary(
            best_global_metrics,
            "global",
            "validation_best_macro_f1",
            float(best_global_threshold),
        ),
        baseline,
        config,
    )
    per_label = add_comparison_fields(
        metric_summary(
            per_label_metrics,
            "per_label",
            "validation_selected_per_label",
            [float(v) for v in per_label_thresholds],
        ),
        baseline,
        config,
    )
    baseline = add_comparison_fields(baseline, baseline, config)
    report = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model": model_description(config),
        "ablation_dataset": config.get("ablation_dataset"),
        "ablation_variant": config.get("ablation_variant"),
        "side_evidence_enabled": bool(config.get("side_evidence_enabled", False)),
        "coefficient_scale": config.get("coefficient_scale"),
        "use_chunk_context": bool(config.get("use_chunk_context", False)),
        "chunk_context_num_layers": int(config.get("chunk_context_num_layers", 0)),
        "chunk_context_num_heads": int(config.get("chunk_context_num_heads", 0)),
        "feature_pooling": config.get("feature_pooling"),
        "recognition_aggregation": config.get("recognition_aggregation"),
        "feature_dir": config["feature_dir"],
        "semantic_feature_dir": config.get("semantic_feature_dir"),
        "use_effect_flow_semantics": config.get("model_type") in EFFECT_FLOW_MODEL_TYPES,
        "effect_flow_semantic_mode": config.get("effect_flow_semantic_mode"),
        "risk_evidence_weight": config.get("risk_evidence_weight"),
        "missing_check_evidence_weight": config.get("missing_check_evidence_weight"),
        "protective_evidence_weight": config.get("protective_evidence_weight"),
        "effect_type_evidence_weight": config.get("effect_type_evidence_weight"),
        "relation_evidence_weight": config.get("relation_evidence_weight"),
        "behavior_weight_path": config.get("behavior_weight_path"),
        "use_weighted_behavior_scoring": bool(config.get("behavior_weight_path")),
        "front_special_feature_dir": config.get("front_special_feature_dir"),
        "front_running_special_enabled": bool(
            config.get("front_running_special_enabled", False)
        ),
        "front_running_generic_pattern_scale": config.get(
            "front_running_generic_pattern_scale"
        ),
        "front_running_confounder_suppression_weight": config.get(
            "front_running_confounder_suppression_weight"
        ),
        "front_hard_negative_loss_enabled": bool(
            config.get("front_hard_negative_loss_enabled", False)
        ),
        "front_hard_negative_lambda": config.get("front_hard_negative_lambda"),
        "front_contrastive_loss_enabled": bool(
            config.get("front_contrastive_loss_enabled", False)
        ),
        "front_contrastive_lambda": config.get("front_contrastive_lambda"),
        "front_contrastive_temperature": config.get("front_contrastive_temperature"),
        "front_contrastive_enable_epoch": config.get("front_contrastive_enable_epoch"),
        "front_contrastive_negative_mining": config.get(
            "front_contrastive_negative_mining"
        ),
        "front_contrastive_hard_negative_ratio": config.get(
            "front_contrastive_hard_negative_ratio"
        ),
        "front_contrastive_hard_negative_min_k": config.get(
            "front_contrastive_hard_negative_min_k"
        ),
        "front_contrastive_hard_negative_max_k": config.get(
            "front_contrastive_hard_negative_max_k"
        ),
        "beta_reliable_init": config.get("beta_reliable_init"),
        "gamma_reliable_init": config.get("gamma_reliable_init"),
        "is_transductive_pretraining": bool(config.get("is_transductive_pretraining", True)),
        "threshold_source": "validation set",
        "evaluated_samples": len(predictions["ids"]),
        "single_gpu_evaluation": True,
        "batch_size": int(config["batch_size"]),
        "max_chunks": int(config["max_chunks"]),
        "effective_chunks_per_batch": int(config["batch_size"])
        * int(config["max_chunks"]),
        "num_workers": int(config.get("num_workers", 0)),
        "loss": predictions["loss"],
        "threshold_0_5_baseline": baseline,
        "best_global_threshold_result": best_global,
        "per_label_threshold_result": per_label,
        "per_label_metrics": label_table(config, per_label_metrics),
        "per_label_thresholds": [
            {
                "label_id": idx,
                "label_name": label_names[idx],
                "threshold": float(value),
            }
            for idx, value in enumerate(per_label_thresholds)
        ],
        "comparison_baselines": {
            "nonoverlap_labelattn_calibrated": {
                "micro_f1": config.get("baseline_comparison", {}).get(
                    "nonoverlap_labelattn_calibrated_micro_f1",
                    0.5918,
                ),
                "macro_f1": config.get("baseline_comparison", {}).get(
                    "nonoverlap_labelattn_calibrated_macro_f1",
                    0.4421,
                ),
            },
            "nonoverlap_labelattn_threshold_0_5": {
                "micro_f1": config.get("baseline_comparison", {}).get(
                    "nonoverlap_labelattn_threshold05_micro_f1",
                    0.5823,
                ),
                "macro_f1": config.get("baseline_comparison", {}).get(
                    "nonoverlap_labelattn_threshold05_macro_f1",
                    0.4375,
                ),
                "detection_f1": config.get("baseline_comparison", {}).get(
                    "nonoverlap_labelattn_threshold05_detection_f1",
                    0.6752,
                ),
            },
            "weighted_global_threshold_0_5": {
                "micro_f1": 0.5823,
                "macro_f1": 0.4375,
                "detection_f1": 0.6752,
                "predicted_positive_total": 7106,
            },
            "evm_bert_first512_weighted": {
                "micro_f1": 0.5659,
                "macro_f1": 0.4326,
                "detection_f1": 0.6895,
                "predicted_positive_total": 5806,
            },
            "legacy_weighted_baseline": {
                "micro_f1": 0.5604,
                "macro_f1": 0.4215,
                "detection_f1": 0.6430,
            },
        },
        "warnings": warning_rows(per_label_metrics, label_names, len(predictions["ids"])),
        "top_chunks_test_calibrated": str(calibrated_top_chunks),
    }
    prefix = result_dir / "test_threshold_calibration_metrics"
    write_report(prefix, report)
    return report


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_path = None
    if config.get("semantic_feature_dir"):
        semantic_path = Path(config["semantic_feature_dir"]) / f"{args.split}.pt"
    front_special_path = None
    if config.get("front_special_feature_dir"):
        front_special_path = Path(config["front_special_feature_dir"]) / f"{args.split}.pt"
    dataset = ChunkFeatureDataset(
        feature_path(config, args.split),
        seed=config.get("seed", 42),
        num_labels=config.get("num_labels"),
        semantic_path=semantic_path,
        front_special_path=front_special_path,
    )
    loader = make_loader(dataset, config)
    model, checkpoint = load_model(config, args.checkpoint, device)
    result_dir = Path(args.result_dir_override or config["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.threshold == "auto":
        threshold = checkpoint.get("best_threshold_by_macro_f1") or config.get("threshold", 0.5)
        threshold_source = "checkpoint_best_threshold_by_macro_f1"
    else:
        threshold = float(args.threshold)
        threshold_source = "cli"
    threshold = float(threshold)
    predictions = collect_predictions(model, loader, device)

    if args.split == "valid" and args.threshold_search == "global_and_per_label":
        thresholds = threshold_candidates()
        scan = save_global_threshold_scan(result_dir, predictions, thresholds)
        probs = sigmoid(predictions["recognition_logits"])
        selection = select_per_label_thresholds(
            predictions["multi_labels"],
            probs,
            thresholds,
            config.get("label_names", [f"label_{idx}" for idx in range(10)]),
            global_threshold=0.5,
        )
        selection["global_threshold_scan_path"] = str(
            result_dir / "valid_global_threshold_scan.json"
        )
        selection["best_global_macro_threshold"] = scan["best_macro_f1_threshold"]
        selection["best_global_micro_threshold"] = scan["best_micro_f1_threshold"]
        save_per_label_thresholds(result_dir, selection)
        return

    if args.split == "test" and args.threshold_file and args.global_threshold_file:
        per_label_thresholds = load_threshold_selection(args.threshold_file)
        best_global_threshold = load_best_global_macro_threshold(args.global_threshold_file)
        baseline_metrics = full_metrics_from_predictions(predictions, 0.5)
        best_global_metrics = full_metrics_from_predictions(
            predictions,
            best_global_threshold,
        )
        per_label_metrics = full_metrics_from_predictions(
            predictions,
            per_label_thresholds,
        )
        report = write_threshold_calibration_report(
            result_dir,
            config,
            args,
            checkpoint,
            predictions,
            baseline_metrics,
            best_global_threshold,
            best_global_metrics,
            per_label_thresholds,
            per_label_metrics,
        )
        if args.save_predictions:
            fixed_prediction_path = result_dir / "test_predictions_threshold_0_5.jsonl"
            write_prediction_jsonl(
                fixed_prediction_path,
                predictions,
                0.5,
                "global_fixed_0_5",
            )
            best_global_prediction_path = (
                result_dir / "test_predictions_best_global.jsonl"
            )
            write_prediction_jsonl(
                best_global_prediction_path,
                predictions,
                best_global_threshold,
                "global_validation_best_macro",
            )
            default_prediction_path = result_dir / "test_predictions.jsonl"
            write_prediction_jsonl(
                default_prediction_path,
                predictions,
                best_global_threshold,
                "global_validation_best_macro",
            )
            per_label_prediction_path = (
                result_dir / "test_predictions_per_label.jsonl"
            )
            write_prediction_jsonl(
                per_label_prediction_path,
                predictions,
                per_label_thresholds,
                "per_label_validation_selected",
            )
            report["fixed_threshold_prediction_path"] = str(fixed_prediction_path)
            report["best_global_prediction_path"] = str(best_global_prediction_path)
            report["prediction_path"] = str(default_prediction_path)
            report["per_label_prediction_path"] = str(per_label_prediction_path)
            write_report(result_dir / "test_threshold_calibration_metrics", report)
            print(f"[OK] wrote {fixed_prediction_path}")
            print(f"[OK] wrote {best_global_prediction_path}")
            print(f"[OK] wrote {default_prediction_path}")
            print(f"[OK] wrote {per_label_prediction_path}")
        return

    metrics = compute_metrics(
        predictions["detection_logits"],
        predictions["binary_labels"],
        predictions["recognition_logits"],
        predictions["multi_labels"],
        threshold=threshold,
        scan_thresholds=config.get("thresholds", [threshold]),
    )
    top_chunks_path = result_dir / f"top_chunks_{args.split}.jsonl"
    write_top_chunks(top_chunks_path, config, predictions, threshold)
    baseline = config.get("baseline_comparison", {})
    report = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "model_type": config.get("model_type", "evm_chunk_mil"),
        "ablation_dataset": config.get("ablation_dataset"),
        "ablation_variant": config.get("ablation_variant"),
        "side_evidence_enabled": bool(config.get("side_evidence_enabled", False)),
        "coefficient_scale": config.get("coefficient_scale"),
        "encoder_frozen": True,
        "feature_dir": config["feature_dir"],
        "semantic_feature_dir": config.get("semantic_feature_dir"),
        "use_effect_flow_semantics": config.get("model_type") in EFFECT_FLOW_MODEL_TYPES,
        "effect_flow_semantic_mode": config.get("effect_flow_semantic_mode"),
        "risk_evidence_weight": config.get("risk_evidence_weight"),
        "missing_check_evidence_weight": config.get("missing_check_evidence_weight"),
        "protective_evidence_weight": config.get("protective_evidence_weight"),
        "effect_type_evidence_weight": config.get("effect_type_evidence_weight"),
        "relation_evidence_weight": config.get("relation_evidence_weight"),
        "behavior_weight_path": config.get("behavior_weight_path"),
        "use_weighted_behavior_scoring": bool(config.get("behavior_weight_path")),
        "front_special_feature_dir": config.get("front_special_feature_dir"),
        "front_running_special_enabled": bool(
            config.get("front_running_special_enabled", False)
        ),
        "front_running_generic_pattern_scale": config.get(
            "front_running_generic_pattern_scale"
        ),
        "front_running_confounder_suppression_weight": config.get(
            "front_running_confounder_suppression_weight"
        ),
        "front_hard_negative_loss_enabled": bool(
            config.get("front_hard_negative_loss_enabled", False)
        ),
        "front_hard_negative_lambda": config.get("front_hard_negative_lambda"),
        "front_contrastive_loss_enabled": bool(
            config.get("front_contrastive_loss_enabled", False)
        ),
        "front_contrastive_lambda": config.get("front_contrastive_lambda"),
        "front_contrastive_temperature": config.get("front_contrastive_temperature"),
        "front_contrastive_enable_epoch": config.get("front_contrastive_enable_epoch"),
        "front_contrastive_negative_mining": config.get(
            "front_contrastive_negative_mining"
        ),
        "front_contrastive_hard_negative_ratio": config.get(
            "front_contrastive_hard_negative_ratio"
        ),
        "front_contrastive_hard_negative_min_k": config.get(
            "front_contrastive_hard_negative_min_k"
        ),
        "front_contrastive_hard_negative_max_k": config.get(
            "front_contrastive_hard_negative_max_k"
        ),
        "beta_reliable_init": config.get("beta_reliable_init"),
        "gamma_reliable_init": config.get("gamma_reliable_init"),
        "evaluated_samples": len(dataset),
        "single_gpu_evaluation": True,
        "batch_size": int(config["batch_size"]),
        "max_chunks": int(config["max_chunks"]),
        "effective_chunks_per_batch": int(config["batch_size"])
        * int(config["max_chunks"]),
        "num_workers": int(config.get("num_workers", 0)),
        "loss": predictions["loss"],
        "detection_precision": metrics["detection_precision"],
        "detection_recall": metrics["detection_recall"],
        "detection_f1": metrics["detection_f1"],
        "recognition_micro_precision": metrics["recognition_micro_precision"],
        "recognition_micro_recall": metrics["recognition_micro_recall"],
        "recognition_micro_f1": metrics["recognition_micro_f1"],
        "recognition_macro_precision": metrics["recognition_macro_precision"],
        "recognition_macro_recall": metrics["recognition_macro_recall"],
        "recognition_macro_f1": metrics["recognition_macro_f1"],
        "predicted_positive_total": metrics["predicted_positive_total"],
        "per_label_predicted_positive_count": metrics["per_label_predicted_positive_count"],
        "per_label_true_positive_count": metrics["per_label_true_positive_count"],
        "per_label_mean_pred_prob": metrics["per_label_mean_pred_prob"],
        "per_label": label_table(config, metrics),
        "threshold_scan": metrics.get("threshold_scan", {}),
        "top_chunks_path": str(top_chunks_path),
        "baseline_comparison": baseline,
        "macro_f1_change_over_evm_bert_first512": (
            metrics["recognition_macro_f1"] - baseline.get("evm_bert_first512_macro_f1", 0.0)
        ),
        "micro_f1_change_over_evm_bert_first512": (
            metrics["recognition_micro_f1"] - baseline.get("evm_bert_first512_micro_f1", 0.0)
        ),
        "detection_f1_change_over_evm_bert_first512": (
            metrics["detection_f1"] - baseline.get("evm_bert_first512_detection_f1", 0.0)
        ),
        "is_transductive_pretraining": bool(config.get("is_transductive_pretraining", True)),
    }
    prefix = result_dir / f"{args.split}_best_macro_metrics"
    write_report(prefix, report)
    if args.save_predictions:
        prediction_path = result_dir / f"{args.split}_predictions.jsonl"
        write_prediction_jsonl(prediction_path, predictions, threshold, "global")
        report["prediction_path"] = str(prediction_path)
        write_report(prefix, report)
        print(f"[OK] wrote {prediction_path}")


if __name__ == "__main__":
    main()
