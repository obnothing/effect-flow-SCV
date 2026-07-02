import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chunk_feature_dataset import ChunkFeatureDataset  # noqa: E402
from evaluate_chunk_mil import (  # noqa: E402
    collect_predictions,
    feature_path,
    json_default,
    load_config,
    load_model,
    make_loader,
    threshold_candidates,
)
from metrics import compute_multilabel_metrics_from_probs, precision_recall_f1, sigmoid  # noqa: E402


BASELINE = {
    "micro_f1": 0.8240858035638883,
    "macro_f1": 0.7452721843450687,
    "detection_f1": 0.9465041054988804,
    "front_f1": 0.5384615384615384,
    "bad_randomness_f1": 0.5882352941176471,
}


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path):
    with resolve(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json_txt(prefix, title, report):
    ensure_dir(prefix.parent)
    json_path = prefix.with_suffix(".json")
    txt_path = prefix.with_suffix(".txt")
    json_path.write_text(
        json.dumps(report, indent=2, default=json_default), encoding="utf-8"
    )
    lines = [title, ""]
    for key, value in report.items():
        if key == "per_label_metrics":
            lines.append("Per-label metrics:")
            for row in value:
                lines.append(
                    f"{row['label_id']} | {row['label_name']} | "
                    f"precision={row['precision']:.6f} recall={row['recall']:.6f} "
                    f"f1={row['f1']:.6f} support={row['support']} "
                    f"predicted={row['predicted_positive_count']} "
                    f"tp={row['true_positive_count']} fp={row['false_positive_count']} "
                    f"fn={row['false_negative_count']}"
                )
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


def detection_metrics_from_probs(probs, labels, threshold=0.5):
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    total = tp + fp + fn + tn
    return {
        "detection_accuracy": (tp + tn) / total if total else 0.0,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "detection_tp": tp,
        "detection_fp": fp,
        "detection_fn": fn,
        "detection_tn": tn,
        "detection_predicted_positive_count": int(preds.sum()),
        "detection_true_positive_count": int(labels.sum()),
    }


def metrics_from_probs(ensemble, thresholds):
    metrics = detection_metrics_from_probs(
        ensemble["detection_probs"],
        ensemble["binary_labels"],
        threshold=0.5,
    )
    metrics.update(
        compute_multilabel_metrics_from_probs(
            ensemble["multi_labels"],
            ensemble["recognition_probs"],
            thresholds,
        )
    )
    return metrics


def label_rows(label_names, metrics):
    rows = []
    for idx, name in enumerate(label_names):
        rows.append(
            {
                "label_id": idx,
                "label_name": name,
                "precision": metrics["per_label_precision"][idx],
                "recall": metrics["per_label_recall"][idx],
                "accuracy": metrics["per_label_accuracy"][idx],
                "f1": metrics["per_label_f1"][idx],
                "support": metrics["per_label_support"][idx],
                "predicted_positive_count": metrics[
                    "per_label_predicted_positive_count"
                ][idx],
                "true_positive_count": metrics["per_label_true_positive_count"][idx],
                "false_positive_count": metrics["per_label_false_positive_count"][idx],
                "false_negative_count": metrics["per_label_false_negative_count"][idx],
                "true_negative_count": metrics["per_label_true_negative_count"][idx],
            }
        )
    return rows


def threshold_array(thresholds, num_labels):
    values = np.asarray(thresholds, dtype=float)
    if values.ndim == 0:
        return np.full(num_labels, float(values))
    if values.shape[0] != num_labels:
        raise ValueError(
            f"threshold count {values.shape[0]} does not match num_labels {num_labels}"
        )
    return values


def label_stats(targets, probs, threshold):
    targets = np.asarray(targets).astype(int)
    preds = (np.asarray(probs) >= float(threshold)).astype(int)
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": int(targets.sum()),
        "predicted_positive_count": int(preds.sum()),
    }


def bootstrap_threshold_score(targets, probs, threshold, iterations, rng):
    targets = np.asarray(targets).astype(int)
    probs = np.asarray(probs)
    pos_idx = np.flatnonzero(targets == 1)
    neg_idx = np.flatnonzero(targets == 0)
    values = []
    for _ in range(iterations):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sampled = np.concatenate([sampled_pos, sampled_neg])
        values.append(label_stats(targets[sampled], probs[sampled], threshold)["f1"])
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def regular_threshold_selection(targets, probs, thresholds):
    rows = [label_stats(targets, probs, threshold) for threshold in thresholds]
    best = max(rows, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5)))
    best = dict(best)
    best["selection_reason"] = "max_valid_f1_then_nearest_0_5"
    return best


def rare_threshold_selection(
    targets,
    probs,
    thresholds,
    iterations,
    bootstrap_seed,
    label_id,
):
    base_rows = [label_stats(targets, probs, threshold) for threshold in thresholds]
    support = int(np.asarray(targets).sum())
    min_pred = 0.5 * support
    max_pred = 1.5 * support
    candidates = [
        row
        for row in base_rows
        if row["recall"] >= 0.45
        and min_pred <= row["predicted_positive_count"] <= max_pred
    ]
    if not candidates:
        fallback = regular_threshold_selection(targets, probs, thresholds)
        fallback["selection_reason"] = (
            "rare_bootstrap_no_candidate_fallback_to_max_valid_f1"
        )
        fallback["bootstrap_warning"] = (
            "No threshold passed recall>=0.45 and predicted-positive-count guard."
        )
        return fallback

    rng = np.random.default_rng(int(bootstrap_seed) + int(label_id))
    scored = []
    for row in candidates:
        mean_f1, std_f1 = bootstrap_threshold_score(
            targets,
            probs,
            row["threshold"],
            iterations,
            rng,
        )
        item = dict(row)
        item["bootstrap_mean_f1"] = mean_f1
        item["bootstrap_std_f1"] = std_f1
        item["bootstrap_score"] = mean_f1 - 0.5 * std_f1
        scored.append(item)

    top_score = max(row["bootstrap_score"] for row in scored)
    near_best = [
        row for row in scored if top_score - row["bootstrap_score"] <= 0.005
    ]
    best = max(
        near_best,
        key=lambda row: (
            row["precision"],
            row["f1"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    best = dict(best)
    best["selection_reason"] = (
        "rare_bootstrap_mean_f1_minus_0_5_std_then_precision"
    )
    best["bootstrap_iterations"] = int(iterations)
    return best


def select_stable_thresholds(
    labels,
    probs,
    thresholds,
    label_names,
    rare_support_threshold,
    bootstrap_iterations,
    bootstrap_seed,
):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    selected = []
    rows = []
    for label_id, label_name in enumerate(label_names):
        targets = labels[:, label_id]
        label_probs = probs[:, label_id]
        support = int(targets.sum())
        if support == 0:
            row = label_stats(targets, label_probs, 0.5)
            row["selection_reason"] = "support_zero_use_0_5"
        elif support < rare_support_threshold:
            row = rare_threshold_selection(
                targets,
                label_probs,
                thresholds,
                bootstrap_iterations,
                bootstrap_seed,
                label_id,
            )
        else:
            row = regular_threshold_selection(targets, label_probs, thresholds)
        row.update(
            {
                "label_id": int(label_id),
                "label_name": label_name,
                "is_rare_label": bool(support < rare_support_threshold),
                "rare_support_threshold": int(rare_support_threshold),
                "best_threshold": float(row["threshold"]),
                "best_valid_precision": float(row["precision"]),
                "best_valid_recall": float(row["recall"]),
                "best_valid_f1": float(row["f1"]),
                "predicted_positive_count_at_best_threshold": int(
                    row["predicted_positive_count"]
                ),
            }
        )
        selected.append(float(row["threshold"]))
        rows.append(row)
    return {"thresholds": selected, "per_label": rows}


def save_global_threshold_scan(result_dir, ensemble, thresholds):
    rows = []
    for threshold in thresholds:
        metrics = metrics_from_probs(ensemble, float(threshold))
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
    result = {
        "threshold_source": "validation ensemble",
        "thresholds": thresholds,
        "best_macro_f1_threshold": best_macro["threshold"],
        "best_macro_f1_value": best_macro["macro_f1"],
        "best_micro_f1_threshold": best_micro["threshold"],
        "best_micro_f1_value": best_micro["micro_f1"],
        "rows": rows,
    }
    ensure_dir(result_dir)
    json_path = result_dir / "valid_global_threshold_scan.json"
    txt_path = result_dir / "valid_global_threshold_scan.txt"
    csv_path = result_dir / "threshold_curve_data.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "DIVE seed ensemble validation global threshold scan",
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
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")
    return result


def save_per_label_thresholds(result_dir, selection):
    ensure_dir(result_dir)
    json_path = result_dir / "per_label_thresholds_valid.json"
    txt_path = result_dir / "per_label_thresholds_valid.txt"
    json_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    lines = [
        "DIVE seed ensemble validation-selected per-label thresholds",
        "",
        "label_name | support | best_threshold | valid_precision | valid_recall | valid_f1 | predicted_positive_count | selection_reason",
        "-" * 150,
    ]
    for row in selection["per_label"]:
        lines.append(
            f"{row['label_name']} | {row['support']} | "
            f"{row['best_threshold']:.2f} | {row['best_valid_precision']:.6f} | "
            f"{row['best_valid_recall']:.6f} | {row['best_valid_f1']:.6f} | "
            f"{row['predicted_positive_count_at_best_threshold']} | "
            f"{row['selection_reason']}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


def load_member_predictions(config_path, checkpoint_name, split, device):
    config = load_config(str(resolve(config_path)))
    semantic_path = None
    if config.get("semantic_feature_dir"):
        semantic_path = Path(config["semantic_feature_dir"]) / f"{split}.pt"
    front_special_path = None
    if config.get("front_special_feature_dir"):
        front_special_path = Path(config["front_special_feature_dir"]) / f"{split}.pt"
    dataset = ChunkFeatureDataset(
        feature_path(config, split),
        seed=config.get("seed", 42),
        num_labels=config.get("num_labels"),
        semantic_path=semantic_path,
        front_special_path=front_special_path,
    )
    loader = make_loader(dataset, config)
    checkpoint_path = resolve(config["checkpoint_dir"]) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    model, checkpoint = load_model(config, str(checkpoint_path), device)
    predictions = collect_predictions(model, loader, device)
    return {
        "config": config,
        "config_path": str(config_path),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": split,
        "ids": [str(value) for value in predictions["ids"]],
        "metadata": predictions["metadata"],
        "binary_labels": predictions["binary_labels"].astype(int),
        "multi_labels": predictions["multi_labels"].astype(int),
        "detection_probs": sigmoid(predictions["detection_logits"]),
        "recognition_probs": sigmoid(predictions["recognition_logits"]),
        "loss": predictions["loss"],
    }


def validate_member_alignment(reference, member):
    if reference["ids"] != member["ids"]:
        raise ValueError(
            f"id order mismatch: {reference['config_path']} vs {member['config_path']}"
        )
    if not np.array_equal(reference["binary_labels"], member["binary_labels"]):
        raise ValueError(
            f"binary labels mismatch: {reference['config_path']} vs {member['config_path']}"
        )
    if not np.array_equal(reference["multi_labels"], member["multi_labels"]):
        raise ValueError(
            f"multi labels mismatch: {reference['config_path']} vs {member['config_path']}"
        )


def build_ensemble(member_predictions):
    if not member_predictions:
        raise ValueError("no member predictions")
    reference = member_predictions[0]
    for member in member_predictions[1:]:
        validate_member_alignment(reference, member)
    recognition_probs = np.stack(
        [member["recognition_probs"] for member in member_predictions], axis=0
    ).mean(axis=0)
    detection_probs = np.stack(
        [member["detection_probs"] for member in member_predictions], axis=0
    ).mean(axis=0)
    return {
        "ids": reference["ids"],
        "metadata": reference["metadata"],
        "binary_labels": reference["binary_labels"],
        "multi_labels": reference["multi_labels"],
        "recognition_probs": recognition_probs,
        "detection_probs": detection_probs,
        "member_count": len(member_predictions),
        "members": [
            {
                "config": member["config_path"],
                "checkpoint": member["checkpoint"],
                "checkpoint_epoch": member["checkpoint_epoch"],
                "loss": member["loss"],
                "seed": member["config"].get("seed"),
            }
            for member in member_predictions
        ],
    }


def write_prediction_jsonl(path, ensemble, thresholds, threshold_mode):
    ensure_dir(path.parent)
    thresholds = threshold_array(thresholds, ensemble["multi_labels"].shape[1])
    multi_preds = (
        ensemble["recognition_probs"] >= thresholds.reshape(1, -1)
    ).astype(int)
    binary_preds = (ensemble["detection_probs"] >= 0.5).astype(int)
    with path.open("w", encoding="utf-8") as f:
        for idx, sample_id in enumerate(ensemble["ids"]):
            record = {
                "id": sample_id,
                "binary_true": int(ensemble["binary_labels"][idx]),
                "binary_prob": float(ensemble["detection_probs"][idx]),
                "binary_pred": int(binary_preds[idx]),
                "multi_true": [
                    int(value) for value in ensemble["multi_labels"][idx].tolist()
                ],
                "multi_prob": [
                    float(value) for value in ensemble["recognition_probs"][idx].tolist()
                ],
                "multi_pred": [int(value) for value in multi_preds[idx].tolist()],
                "threshold_mode": threshold_mode,
                "thresholds": [float(value) for value in thresholds.tolist()],
                "member_count": int(ensemble["member_count"]),
                "metadata": ensemble["metadata"][idx],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {path.relative_to(PROJECT_ROOT)}")


def report_for_thresholds(
    split,
    checkpoint_name,
    ensemble,
    thresholds,
    threshold_mode,
    threshold_source,
    label_names,
):
    metrics = metrics_from_probs(ensemble, thresholds)
    return {
        "split": split,
        "checkpoint_name": checkpoint_name,
        "threshold_mode": threshold_mode,
        "threshold_source": threshold_source,
        "thresholds": (
            [float(value) for value in thresholds]
            if isinstance(thresholds, (list, tuple, np.ndarray))
            else float(thresholds)
        ),
        "member_count": int(ensemble["member_count"]),
        "evaluated_samples": int(len(ensemble["ids"])),
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
        "per_label_false_positive_count": metrics["per_label_false_positive_count"],
        "per_label_false_negative_count": metrics["per_label_false_negative_count"],
        "per_label_true_negative_count": metrics["per_label_true_negative_count"],
        "per_label_mean_pred_prob": metrics["per_label_mean_pred_prob"],
        "per_label_metrics": label_rows(label_names, metrics),
    }


def write_summary(result_dir, test_report, valid_report, label_names):
    label_to_row = {row["label_name"]: row for row in test_report["per_label_metrics"]}
    front = label_to_row.get("Front Running")
    bad = label_to_row.get("Bad Randomness")
    rare_mean = None
    baseline_rare_mean = (BASELINE["front_f1"] + BASELINE["bad_randomness_f1"]) / 2
    if front and bad:
        rare_mean = (front["f1"] + bad["f1"]) / 2
    macro = test_report["recognition_macro_f1"]
    micro = test_report["recognition_micro_f1"]
    acceptance = {
        "macro_improves": macro > BASELINE["macro_f1"],
        "micro_drop_within_0_002": micro >= BASELINE["micro_f1"] - 0.002,
        "rare_mean_not_lower": rare_mean is not None and rare_mean >= baseline_rare_mean,
        "main_candidate": macro >= 0.75 and micro >= BASELINE["micro_f1"],
    }
    acceptance["adopt_as_main"] = all(
        [
            acceptance["macro_improves"],
            acceptance["micro_drop_within_0_002"],
            acceptance["rare_mean_not_lower"],
        ]
    )
    summary = {
        "title": "DIVE seed ensemble summary",
        "baseline": BASELINE,
        "member_count": test_report["member_count"],
        "test_micro_f1": micro,
        "test_macro_f1": macro,
        "test_detection_f1": test_report["detection_f1"],
        "front_running_f1": None if front is None else front["f1"],
        "front_running_tp": None if front is None else front["true_positive_count"],
        "front_running_fp": None if front is None else front["false_positive_count"],
        "front_running_fn": None if front is None else front["false_negative_count"],
        "bad_randomness_f1": None if bad is None else bad["f1"],
        "bad_randomness_tp": None if bad is None else bad["true_positive_count"],
        "bad_randomness_fp": None if bad is None else bad["false_positive_count"],
        "bad_randomness_fn": None if bad is None else bad["false_negative_count"],
        "rare_mean_f1": rare_mean,
        "baseline_rare_mean_f1": baseline_rare_mean,
        "macro_delta_vs_baseline": macro - BASELINE["macro_f1"],
        "micro_delta_vs_baseline": micro - BASELINE["micro_f1"],
        "acceptance": acceptance,
        "valid_macro_f1": valid_report["recognition_macro_f1"],
        "valid_micro_f1": valid_report["recognition_micro_f1"],
        "label_names": label_names,
    }
    write_json_txt(result_dir / "summary", "DIVE seed ensemble summary", summary)
    csv_path = result_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a DIVE probability ensemble over seed checkpoints."
    )
    parser.add_argument(
        "--manifest",
        default="configs/generated/dive_seed_ensemble/manifest.yaml",
    )
    parser.add_argument("--checkpoint_name", default="best_macro_f1.pt")
    parser.add_argument(
        "--result_dir",
        default="results/dive_seed_ensemble/ensemble_best_macro",
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=500)
    parser.add_argument("--bootstrap_seed", type=int, default=20260702)
    parser.add_argument("--rare_support_threshold", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    configs = [item["config"] for item in manifest["configs"]]
    if not configs:
        raise ValueError(f"no configs in manifest: {args.manifest}")
    result_dir = resolve(args.result_dir)
    ensure_dir(result_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = load_config(str(resolve(configs[0]))).get(
        "label_names", [f"label_{idx}" for idx in range(8)]
    )
    thresholds = threshold_candidates()

    ensembles = {}
    for split in ("valid", "test"):
        print(f"[INFO] collecting {split} predictions for {len(configs)} members")
        member_predictions = [
            load_member_predictions(config, args.checkpoint_name, split, device)
            for config in configs
        ]
        ensembles[split] = build_ensemble(member_predictions)

    global_scan = save_global_threshold_scan(result_dir, ensembles["valid"], thresholds)
    selection = select_stable_thresholds(
        ensembles["valid"]["multi_labels"],
        ensembles["valid"]["recognition_probs"],
        thresholds,
        label_names,
        args.rare_support_threshold,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    selection["global_threshold_scan_path"] = str(
        result_dir / "valid_global_threshold_scan.json"
    )
    selection["best_global_macro_threshold"] = global_scan[
        "best_macro_f1_threshold"
    ]
    selection["best_global_micro_threshold"] = global_scan[
        "best_micro_f1_threshold"
    ]
    selection["bootstrap_seed"] = int(args.bootstrap_seed)
    save_per_label_thresholds(result_dir, selection)

    per_label_thresholds = selection["thresholds"]
    best_global_threshold = global_scan["best_macro_f1_threshold"]
    valid_report = report_for_thresholds(
        "valid",
        args.checkpoint_name,
        ensembles["valid"],
        per_label_thresholds,
        "per_label",
        "validation_ensemble_stable",
        label_names,
    )
    test_fixed = report_for_thresholds(
        "test",
        args.checkpoint_name,
        ensembles["test"],
        0.5,
        "global",
        "fixed_0_5_baseline",
        label_names,
    )
    test_global = report_for_thresholds(
        "test",
        args.checkpoint_name,
        ensembles["test"],
        best_global_threshold,
        "global",
        "validation_ensemble_best_macro",
        label_names,
    )
    test_per_label = report_for_thresholds(
        "test",
        args.checkpoint_name,
        ensembles["test"],
        per_label_thresholds,
        "per_label",
        "validation_ensemble_stable",
        label_names,
    )
    test_report = {
        "split": "test",
        "checkpoint_name": args.checkpoint_name,
        "threshold_source": "validation ensemble only",
        "member_manifest": args.manifest,
        "members": ensembles["test"]["members"],
        "fixed_0_5_baseline": test_fixed,
        "best_global_threshold_result": test_global,
        "per_label_threshold_result": test_per_label,
        "per_label_metrics": test_per_label["per_label_metrics"],
    }
    write_json_txt(
        result_dir / "valid_threshold_metrics",
        "DIVE seed ensemble validation metrics",
        valid_report,
    )
    write_json_txt(
        result_dir / "test_threshold_calibration_metrics",
        "DIVE seed ensemble strict test metrics",
        test_report,
    )
    write_prediction_jsonl(
        result_dir / "valid_predictions_per_label.jsonl",
        ensembles["valid"],
        per_label_thresholds,
        "per_label_validation_ensemble_stable",
    )
    write_prediction_jsonl(
        result_dir / "test_predictions_threshold_0_5.jsonl",
        ensembles["test"],
        0.5,
        "global_fixed_0_5",
    )
    write_prediction_jsonl(
        result_dir / "test_predictions_best_global.jsonl",
        ensembles["test"],
        best_global_threshold,
        "global_validation_ensemble_best_macro",
    )
    write_prediction_jsonl(
        result_dir / "test_predictions_per_label.jsonl",
        ensembles["test"],
        per_label_thresholds,
        "per_label_validation_ensemble_stable",
    )
    write_summary(result_dir, test_per_label, valid_report, label_names)


if __name__ == "__main__":
    main()
