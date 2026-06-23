import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_GRID = np.arange(0.05, 1.0, 0.05)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-selected probability/logit late fusion."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_prediction_file(path):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction artifact: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "id",
                "binary_true",
                "binary_prob",
                "multi_true",
                "multi_prob",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            rows.append(row)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate ids; id alignment is ambiguous")
    return rows


def align_model_predictions(model_specs, split):
    loaded = {
        spec["name"]: load_prediction_file(spec[f"{split}_path"])
        for spec in model_specs
    }
    base_name = model_specs[0]["name"]
    base_rows = loaded[base_name]
    base_ids = [str(row["id"]) for row in base_rows]
    base_set = set(base_ids)
    aligned = {}
    base_binary_true = np.asarray([row["binary_true"] for row in base_rows], dtype=int)
    base_multi_true = np.asarray([row["multi_true"] for row in base_rows], dtype=int)

    for spec in model_specs:
        name = spec["name"]
        rows_by_id = {str(row["id"]): row for row in loaded[name]}
        if set(rows_by_id) != base_set:
            missing = len(base_set - set(rows_by_id))
            extra = len(set(rows_by_id) - base_set)
            raise ValueError(
                f"{split} id set mismatch for {name}: missing={missing}, extra={extra}"
            )
        rows = [rows_by_id[sample_id] for sample_id in base_ids]
        binary_true = np.asarray([row["binary_true"] for row in rows], dtype=int)
        multi_true = np.asarray([row["multi_true"] for row in rows], dtype=int)
        if not np.array_equal(binary_true, base_binary_true):
            raise ValueError(f"{split} binary_true mismatch for model {name}")
        if not np.array_equal(multi_true, base_multi_true):
            raise ValueError(f"{split} multi_true mismatch for model {name}")
        aligned[name] = {
            "binary_prob": np.asarray([row["binary_prob"] for row in rows], dtype=float),
            "multi_prob": np.asarray([row["multi_prob"] for row in rows], dtype=float),
        }
    return {
        "ids": base_ids,
        "binary_true": base_binary_true,
        "multi_true": base_multi_true,
        "models": aligned,
    }


def logit(values):
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def fuse_arrays(arrays, weights, fusion_type):
    stacked = np.stack(arrays, axis=0)
    weight_shape = (len(weights),) + (1,) * (stacked.ndim - 1)
    weights = np.asarray(weights, dtype=float).reshape(weight_shape)
    if fusion_type == "probability":
        return np.sum(weights * stacked, axis=0)
    if fusion_type == "logit":
        return sigmoid(np.sum(weights * logit(stacked), axis=0))
    raise ValueError(f"Unsupported fusion_type: {fusion_type}")


def weight_grid(size):
    if size == 2:
        return [
            (float(round(value, 2)), float(round(1.0 - value, 2)))
            for value in np.arange(0, 1.01, 0.05)
        ]
    if size == 3:
        rows = []
        for first in range(11):
            for second in range(11 - first):
                third = 10 - first - second
                rows.append((first / 10.0, second / 10.0, third / 10.0))
        return rows
    raise ValueError("Stage 14A grid search supports two or three models")


def multilabel_metrics(y_true, probabilities, thresholds):
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.full(y_true.shape[1], float(thresholds))
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )
    return {
        "predictions": predictions,
        "micro_precision": precision_score(y_true, predictions, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, predictions, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, predictions, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
        "samples_f1": f1_score(y_true, predictions, average="samples", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, predictions),
        "hamming_loss": hamming_loss(y_true, predictions),
        "predicted_positive_total": int(predictions.sum()),
        "true_positive_total": int(y_true.sum()),
        "per_label_precision": precision.tolist(),
        "per_label_recall": recall.tolist(),
        "per_label_f1": f1.tolist(),
        "per_label_support": support.astype(int).tolist(),
        "per_label_predicted_positive_count": predictions.sum(axis=0).astype(int).tolist(),
    }


def binary_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= float(threshold)).astype(int)
    return {
        "predictions": predictions,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "accuracy": accuracy_score(y_true, predictions),
        "predicted_positive_total": int(predictions.sum()),
        "true_positive_total": int(y_true.sum()),
    }


def select_global_threshold(y_true, probabilities):
    grid_predictions, _, _, _, per_label_f1 = threshold_grid_counts(
        y_true,
        probabilities,
    )
    macro_f1 = per_label_f1.mean(axis=0)
    predicted_totals = grid_predictions.sum(axis=(0, 1))
    true_total = int(y_true.sum())
    best_idx = max(
        range(len(THRESHOLD_GRID)),
        key=lambda idx: (
            macro_f1[idx],
            -abs(int(predicted_totals[idx]) - true_total),
            -abs(float(THRESHOLD_GRID[idx]) - 0.5),
        ),
    )
    threshold = float(THRESHOLD_GRID[best_idx])
    return threshold, multilabel_metrics(y_true, probabilities, threshold)


def select_per_label_thresholds(y_true, probabilities, fallback=0.5):
    grid_predictions, precision, recall, _, per_label_f1 = threshold_grid_counts(
        y_true,
        probabilities,
    )
    thresholds = []
    for label_id in range(y_true.shape[1]):
        support = int(y_true[:, label_id].sum())
        if support == 0:
            thresholds.append(float(fallback))
            continue
        predicted_counts = grid_predictions[:, label_id, :].sum(axis=0)
        best_idx = max(
            range(len(THRESHOLD_GRID)),
            key=lambda idx: (
                per_label_f1[label_id, idx],
                -abs(precision[label_id, idx] - recall[label_id, idx]),
                -abs(int(predicted_counts[idx]) - support),
                -abs(float(THRESHOLD_GRID[idx]) - 0.5),
            ),
        )
        thresholds.append(float(THRESHOLD_GRID[best_idx]))
    return thresholds, multilabel_metrics(y_true, probabilities, thresholds)


def select_binary_threshold(y_true, probabilities):
    predictions = probabilities[:, None] >= THRESHOLD_GRID[None, :]
    targets = y_true[:, None].astype(bool)
    tp = np.logical_and(predictions, targets).sum(axis=0)
    fp = np.logical_and(predictions, ~targets).sum(axis=0)
    fn = np.logical_and(~predictions, targets).sum(axis=0)
    denominator = 2 * tp + fp + fn
    f1_values = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator > 0,
    )
    predicted_totals = predictions.sum(axis=0)
    true_total = int(y_true.sum())
    best_idx = max(
        range(len(THRESHOLD_GRID)),
        key=lambda idx: (
            f1_values[idx],
            -abs(int(predicted_totals[idx]) - true_total),
            -abs(float(THRESHOLD_GRID[idx]) - 0.5),
        ),
    )
    threshold = float(THRESHOLD_GRID[best_idx])
    return threshold, binary_metrics(y_true, probabilities, threshold)


def threshold_grid_counts(y_true, probabilities):
    predictions = probabilities[:, :, None] >= THRESHOLD_GRID[None, None, :]
    targets = y_true[:, :, None].astype(bool)
    tp = np.logical_and(predictions, targets).sum(axis=0)
    fp = np.logical_and(predictions, ~targets).sum(axis=0)
    fn = np.logical_and(~predictions, targets).sum(axis=0)
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) > 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fn) > 0,
    )
    per_label_f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) > 0,
    )
    return predictions, precision, recall, tp, per_label_f1


def threshold_protocols(y_true, probabilities):
    fixed = multilabel_metrics(y_true, probabilities, 0.5)
    grid_predictions, precision, recall, _, grid_f1 = threshold_grid_counts(
        y_true,
        probabilities,
    )
    predicted_totals = grid_predictions.sum(axis=(0, 1))
    true_total = int(y_true.sum())
    macro_f1 = grid_f1.mean(axis=0)
    global_idx = max(
        range(len(THRESHOLD_GRID)),
        key=lambda idx: (
            macro_f1[idx],
            -abs(int(predicted_totals[idx]) - true_total),
            -abs(float(THRESHOLD_GRID[idx]) - 0.5),
        ),
    )
    global_threshold = float(THRESHOLD_GRID[global_idx])
    global_metrics = multilabel_metrics(y_true, probabilities, global_threshold)

    per_label_thresholds = []
    for label_id in range(y_true.shape[1]):
        support = int(y_true[:, label_id].sum())
        if support == 0:
            per_label_thresholds.append(global_threshold)
            continue
        label_predicted_totals = grid_predictions[:, label_id, :].sum(axis=0)
        best_idx = max(
            range(len(THRESHOLD_GRID)),
            key=lambda idx: (
                grid_f1[label_id, idx],
                -abs(precision[label_id, idx] - recall[label_id, idx]),
                -abs(int(label_predicted_totals[idx]) - support),
                -abs(float(THRESHOLD_GRID[idx]) - 0.5),
            ),
        )
        per_label_thresholds.append(float(THRESHOLD_GRID[best_idx]))
    per_label_metrics = multilabel_metrics(
        y_true,
        probabilities,
        per_label_thresholds,
    )
    return [
        ("fixed_0.5", [0.5] * y_true.shape[1], fixed),
        ("best_global", [global_threshold] * y_true.shape[1], global_metrics),
        ("per_label", per_label_thresholds, per_label_metrics),
    ]


def threshold_mode_rank(mode):
    return {"fixed_0.5": 0, "best_global": 1, "per_label": 2}[mode]


def candidate_key(candidate, true_positive_total):
    return (
        candidate["valid_macro_f1"],
        -abs(candidate["valid_predicted_positives"] - true_positive_total),
        -threshold_mode_rank(candidate["threshold_mode"]),
        -len(candidate["model_names"]),
    )


def search_fusion(valid, config):
    names = [spec["name"] for spec in config["models"]]
    max_models = min(int(config.get("max_models_per_fusion", 3)), len(names))
    fusion_types = config.get("fusion_types", ["logit", "probability"])
    rows = []
    detection_candidates = []
    y_multi = valid["multi_true"]
    y_binary = valid["binary_true"]

    for size in range(2, max_models + 1):
        for combination in itertools.combinations(names, size):
            multi_arrays = [valid["models"][name]["multi_prob"] for name in combination]
            binary_arrays = [valid["models"][name]["binary_prob"] for name in combination]
            for fusion_type in fusion_types:
                for weights in weight_grid(size):
                    multi_prob = fuse_arrays(multi_arrays, weights, fusion_type)
                    binary_prob = fuse_arrays(binary_arrays, weights, fusion_type)
                    detection_threshold, detection = select_binary_threshold(
                        y_binary,
                        binary_prob,
                    )
                    detection_candidates.append(
                        {
                            "model_names": list(combination),
                            "fusion_type": fusion_type,
                            "weights": list(weights),
                            "threshold": detection_threshold,
                            "valid_detection_f1": detection["f1"],
                            "valid_predicted_positives": detection["predicted_positive_total"],
                        }
                    )
                    for threshold_mode, thresholds, metrics in threshold_protocols(
                        y_multi,
                        multi_prob,
                    ):
                        rows.append(
                            {
                                "fusion_type": fusion_type,
                                "model_combination": "+".join(combination),
                                "model_names": list(combination),
                                "weights": list(weights),
                                "threshold_mode": threshold_mode,
                                "thresholds": [float(value) for value in thresholds],
                                "valid_micro_f1": metrics["micro_f1"],
                                "valid_macro_f1": metrics["macro_f1"],
                                "valid_macro_precision": metrics["macro_precision"],
                                "valid_macro_recall": metrics["macro_recall"],
                                "valid_predicted_positives": metrics["predicted_positive_total"],
                                "valid_detection_f1": detection["f1"],
                            }
                        )

    selected_recognition = max(
        rows,
        key=lambda row: candidate_key(row, int(y_multi.sum())),
    )
    selected_detection = max(
        detection_candidates,
        key=lambda row: (
            row["valid_detection_f1"],
            -abs(row["valid_predicted_positives"] - int(y_binary.sum())),
            -len(row["model_names"]),
        ),
    )
    return rows, selected_recognition, selected_detection


def select_individual_models(valid, config):
    results = {}
    y_multi = valid["multi_true"]
    y_binary = valid["binary_true"]
    for spec in config["models"]:
        name = spec["name"]
        probs = valid["models"][name]
        candidates = []
        for mode, thresholds, metrics in threshold_protocols(y_multi, probs["multi_prob"]):
            candidates.append(
                {
                    "threshold_mode": mode,
                    "thresholds": thresholds,
                    "valid_micro_f1": metrics["micro_f1"],
                    "valid_macro_f1": metrics["macro_f1"],
                    "valid_predicted_positives": metrics["predicted_positive_total"],
                }
            )
        selected = max(
            candidates,
            key=lambda row: candidate_key(
                {**row, "model_names": [name]},
                int(y_multi.sum()),
            ),
        )
        detection_threshold, detection = select_binary_threshold(
            y_binary,
            probs["binary_prob"],
        )
        selected["detection_threshold"] = detection_threshold
        selected["valid_detection_f1"] = detection["f1"]
        results[name] = selected
    strongest_name = max(
        results,
        key=lambda name: (
            results[name]["valid_macro_f1"],
            results[name]["valid_micro_f1"],
        ),
    )
    return results, strongest_name


def apply_selected_fusion(split_data, selection, probability_key):
    arrays = [
        split_data["models"][name][probability_key]
        for name in selection["model_names"]
    ]
    return fuse_arrays(arrays, selection["weights"], selection["fusion_type"])


def per_label_table(label_names, metrics):
    return [
        {
            "label_id": idx,
            "label_name": label_names[idx],
            "precision": metrics["per_label_precision"][idx],
            "recall": metrics["per_label_recall"][idx],
            "f1": metrics["per_label_f1"][idx],
            "support": metrics["per_label_support"][idx],
            "predicted_positive_count": metrics["per_label_predicted_positive_count"][idx],
        }
        for idx in range(len(label_names))
    ]


def evaluate_individuals(test, valid_selections, config):
    rows = {}
    for spec in config["models"]:
        name = spec["name"]
        selection = valid_selections[name]
        recognition = multilabel_metrics(
            test["multi_true"],
            test["models"][name]["multi_prob"],
            selection["thresholds"],
        )
        detection = binary_metrics(
            test["binary_true"],
            test["models"][name]["binary_prob"],
            selection["detection_threshold"],
        )
        rows[name] = {
            "selected_on_validation": selection,
            "test_recognition": {key: value for key, value in recognition.items() if key != "predictions"},
            "test_detection": {key: value for key, value in detection.items() if key != "predictions"},
            "test_recognition_predictions": recognition["predictions"],
            "test_detection_predictions": detection["predictions"],
        }
    return rows


def simple_f1_scores(y_multi, pred_multi, y_binary, pred_binary):
    return np.asarray(
        [
            f1_score(y_multi, pred_multi, average="micro", zero_division=0),
            f1_score(y_multi, pred_multi, average="macro", zero_division=0),
            f1_score(y_binary, pred_binary, zero_division=0),
        ]
    )


def paired_bootstrap(
    y_multi,
    fusion_multi,
    individual_multi,
    y_binary,
    fusion_binary,
    individual_binary,
    samples=1000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    observed = simple_f1_scores(y_multi, fusion_multi, y_binary, fusion_binary) - simple_f1_scores(
        y_multi,
        individual_multi,
        y_binary,
        individual_binary,
    )
    deltas = np.empty((samples, 3), dtype=float)
    count = len(y_multi)
    for sample_idx in range(samples):
        indices = rng.integers(0, count, count)
        deltas[sample_idx] = simple_f1_scores(
            y_multi[indices],
            fusion_multi[indices],
            y_binary[indices],
            fusion_binary[indices],
        ) - simple_f1_scores(
            y_multi[indices],
            individual_multi[indices],
            y_binary[indices],
            individual_binary[indices],
        )
    names = ["micro_f1", "macro_f1", "detection_f1"]
    report = {"bootstrap_samples": samples, "seed": seed, "metrics": {}}
    for idx, name in enumerate(names):
        lower, upper = np.percentile(deltas[:, idx], [2.5, 97.5])
        report["metrics"][name] = {
            "observed_delta": float(observed[idx]),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
            "ci_crosses_zero": bool(lower <= 0.0 <= upper),
        }
    return report


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot JSON encode {type(value)}")


def write_search_csv(path, rows):
    fields = [
        "fusion_type",
        "model_combination",
        "weights",
        "threshold_mode",
        "thresholds",
        "valid_micro_f1",
        "valid_macro_f1",
        "valid_macro_precision",
        "valid_macro_recall",
        "valid_predicted_positives",
        "valid_detection_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field])
                    if field in {"weights", "thresholds"}
                    else row[field]
                    for field in fields
                }
            )


def write_predictions(path, test, multi_prob, multi_pred, binary_prob, binary_pred, best_config):
    with path.open("w", encoding="utf-8") as f:
        for idx, sample_id in enumerate(test["ids"]):
            row = {
                "id": sample_id,
                "binary_true": int(test["binary_true"][idx]),
                "binary_prob": float(binary_prob[idx]),
                "binary_pred": int(binary_pred[idx]),
                "multi_true": test["multi_true"][idx].astype(int).tolist(),
                "multi_prob": multi_prob[idx].astype(float).tolist(),
                "multi_pred": multi_pred[idx].astype(int).tolist(),
                "recognition_models": best_config["selected_model_combination"],
                "recognition_weights": best_config["recognition_weights"],
                "detection_models": best_config["detection_model_combination"],
                "detection_weights": best_config["detection_weights"],
                "threshold_mode": best_config["threshold_mode"],
                "thresholds": best_config["thresholds"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text_report(path, report):
    metrics = report["fusion_test_metrics"]
    lines = [
        "Stage 14A validation-selected late fusion report",
        "",
        f"experiment_name: {report['experiment_name']}",
        f"split_policy: {report['split_policy']}",
        f"strict_grouped_split: {report['strict_grouped_split']}",
        f"test_leakage: {report['test_leakage']}",
        f"models: {report['participating_models']}",
        f"selected_recognition: {report['best_config']['selected_model_combination']}",
        f"recognition_fusion_type: {report['best_config']['fusion_type']}",
        f"recognition_weights: {report['best_config']['recognition_weights']}",
        f"threshold_mode: {report['best_config']['threshold_mode']}",
        f"thresholds: {report['best_config']['thresholds']}",
        f"selected_detection: {report['best_config']['detection_model_combination']}",
        f"detection_fusion_type: {report['best_config']['detection_fusion_type']}",
        f"detection_weights: {report['best_config']['detection_weights']}",
        f"detection_threshold: {report['best_config']['detection_threshold']}",
        "",
        f"test_micro_precision: {metrics['recognition']['micro_precision']:.6f}",
        f"test_micro_recall: {metrics['recognition']['micro_recall']:.6f}",
        f"test_micro_f1: {metrics['recognition']['micro_f1']:.6f}",
        f"test_macro_precision: {metrics['recognition']['macro_precision']:.6f}",
        f"test_macro_recall: {metrics['recognition']['macro_recall']:.6f}",
        f"test_macro_f1: {metrics['recognition']['macro_f1']:.6f}",
        f"test_weighted_f1: {metrics['recognition']['weighted_f1']:.6f}",
        f"test_samples_f1: {metrics['recognition']['samples_f1']:.6f}",
        f"test_subset_accuracy: {metrics['recognition']['subset_accuracy']:.6f}",
        f"test_hamming_loss: {metrics['recognition']['hamming_loss']:.6f}",
        f"test_detection_precision: {metrics['detection']['precision']:.6f}",
        f"test_detection_recall: {metrics['detection']['recall']:.6f}",
        f"test_detection_f1: {metrics['detection']['f1']:.6f}",
        f"predicted_positive_total: {metrics['recognition']['predicted_positive_total']}",
        f"true_positive_total: {metrics['recognition']['true_positive_total']}",
        f"strongest_individual_selected_on_validation: {report['strongest_individual_name']}",
        f"fusion_micro_f1_delta: {report['fusion_vs_strongest_individual']['micro_f1_delta']:+.6f}",
        f"fusion_macro_f1_delta: {report['fusion_vs_strongest_individual']['macro_f1_delta']:+.6f}",
        f"fusion_detection_f1_delta: {report['fusion_vs_strongest_individual']['detection_f1_delta']:+.6f}",
        f"recommend_as_current_best: {report['recommend_as_current_best']}",
        f"recommend_as_random_upper_bound: {report['recommend_as_random_upper_bound']}",
        "",
        "Individual model test metrics (model and thresholds selected on validation):",
        "model | threshold_mode | micro_f1 | macro_f1 | detection_f1",
    ]
    for model_name, row in report["individual_models"].items():
        lines.append(
            f"{model_name} | {row['selected_on_validation']['threshold_mode']} | "
            f"{row['test_recognition']['micro_f1']:.6f} | "
            f"{row['test_recognition']['macro_f1']:.6f} | "
            f"{row['test_detection']['f1']:.6f}"
        )
    lines.extend(
        [
            "",
            "Paired bootstrap versus strongest validation-selected individual:",
        ]
    )
    for metric_name, row in report["paired_bootstrap"]["metrics"].items():
        lines.append(
            f"{metric_name}: delta={row['observed_delta']:+.6f} "
            f"95%CI=[{row['ci_95_lower']:+.6f}, {row['ci_95_upper']:+.6f}] "
            f"crosses_zero={row['ci_crosses_zero']}"
        )
    lines.extend(
        [
            "",
            f"reference_baselines: {report['reference_baselines']}",
            "",
        "Per-label metrics:",
        "label | precision | recall | f1 | support | predicted",
        ]
    )
    for row in report["per_label"]:
        lines.append(
            f"{row['label_name']} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {row['support']} | {row['predicted_positive_count']}"
        )
    lines.extend(["", f"warning: {report['warning']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test predictions are intentionally not loaded until all validation-only
    # model, fusion-weight, and threshold selections have been frozen.
    valid = align_model_predictions(config["models"], "valid")
    if valid["multi_true"].shape[1] != len(config["label_names"]):
        raise ValueError("Validation label dimension does not match label_names")
    rows, selected_recognition, selected_detection = search_fusion(valid, config)
    individual_valid, strongest_individual_name = select_individual_models(valid, config)

    best_config = {
        "selected_model_combination": selected_recognition["model_names"],
        "fusion_type": selected_recognition["fusion_type"],
        "recognition_weights": selected_recognition["weights"],
        "detection_model_combination": selected_detection["model_names"],
        "detection_fusion_type": selected_detection["fusion_type"],
        "detection_weights": selected_detection["weights"],
        "detection_threshold": selected_detection["threshold"],
        "threshold_mode": selected_recognition["threshold_mode"],
        "thresholds": selected_recognition["thresholds"],
        "selected_by": "validation_macro_f1",
        "validation_macro_f1": selected_recognition["valid_macro_f1"],
        "validation_micro_f1": selected_recognition["valid_micro_f1"],
        "validation_detection_f1": selected_detection["valid_detection_f1"],
    }
    write_search_csv(output_dir / "fusion_valid_search_results.csv", rows)
    (output_dir / "fusion_best_config.json").write_text(
        json.dumps(best_config, indent=2),
        encoding="utf-8",
    )

    test = align_model_predictions(config["models"], "test")
    if test["multi_true"].shape[1] != len(config["label_names"]):
        raise ValueError("Test label dimension does not match label_names")
    fusion_multi_prob = apply_selected_fusion(test, selected_recognition, "multi_prob")
    fusion_binary_prob = apply_selected_fusion(test, selected_detection, "binary_prob")
    recognition = multilabel_metrics(
        test["multi_true"],
        fusion_multi_prob,
        selected_recognition["thresholds"],
    )
    detection = binary_metrics(
        test["binary_true"],
        fusion_binary_prob,
        selected_detection["threshold"],
    )
    individual_test = evaluate_individuals(test, individual_valid, config)
    strongest = individual_test[strongest_individual_name]
    comparison = {
        "micro_f1_delta": recognition["micro_f1"] - strongest["test_recognition"]["micro_f1"],
        "macro_f1_delta": recognition["macro_f1"] - strongest["test_recognition"]["macro_f1"],
        "detection_f1_delta": detection["f1"] - strongest["test_detection"]["f1"],
    }
    bootstrap = paired_bootstrap(
        test["multi_true"],
        recognition["predictions"],
        strongest["test_recognition_predictions"],
        test["binary_true"],
        detection["predictions"],
        strongest["test_detection_predictions"],
        samples=int(config.get("bootstrap_samples", 1000)),
        seed=int(config.get("seed", 42)),
    )
    for row in individual_test.values():
        row.pop("test_recognition_predictions", None)
        row.pop("test_detection_predictions", None)

    strict = bool(config.get("strict_grouped_split", False))
    current_best = float(config.get("current_best_macro_f1", 0.0))
    recommendation = bool(strict and recognition["macro_f1"] > current_best)
    random_upper_bound_recommendation = bool(
        not strict and recognition["macro_f1"] > current_best
    )
    report = {
        "experiment_name": config["experiment_name"],
        "participating_models": [spec["name"] for spec in config["models"]],
        "split_policy": config["split_policy"],
        "strict_grouped_split": strict,
        "test_leakage": bool(config.get("test_leakage", False)),
        "evaluated_samples": len(test["ids"]),
        "best_config": best_config,
        "fusion_test_metrics": {
            "recognition": {key: value for key, value in recognition.items() if key != "predictions"},
            "detection": {key: value for key, value in detection.items() if key != "predictions"},
        },
        "per_label": per_label_table(config["label_names"], recognition),
        "individual_models": individual_test,
        "strongest_individual_name": strongest_individual_name,
        "fusion_vs_strongest_individual": comparison,
        "paired_bootstrap": bootstrap,
        "reference_baselines": config.get("reference_baselines", {}),
        "recommend_as_current_best": recommendation,
        "recommend_as_random_upper_bound": random_upper_bound_recommendation,
        "success_criteria": {
            "beats_current_strict_best": bool(strict and recognition["macro_f1"] > current_best),
            "strict_macro_f1_at_least_0_50": bool(strict and recognition["macro_f1"] >= 0.50),
            "beats_current_random_upper_bound": random_upper_bound_recommendation,
        },
        "warning": config.get("warning", "none"),
    }
    metrics_json = output_dir / "fusion_test_metrics.json"
    metrics_txt = output_dir / "fusion_test_metrics.txt"
    metrics_json.write_text(
        json.dumps(report, indent=2, default=json_ready),
        encoding="utf-8",
    )
    write_text_report(metrics_txt, report)
    write_predictions(
        output_dir / "fusion_test_predictions.jsonl",
        test,
        fusion_multi_prob,
        recognition["predictions"],
        fusion_binary_prob,
        detection["predictions"],
        best_config,
    )

    bootstrap_json = output_dir / "fusion_bootstrap_report.json"
    bootstrap_txt = output_dir / "fusion_bootstrap_report.txt"
    bootstrap_json.write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    bootstrap_lines = [
        "Stage 14A paired bootstrap report",
        "",
        f"strongest_individual: {strongest_individual_name}",
        f"bootstrap_samples: {bootstrap['bootstrap_samples']}",
        f"seed: {bootstrap['seed']}",
    ]
    for name, row in bootstrap["metrics"].items():
        bootstrap_lines.append(
            f"{name}: delta={row['observed_delta']:+.6f} "
            f"95%CI=[{row['ci_95_lower']:+.6f}, {row['ci_95_upper']:+.6f}] "
            f"crosses_zero={row['ci_crosses_zero']}"
        )
    bootstrap_txt.write_text("\n".join(bootstrap_lines) + "\n", encoding="utf-8")

    summary_prefix = resolve_path(config["summary_path"])
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_json = summary_prefix.with_suffix(".json")
    summary_txt = summary_prefix.with_suffix(".txt")
    summary_json.write_text(
        json.dumps(report, indent=2, default=json_ready),
        encoding="utf-8",
    )
    write_text_report(summary_txt, report)
    print(f"[OK] wrote {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {summary_txt.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
