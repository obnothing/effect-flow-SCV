import numpy as np


def sigmoid(logits):
    logits = np.asarray(logits)
    return 1 / (1 + np.exp(-logits))


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0


def binary_detection_accuracy(logits, labels, threshold=0.5):
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    return float((preds == labels).mean()) if labels.size > 0 else 0.0


def binary_detection_metrics(logits, labels, threshold=0.5):
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    return {
        "detection_accuracy": accuracy,
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


def multilabel_metrics(logits, labels, threshold=0.5):
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    micro_precision = safe_divide(tp, tp + fp)
    micro_recall = safe_divide(tp, tp + fn)
    micro_f1 = safe_divide(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )

    per_label_accuracy = []
    per_label_precision = []
    per_label_recall = []
    per_label_f1 = []
    for label_idx in range(labels.shape[1]):
        label_preds = preds[:, label_idx]
        label_targets = labels[:, label_idx]
        label_tp = int(((label_preds == 1) & (label_targets == 1)).sum())
        label_fp = int(((label_preds == 1) & (label_targets == 0)).sum())
        label_fn = int(((label_preds == 0) & (label_targets == 1)).sum())
        label_tn = int(((label_preds == 0) & (label_targets == 0)).sum())
        accuracy = safe_divide(label_tp + label_tn, len(label_targets))
        precision = safe_divide(label_tp, label_tp + label_fp)
        recall = safe_divide(label_tp, label_tp + label_fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label_accuracy.append(accuracy)
        per_label_precision.append(precision)
        per_label_recall.append(recall)
        per_label_f1.append(f1)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": (
            float(np.mean(per_label_precision)) if per_label_precision else 0.0
        ),
        "macro_recall": (
            float(np.mean(per_label_recall)) if per_label_recall else 0.0
        ),
        "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "per_label_accuracy": [float(value) for value in per_label_accuracy],
        "per_label_precision": [float(value) for value in per_label_precision],
        "per_label_recall": [float(value) for value in per_label_recall],
        "per_label_f1": [float(value) for value in per_label_f1],
        "per_label_support": [int(value) for value in labels.sum(axis=0)],
    }


def prediction_distribution(logits, labels, threshold=0.5):
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    return {
        "predicted_positive_total": int(preds.sum()),
        "per_label_predicted_positive_count": [
            int(value) for value in preds.sum(axis=0)
        ],
        "per_label_true_positive_count": [
            int(value) for value in labels.sum(axis=0)
        ],
        "per_label_mean_pred_prob": [
            float(value) for value in probs.mean(axis=0)
        ],
    }


def threshold_scan(logits, labels, thresholds):
    results = {}
    for threshold in thresholds:
        multi = multilabel_metrics(logits, labels, threshold=threshold)
        dist = prediction_distribution(logits, labels, threshold=threshold)
        results[str(threshold)] = {
            "micro_precision": multi["micro_precision"],
            "micro_recall": multi["micro_recall"],
            "micro_f1": multi["micro_f1"],
            "macro_precision": multi["macro_precision"],
            "macro_recall": multi["macro_recall"],
            "macro_f1": multi["macro_f1"],
            "predicted_positive_total": dist["predicted_positive_total"],
        }
    return results


def compute_metrics(
    detection_logits,
    binary_labels,
    recognition_logits,
    multi_labels,
    threshold=0.5,
    scan_thresholds=None,
):
    multi = multilabel_metrics(recognition_logits, multi_labels, threshold=threshold)
    detection = binary_detection_metrics(
        detection_logits, binary_labels, threshold=threshold
    )
    metrics = {
        "recognition_micro_f1": multi["micro_f1"],
        "recognition_micro_precision": multi["micro_precision"],
        "recognition_micro_recall": multi["micro_recall"],
        "recognition_macro_f1": multi["macro_f1"],
        "recognition_macro_precision": multi["macro_precision"],
        "recognition_macro_recall": multi["macro_recall"],
        "per_label_accuracy": multi["per_label_accuracy"],
        "per_label_precision": multi["per_label_precision"],
        "per_label_recall": multi["per_label_recall"],
        "per_label_f1": multi["per_label_f1"],
        "per_label_support": multi["per_label_support"],
    }
    metrics.update(detection)
    metrics.update(
        prediction_distribution(recognition_logits, multi_labels, threshold=threshold)
    )
    if scan_thresholds:
        metrics["threshold_scan"] = threshold_scan(
            recognition_logits, multi_labels, scan_thresholds
        )
    return metrics
