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

    per_label_f1 = []
    for label_idx in range(labels.shape[1]):
        label_preds = preds[:, label_idx]
        label_targets = labels[:, label_idx]
        label_tp = int(((label_preds == 1) & (label_targets == 1)).sum())
        label_fp = int(((label_preds == 1) & (label_targets == 0)).sum())
        label_fn = int(((label_preds == 0) & (label_targets == 1)).sum())
        precision = safe_divide(label_tp, label_tp + label_fp)
        recall = safe_divide(label_tp, label_tp + label_fn)
        per_label_f1.append(safe_divide(2 * precision * recall, precision + recall))

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
    }


def compute_metrics(
    detection_logits, binary_labels, recognition_logits, multi_labels, threshold=0.5
):
    multi = multilabel_metrics(recognition_logits, multi_labels, threshold=threshold)
    return {
        "detection_accuracy": binary_detection_accuracy(
            detection_logits, binary_labels, threshold=threshold
        ),
        "recognition_micro_f1": multi["micro_f1"],
        "recognition_macro_f1": multi["macro_f1"],
    }
