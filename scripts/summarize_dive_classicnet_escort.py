import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation",
]
MAJOR_LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
]
INCREMENTAL_LABEL_ORDER = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
    "Front Running",
    "Bad Randomness",
]
BASELINE = {
    "micro_f1": 0.8240858035638883,
    "macro_f1": 0.7452721843450687,
    "large_mean_f1": 0.8059134403635607,
}
BRANCH50 = {
    "micro_f1": 0.829676,
    "macro_f1": 0.749556,
    "large_mean_f1": 0.808770,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_dir", required=True)
    parser.add_argument("--joint_dir", required=True)
    parser.add_argument("--output_prefix", required=True)
    return parser.parse_args()


def load_result(path):
    path = Path(path)
    predictions = [
        json.loads(line)
        for line in (path / "test_predictions_per_label.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    y_true = np.asarray([row["multi_true"] for row in predictions], dtype=int)
    y_prob = np.asarray([row["multi_prob"] for row in predictions], dtype=float)
    y_pred = np.asarray([row["multi_pred"] for row in predictions], dtype=int)
    per_label = {}
    for source_idx, label_name in enumerate(INCREMENTAL_LABEL_ORDER):
        targets = y_true[:, source_idx]
        probs = y_prob[:, source_idx]
        preds = y_pred[:, source_idx]
        tp = int(((preds == 1) & (targets == 1)).sum())
        fp = int(((preds == 1) & (targets == 0)).sum())
        fn = int(((preds == 0) & (targets == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(targets.sum()),
            "true_positive_count": tp,
            "false_positive_count": fp,
            "false_negative_count": fn,
        }
        metrics["average_precision"] = float(average_precision_score(targets, probs))
        metrics["roc_auc"] = (
            float(roc_auc_score(targets, probs))
            if len(np.unique(targets)) > 1
            else None
        )
        per_label[label_name] = metrics
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    micro_tp = int(((flat_pred == 1) & (flat_true == 1)).sum())
    micro_fp = int(((flat_pred == 1) & (flat_true == 0)).sum())
    micro_fn = int(((flat_pred == 0) & (flat_true == 1)).sum())
    micro_precision = micro_tp / max(micro_tp + micro_fp, 1)
    micro_recall = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(
        micro_precision + micro_recall, 1e-12
    )
    binary_true = y_true.any(axis=1).astype(int)
    binary_pred = y_pred.any(axis=1).astype(int)
    detection_tp = int(((binary_pred == 1) & (binary_true == 1)).sum())
    detection_fp = int(((binary_pred == 1) & (binary_true == 0)).sum())
    detection_fn = int(((binary_pred == 0) & (binary_true == 1)).sum())
    detection_precision = detection_tp / max(detection_tp + detection_fp, 1)
    detection_recall = detection_tp / max(detection_tp + detection_fn, 1)
    detection_f1 = 2 * detection_precision * detection_recall / max(
        detection_precision + detection_recall, 1e-12
    )
    large_mean = float(np.mean([per_label[name]["f1"] for name in MAJOR_LABELS]))
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean([row["f1"] for row in per_label.values()])),
        "detection_f1": float(detection_f1),
        "large_mean_f1": large_mean,
        "per_label": per_label,
    }


def main():
    args = parse_args()
    results = {
        "incremental_final": load_result(args.final_dir),
        "joint8_bs128_control": load_result(args.joint_dir),
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline": BASELINE,
        "branch50": BRANCH50,
        "results": results,
        "incremental_minus_joint": {
            metric: results["incremental_final"][metric]
            - results["joint8_bs128_control"][metric]
            for metric in ("micro_f1", "macro_f1", "large_mean_f1")
        },
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    csv_rows = []
    for model_name, result in results.items():
        for label_name in LABELS:
            row = result["per_label"][label_name]
            csv_rows.append(
                {
                    "model": model_name,
                    "label": label_name,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "support": row["support"],
                    "tp": row["true_positive_count"],
                    "fp": row["false_positive_count"],
                    "fn": row["false_negative_count"],
                    "ap": row["average_precision"],
                    "auc": row["roc_auc"],
                }
            )
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "DIVE ClassicNet-ESCORT summary",
        "",
        f"baseline: {BASELINE}",
        f"branch50: {BRANCH50}",
        "",
    ]
    for model_name, result in results.items():
        lines.append(
            f"{model_name}: micro={result['micro_f1']:.6f} "
            f"macro={result['macro_f1']:.6f} "
            f"large_mean={result['large_mean_f1']:.6f} "
            f"detection={result['detection_f1']:.6f}"
        )
        for label_name in LABELS:
            row = result["per_label"][label_name]
            lines.append(
                f"  {label_name}: f1={row['f1']:.6f} "
                f"tp={row['true_positive_count']} fp={row['false_positive_count']} "
                f"fn={row['false_negative_count']} ap={row['average_precision']:.6f} "
                f"auc={row['roc_auc']:.6f}"
            )
    lines.extend(
        [
            "",
            f"incremental_minus_joint: {report['incremental_minus_joint']}",
        ]
    )
    prefix.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {prefix.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
