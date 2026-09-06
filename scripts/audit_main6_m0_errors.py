"""Validation-only error audit for the frozen DIVE Main-6 M0 model.

This script does not train a model and never reads test data.  It separates
ordinary per-label errors from multi-label interference and reports possible
train/valid duplicate normalized opcode hashes.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import (  # noqa: E402
    compute_multilabel_metrics_from_probs,
    select_per_label_thresholds,
)
from train_evidence_retrieval_mvp import (  # noqa: E402
    load_config,
    load_m0,
    load_split,
    make_pos_weight,
    run_epoch,
)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def metric_dict(metric):
    return {
        "macro_f1": float(metric["recognition_macro_f1"]),
        "micro_f1": float(metric["recognition_micro_f1"]),
        "macro_precision": float(metric["recognition_macro_precision"]),
        "macro_recall": float(metric["recognition_macro_recall"]),
        "per_label_f1": [float(x) for x in metric["per_label_f1"]],
        "per_label_precision": [float(x) for x in metric["per_label_precision"]],
        "per_label_recall": [float(x) for x in metric["per_label_recall"]],
        "support": [int(x) for x in metric["per_label_support"]],
    }


def confusion_rows(label_names, labels, probs, thresholds):
    targets = np.asarray(labels).astype(int)
    predictions = (
        np.asarray(probs) >= np.asarray(thresholds, dtype=float).reshape(1, -1)
    ).astype(int)
    rows = []
    for label_id, label_name in enumerate(label_names):
        target = targets[:, label_id]
        prediction = predictions[:, label_id]
        tp = int(((prediction == 1) & (target == 1)).sum())
        fp = int(((prediction == 1) & (target == 0)).sum())
        fn = int(((prediction == 0) & (target == 1)).sum())
        tn = int(((prediction == 0) & (target == 0)).sum())
        rows.append(
            {
                "label_id": label_id,
                "label_name": label_name,
                "threshold": float(thresholds[label_id]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "support": int(target.sum()),
                "predicted_positive": int(prediction.sum()),
                "false_positive_rate": float(fp / max(fp + tn, 1)),
                "false_negative_rate": float(fn / max(fn + tp, 1)),
            }
        )
    return rows


def evaluate_subset(label_names, labels, probs, thresholds, mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"contracts": 0, "metrics": None}
    metric = compute_multilabel_metrics_from_probs(
        np.asarray(labels)[mask], np.asarray(probs)[mask], thresholds
    )
    return {"contracts": int(mask.sum()), "metrics": metric_dict(metric)}


def conditional_rows(label_names, labels, probs, thresholds):
    labels = np.asarray(labels).astype(int)
    predictions = (
        np.asarray(probs) >= np.asarray(thresholds).reshape(1, -1)
    ).astype(int)
    rows = []
    for target_id, target_name in enumerate(label_names):
        for other_id, other_name in enumerate(label_names):
            if target_id == other_id:
                continue
            negative_mask = (labels[:, target_id] == 0) & (labels[:, other_id] == 1)
            positive_mask = (labels[:, target_id] == 1) & (labels[:, other_id] == 1)
            negative_count = int(negative_mask.sum())
            positive_count = int(positive_mask.sum())
            conditional_fp = int(predictions[negative_mask, target_id].sum())
            conditional_fn = int(
                (predictions[positive_mask, target_id] == 0).sum()
            )
            rows.append(
                {
                    "target_label": target_name,
                    "conditioning_label": other_name,
                    "target_negative_conditioning_positive_count": negative_count,
                    "conditional_false_positive_count": conditional_fp,
                    "conditional_fpr": float(conditional_fp / max(negative_count, 1)),
                    "target_positive_conditioning_positive_count": positive_count,
                    "conditional_false_negative_count": conditional_fn,
                    "conditional_fnr": float(conditional_fn / max(positive_count, 1)),
                }
            )
    return rows


def error_pair_rows(label_names, labels, probs, thresholds):
    labels = np.asarray(labels).astype(int)
    predictions = (
        np.asarray(probs) >= np.asarray(thresholds).reshape(1, -1)
    ).astype(int)
    errors = predictions != labels
    rows = []
    for left in range(len(label_names)):
        for right in range(left + 1, len(label_names)):
            rows.append(
                {
                    "label_left": label_names[left],
                    "label_right": label_names[right],
                    "both_correct": int((~errors[:, left] & ~errors[:, right]).sum()),
                    "both_wrong": int((errors[:, left] & errors[:, right]).sum()),
                    "left_only_wrong": int((errors[:, left] & ~errors[:, right]).sum()),
                    "right_only_wrong": int((~errors[:, left] & errors[:, right]).sum()),
                    "left_fp_right_fn": int(
                        (errors[:, left] & (predictions[:, left] == 1) & (labels[:, left] == 0)
                         & errors[:, right] & (predictions[:, right] == 0) & (labels[:, right] == 1)).sum()
                    ),
                    "left_fn_right_fp": int(
                        (errors[:, left] & (predictions[:, left] == 0) & (labels[:, left] == 1)
                         & errors[:, right] & (predictions[:, right] == 1) & (labels[:, right] == 0)).sum()
                    ),
                }
            )
    return rows


def load_opcode_hashes(path):
    path = resolve(path)
    if not path.exists():
        return {"status": "missing", "path": str(path), "ids": {}, "hashes": {}}
    ids = {}
    hashes = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item["id"])
            opcode = " ".join(str(item.get("opcode", "")).split())
            digest = hashlib.sha256(opcode.encode("utf-8")).hexdigest()
            ids[sample_id] = digest
            hashes.setdefault(digest, []).append(sample_id)
    return {
        "status": "ok",
        "path": str(path),
        "records": len(ids),
        "ids": ids,
        "hashes": hashes,
    }


def overlap_report(train_source, valid_source, train_ids, valid_ids):
    train_hashes = set(train_source.get("hashes", {}))
    valid_hashes = set(valid_source.get("hashes", {}))
    train_id_set = set(map(str, train_ids))
    valid_id_set = set(map(str, valid_ids))
    shared_hashes = train_hashes & valid_hashes
    duplicate_train = [x for x in train_source.get("hashes", {}).values() if len(x) > 1]
    duplicate_valid = [x for x in valid_source.get("hashes", {}).values() if len(x) > 1]
    return {
        "feature_train_count": len(train_id_set),
        "feature_valid_count": len(valid_id_set),
        "feature_exact_id_overlap": len(train_id_set & valid_id_set),
        "source_train_count": train_source.get("records", 0),
        "source_valid_count": valid_source.get("records", 0),
        "source_exact_id_overlap": len(
            set(train_source.get("ids", {})) & set(valid_source.get("ids", {}))
        ),
        "normalized_opcode_hash_overlap_count": len(shared_hashes),
        "normalized_opcode_hash_overlap_rate_valid": float(
            len(shared_hashes) / max(len(valid_hashes), 1)
        ),
        "within_train_duplicate_hash_groups": len(duplicate_train),
        "within_valid_duplicate_hash_groups": len(duplicate_valid),
        "within_train_duplicate_sample_count": int(sum(len(x) for x in duplicate_train)),
        "within_valid_duplicate_sample_count": int(sum(len(x) for x in duplicate_valid)),
        "warning": (
            "A normalized opcode hash match is a leakage risk indicator, not proof "
            "of identical semantic contracts."
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evidence_retrieval_mvp.yaml")
    parser.add_argument(
        "--output",
        default="results/main6_random_090_mlm8/error_audit/m0_error_audit.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("allow_test") or config.get("ALLOW_TEST") == 1:
        raise ValueError("M0 error audit is validation-only and keeps test locked")
    train = load_split(config, "train")
    valid = load_split(config, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_m0(config, device)
    pos_weight = make_pos_weight(train["labels"], config)
    with torch.no_grad():
        valid_logits, valid_loss = run_epoch(
            model,
            valid,
            device,
            args.batch_size,
            pos_weight,
            config,
        )
    labels = valid["labels"].numpy().astype(int)
    probs = torch.sigmoid(valid_logits).numpy()
    label_names = list(config["label_names"])
    fixed_thresholds = [0.5] * len(label_names)
    tuned = select_per_label_thresholds(
        labels,
        probs,
        config["thresholds"],
        label_names,
        global_threshold=0.2,
    )
    tuned_thresholds = [float(x) for x in tuned["thresholds"]]
    fixed_metric = compute_multilabel_metrics_from_probs(labels, probs, fixed_thresholds)
    tuned_metric = compute_multilabel_metrics_from_probs(labels, probs, tuned_thresholds)

    cardinality = labels.sum(axis=1)
    cardinality_rows = []
    for name, mask in {
        "zero": cardinality == 0,
        "one": cardinality == 1,
        "two": cardinality == 2,
        "three": cardinality == 3,
        "four_plus": cardinality >= 4,
    }.items():
        item = evaluate_subset(label_names, labels, probs, tuned_thresholds, mask)
        item["group"] = name
        cardinality_rows.append(item)

    lengths = valid["mask"].sum(dim=1).numpy()
    q1, q2 = np.quantile(lengths, [1 / 3, 2 / 3])
    length_rows = []
    for name, mask in {
        "short": lengths <= q1,
        "medium": (lengths > q1) & (lengths <= q2),
        "long": lengths > q2,
    }.items():
        item = evaluate_subset(label_names, labels, probs, tuned_thresholds, mask)
        item.update({"group": name, "min_chunks": int(lengths[mask].min()) if mask.any() else None,
                     "max_chunks": int(lengths[mask].max()) if mask.any() else None})
        length_rows.append(item)

    train_source = load_opcode_hashes(Path(config["data_dir"]) / "train.jsonl")
    valid_source = load_opcode_hashes(Path(config["data_dir"]) / "valid.jsonl")
    overlap = overlap_report(train_source, valid_source, train["ids"], valid["ids"])
    overlap["train_source_status"] = train_source["status"]
    overlap["valid_source_status"] = valid_source["status"]

    examples = {}
    for label_id, label_name in enumerate(label_names):
        target = labels[:, label_id]
        false_positive = np.flatnonzero((target == 0) & (probs[:, label_id] >= tuned_thresholds[label_id]))
        false_negative = np.flatnonzero((target == 1) & (probs[:, label_id] < tuned_thresholds[label_id]))
        false_positive = false_positive[np.argsort(-probs[false_positive, label_id])[:20]]
        false_negative = false_negative[np.argsort(probs[false_negative, label_id])[:20]]
        examples[label_name] = {
            "false_positive_highest_probability": [
                {"id": valid["ids"][int(i)], "probability": float(probs[i, label_id])}
                for i in false_positive
            ],
            "false_negative_lowest_probability": [
                {"id": valid["ids"][int(i)], "probability": float(probs[i, label_id])}
                for i in false_negative
            ],
        }

    report = {
        "route": "DIVE Main6 M0 validation error audit",
        "dataset": "DIVE Main6 random split",
        "seed": 42,
        "checkpoint": str(resolve(config["baseline_checkpoint"])),
        "validation_only": True,
        "test_checked": False,
        "model_changed": False,
        "valid_loss": float(valid_loss),
        "label_names": label_names,
        "fixed_thresholds": fixed_thresholds,
        "tuned_thresholds": tuned_thresholds,
        "tuned_threshold_selection_note": "Descriptive validation-only selection; not an unbiased test estimate.",
        "overall": {"fixed": metric_dict(fixed_metric), "tuned": metric_dict(tuned_metric)},
        "per_label_fixed": confusion_rows(label_names, labels, probs, fixed_thresholds),
        "per_label_tuned": confusion_rows(label_names, labels, probs, tuned_thresholds),
        "cardinality_tuned": cardinality_rows,
        "length_tuned": {"tertiles": [float(q1), float(q2)], "groups": length_rows},
        "conditional_error_tuned": conditional_rows(label_names, labels, probs, tuned_thresholds),
        "error_pair_tuned": error_pair_rows(label_names, labels, probs, tuned_thresholds),
        "duplicate_overlap": overlap,
        "error_examples_tuned": examples,
        "interpretation": {
            "conditional_fpr": "Target-label false positive rate among contracts carrying the conditioning label.",
            "conditional_fnr": "Target-label false negative rate among contracts carrying both labels.",
            "cardinality": "Separates single-label and multi-label contract interference.",
            "warning": "All results are validation diagnostics; they are not evidence localization or test performance.",
        },
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output.with_name("per_label_confusion.csv"), report["per_label_tuned"])
    write_csv(output.with_name("cardinality_metrics.csv"), cardinality_rows)
    write_csv(output.with_name("length_metrics.csv"), length_rows)
    write_csv(output.with_name("conditional_error.csv"), report["conditional_error_tuned"])
    write_csv(output.with_name("error_pair.csv"), report["error_pair_tuned"])
    print(json.dumps({
        "output": str(output),
        "device": str(device),
        "valid_loss": float(valid_loss),
        "fixed_macro_f1": report["overall"]["fixed"]["macro_f1"],
        "tuned_macro_f1": report["overall"]["tuned"]["macro_f1"],
        "test_checked": False,
    }, indent=2))


if __name__ == "__main__":
    main()
