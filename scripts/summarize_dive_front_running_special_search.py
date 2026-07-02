import argparse
import csv
import json
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONT_LABEL = "Front Running"
BASELINE = {
    "micro_f1": 0.8240858035638883,
    "macro_f1": 0.7452721843450687,
    "detection_f1": 0.9465041054988804,
    "front_f1": 0.5384615384615384,
}


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def load_json(path):
    path = resolve(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    path = resolve(path)
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def front_metric_row(test_report, predictions=None, label_names=None):
    if predictions and label_names and FRONT_LABEL in label_names:
        front_id = label_names.index(FRONT_LABEL)
        tp = fp = fn = 0
        support = predicted = 0
        for row in predictions:
            true = int(row["multi_true"][front_id])
            pred = int(row["multi_pred"][front_id])
            support += true
            predicted += pred
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 1 and pred == 0:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "front_precision": precision,
            "front_recall": recall,
            "front_f1": f1,
            "front_support": support,
            "front_predicted_positive_count": predicted,
            "front_tp": tp,
            "front_fp": fp,
            "front_fn": fn,
        }
    for row in test_report.get("per_label_metrics", []):
        if row["label_name"] == FRONT_LABEL:
            tp = int(row.get("true_positive_count", round(row["recall"] * row["support"])))
            fp = int(row["predicted_positive_count"] - tp)
            fn = int(row["support"] - tp)
            return {
                "front_precision": row["precision"],
                "front_recall": row["recall"],
                "front_f1": row["f1"],
                "front_support": row["support"],
                "front_predicted_positive_count": row["predicted_positive_count"],
                "front_tp": tp,
                "front_fp": fp,
                "front_fn": fn,
            }
    raise ValueError("Front Running row not found in per_label_metrics")


def load_special_cache(config):
    special_dir = config.get("front_special_feature_dir")
    if not special_dir:
        return None
    path = resolve(special_dir) / "test.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def feature_means_by_front_group(config, prediction_path):
    cache = load_special_cache(config)
    predictions = read_jsonl(prediction_path)
    if not cache or not predictions:
        return {}
    id_to_index = {str(sample_id): idx for idx, sample_id in enumerate(cache["ids"])}
    feature_names = list(cache["feature_names"])
    features = cache["front_special_features"].float()
    mask = cache["chunk_mask"].bool()
    front_id = config["label_names"].index(FRONT_LABEL)
    grouped = {"tp": [], "fp": [], "fn": [], "tn": []}
    for row in predictions:
        idx = id_to_index.get(str(row["id"]))
        if idx is None:
            continue
        true = int(row["multi_true"][front_id])
        pred = int(row["multi_pred"][front_id])
        if true == 1 and pred == 1:
            group = "tp"
        elif true == 0 and pred == 1:
            group = "fp"
        elif true == 1 and pred == 0:
            group = "fn"
        else:
            group = "tn"
        sample_features = features[idx]
        sample_mask = mask[idx]
        if sample_mask.any():
            # Contract-level presence: max over active chunks, then average by group.
            grouped[group].append(sample_features[sample_mask].max(dim=0).values)
    output = {}
    for group, tensors in grouped.items():
        if not tensors:
            output[group] = {
                "count": 0,
                "feature_means": {name: None for name in feature_names},
            }
            continue
        stacked = torch.stack(tensors)
        means = stacked.mean(dim=0).tolist()
        output[group] = {
            "count": len(tensors),
            "feature_means": {
                name: float(means[idx]) for idx, name in enumerate(feature_names)
            },
        }
    return output


def collect_row(item):
    config = load_yaml(item["config"])
    result_dir = resolve(config["result_dir"])
    test_report = load_json(result_dir / "test_threshold_calibration_metrics.json")
    checkpoint_summary = load_json(result_dir / "checkpoint_summary.json")
    behavior_report = load_json(result_dir / "behavior_contribution_summary.json")
    row = {
        "variant": item["variant"],
        "config": item["config"],
        "result_dir": config["result_dir"],
        "status": "missing",
        "front_running_confounder_suppression_weight": config.get(
            "front_running_confounder_suppression_weight"
        ),
        "front_hard_negative_lambda": config.get("front_hard_negative_lambda"),
        "front_running_generic_pattern_scale": config.get(
            "front_running_generic_pattern_scale"
        ),
    }
    if not test_report or not checkpoint_summary:
        row["missing_file"] = str(result_dir / "test_threshold_calibration_metrics.json")
        return row
    per_label = test_report["per_label_threshold_result"]
    row.update(
        {
            "status": "ok",
            "checkpoint_epoch": test_report.get("checkpoint_epoch"),
            "micro_f1": per_label["recognition_micro_f1"],
            "macro_f1": per_label["recognition_macro_f1"],
            "detection_f1": per_label["detection_f1"],
            "predicted_positive_total": per_label["predicted_positive_total"],
            "macro_delta_vs_baseline": per_label["recognition_macro_f1"] - BASELINE["macro_f1"],
            "micro_delta_vs_baseline": per_label["recognition_micro_f1"] - BASELINE["micro_f1"],
            "detection_delta_vs_baseline": per_label["detection_f1"] - BASELINE["detection_f1"],
            "front_hard_negative_active_count_total": checkpoint_summary.get(
                "front_hard_negative_active_count_total",
                0.0,
            ),
            "front_hard_negative_loss_mean": checkpoint_summary.get(
                "front_hard_negative_loss_mean",
                0.0,
            ),
        }
    )
    prediction_path = result_dir / "test_predictions_per_label.jsonl"
    prediction_rows = read_jsonl(prediction_path)
    row.update(
        front_metric_row(
            test_report,
            predictions=prediction_rows,
            label_names=config.get("label_names", []),
        )
    )
    row["front_f1_delta_vs_baseline"] = row["front_f1"] - BASELINE["front_f1"]
    row["front_special_feature_mean_by_tp_fp_fn"] = feature_means_by_front_group(
        config,
        prediction_path,
    )
    front_behavior = {}
    if behavior_report:
        for behavior_row in behavior_report.get("per_label", []):
            if behavior_row.get("label_name") == FRONT_LABEL:
                front_behavior = {
                    "tp_final": (behavior_row.get("tp_contribution") or {}).get(
                        "final_evidence_mean"
                    ),
                    "fp_final": (behavior_row.get("fp_contribution") or {}).get(
                        "final_evidence_mean"
                    ),
                    "fn_final": (behavior_row.get("fn_contribution") or {}).get(
                        "final_evidence_mean"
                    ),
                }
                break
    row["front_behavior_contribution"] = front_behavior
    return row


def write_outputs(rows, output_prefix):
    output_prefix = resolve(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    txt_path = output_prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {
                    "front_special_feature_mean_by_tp_fp_fn",
                    "front_behavior_contribution",
                }
            }
        )
    fields = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows.sort(key=lambda row: row["front_f1"], reverse=True)
    lines = [
        "DIVE Front Running special evidence search summary",
        "",
        "baseline side_scale_150_ep50:",
        f"micro_f1={BASELINE['micro_f1']:.6f}",
        f"macro_f1={BASELINE['macro_f1']:.6f}",
        f"detection_f1={BASELINE['detection_f1']:.6f}",
        f"front_f1={BASELINE['front_f1']:.6f}",
        "",
    ]
    if ok_rows:
        best = ok_rows[0]
        lines.extend(
            [
                f"best_variant_by_front_f1: {best['variant']}",
                f"best_front_f1: {best['front_f1']:.6f}",
                f"best_macro_f1: {best['macro_f1']:.6f}",
                "",
            ]
        )
    lines.append(
        "variant | micro_f1 | macro_f1 | detection_f1 | front_p | front_r | "
        "front_f1 | front_tp | front_fp | front_fn | front_delta | macro_delta | hn_active"
    )
    lines.append("-" * 170)
    for row in ok_rows:
        lines.append(
            f"{row['variant']} | {row['micro_f1']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['detection_f1']:.6f} | {row['front_precision']:.6f} | "
            f"{row['front_recall']:.6f} | {row['front_f1']:.6f} | "
            f"{row['front_tp']} | {row['front_fp']} | {row['front_fn']} | "
            f"{row['front_f1_delta_vs_baseline']:.6f} | "
            f"{row['macro_delta_vs_baseline']:.6f} | "
            f"{row['front_hard_negative_active_count_total']}"
        )
    lines.extend(["", "Front special feature means by TP/FP/FN:"])
    for row in ok_rows:
        lines.append("")
        lines.append(f"[{row['variant']}]")
        means = row.get("front_special_feature_mean_by_tp_fp_fn", {})
        for group in ("tp", "fp", "fn"):
            group_row = means.get(group, {})
            feature_means = group_row.get("feature_means", {})
            ranked = sorted(
                feature_means.items(),
                key=lambda item: -1 if item[1] is None else item[1],
                reverse=True,
            )[:8]
            text = ", ".join(
                f"{name}={value:.4f}" if value is not None else f"{name}=NA"
                for name, value in ranked
            )
            lines.append(f"{group}: count={group_row.get('count', 0)} {text}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize DIVE Front Running special evidence search."
    )
    parser.add_argument(
        "--manifest",
        default="configs/generated/dive_front_running_special_search/manifest.yaml",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/dive_front_running_special_search/summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    rows = [collect_row(item) for item in manifest["configs"]]
    write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
