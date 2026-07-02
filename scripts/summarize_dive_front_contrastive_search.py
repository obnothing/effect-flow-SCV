import argparse
import csv
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation",
]
FRONT_ID = LABEL_NAMES.index("Front Running")
BAD_RANDOMNESS_ID = LABEL_NAMES.index("Bad Randomness")
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


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def auc_score(labels, scores):
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    ranks = [0.0] * len(pairs)
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        average_rank = (idx + 1 + end) / 2.0
        for rank_idx in range(idx, end):
            ranks[rank_idx] = average_rank
        idx = end
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def ap_score(labels, scores):
    positive = sum(labels)
    if positive == 0:
        return 0.0
    order = sorted(range(len(labels)), key=lambda idx: scores[idx], reverse=True)
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            hits += 1
            total += hits / rank
    return total / positive


def metrics_from_predictions(predictions):
    tp = [0] * len(LABEL_NAMES)
    fp = [0] * len(LABEL_NAMES)
    fn = [0] * len(LABEL_NAMES)
    binary_tp = binary_fp = binary_fn = binary_tn = 0
    for row in predictions:
        y_true = row["multi_true"]
        y_pred = row["multi_pred"]
        for label_id, (true, pred) in enumerate(zip(y_true, y_pred)):
            true = int(true)
            pred = int(pred)
            if true and pred:
                tp[label_id] += 1
            elif not true and pred:
                fp[label_id] += 1
            elif true and not pred:
                fn[label_id] += 1
        binary_true = int(row.get("binary_true", int(any(y_true))))
        binary_pred = int(row.get("binary_pred", int(any(y_pred))))
        if binary_true and binary_pred:
            binary_tp += 1
        elif not binary_true and binary_pred:
            binary_fp += 1
        elif binary_true and not binary_pred:
            binary_fn += 1
        else:
            binary_tn += 1
    per_label = []
    for label_id, label_name in enumerate(LABEL_NAMES):
        precision, recall, f1 = prf(tp[label_id], fp[label_id], fn[label_id])
        labels = [int(row["multi_true"][label_id]) for row in predictions]
        scores = [float(row["multi_prob"][label_id]) for row in predictions]
        per_label.append(
            {
                "label_name": label_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp[label_id],
                "fp": fp[label_id],
                "fn": fn[label_id],
                "support": tp[label_id] + fn[label_id],
                "predicted": tp[label_id] + fp[label_id],
                "auc": auc_score(labels, scores),
                "ap": ap_score(labels, scores),
            }
        )
    micro_precision, micro_recall, micro_f1 = prf(sum(tp), sum(fp), sum(fn))
    _, _, detection_f1 = prf(binary_tp, binary_fp, binary_fn)
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": sum(row["f1"] for row in per_label) / len(per_label),
        "detection_f1": detection_f1,
        "per_label": per_label,
        "front": per_label[FRONT_ID],
        "bad_randomness": per_label[BAD_RANDOMNESS_ID],
    }


def valid_front_f1(eval_dir):
    valid_report = load_json(eval_dir / "valid_threshold_metrics.json")
    if not valid_report:
        return 0.0
    for row in valid_report.get("per_label_metrics", []):
        if row.get("label_name") == "Front Running":
            return float(row.get("f1", 0.0))
    return 0.0


def collect_eval_row(item, checkpoint_tag, mode, filename):
    eval_dir = resolve(item["eval_result_root"]) / checkpoint_tag
    predictions = read_jsonl(eval_dir / filename)
    if not predictions:
        return {
            "status": "missing",
            "variant": item["variant"],
            "checkpoint_tag": checkpoint_tag,
            "threshold_mode": mode,
            "missing_path": str(eval_dir / filename),
        }
    metrics = metrics_from_predictions(predictions)
    train_summary = load_json(resolve(item["train_result_dir"]) / "checkpoint_summary.json") or {}
    valid_report = load_json(eval_dir / "valid_threshold_metrics.json") or {}
    row = {
        "status": "ok",
        "variant": item["variant"],
        "lambda": item.get("lambda"),
        "temperature": item.get("temperature"),
        "checkpoint_tag": checkpoint_tag,
        "threshold_mode": mode,
        "eval_dir": str(eval_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "valid_macro_f1": valid_report.get("recognition_macro_f1"),
        "valid_micro_f1": valid_report.get("recognition_micro_f1"),
        "valid_front_f1": valid_front_f1(eval_dir),
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "detection_f1": metrics["detection_f1"],
        "front_precision": metrics["front"]["precision"],
        "front_recall": metrics["front"]["recall"],
        "front_f1": metrics["front"]["f1"],
        "front_tp": metrics["front"]["tp"],
        "front_fp": metrics["front"]["fp"],
        "front_fn": metrics["front"]["fn"],
        "front_ap": metrics["front"]["ap"],
        "front_auc": metrics["front"]["auc"],
        "bad_randomness_f1": metrics["bad_randomness"]["f1"],
        "bad_randomness_tp": metrics["bad_randomness"]["tp"],
        "bad_randomness_fp": metrics["bad_randomness"]["fp"],
        "bad_randomness_fn": metrics["bad_randomness"]["fn"],
        "bad_randomness_ap": metrics["bad_randomness"]["ap"],
        "rare_mean_f1": (
            metrics["front"]["f1"] + metrics["bad_randomness"]["f1"]
        )
        / 2.0,
        "macro_delta_vs_baseline": metrics["macro_f1"] - BASELINE["macro_f1"],
        "micro_delta_vs_baseline": metrics["micro_f1"] - BASELINE["micro_f1"],
        "front_delta_vs_baseline": metrics["front"]["f1"] - BASELINE["front_f1"],
        "bad_delta_vs_baseline": (
            metrics["bad_randomness"]["f1"] - BASELINE["bad_randomness_f1"]
        ),
        "front_contrastive_loss_mean": train_summary.get(
            "front_contrastive_loss_mean"
        ),
        "front_contrastive_anchor_count_total": train_summary.get(
            "front_contrastive_anchor_count_total"
        ),
        "front_contrastive_hard_negative_count_total": train_summary.get(
            "front_contrastive_hard_negative_count_total"
        ),
        "front_contrastive_active_batches_total": train_summary.get(
            "front_contrastive_active_batches_total"
        ),
    }
    return row


def collect_rows(manifest):
    modes = [
        ("fixed_0_5", "test_predictions_threshold_0_5.jsonl"),
        ("best_global", "test_predictions_best_global.jsonl"),
        ("per_label", "test_predictions_per_label.jsonl"),
    ]
    checkpoint_tags = ["best_micro_f1", "best_macro_f1"]
    rows = []
    for item in manifest["configs"]:
        for checkpoint_tag in checkpoint_tags:
            for mode, filename in modes:
                rows.append(collect_eval_row(item, checkpoint_tag, mode, filename))
    return rows


def mark_validation_winner(rows):
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("threshold_mode") == "per_label"
        and row.get("checkpoint_tag") == "best_micro_f1"
    ]
    if not ok_rows:
        ok_rows = [
            row
            for row in rows
            if row.get("status") == "ok" and row.get("threshold_mode") == "per_label"
        ]
    ok_rows.sort(
        key=lambda row: (
            float(row.get("valid_macro_f1") or 0.0),
            float(row.get("valid_front_f1") or 0.0),
            float(row.get("macro_f1") or 0.0),
        ),
        reverse=True,
    )
    winner_key = None
    if ok_rows:
        winner = ok_rows[0]
        winner_key = (
            winner["variant"],
            winner["checkpoint_tag"],
            winner["threshold_mode"],
        )
    for row in rows:
        row["validation_winner"] = (
            row.get("variant"),
            row.get("checkpoint_tag"),
            row.get("threshold_mode"),
        ) == winner_key
    return rows


def write_outputs(rows, output_prefix):
    output_prefix = resolve(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = mark_validation_winner(rows)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    txt_path = output_prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows.sort(
        key=lambda row: (
            row["validation_winner"],
            row["macro_f1"],
            row["front_f1"],
        ),
        reverse=True,
    )
    lines = [
        "DIVE Front conditional contrastive search summary",
        "",
        "baseline:",
        f"micro_f1={BASELINE['micro_f1']:.6f}",
        f"macro_f1={BASELINE['macro_f1']:.6f}",
        f"detection_f1={BASELINE['detection_f1']:.6f}",
        f"front_f1={BASELINE['front_f1']:.6f}",
        f"bad_randomness_f1={BASELINE['bad_randomness_f1']:.6f}",
        "",
        "variant | checkpoint | mode | valid_macro | micro | macro | detection | "
        "front_f1 | front_tp | front_fp | front_fn | front_ap | bad_f1 | "
        "rare_mean | winner",
        "-" * 180,
    ]
    for row in ok_rows:
        lines.append(
            f"{row['variant']} | {row['checkpoint_tag']} | {row['threshold_mode']} | "
            f"{float(row.get('valid_macro_f1') or 0.0):.6f} | "
            f"{row['micro_f1']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['detection_f1']:.6f} | {row['front_f1']:.6f} | "
            f"{row['front_tp']} | {row['front_fp']} | {row['front_fn']} | "
            f"{row['front_ap']:.6f} | {row['bad_randomness_f1']:.6f} | "
            f"{row['rare_mean_f1']:.6f} | {row['validation_winner']}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize DIVE Front conditional contrastive search."
    )
    parser.add_argument(
        "--manifest",
        default="configs/generated/dive_front_contrastive_search/manifest.yaml",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/dive_front_contrastive_search/summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    rows = collect_rows(manifest)
    write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
