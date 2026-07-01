import argparse
import ast
import csv
import json
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_RESULT_DIR = Path("results/dive_side_scale_search/train_dive_side_scale_150_ep50")
BASELINE = {
    "variant": "side_scale_150_ep50_reference",
    "micro_f1": 0.8240858035638883,
    "macro_f1": 0.7452721843450687,
    "detection_f1": 0.9465041054988804,
}
FOCUS_LABELS = {"Bad Randomness", "Front Running"}


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path):
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


def extract_literal(text, start, end):
    match = re.search(rf"{re.escape(start)}: (.*?)\n{re.escape(end)}:", text, re.S)
    if not match:
        return None
    return ast.literal_eval(match.group(1))


def load_metrics(result_dir):
    result_dir = resolve_path(result_dir)
    json_path = result_dir / "test_threshold_calibration_metrics.json"
    txt_path = result_dir / "test_threshold_calibration_metrics.txt"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return payload["per_label_threshold_result"], payload.get("per_label_metrics", [])
    if not txt_path.exists():
        return None, None
    text = txt_path.read_text(encoding="utf-8")
    result = extract_literal(text, "per_label_threshold_result", "per_label_metrics")
    labels = extract_literal(text, "per_label_metrics", "per_label_thresholds")
    return result, labels


def collect_row(variant, result_dir, config=None):
    result, labels = load_metrics(result_dir)
    if not result:
        return {
            "variant": variant,
            "status": "missing",
            "missing_file": str(resolve_path(result_dir) / "test_threshold_calibration_metrics.txt"),
        }
    per_label = {item["label_name"]: item for item in labels}
    row = {
        "variant": variant,
        "status": "ok",
        "result_dir": str(result_dir).replace("\\", "/"),
        "recognition_loss_type": (config or {}).get("recognition_loss_type", "bce"),
        "train_sampler": (config or {}).get("train_sampler", "shuffle"),
        "use_pos_weight": bool((config or {}).get("use_pos_weight", True)),
        "micro_f1": result["recognition_micro_f1"],
        "macro_f1": result["recognition_macro_f1"],
        "detection_f1": result["detection_f1"],
        "predicted_positive_total": result["predicted_positive_total"],
        "macro_delta_vs_side_scale_150": result["recognition_macro_f1"] - BASELINE["macro_f1"],
        "micro_delta_vs_side_scale_150": result["recognition_micro_f1"] - BASELINE["micro_f1"],
        "per_label_metrics": labels,
    }
    for label in FOCUS_LABELS:
        metrics = per_label.get(label, {})
        key = label.lower().replace(" ", "_")
        row[f"{key}_precision"] = metrics.get("precision")
        row[f"{key}_recall"] = metrics.get("recall")
        row[f"{key}_f1"] = metrics.get("f1")
        row[f"{key}_support"] = metrics.get("support")
        row[f"{key}_predicted"] = metrics.get("predicted_positive_count")
    return row


def write_outputs(rows, output_prefix):
    output_prefix = resolve_path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    txt_path = output_prefix.with_suffix(".txt")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    flat_rows = [
        {key: value for key, value in row.items() if key != "per_label_metrics"}
        for row in rows
    ]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    best_rows = sorted(ok_rows, key=lambda row: row["macro_f1"], reverse=True)
    lines = [
        "DIVE label-balanced sampler + ASL summary",
        "",
        "baseline: side_scale_150_ep50",
        f"baseline_micro_f1: {BASELINE['micro_f1']:.6f}",
        f"baseline_macro_f1: {BASELINE['macro_f1']:.6f}",
        f"baseline_detection_f1: {BASELINE['detection_f1']:.6f}",
        "",
    ]
    if best_rows:
        best = best_rows[0]
        lines.extend(
            [
                f"best_variant_by_macro_f1: {best['variant']}",
                f"best_micro_f1: {best['micro_f1']:.6f}",
                f"best_macro_f1: {best['macro_f1']:.6f}",
                f"best_detection_f1: {best['detection_f1']:.6f}",
                "",
            ]
        )

    lines.extend(
        [
            "variant | sampler | loss | pos_weight | micro_f1 | macro_f1 | detection_f1 | macro_delta | pred_pos | bad_rand_f1 | front_running_f1",
            "-" * 150,
        ]
    )
    for row in sorted(ok_rows, key=lambda item: item["macro_f1"], reverse=True):
        lines.append(
            f"{row['variant']} | {row['train_sampler']} | "
            f"{row['recognition_loss_type']} | {row['use_pos_weight']} | "
            f"{row['micro_f1']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['detection_f1']:.6f} | "
            f"{row['macro_delta_vs_side_scale_150']:.6f} | "
            f"{row['predicted_positive_total']} | "
            f"{row.get('bad_randomness_f1')} | "
            f"{row.get('front_running_f1')}"
        )

    lines.extend(["", "Focus label precision/recall/F1:", ""])
    for row in sorted(ok_rows, key=lambda item: item["macro_f1"], reverse=True):
        lines.append(
            f"{row['variant']}: "
            f"Bad Randomness P/R/F1="
            f"{row.get('bad_randomness_precision')}/"
            f"{row.get('bad_randomness_recall')}/"
            f"{row.get('bad_randomness_f1')}; "
            f"Front Running P/R/F1="
            f"{row.get('front_running_precision')}/"
            f"{row.get('front_running_recall')}/"
            f"{row.get('front_running_f1')}"
        )

    missing = [row for row in rows if row["status"] != "ok"]
    if missing:
        lines.extend(["", "Missing variants:"])
        for row in missing:
            lines.append(f"- {row['variant']}: {row.get('missing_file')}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize DIVE sampler/ASL search.")
    parser.add_argument(
        "--manifest",
        default="configs/generated/dive_sampler_asl_search/manifest.yaml",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/dive_sampler_asl_search/summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    rows = [
        collect_row(
            BASELINE["variant"],
            BASELINE_RESULT_DIR,
            {
                "recognition_loss_type": "bce",
                "train_sampler": "shuffle",
                "use_pos_weight": True,
            },
        )
    ]
    for item in manifest["configs"]:
        config = load_yaml(item["config"])
        rows.append(collect_row(item["variant"], config["result_dir"], config))
    write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
