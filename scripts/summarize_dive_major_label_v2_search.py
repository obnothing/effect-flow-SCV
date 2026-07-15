import argparse
import ast
import csv
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAJOR_LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
]
BASELINE = {
    "micro_f1": 0.8240858035638883,
    "macro_f1": 0.7452721843450687,
    "large_mean_f1": 0.8059134403635607,
}
ENSEMBLE_BEST_MICRO = {
    "micro_f1": 0.8360528360528361,
    "macro_f1": 0.7531034127883588,
    "large_mean_f1": 0.8187998694577289,
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def parse_eval_metrics(eval_dir):
    report_path = resolve(eval_dir) / "test_threshold_calibration_metrics.txt"
    if not report_path.exists():
        return None
    threshold_payload = None
    per_label = []
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("per_label_threshold_result: "):
            threshold_payload = ast.literal_eval(line.split(": ", 1)[1])
        elif line.startswith("per_label_metrics: "):
            per_label = ast.literal_eval(line.split(": ", 1)[1])
    if threshold_payload is None:
        return None
    if not per_label:
        per_label = threshold_payload.get("per_label_metrics", [])
    large = [
        row
        for row in per_label
        if row.get("label_name") in MAJOR_LABELS
    ]
    large_mean = (
        sum(float(row.get("f1", 0.0)) for row in large) / len(large)
        if large
        else 0.0
    )
    return {
        "micro_f1": float(threshold_payload.get("recognition_micro_f1", 0.0)),
        "macro_f1": float(threshold_payload.get("recognition_macro_f1", 0.0)),
        "detection_f1": float(threshold_payload.get("detection_f1", 0.0)),
        "large_mean_f1": large_mean,
        "per_label": per_label,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize DIVE major-label v2 search.")
    parser.add_argument(
        "--manifest",
        default="configs/generated/dive_major_label_v2/manifest.yaml",
    )
    parser.add_argument(
        "--result_root",
        default="results/dive_major_label_v2",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/dive_major_label_v2/summary",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=["best_micro_f1", "best_macro_f1"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    rows = []
    for item in manifest.get("variants", []):
        variant = item["variant"]
        for checkpoint_tag in args.checkpoints:
            eval_dir = resolve(args.result_root) / variant / checkpoint_tag
            metrics = parse_eval_metrics(eval_dir)
            if metrics is None:
                continue
            row = {
                "variant": variant,
                "checkpoint": checkpoint_tag,
                "eval_dir": str(eval_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "detection_f1": metrics["detection_f1"],
                "large_mean_f1": metrics["large_mean_f1"],
                "micro_delta_vs_baseline": metrics["micro_f1"] - BASELINE["micro_f1"],
                "macro_delta_vs_baseline": metrics["macro_f1"] - BASELINE["macro_f1"],
                "large_mean_delta_vs_baseline": metrics["large_mean_f1"] - BASELINE["large_mean_f1"],
                "micro_delta_vs_ensemble_best_micro": metrics["micro_f1"] - ENSEMBLE_BEST_MICRO["micro_f1"],
                "macro_delta_vs_ensemble_best_micro": metrics["macro_f1"] - ENSEMBLE_BEST_MICRO["macro_f1"],
                "large_mean_delta_vs_ensemble_best_micro": metrics["large_mean_f1"] - ENSEMBLE_BEST_MICRO["large_mean_f1"],
            }
            for label in metrics["per_label"]:
                name = label.get("label_name")
                if name in MAJOR_LABELS:
                    key = name.lower().replace(" ", "_")
                    row[f"{key}_f1"] = float(label.get("f1", 0.0))
                    row[f"{key}_precision"] = float(label.get("precision", 0.0))
                    row[f"{key}_recall"] = float(label.get("recall", 0.0))
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["micro_f1"],
            row["large_mean_f1"],
            row["macro_f1"],
        ),
        reverse=True,
    )
    output_prefix = resolve(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    txt_path = output_prefix.with_suffix(".txt")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["DIVE major-label v2 summary", ""]
    lines.append(f"baseline: {BASELINE}")
    lines.append(f"ensemble_best_micro: {ENSEMBLE_BEST_MICRO}")
    lines.append("")
    for row in rows:
        lines.append(
            "{variant}/{checkpoint}: micro={micro_f1:.6f} macro={macro_f1:.6f} "
            "large_mean={large_mean_f1:.6f} detection={detection_f1:.6f}".format(**row)
        )
        label_parts = [
            f"{label}={row.get(label.lower().replace(' ', '_') + '_f1', 0.0):.4f}"
            for label in MAJOR_LABELS
        ]
        lines.append("  " + " | ".join(label_parts))
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
