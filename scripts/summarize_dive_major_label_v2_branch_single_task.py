import argparse
import ast
import csv
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
BASELINE = {
    "micro_f1": 0.8240858035638883,
    "macro_f1": 0.7452721843450687,
    "detection_f1": 0.9465041054988804,
    "large_mean_f1": 0.8059134403635607,
}
FULL_RECOGNITION_ONLY = {
    "micro_f1": 0.8250733897427043,
    "macro_f1": 0.7332213136469405,
    "detection_f1": 0.9481155163974546,
    "large_mean_f1": 0.8033220889476849,
}
SEED_ENSEMBLE_MAIN = {
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
    f1_by_label = {row["label_name"]: float(row["f1"]) for row in per_label}
    large_mean = sum(f1_by_label[label] for label in MAJOR_LABELS) / len(MAJOR_LABELS)
    row = {
        "micro_f1": float(threshold_payload["recognition_micro_f1"]),
        "macro_f1": float(threshold_payload["recognition_macro_f1"]),
        "detection_f1": float(threshold_payload["detection_f1"]),
        "detection_source": threshold_payload.get("detection_source", "unknown"),
        "large_mean_f1": large_mean,
    }
    for label in LABELS:
        key = label.lower().replace(" ", "_")
        row[f"{key}_f1"] = f1_by_label[label]
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize DIVE major-label v2 branch single-task search."
    )
    parser.add_argument(
        "--manifest",
        default="configs/generated/dive_major_label_v2_branch_single_task/manifest.yaml",
    )
    parser.add_argument(
        "--result_root",
        default="results/dive_major_label_v2_branch_single_task",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/dive_major_label_v2_branch_single_task/summary",
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
                **metrics,
                "micro_delta_vs_baseline": metrics["micro_f1"] - BASELINE["micro_f1"],
                "macro_delta_vs_baseline": metrics["macro_f1"] - BASELINE["macro_f1"],
                "detection_delta_vs_baseline": (
                    metrics["detection_f1"] - BASELINE["detection_f1"]
                ),
                "large_mean_delta_vs_baseline": (
                    metrics["large_mean_f1"] - BASELINE["large_mean_f1"]
                ),
                "micro_delta_vs_full_recognition_only": (
                    metrics["micro_f1"] - FULL_RECOGNITION_ONLY["micro_f1"]
                ),
                "macro_delta_vs_full_recognition_only": (
                    metrics["macro_f1"] - FULL_RECOGNITION_ONLY["macro_f1"]
                ),
                "large_mean_delta_vs_full_recognition_only": (
                    metrics["large_mean_f1"] - FULL_RECOGNITION_ONLY["large_mean_f1"]
                ),
                "micro_delta_vs_seed_ensemble_main": (
                    metrics["micro_f1"] - SEED_ENSEMBLE_MAIN["micro_f1"]
                ),
                "macro_delta_vs_seed_ensemble_main": (
                    metrics["macro_f1"] - SEED_ENSEMBLE_MAIN["macro_f1"]
                ),
                "large_mean_delta_vs_seed_ensemble_main": (
                    metrics["large_mean_f1"] - SEED_ENSEMBLE_MAIN["large_mean_f1"]
                ),
            }
            rows.append(row)
    rows.sort(
        key=lambda row: (row["macro_f1"], row["large_mean_f1"], row["micro_f1"]),
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

    lines = ["DIVE major-label v2 branch single-task summary", ""]
    lines.append(f"baseline: {BASELINE}")
    lines.append(f"full_recognition_only: {FULL_RECOGNITION_ONLY}")
    lines.append(f"seed_ensemble_main: {SEED_ENSEMBLE_MAIN}")
    lines.append("")
    for row in rows:
        lines.append(
            "{variant}/{checkpoint}: micro={micro_f1:.6f} macro={macro_f1:.6f} "
            "large_mean={large_mean_f1:.6f} detection={detection_f1:.6f} "
            "detection_source={detection_source}".format(**row)
        )
        lines.append(
            "  "
            + " | ".join(
                f"{label}={row[label.lower().replace(' ', '_') + '_f1']:.4f}"
                for label in LABELS
            )
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
