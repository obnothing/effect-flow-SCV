import argparse
import csv
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path):
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


def load_json_if_exists(path):
    path = resolve_path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime_seconds(result_dir):
    path = result_dir / "train_runtime_seconds.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    try:
        return float(text)
    except ValueError:
        return None


def collect_row(item):
    config = load_yaml(item["config"])
    result_dir = resolve_path(config["result_dir"])
    test_report = load_json_if_exists(result_dir / "test_threshold_calibration_metrics.json")
    checkpoint_summary = load_json_if_exists(result_dir / "checkpoint_summary.json")
    row = {
        "variant": item["variant"],
        "config": item["config"],
        "result_dir": config["result_dir"],
        "status": "missing",
        "batch_size": int(config["batch_size"]),
        "max_chunks": int(config["max_chunks"]),
        "effective_chunks_per_batch": int(config["batch_size"]) * int(config["max_chunks"]),
        "epochs_configured": int(config["epochs"]),
        "early_stopping_patience": int(config.get("early_stopping_patience", 0)),
        "coefficient_scale": config.get("coefficient_scale"),
        "risk_evidence_weight": config.get("risk_evidence_weight"),
        "beta_reliable_init": config.get("beta_reliable_init"),
        "gamma_reliable_init": config.get("gamma_reliable_init"),
    }
    if not test_report:
        row["missing_file"] = str(result_dir / "test_threshold_calibration_metrics.json")
        return row
    if not checkpoint_summary:
        row["missing_file"] = str(result_dir / "checkpoint_summary.json")
        return row

    per_label = test_report["per_label_threshold_result"]
    runtime_seconds = load_runtime_seconds(result_dir)
    trained_epochs = checkpoint_summary.get("last_epoch") or checkpoint_summary.get(
        "best_macro_f1_epoch"
    )
    if trained_epochs is None:
        epoch_history = load_json_if_exists(result_dir / "epoch_history.json")
        trained_epochs = len(epoch_history) if epoch_history else None

    row.update(
        {
            "status": "ok",
            "checkpoint_epoch": test_report.get("checkpoint_epoch"),
            "best_loss_epoch": checkpoint_summary.get("best_loss_epoch"),
            "best_macro_f1_epoch": checkpoint_summary.get("best_macro_f1_epoch"),
            "valid_best_macro_f1": checkpoint_summary.get("best_macro_f1_value"),
            "valid_best_macro_threshold": checkpoint_summary.get(
                "best_macro_f1_threshold"
            ),
            "per_label_micro_f1": per_label["recognition_micro_f1"],
            "per_label_macro_f1": per_label["recognition_macro_f1"],
            "detection_f1": per_label["detection_f1"],
            "predicted_positive_total": per_label["predicted_positive_total"],
            "max_memory_allocated_mb": checkpoint_summary.get(
                "max_memory_allocated_mb"
            ),
            "max_memory_reserved_mb": checkpoint_summary.get("max_memory_reserved_mb"),
            "train_runtime_seconds": runtime_seconds,
            "train_runtime_minutes": None
            if runtime_seconds is None
            else runtime_seconds / 60.0,
            "trained_epochs": trained_epochs,
            "seconds_per_trained_epoch": None
            if runtime_seconds is None or not trained_epochs
            else runtime_seconds / float(trained_epochs),
            "macro_delta_vs_batch1024_old": per_label["recognition_macro_f1"]
            - 0.73989125485349,
            "micro_delta_vs_batch1024_old": per_label["recognition_micro_f1"]
            - 0.8249630080946994,
            "per_label_f1": {
                item["label_name"]: item["f1"]
                for item in test_report.get("per_label_metrics", [])
            },
        }
    )
    return row


def write_outputs(output_prefix, rows):
    output_prefix = resolve_path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    txt_path = output_prefix.with_suffix(".txt")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    flat_rows = [
        {key: value for key, value in row.items() if key != "per_label_f1"}
        for row in rows
    ]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_macro = sorted(ok_rows, key=lambda row: row["per_label_macro_f1"], reverse=True)
    by_speed = sorted(
        ok_rows,
        key=lambda row: (
            float("inf")
            if row.get("seconds_per_trained_epoch") is None
            else row["seconds_per_trained_epoch"]
        ),
    )
    lines = [
        "DIVE side_scale_200 batch-size search summary",
        "",
        "baseline_old_batch1024_side_scale_200_macro_f1: 0.739891",
        "baseline_old_batch1024_side_scale_200_micro_f1: 0.824963",
        "",
    ]
    if by_macro:
        best = by_macro[0]
        lines.extend(
            [
                f"best_batch_by_test_macro_f1: {best['batch_size']}",
                f"best_macro_f1: {best['per_label_macro_f1']:.6f}",
                f"best_micro_f1: {best['per_label_micro_f1']:.6f}",
                f"best_detection_f1: {best['detection_f1']:.6f}",
                "",
            ]
        )
    if by_speed:
        fastest = by_speed[0]
        lines.extend(
            [
                f"fastest_batch_by_seconds_per_epoch: {fastest['batch_size']}",
                f"fastest_seconds_per_epoch: {fastest['seconds_per_trained_epoch']}",
                "",
            ]
        )

    lines.extend(
        [
            "batch | eff_chunks | epoch | micro_f1 | macro_f1 | detection_f1 | train_minutes | sec_per_epoch | max_alloc_mb | max_reserved_mb | pred_pos",
            "-" * 150,
        ]
    )
    for row in sorted(ok_rows, key=lambda item: item["batch_size"]):
        lines.append(
            f"{row['batch_size']} | {row['effective_chunks_per_batch']} | "
            f"{row['checkpoint_epoch']} | {row['per_label_micro_f1']:.6f} | "
            f"{row['per_label_macro_f1']:.6f} | {row['detection_f1']:.6f} | "
            f"{row['train_runtime_minutes']} | {row['seconds_per_trained_epoch']} | "
            f"{row['max_memory_allocated_mb']} | {row['max_memory_reserved_mb']} | "
            f"{row['predicted_positive_total']}"
        )

    focus_labels = ["DoS", "Bad Randomness", "Front Running", "Time manipulation"]
    lines.extend(["", "Focus per-label F1:", ""])
    for row in sorted(ok_rows, key=lambda item: item["batch_size"]):
        pieces = [
            f"{label}={row.get('per_label_f1', {}).get(label, 'n/a')}"
            for label in focus_labels
        ]
        lines.append(f"batch={row['batch_size']}: " + ", ".join(pieces))

    missing = [row for row in rows if row.get("status") != "ok"]
    if missing:
        lines.extend(["", "Missing variants:"])
        for row in missing:
            lines.append(f"- batch={row['batch_size']}: {row.get('missing_file')}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize DIVE side_scale_200 batch-size search."
    )
    parser.add_argument(
        "--manifest",
        default="configs/generated/dive_side_scale_200_batch_search/manifest.yaml",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/dive_side_scale_200_batch_search/summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    rows = [collect_row(item) for item in manifest["configs"]]
    write_outputs(args.output_prefix, rows)


if __name__ == "__main__":
    main()
