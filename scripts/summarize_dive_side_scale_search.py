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


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def gate_means(checkpoint_summary):
    if not checkpoint_summary:
        return None, None
    gate = checkpoint_summary.get("gate_values") or {}
    return mean(gate.get("beta_reliable") or []), mean(gate.get("gamma_reliable") or [])


def collect_row(item):
    config = load_yaml(item["config"])
    result_dir = resolve_path(config["result_dir"])
    test_report = load_json_if_exists(result_dir / "test_threshold_calibration_metrics.json")
    checkpoint_summary = load_json_if_exists(result_dir / "checkpoint_summary.json")
    behavior_report = load_json_if_exists(result_dir / "behavior_contribution_summary.json")
    row = {
        "variant": item["variant"],
        "config": item["config"],
        "result_dir": config["result_dir"],
        "status": "missing",
        "coefficient_scale": config.get("coefficient_scale"),
        "risk_evidence_weight": config.get("risk_evidence_weight"),
        "missing_check_evidence_weight": config.get("missing_check_evidence_weight"),
        "protective_evidence_weight": config.get("protective_evidence_weight"),
        "effect_type_evidence_weight": config.get("effect_type_evidence_weight"),
        "relation_evidence_weight": config.get("relation_evidence_weight"),
        "beta_reliable_init": config.get("beta_reliable_init"),
        "gamma_reliable_init": config.get("gamma_reliable_init"),
        "epochs": config.get("epochs"),
        "early_stopping_patience": config.get("early_stopping_patience"),
    }
    if not test_report:
        row["missing_file"] = str(result_dir / "test_threshold_calibration_metrics.json")
        return row
    if not checkpoint_summary:
        row["missing_file"] = str(result_dir / "checkpoint_summary.json")
        return row

    per_label = test_report["per_label_threshold_result"]
    best_global = test_report["best_global_threshold_result"]
    fixed = test_report["threshold_0_5_baseline"]
    beta_mean, gamma_mean = gate_means(checkpoint_summary)
    per_label_f1 = {
        item["label_name"]: item["f1"] for item in test_report.get("per_label_metrics", [])
    }
    behavior_means = {}
    if behavior_report:
        for behavior_row in behavior_report.get("per_label", []):
            contribution = behavior_row.get("predicted_positive_contribution") or {}
            behavior_means[behavior_row["label_name"]] = {
                "risk_mean": contribution.get("risk_mean"),
                "protective_mean": contribution.get("protective_mean"),
                "missing_check_mean": contribution.get("missing_check_mean"),
                "final_evidence_mean": contribution.get("final_evidence_mean"),
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
            "fixed_0_5_micro_f1": fixed["recognition_micro_f1"],
            "fixed_0_5_macro_f1": fixed["recognition_macro_f1"],
            "best_global_threshold": best_global["thresholds"],
            "best_global_micro_f1": best_global["recognition_micro_f1"],
            "best_global_macro_f1": best_global["recognition_macro_f1"],
            "per_label_micro_f1": per_label["recognition_micro_f1"],
            "per_label_macro_f1": per_label["recognition_macro_f1"],
            "detection_f1": per_label["detection_f1"],
            "predicted_positive_total": per_label["predicted_positive_total"],
            "beta_reliable_mean": beta_mean,
            "gamma_reliable_mean": gamma_mean,
            "per_label_f1": per_label_f1,
            "behavior_means": behavior_means,
            "warnings": test_report.get("warnings", []),
        }
    )
    row["macro_delta_vs_old_side_scale_200"] = row["per_label_macro_f1"] - 0.73989125485349
    row["micro_delta_vs_old_side_scale_200"] = row["per_label_micro_f1"] - 0.8249630080946994
    return row


def write_outputs(output_prefix, rows):
    output_prefix = resolve_path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    txt_path = output_prefix.with_suffix(".txt")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"per_label_f1", "behavior_means", "warnings"}
            }
        )
    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows = sorted(ok_rows, key=lambda row: row["per_label_macro_f1"], reverse=True)
    best = ok_rows[0] if ok_rows else None
    lines = [
        "DIVE side-scale parameter search summary",
        "",
        "baseline_old_side_scale_200_per_label_macro_f1: 0.739891",
        "baseline_old_side_scale_200_per_label_micro_f1: 0.824963",
        "",
    ]
    if best:
        lines.extend(
            [
                f"best_variant_by_test_macro_f1: {best['variant']}",
                f"best_macro_f1: {best['per_label_macro_f1']:.6f}",
                f"best_micro_f1: {best['per_label_micro_f1']:.6f}",
                f"best_detection_f1: {best['detection_f1']:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "variant | scale | beta_init | gamma_init | epoch | micro_f1 | macro_f1 | detection_f1 | macro_delta_vs_old_200 | predicted_positive_total",
            "-" * 145,
        ]
    )
    for row in ok_rows:
        lines.append(
            f"{row['variant']} | {row['coefficient_scale']} | "
            f"{row['beta_reliable_init']} | {row['gamma_reliable_init']} | "
            f"{row['checkpoint_epoch']} | {row['per_label_micro_f1']:.6f} | "
            f"{row['per_label_macro_f1']:.6f} | {row['detection_f1']:.6f} | "
            f"{row['macro_delta_vs_old_side_scale_200']:.6f} | "
            f"{row['predicted_positive_total']}"
        )

    focus_labels = ["DoS", "Bad Randomness", "Front Running", "Time manipulation"]
    lines.extend(["", "Focus per-label F1:", ""])
    for row in ok_rows:
        label_bits = [
            f"{label}={row.get('per_label_f1', {}).get(label, 'n/a')}"
            for label in focus_labels
        ]
        lines.append(f"{row['variant']}: " + ", ".join(label_bits))

    missing = [row for row in rows if row.get("status") != "ok"]
    if missing:
        lines.extend(["", "Missing variants:"])
        for row in missing:
            lines.append(f"- {row['variant']}: {row.get('missing_file')}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize DIVE side-scale search.")
    parser.add_argument(
        "--manifest",
        default="configs/generated/dive_side_scale_search/manifest.yaml",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/dive_side_scale_search/summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    rows = [collect_row(item) for item in manifest["configs"]]
    write_outputs(args.output_prefix, rows)


if __name__ == "__main__":
    main()
