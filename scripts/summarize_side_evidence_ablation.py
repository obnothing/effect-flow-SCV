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
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_if_exists(path):
    path = resolve_path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_none(values):
    if not values:
        return None
    return sum(values) / len(values)


def gate_means(checkpoint_summary):
    if not checkpoint_summary:
        return None, None
    gate = checkpoint_summary.get("gate_values") or {}
    beta = gate.get("beta_reliable")
    gamma = gate.get("gamma_reliable")
    return mean_or_none(beta), mean_or_none(gamma)


def behavior_means(behavior_report):
    if not behavior_report:
        return None, None, None, None
    rows = behavior_report.get("per_label", [])
    risk = []
    protective = []
    missing = []
    final = []
    for row in rows:
        contribution = row.get("predicted_positive_contribution") or {}
        for target, key in (
            (risk, "risk_mean"),
            (protective, "protective_mean"),
            (missing, "missing_check_mean"),
            (final, "final_evidence_mean"),
        ):
            value = contribution.get(key)
            if value is not None:
                target.append(float(value))
    return (
        mean_or_none(risk),
        mean_or_none(protective),
        mean_or_none(missing),
        mean_or_none(final),
    )


def collect_rows(manifest):
    rows = []
    for item in manifest["configs"]:
        config = load_yaml(item["config"])
        result_dir = resolve_path(config["result_dir"])
        eval_report = load_json_if_exists(
            result_dir / "test_threshold_calibration_metrics.json"
        )
        checkpoint_summary = load_json_if_exists(result_dir / "checkpoint_summary.json")
        behavior_report = load_json_if_exists(
            result_dir / "behavior_contribution_summary.json"
        )
        if not eval_report:
            rows.append(
                {
                    "dataset": item["dataset"],
                    "variant": item["variant"],
                    "side_evidence": item["side_evidence"],
                    "status": "missing_test_report",
                }
            )
            continue
        baseline = eval_report["threshold_0_5_baseline"]
        best_global = eval_report["best_global_threshold_result"]
        per_label = eval_report["per_label_threshold_result"]
        beta_mean, gamma_mean = gate_means(checkpoint_summary)
        risk_mean, protective_mean, missing_mean, final_mean = behavior_means(
            behavior_report
        )
        rows.append(
            {
                "dataset": item["dataset"],
                "variant": item["variant"],
                "side_evidence": item["side_evidence"],
                "status": "ok",
                "risk_evidence_weight": config.get("risk_evidence_weight"),
                "missing_check_evidence_weight": config.get(
                    "missing_check_evidence_weight"
                ),
                "protective_evidence_weight": config.get(
                    "protective_evidence_weight"
                ),
                "effect_type_evidence_weight": config.get(
                    "effect_type_evidence_weight"
                ),
                "relation_evidence_weight": config.get("relation_evidence_weight"),
                "beta_reliable_init": config.get("beta_reliable_init"),
                "gamma_reliable_init": config.get("gamma_reliable_init"),
                "beta_reliable_mean": beta_mean,
                "gamma_reliable_mean": gamma_mean,
                "threshold_0_5_micro_f1": baseline["recognition_micro_f1"],
                "threshold_0_5_macro_f1": baseline["recognition_macro_f1"],
                "best_global_threshold": best_global["thresholds"],
                "best_global_micro_f1": best_global["recognition_micro_f1"],
                "best_global_macro_f1": best_global["recognition_macro_f1"],
                "per_label_micro_f1": per_label["recognition_micro_f1"],
                "per_label_macro_f1": per_label["recognition_macro_f1"],
                "detection_f1": per_label["detection_f1"],
                "predicted_positive_total": per_label["predicted_positive_total"],
                "predicted_positive_risk_mean": risk_mean,
                "predicted_positive_protective_mean": protective_mean,
                "predicted_positive_missing_check_mean": missing_mean,
                "predicted_positive_final_evidence_mean": final_mean,
                "result_dir": config["result_dir"],
            }
        )
    return rows


def write_summary(output_prefix, rows):
    output_prefix = resolve_path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    txt_path = output_prefix.with_suffix(".txt")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "EVEF-MVD v2 side-evidence ablation summary",
        "",
        "dataset | variant | side | per_label_micro_f1 | per_label_macro_f1 | detection_f1 | beta_mean | gamma_mean | final_evidence_mean",
        "-" * 150,
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"{row['dataset']} | {row['variant']} | {row.get('side_evidence')} | {row['status']}"
            )
            continue
        lines.append(
            f"{row['dataset']} | {row['variant']} | {row['side_evidence']} | "
            f"{row['per_label_micro_f1']:.6f} | {row['per_label_macro_f1']:.6f} | "
            f"{row['detection_f1']:.6f} | "
            f"{row['beta_reliable_mean'] if row['beta_reliable_mean'] is not None else 'n/a'} | "
            f"{row['gamma_reliable_mean'] if row['gamma_reliable_mean'] is not None else 'n/a'} | "
            f"{row['predicted_positive_final_evidence_mean'] if row['predicted_positive_final_evidence_mean'] is not None else 'n/a'}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {csv_path.relative_to(PROJECT_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize EVEF-MVD side-evidence ablation results."
    )
    parser.add_argument(
        "--manifest",
        default="configs/generated/evef_mvd_v2_ablation/manifest.yaml",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/ablation_evef_mvd_v2/side_evidence_ablation_summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    rows = collect_rows(manifest)
    write_summary(args.output_prefix, rows)


if __name__ == "__main__":
    main()
