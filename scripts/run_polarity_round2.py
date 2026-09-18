"""Run the P11 error-driven ablation queue with incremental reports."""

import csv
import json
import sys

import run_polarity_queries as base


ROOT = base.ROOT
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
REFERENCE = ROOT / "results/light_label/polarity_queries_followup_p11/P11/metrics.json"
TRIALS = (
    ("P11_reference", None),
    ("R1_dos_main15", ROOT / "configs/light_label/polarity_queries_round2_r1_dos_main15.yaml"),
    ("R2_dos_positive15", ROOT / "configs/light_label/polarity_queries_round2_r2_dos_positive15.yaml"),
    ("R3_dos_soft0905", ROOT / "configs/light_label/polarity_queries_round2_r3_dos_soft0905.yaml"),
    ("R4_negative_fp15", ROOT / "configs/light_label/polarity_queries_round2_r4_negative_fp15.yaml"),
    ("R5_aux005", ROOT / "configs/light_label/polarity_queries_round2_r5_aux005.yaml"),
    ("R6_max12288", ROOT / "configs/light_label/polarity_queries_round2_r6_max12288.yaml"),
)


def read_metrics(path, trial, config_path=None):
    value = json.loads(path.read_text(encoding="utf-8"))
    metrics = value["metrics"]
    requested = value["requested_config"]
    return {
        "trial": trial,
        "source": str(path.relative_to(ROOT)),
        "macro_f1": float(metrics["tuned"]["macro_f1"]),
        "micro_f1": float(metrics["tuned"]["micro_f1"]),
        "fixed_macro_f1": float(metrics["fixed"]["macro_f1"]),
        "best_epoch": value.get("best_epoch"),
        "learning_rate": float(requested["learning_rate"]),
        "weight_decay": float(requested["weight_decay"]),
        "scheduler": requested.get("scheduler", "none"),
        "auxiliary_weight": float(requested["auxiliary_weight"]),
        "dos_weight_multiplier": float(requested.get("dos_weight_multiplier", 1.0)),
        "positive_auxiliary_label_multiplier": requested.get("positive_auxiliary_label_multiplier"),
        "negative_auxiliary_label_multiplier": requested.get("negative_auxiliary_label_multiplier"),
        "dos_soft_positive": float(requested.get("dos_soft_positive", 0.8)),
        "dos_soft_negative": float(requested.get("dos_soft_negative", 0.1)),
        "max_len": int(requested["max_len"]),
        "per_label_f1": metrics["tuned"]["per_label_f1"],
        "test_checked": False,
        "config": str(config_path.relative_to(ROOT)) if config_path else "historical P11 config",
    }


def write_report(rows):
    out = ROOT / "results/light_label/polarity_queries_round2"
    out.mkdir(parents=True, exist_ok=True)
    reference = rows[0]["macro_f1"]
    enriched = []
    for row in rows:
        copy = dict(row)
        copy["delta_vs_p11"] = copy["macro_f1"] - reference
        enriched.append(copy)
    (out / "summary.json").write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    fields = ["trial", "macro_f1", "micro_f1", "fixed_macro_f1", "delta_vs_p11", "best_epoch",
              "scheduler", "learning_rate", "weight_decay", "auxiliary_weight", "dos_weight_multiplier",
              "max_len", "test_checked", "source"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in enriched)
    lines = [
        "# P11 Error-Driven Polarity Query Round 2",
        "",
        "DIVE Main-6 process01; seed 42; validation only; test_checked=false.",
        "All trials retain independent q+ and q- evidence locations; no q_rel is used.",
        "",
        "| Trial | Main change | Macro-F1 | Delta vs P11 | Micro-F1 | Fixed macro | Best epoch | Max len |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in enriched:
        lines.append(
            f"| {row['trial']} | {row['scheduler']} / aux={row['auxiliary_weight']:.3g} / DoS={row['dos_weight_multiplier']:.3g} | "
            f"{row['macro_f1']:.6f} | {row['delta_vs_p11']:+.6f} | {row['micro_f1']:.6f} | "
            f"{row['fixed_macro_f1']:.6f} | {row['best_epoch']} | {row['max_len']} |"
        )
    lines.extend(["", "Per-label F1 order: " + ", ".join(LABELS), ""])
    for row in enriched:
        lines.append(row["trial"] + ": " + ", ".join(f"{x:.6f}" for x in row["per_label_f1"]))
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not REFERENCE.exists():
        raise FileNotFoundError(f"P11 reference is missing: {REFERENCE}")
    rows = [read_metrics(REFERENCE, TRIALS[0][0])]
    write_report(rows)
    for trial, config in TRIALS[1:]:
        base.CONFIG = config
        base.VARIANTS = ("P11",)
        sys.argv = [sys.argv[0]]
        print(f"[round2] starting {trial} config={config}", flush=True)
        base.main()
        trial_config = base.load_config(config)
        metrics_path = ROOT / trial_config["result_root"] / "P11" / "metrics.json"
        rows.append(read_metrics(metrics_path, trial, config))
        write_report(rows)
        print(f"[round2] completed {trial} macro={rows[-1]['macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
