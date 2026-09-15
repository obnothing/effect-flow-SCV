"""Run P11 optimization trials and write one incremental comparison report."""

import csv
import json
import sys

import run_polarity_queries as base


ROOT = base.ROOT
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
REFERENCE = ROOT / "results/light_label/polarity_queries_followup_p11/P11/metrics.json"
TRIALS = (
    ("fixed_lr_p11_reference", None),
    ("S1_warmup_cosine", ROOT / "configs/light_label/polarity_queries_round1_s1_cosine.yaml"),
    ("S2_warmup_constant", ROOT / "configs/light_label/polarity_queries_round1_s2_warmup_constant.yaml"),
    ("S3_weight_decay_3e4", ROOT / "configs/light_label/polarity_queries_round1_s3_weight_decay.yaml"),
    ("S4_lr_5e4", ROOT / "configs/light_label/polarity_queries_round1_s4_lr05.yaml"),
)


def read_metrics(path, trial):
    value = json.loads(path.read_text(encoding="utf-8"))
    metrics = value["metrics"]
    return {"trial": trial, "source": str(path.relative_to(ROOT)),
            "macro_f1": metrics["tuned"]["macro_f1"],
            "micro_f1": metrics["tuned"]["micro_f1"],
            "fixed_macro_f1": metrics["fixed"]["macro_f1"],
            "best_epoch": value.get("best_epoch"),
            "learning_rate": value["requested_config"]["learning_rate"],
            "weight_decay": value["requested_config"]["weight_decay"],
            "scheduler": value["requested_config"].get("scheduler", "none"),
            "per_label_f1": metrics["tuned"]["per_label_f1"],
            "test_checked": False}


def write_report(rows):
    out = ROOT / "results/light_label/polarity_queries_round1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = ["trial", "source", "macro_f1", "micro_f1", "fixed_macro_f1", "best_epoch",
              "learning_rate", "weight_decay", "scheduler", "test_checked"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in rows])
    reference = rows[0]["macro_f1"]
    lines = ["# P11 Optimization Round 1", "", "process01; seed 42; validation only; test_checked=false.", "",
             "| Trial | Scheduler | LR | Weight decay | Macro-F1 | Micro-F1 | Delta vs fixed P11 | Best epoch |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['trial']} | {row['scheduler']} | {row['learning_rate']:.6g} | {row['weight_decay']:.6g} | "
                     f"{row['macro_f1']:.6f} | {row['micro_f1']:.6f} | {row['macro_f1']-reference:+.6f} | {row['best_epoch']} |")
    lines.extend(["", "Per-label F1 order: " + ", ".join(LABELS), ""])
    for row in rows:
        lines.append(row["trial"] + ": " + ", ".join(f"{x:.6f}" for x in row["per_label_f1"]))
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not REFERENCE.exists():
        raise FileNotFoundError(f"P11 fixed-LR reference is missing: {REFERENCE}")
    rows = [read_metrics(REFERENCE, TRIALS[0][0])]
    write_report(rows)
    for trial, config in TRIALS[1:]:
        base.CONFIG = config
        base.VARIANTS = ("P11",)
        sys.argv = [sys.argv[0]]
        base.main()
        trial_config = base.load_config(config)
        metrics_path = ROOT / trial_config["result_root"] / "P11" / "metrics.json"
        rows.append(read_metrics(metrics_path, trial))
        write_report(rows)


if __name__ == "__main__":
    main()
