"""Create the unified validation comparison for the mutually exclusive E1-E5 runs."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
EXTENSIONS = [
    ("e1_vcfm", "E1 VCFM", "ACoL"),
    ("e2_vrop", "E2 VROP", "Representative snippet propagation"),
    ("e3_vasm", "E3 VASM", "ASM-Loc"),
    ("e4_pgvr", "E4 PGVR", "PivoTAL"),
    ("e5_tdvp", "E5 TDVP", "IBMIL"),
]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_from_metrics(variant, display, inspiration, report):
    tuned = report["metrics"]["tuned"]
    fixed = report["metrics"]["fixed"]
    return {
        "variant": variant,
        "display": display,
        "inspiration": inspiration,
        "fixed_macro_f1": float(fixed["macro_f1"]),
        "tuned_macro_f1": float(tuned["macro_f1"]),
        "tuned_micro_f1": float(tuned["micro_f1"]),
        "detection_f1": float(report["metrics"]["detection_f1"]),
        "per_label_f1": [float(value) for value in tuned["per_label_f1"]],
        "thresholds": [float(value) for value in report["metrics"]["thresholds"]],
        "total_params": int(report["total_params"]),
        "trainable_params": int(report["trainable_params"]),
        "peak_memory_mb": float(report["peak_memory_mb"]),
        "mean_epoch_seconds": float(report["mean_epoch_seconds"]),
        "inference_seconds_per_batch": float(report["inference_seconds_per_batch"]),
        "best_epoch": int(report["best_epoch"]),
        "diagnostics_path": str(resolve(f"results/light_label/extensions/{variant}/full/diagnostics.json")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="full", choices=["smoke", "full"])
    parser.add_argument("--min-gain", type=float, default=0.01)
    args = parser.parse_args()

    base_path = resolve(f"results/light_label/b2_label_attention/{args.run}/metrics.json")
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    base_report = read_json(base_path)
    records = [record_from_metrics("e0_b2", "E0 B2", "baseline", base_report)]
    missing = []
    for variant, display, inspiration in EXTENSIONS:
        path = resolve(f"results/light_label/extensions/{variant}/{args.run}/metrics.json")
        if not path.exists():
            missing.append(str(path))
            continue
        records.append(record_from_metrics(variant, display, inspiration, read_json(path)))

    baseline = records[0]["tuned_macro_f1"]
    for record in records:
        record["delta_macro_vs_b2"] = record["tuned_macro_f1"] - baseline
        record["noise_gate_pass"] = record["variant"] == "e0_b2" or record["delta_macro_vs_b2"] >= args.min_gain
        diagnostic_path = Path(record["diagnostics_path"])
        record["diagnostics_available"] = diagnostic_path.exists()

    candidates = [record for record in records if record["variant"] != "e0_b2"]
    ranked = sorted(candidates, key=lambda item: item["delta_macro_vs_b2"], reverse=True)
    qualified = [item for item in ranked if item["noise_gate_pass"]]
    recommendation = qualified[0]["display"] if qualified else "F: None of the five is sufficiently supported"
    output_dir = resolve("results/light_label/extensions")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = resolve("reports/light_label_model/extensions")
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": "DIVE_main6_opcode_process01",
        "run": args.run,
        "seed": 42,
        "test_checked": False,
        "b2_reference_macro_f1": baseline,
        "minimum_gain_gate": args.min_gain,
        "records": records,
        "ranking": [item["display"] for item in ranked],
        "qualified_by_gain_gate": [item["display"] for item in qualified],
        "recommendation": recommendation,
        "missing_metrics": missing,
        "interpretation": "A gain below 0.01 is treated as performance fluctuation under the project decision rule. Mechanism diagnostics must still be checked before adoption.",
    }
    (output_dir / f"comparison_{args.run}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    metric_rows = []
    per_label_rows = []
    for record in records:
        metric_rows.append({key: record[key] for key in ("variant", "display", "inspiration", "fixed_macro_f1", "tuned_macro_f1", "tuned_micro_f1", "detection_f1", "total_params", "trainable_params", "peak_memory_mb", "mean_epoch_seconds", "inference_seconds_per_batch", "best_epoch", "delta_macro_vs_b2", "noise_gate_pass")})
        for label, value in zip(LABELS, record["per_label_f1"]):
            per_label_rows.append({"variant": record["variant"], "label": label, "f1": value, "delta_vs_b2": value - records[0]["per_label_f1"][LABELS.index(label)]})
    with (output_dir / f"comparison_{args.run}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output_dir / f"per_label_{args.run}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_label_rows[0]))
        writer.writeheader()
        writer.writerows(per_label_rows)

    lines = [
        "# E1-E5 Light Extension Comparison",
        "",
        "Dataset: `DIVE_main6_opcode_process01`; seed: `42`; validation only; `test_checked=false`.",
        "",
        "All five candidates are mutually exclusive B2 extensions initialized from the same B2 checkpoint.",
        f"A Macro-F1 gain below `{args.min_gain:.2f}` is treated as performance fluctuation.",
        "",
        "## Main Results",
        "",
        "| Variant | Inspiration | Tuned Macro-F1 | Micro-F1 | Detection-F1 | Params | Peak MB | Epoch s | Delta vs B2 | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        lines.append(f"| {item['display']} | {item['inspiration']} | {item['tuned_macro_f1']:.6f} | {item['tuned_micro_f1']:.6f} | {item['detection_f1']:.6f} | {item['total_params']:,} | {item['peak_memory_mb']:.1f} | {item['mean_epoch_seconds']:.2f} | {item['delta_macro_vs_b2']:+.6f} | {'PASS' if item['noise_gate_pass'] else 'noise'} |")
    lines += ["", "## Per-label F1", "", "| Variant | " + " | ".join(LABELS) + " |", "|---|" + "---:|" * len(LABELS)]
    for item in records:
        lines.append("| " + item["display"] + " | " + " | ".join(f"{value:.6f}" for value in item["per_label_f1"]) + " |")
    lines += ["", "## Provisional Selection", "", f"Ranking by validation Macro-F1: `{', '.join(item['display'] for item in ranked) or 'none'}`.", f"Noise-gate recommendation: **{recommendation}**.", "This is not a final paper claim; mechanism diagnostics and failure analysis must be checked before adopting any extension.", "", "## Missing Runs", ""]
    lines += [f"- `{path}`" for path in missing] if missing else ["- None"]
    (report_dir / f"comparison_{args.run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run": args.run, "records": len(records), "missing": missing, "recommendation": recommendation, "test_checked": False}, indent=2))


if __name__ == "__main__":
    main()

