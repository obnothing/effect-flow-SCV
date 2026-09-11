"""Aggregate E5 TDVP control-gate and paired-seed validation results."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
VARIANTS = ["d0_b2", "d1_param_control", "d2_global_mean", "d3_shuffled_dictionary", "d4_tdvp"]
SEEDS = [42, 43, 44]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(variant, seed, report, source):
    tuned = report["metrics"]["tuned"]
    fixed = report["metrics"]["fixed"]
    return {"variant": variant, "seed": int(seed), "source": source,
            "fixed_macro_f1": float(fixed["macro_f1"]), "tuned_macro_f1": float(tuned["macro_f1"]),
            "tuned_micro_f1": float(tuned["micro_f1"]), "detection_f1": float(report["metrics"]["detection_f1"]),
            "macro_precision": float(tuned["macro_precision"]), "macro_recall": float(tuned["macro_recall"]),
            "per_label_f1": [float(x) for x in tuned["per_label_f1"]], "thresholds": [float(x) for x in report["metrics"]["thresholds"]],
            "best_epoch": int(report["best_epoch"]), "total_params": int(report["total_params"]),
            "trainable_params": int(report["trainable_params"]), "peak_memory_mb": float(report["peak_memory_mb"]),
            "mean_epoch_seconds": float(report["mean_epoch_seconds"])}


def main():
    records = []
    missing = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = ROOT / f"results/e5_tdvp/random/{variant}/seed_{seed}/metrics.json"
            if path.exists():
                records.append(record(variant, seed, read_json(path), str(path)))
            elif variant == "d0_b2" and seed == 42:
                path = ROOT / "results/light_label/b2_label_attention/full/metrics.json"
                if path.exists():
                    records.append(record(variant, seed, read_json(path), "historical B2 reference"))
                else:
                    missing.append(str(path))
            else:
                missing.append(str(path))
    by_key = {(item["variant"], item["seed"]): item for item in records}
    paired = []
    for seed in SEEDS:
        baseline = by_key.get(("d0_b2", seed))
        if not baseline:
            continue
        for variant in VARIANTS[1:]:
            candidate = by_key.get((variant, seed))
            if not candidate:
                continue
            paired.append({"seed": seed, "variant": variant, "delta_macro_vs_b2": candidate["tuned_macro_f1"] - baseline["tuned_macro_f1"],
                           "delta_micro_vs_b2": candidate["tuned_micro_f1"] - baseline["tuned_micro_f1"],
                           "delta_detection_vs_b2": candidate["detection_f1"] - baseline["detection_f1"]})
    summary = []
    for variant in VARIANTS:
        group = [item for item in records if item["variant"] == variant]
        if not group:
            continue
        summary.append({"variant": variant, "seeds": [item["seed"] for item in group],
                        "tuned_macro_mean": float(np.mean([item["tuned_macro_f1"] for item in group])),
                        "tuned_macro_std": float(np.std([item["tuned_macro_f1"] for item in group], ddof=0)),
                        "tuned_micro_mean": float(np.mean([item["tuned_micro_f1"] for item in group])),
                        "tuned_micro_std": float(np.std([item["tuned_micro_f1"] for item in group], ddof=0)),
                        "detection_mean": float(np.mean([item["detection_f1"] for item in group])),
                        "detection_std": float(np.std([item["detection_f1"] for item in group], ddof=0)),
                        "per_label_f1_mean": [float(np.mean([item["per_label_f1"][index] for item in group])) for index in range(6)],
                        "per_label_f1_std": [float(np.std([item["per_label_f1"][index] for item in group], ddof=0)) for index in range(6)]})
    seed42 = {item["variant"]: item for item in records if item["seed"] == 42}
    gate = {"available": all(variant in seed42 for variant in VARIANTS),
            "e5_beats_d1": False, "e5_beats_d2": False, "e5_beats_d3": False, "proceed_multiseed": False}
    if gate["available"]:
        e5 = seed42["d4_tdvp"]["tuned_macro_f1"]
        gate["e5_beats_d1"] = e5 > seed42["d1_param_control"]["tuned_macro_f1"]
        gate["e5_beats_d2"] = e5 > seed42["d2_global_mean"]["tuned_macro_f1"]
        gate["e5_beats_d3"] = e5 > seed42["d3_shuffled_dictionary"]["tuned_macro_f1"]
        gate["proceed_multiseed"] = gate["e5_beats_d1"] and gate["e5_beats_d2"] and gate["e5_beats_d3"]
    output = ROOT / "results/e5_tdvp"
    report_dir = ROOT / "reports/light_label_model/e5_final"
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "DIVE_main6_opcode_process01", "protocol": "random", "records": records,
               "summary": summary, "paired_deltas": paired, "control_gate": gate, "missing": missing, "test_checked": False}
    (output / "main_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "per_label.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "seed", "label", "f1"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in records:
            for label, value in zip(LABELS, item["per_label_f1"]):
                writer.writerow({"variant": item["variant"], "seed": item["seed"], "label": label, "f1": value})
    with (output / "paired_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["seed", "variant", "delta_macro_vs_b2", "delta_micro_vs_b2", "delta_detection_vs_b2"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    lines = ["# E5 TDVP Control Comparison", "", "Dataset: `DIVE_main6_opcode_process01`; validation only; `test_checked=false`.", "",
             "## Per-run results", "", "| Variant | Seed | Tuned Macro-F1 | Micro-F1 | Detection-F1 | Best epoch | Params | Peak MB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in records:
        lines.append(f"| {item['variant']} | {item['seed']} | {item['tuned_macro_f1']:.6f} | {item['tuned_micro_f1']:.6f} | {item['detection_f1']:.6f} | {item['best_epoch']} | {item['total_params']:,} | {item['peak_memory_mb']:.1f} |")
    lines += ["", "## Mean and standard deviation", "", "| Variant | Seeds | Macro mean | Macro std | Micro mean | Micro std | Detection mean |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['variant']} | {','.join(str(x) for x in item['seeds'])} | {item['tuned_macro_mean']:.6f} | {item['tuned_macro_std']:.6f} | {item['tuned_micro_mean']:.6f} | {item['tuned_micro_std']:.6f} | {item['detection_mean']:.6f} |")
    lines += ["", "## CONTROL GATE", "", json.dumps(gate, indent=2), "", "Proceed to seeds 43/44 only when the control gate is true. This report does not unlock test."]
    (report_dir / "control_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "missing": missing, "control_gate": gate, "test_checked": False}, indent=2))


if __name__ == "__main__": main()

