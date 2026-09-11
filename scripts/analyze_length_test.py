"""Compare validation-only B2 length-cap experiments."""

import csv
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
EXPERIMENTS = [("len12288", 12288), ("len16384", 16384)]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def coverage(cache_path, limit):
    payload = torch.load(cache_path, map_location="cpu")
    lengths = payload["original_lengths"].numpy()
    return {
        "samples": int(len(lengths)),
        "coverage": float((lengths <= limit).mean()),
        "truncated": int((lengths > limit).sum()),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
    }


def main():
    baseline_path = ROOT / "results/light_label/b2_label_attention/full/metrics.json"
    baseline = read_json(baseline_path)
    base_macro = float(baseline["metrics"]["tuned"]["macro_f1"])
    records = [{
        "variant": "B2_8192",
        "max_len": 8192,
        "tuned_macro_f1": base_macro,
        "tuned_micro_f1": float(baseline["metrics"]["tuned"]["micro_f1"]),
        "detection_f1": float(baseline["metrics"]["detection_f1"]),
        "per_label_f1": [float(x) for x in baseline["metrics"]["tuned"]["per_label_f1"]],
        "total_params": int(baseline["total_params"]),
        "peak_memory_mb": float(baseline["peak_memory_mb"]),
        "mean_epoch_seconds": float(baseline["mean_epoch_seconds"]),
        "best_epoch": int(baseline["best_epoch"]),
        "delta_macro_vs_8192": 0.0,
    }]
    missing = []
    for name, limit in EXPERIMENTS:
        path = ROOT / f"results/light_label/length_test/{name}/full/metrics.json"
        if not path.exists():
            missing.append(str(path))
            continue
        report = read_json(path)
        tuned = report["metrics"]["tuned"]
        records.append({
            "variant": f"B2_{limit}",
            "max_len": limit,
            "tuned_macro_f1": float(tuned["macro_f1"]),
            "tuned_micro_f1": float(tuned["micro_f1"]),
            "detection_f1": float(report["metrics"]["detection_f1"]),
            "per_label_f1": [float(x) for x in tuned["per_label_f1"]],
            "total_params": int(report["total_params"]),
            "peak_memory_mb": float(report["peak_memory_mb"]),
            "mean_epoch_seconds": float(report["mean_epoch_seconds"]),
            "best_epoch": int(report["best_epoch"]),
            "delta_macro_vs_8192": float(tuned["macro_f1"] - base_macro),
        })
    for record in records:
        limit = record["max_len"]
        record["train_coverage"] = coverage(ROOT / f"data/features/light_label_process01/train_max{limit}.pt", limit)
        record["valid_coverage"] = coverage(ROOT / f"data/features/light_label_process01/valid_max{limit}.pt", limit)
        record["gain_gate_pass"] = record["delta_macro_vs_8192"] >= 0.01
    output_dir = ROOT / "results/light_label/length_test"
    report_dir = ROOT / "reports/light_label_model/length_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "DIVE_main6_opcode_process01",
        "seed": 42,
        "records": records,
        "missing": missing,
        "minimum_gain_gate": 0.01,
        "test_checked": False,
        "interpretation": "This is a validation-only length-cap comparison. Gains below 0.01 are treated as fluctuation.",
    }
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = ["variant", "max_len", "tuned_macro_f1", "tuned_micro_f1", "detection_f1", "total_params", "peak_memory_mb", "mean_epoch_seconds", "best_epoch", "delta_macro_vs_8192", "gain_gate_pass"]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: record[field] for field in fields} for record in records])
    with (output_dir / "per_label.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "max_len", "label", "f1", "delta_vs_8192"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for label, value, base_value in zip(LABELS, record["per_label_f1"], records[0]["per_label_f1"]):
                writer.writerow({"variant": record["variant"], "max_len": record["max_len"], "label": label, "f1": value, "delta_vs_8192": value - base_value})
    lines = [
        "# B2 Length-Cap Experiment",
        "",
        "Dataset: `DIVE_main6_opcode_process01`; seed 42; validation only; test remains locked.",
        "",
        "| Variant | Max length | Tuned Macro-F1 | Micro-F1 | Delta vs 8192 | Train coverage | Valid coverage | Peak MB | Epoch s | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(f"| {record['variant']} | {record['max_len']} | {record['tuned_macro_f1']:.6f} | {record['tuned_micro_f1']:.6f} | {record['delta_macro_vs_8192']:+.6f} | {record['train_coverage']['coverage']:.4f} | {record['valid_coverage']['coverage']:.4f} | {record['peak_memory_mb']:.1f} | {record['mean_epoch_seconds']:.2f} | {'PASS' if record['gain_gate_pass'] else 'noise'} |")
    lines += ["", "Gains below 0.01 are performance fluctuation under the project rule.", "", "Missing runs:"]
    lines += [f"- `{item}`" for item in missing] if missing else ["- None"]
    (report_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "missing": missing, "test_checked": False}, indent=2))


if __name__ == "__main__":
    main()

