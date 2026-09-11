"""Aggregate the secondary grouped-development B2 versus TDVP comparison."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    rows = []
    for variant in ("d0_b2", "d4_tdvp"):
        for seed in (42, 43, 44):
            path = ROOT / f"results/e5_tdvp/grouped/{variant}/seed_{seed}/metrics.json"
            if not path.exists():
                continue
            item = read(path)
            tuned = item["metrics"]["tuned"]
            rows.append({"variant": variant, "seed": seed, "macro_f1": float(tuned["macro_f1"]), "micro_f1": float(tuned["micro_f1"]),
                         "detection_f1": float(item["metrics"]["detection_f1"]), "per_label_f1": [float(x) for x in tuned["per_label_f1"]],
                         "best_epoch": int(item["best_epoch"]), "test_checked": False})
    index = {(row["variant"], row["seed"]): row for row in rows}
    deltas = []
    for seed in (42, 43, 44):
        if ("d0_b2", seed) in index and ("d4_tdvp", seed) in index:
            base, candidate = index[("d0_b2", seed)], index[("d4_tdvp", seed)]
            deltas.append({"seed": seed, "delta_macro_f1": candidate["macro_f1"] - base["macro_f1"],
                           "delta_micro_f1": candidate["micro_f1"] - base["micro_f1"],
                           "delta_detection_f1": candidate["detection_f1"] - base["detection_f1"]})
    output = ROOT / "results/e5_tdvp/grouped"
    report_dir = ROOT / "reports/light_label_model/e5_final"
    output.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "DIVE_main6_opcode_process01 train+valid grouped development", "protocol": "grouped_dev", "records": rows,
               "paired_deltas": deltas, "mean_delta_macro": float(np.mean([x["delta_macro_f1"] for x in deltas])) if deltas else None,
               "std_delta_macro": float(np.std([x["delta_macro_f1"] for x in deltas])) if deltas else None,
               "test_checked": False}
    (output / "grouped_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "grouped_per_label.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "seed", "label", "f1"]); writer.writeheader()
        for row in rows:
            for label, value in zip(LABELS, row["per_label_f1"]): writer.writerow({"variant": row["variant"], "seed": row["seed"], "label": label, "f1": value})
    lines = ["# E5 TDVP Grouped Development Comparison", "", "Train+valid development pool only; test remains locked.", "",
             "| Variant | Seed | Macro-F1 | Micro-F1 | Detection-F1 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {row['variant']} | {row['seed']} | {row['macro_f1']:.6f} | {row['micro_f1']:.6f} | {row['detection_f1']:.6f} |" for row in rows]
    lines += ["", "## Paired deltas", "", "| Seed | Delta Macro | Delta Micro | Delta Detection |", "|---:|---:|---:|---:|"]
    lines += [f"| {row['seed']} | {row['delta_macro_f1']:+.6f} | {row['delta_micro_f1']:+.6f} | {row['delta_detection_f1']:+.6f} |" for row in deltas]
    (report_dir / "grouped_generalization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "paired": len(deltas), "test_checked": False}, indent=2))


if __name__ == "__main__": main()

