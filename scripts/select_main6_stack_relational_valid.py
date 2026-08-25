"""Validation-only selection for the isolated stack-relational route."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_relational.yaml")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    baseline_path = resolve(config["baseline_summary"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_score = float(baseline.get("valid_macro_f1", baseline.get("best_macro_f1_value")))
    result_dir = resolve(config["result_dir"])
    summary_path = result_dir / "valid_summary.json"
    candidates = []
    if summary_path.exists():
        row = json.loads(summary_path.read_text(encoding="utf-8")); row["variant"] = "stack_relation_full"; candidates.append(row)
    eligible = [row for row in candidates if float(row["valid_macro_f1"]) >= baseline_score]
    eligible.sort(key=lambda row: (float(row["valid_macro_f1"]), float(row.get("valid_micro_f1", 0))), reverse=True)
    selected = eligible[0] if eligible else None
    payload = {
        "route": config["route_name"], "selection_source": "validation_only", "test_labels_read": False,
        "baseline_variant": "historical_mlm8_slot3", "baseline_summary": str(baseline_path),
        "baseline_valid_macro_f1": baseline_score, "candidates": candidates, "selected": selected,
        "approved_for_single_test": False,
        "note": "Manual artifact review and threshold freeze are required before ALLOW_TEST=1.",
    }
    output = result_dir / "validation_selection.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise SystemExit("No stack-relational candidate reached the frozen baseline")


if __name__ == "__main__":
    main()

