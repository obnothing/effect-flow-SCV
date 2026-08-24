import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_execution_aware_mil.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    baseline_path = resolve(config["baseline_summary"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_score = float(baseline.get("valid_macro_f1", baseline.get("best_macro_f1")))
    summary_path = resolve(config["result_dir"]) / "valid_summary.json"
    candidates = []
    if summary_path.exists():
        item = json.loads(summary_path.read_text(encoding="utf-8")); item["variant"] = "execution_aware_mil"; candidates.append(item)
    eligible = [x for x in candidates if float(x["valid_macro_f1"]) >= baseline_score]
    eligible.sort(key=lambda x: (float(x["valid_macro_f1"]), float(x.get("valid_micro_f1", 0))), reverse=True)
    result = {"route": config["route_name"], "selection_source": "validation_only", "test_labels_read": False, "baseline_valid_macro_f1": baseline_score, "candidates": candidates, "selected": eligible[0] if eligible else None, "approved_for_single_test": False, "note": "Manual review and threshold freeze are required before ALLOW_TEST=1."}
    output = resolve("results/main6_opcode_execution_aware/selection.json"); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"); print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
