"""Validation-only candidate selection for Opcode-LSE-MIL."""

import argparse
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def baseline_score(payload):
    for key in ("valid_macro_f1", "best_macro_f1", "baseline_valid_macro_f1"):
        if key in payload:
            return float(payload[key])
    raise ValueError("Baseline summary has no validation macro-F1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_lse_mil.yaml")
    args = parser.parse_args()
    payload = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    common = payload["common"]
    baseline_path = resolve(common["baseline_summary"])
    baseline = baseline_score(read(baseline_path))
    candidates = []
    for variant, variant_config in payload["variants"].items():
        path = resolve(variant_config["result_dir"]) / "valid_summary.json"
        if not path.exists():
            continue
        item = read(path)
        item["variant"] = variant
        candidates.append(item)
    eligible = [item for item in candidates if float(item["valid_macro_f1"]) >= baseline]
    eligible.sort(key=lambda item: (float(item["valid_macro_f1"]), float(item["valid_micro_f1"])), reverse=True)
    selected = eligible[0] if eligible else None
    result = {
        "route": common["route_name"],
        "selection_source": "validation_only",
        "test_labels_read": False,
        "baseline_variant": "public_mlm_label_mil_round1",
        "baseline_summary": str(baseline_path),
        "baseline_valid_macro_f1": baseline,
        "candidates": candidates,
        "selected": selected,
        "approved_for_single_test": False,
        "note": "Manual review and a dedicated locked final evaluator are required before any test cache is created.",
    }
    output = resolve("results/main6_opcode_lse_mil/selection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
