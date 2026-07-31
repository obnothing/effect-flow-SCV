"""Select exactly one Opcode-CSDG candidate using validation artifacts only."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def baseline_score(payload):
    for key in ("valid_macro_f1", "baseline_valid_macro_f1", "best_macro_f1"):
        if key in payload:
            return float(payload[key])
    history = payload.get("history") or payload.get("epoch_history")
    if isinstance(history, list) and history:
        values = [item.get("best_macro_f1", item.get("macro_f1", -1.0)) for item in history]
        return float(max(values))
    raise ValueError("Baseline summary has no validation macro-F1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--require-ablation", action="store_true")
    args = parser.parse_args()
    payload = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    common = dict(payload.get("common", {}))
    baseline_path = resolve(common["baseline_summary"])
    baseline = baseline_score(read_json(baseline_path))
    graph_dir = resolve(common["graph_cache_dir"])
    sequence_dir = resolve(common["sequence_feature_dir"])
    if (graph_dir / "test.pt").exists() or (sequence_dir / "test.pt").exists():
        raise RuntimeError("Refusing validation selection after a test cache exists")
    candidates = []
    for variant in payload.get("variants", {}):
        result_path = resolve(payload["variants"][variant]["result_dir"]) / "valid_summary.json"
        if not result_path.exists():
            continue
        summary = read_json(result_path)
        if args.require_ablation:
            ablation_path = result_path.parent / "graph_ablation.json"
            if not ablation_path.exists():
                raise RuntimeError(f"Missing graph ablation artifact: {ablation_path}")
            ablation = read_json(ablation_path)
            summary["graph_ablation"] = ablation
            if float(ablation.get("edge_shuffle_macro_drop", 0.0)) <= 0.0:
                continue
        summary["variant"] = variant
        candidates.append(summary)
    if not candidates:
        raise RuntimeError("No graph candidate validation artifacts found")
    eligible = [item for item in candidates if item["variant"] != "graph_mlp_control" and float(item["valid_macro_f1"]) >= baseline]
    eligible.sort(key=lambda item: (float(item["valid_macro_f1"]), float(item.get("valid_micro_f1", -1.0)), float(np_mean(item.get("valid_metrics", {}).get("per_label_average_precision", [])))), reverse=True)
    selected = eligible[0] if eligible else None
    candidate_records = [{
        "variant": "mlm8_slot3_baseline",
        "valid_macro_f1": baseline,
        "baseline": True,
    }] + candidates
    result = {
        "route": common["route_name"],
        "selection_source": "validation_only",
        "test_labels_read": False,
        "baseline_variant": "mlm8_slot3_baseline",
        "baseline_summary": str(baseline_path),
        "baseline_valid_macro_f1": baseline,
        "candidates": candidate_records,
        "selected": selected,
        "approved_for_single_test": selected is not None,
        "thresholds": selected["valid_metrics"]["thresholds"] if selected else None,
    }
    output = resolve(common["result_root"]) / "selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if selected is None:
        raise SystemExit("No candidate passed the validation baseline gate")


def np_mean(values):
    return sum(float(value) for value in values) / max(1, len(values))


if __name__ == "__main__":
    main()
