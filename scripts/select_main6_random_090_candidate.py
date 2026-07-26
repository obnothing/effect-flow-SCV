"""Select one pure-MLM eight-view candidate using validation only."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("mlm8_slot1", "mlm8_slot2", "mlm8_slot3", "mlm8_slot4")
REFERENCE_SUMMARY = ROOT / "results/main6_random_090/mlm_label_mil/checkpoint_summary.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results/main6_random_090_mlm8")
    parser.add_argument("--checkpoint-root", default="checkpoints/main6_random_090_mlm8")
    parser.add_argument("--baseline-summary", default=str(REFERENCE_SUMMARY))
    parser.add_argument("--output", default="results/main6_random_090_mlm8/validation_selection.json")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_summary)
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing frozen pure-MLM baseline summary: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_macro = float(baseline["best_macro_f1_value"])
    root = ROOT / args.result_root
    checkpoint_root = ROOT / args.checkpoint_root
    rows = []
    for variant in CANDIDATES:
        summary_path = root / variant / "checkpoint_summary.json"
        checkpoint = checkpoint_root / variant / "best_macro_f1.pt"
        if not summary_path.exists() or not checkpoint.exists():
            raise FileNotFoundError(f"Missing validation artifact for {variant}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": variant,
                "checkpoint": str(checkpoint),
                "valid_macro_f1": float(summary["best_macro_f1_value"]),
                "valid_micro_f1": float(summary["best_micro_f1_value"]),
                "best_epoch": int(summary["best_macro_f1_epoch"]),
            }
        )
    eligible = [row for row in rows if row["valid_macro_f1"] >= baseline_macro]
    selected = max(
        eligible,
        key=lambda row: (row["valid_macro_f1"], row["valid_micro_f1"]),
    ) if eligible else None
    payload = {
        "selection_source": "validation_only",
        "test_labels_read": False,
        "baseline_variant": "public_mlm_label_mil_round1",
        "baseline_summary": str(baseline_path),
        "baseline_valid_macro_f1": baseline_macro,
        "candidates": rows,
        "selected": selected,
        "approved_for_single_test": selected is not None,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise SystemExit("No MLM8 candidate reached the frozen pure-MLM validation baseline")


if __name__ == "__main__":
    main()
