"""Choose one Main-6 candidate from validation summaries without reading test labels."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("mlm_control", "effectflow_control", "effectflow_multiscale")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="results/main6_random_090")
    parser.add_argument("--minimum_valid_macro", type=float, default=0.7899302632666712)
    parser.add_argument("--output", default="results/main6_random_090/validation_selection.json")
    args = parser.parse_args()
    result_root = PROJECT_ROOT / args.result_root
    rows = []
    for variant in CANDIDATES:
        path = result_root / variant / "checkpoint_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing validation summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": variant,
                "checkpoint": str(
                    PROJECT_ROOT / "checkpoints" / "main6_random_090" / variant / "best_macro_f1.pt"
                ),
                "valid_macro_f1": float(summary["best_macro_f1_value"]),
                "valid_micro_f1": float(summary["best_micro_f1_value"]),
                "best_epoch": int(summary["best_macro_f1_epoch"]),
            }
        )
    selected = max(rows, key=lambda row: (row["valid_macro_f1"], row["valid_micro_f1"]))
    approved = selected["valid_macro_f1"] > float(args.minimum_valid_macro)
    report = {
        "selection_source": "validation_only",
        "test_labels_read": False,
        "minimum_valid_macro": float(args.minimum_valid_macro),
        "candidates": rows,
        "selected": selected if approved else None,
        "approved_for_single_test": approved,
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not approved:
        raise SystemExit("No candidate improves the validation macro-F1 gate; test evaluation is blocked.")


if __name__ == "__main__":
    main()
