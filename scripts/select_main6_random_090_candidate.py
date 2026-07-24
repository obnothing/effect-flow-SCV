"""Select one LD-ETPCA candidate using validation-only performance and gates."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "mlm_label_mil",
    "etp_encoder_mil",
    "etp_concat_mil",
    "ld_etpca",
    "ld_etpca_sep",
)
ELIGIBLE = {"ld_etpca", "ld_etpca_sep"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="results/main6_random_090")
    parser.add_argument("--output", default="results/main6_random_090/validation_selection.json")
    args = parser.parse_args()
    root = ROOT / args.result_root
    rows = []
    for variant in CANDIDATES:
        summary_path = root / variant / "checkpoint_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing validation summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "variant": variant,
            "checkpoint": str(ROOT / "checkpoints" / "main6_random_090" / variant / "best_macro_f1.pt"),
            "valid_macro_f1": float(summary["best_macro_f1_value"]),
            "valid_micro_f1": float(summary["best_micro_f1_value"]),
            "best_epoch": int(summary["best_macro_f1_epoch"]),
        }
        diagnostics_path = root / variant / "valid_interference_analysis.json"
        if variant in ELIGIBLE:
            if not diagnostics_path.exists():
                raise FileNotFoundError(f"Missing validation diagnostics: {diagnostics_path}")
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            row["interpretability_pass"] = bool(diagnostics.get("interpretability_pass", False))
            row["faithfulness_pass"] = bool(diagnostics.get("faithfulness_pass", False))
            values = [item["fpr"] for item in diagnostics.get("conditional_false_positive_rate", [])]
            row["mean_conditional_fpr"] = float(sum(values) / max(1, len(values)))
        rows.append(row)
    baseline = next(row for row in rows if row["variant"] == "mlm_label_mil")
    eligible = [
        row for row in rows
        if row["variant"] in ELIGIBLE
        and row["valid_macro_f1"] >= baseline["valid_macro_f1"]
        and row["interpretability_pass"]
        and row["faithfulness_pass"]
    ]
    selected = max(eligible, key=lambda row: (row["valid_macro_f1"], row["valid_micro_f1"], -row["mean_conditional_fpr"])) if eligible else None
    payload = {
        "selection_source": "validation_only",
        "test_labels_read": False,
        "baseline_variant": baseline["variant"],
        "baseline_valid_macro_f1": baseline["valid_macro_f1"],
        "candidates": rows,
        "selected": selected,
        "approved_for_single_test": selected is not None,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise SystemExit("No LD-ETPCA candidate passed validation performance and interpretation gates")


if __name__ == "__main__":
    main()
