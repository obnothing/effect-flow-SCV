"""Validation-only selection for the independent DIVE Source-Main6 route."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("source_seq_mean", "source_graph_mean", "source_graph_mil", "source_graph_mil_contrast", "source_graph_mil_full")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--result-root", default="results/dive_source_main6"); parser.add_argument("--checkpoint-root", default="checkpoints/dive_source_main6"); parser.add_argument("--output", default="results/dive_source_main6/validation_selection.json"); parser.add_argument("--candidates", nargs="+", default=CANDIDATES); args = parser.parse_args()
    rows = []
    for variant in args.candidates:
        summary = json.loads((ROOT / args.result_root / variant / "checkpoint_summary.json").read_text(encoding="utf-8"))
        checkpoint = ROOT / args.checkpoint_root / variant / "best_macro_f1.pt"
        if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
        rows.append({"variant": variant, "checkpoint": str(checkpoint), "valid_macro_f1": summary["best_macro_f1_value"], "valid_micro_f1": summary["best_micro_f1_value"], "mean_pr_auc": summary["mean_pr_auc"], "best_epoch": summary["best_macro_f1_epoch"]})
    selected = max(rows, key=lambda row: (row["valid_macro_f1"], row["valid_micro_f1"], row["mean_pr_auc"]))
    payload = {"dataset": "DIVE Source-Main6", "selection_source": "validation_only", "test_labels_read": False, "candidates": rows, "selected": selected, "approved_for_single_test": True}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
