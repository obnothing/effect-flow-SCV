import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_chunk_mil import load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--selection", choices=["checkpoint", "valid_label"], required=True)
    parser.add_argument("--label_name", default=None)
    parser.add_argument("--validation_root", default=None)
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_prefix", required=True)
    return parser.parse_args()


def checkpoint_path_for(config, variant, checkpoint_root):
    if checkpoint_root:
        return Path(checkpoint_root) / variant / "best_selection.pt"
    return Path(config["checkpoint_dir"]) / "best_selection.pt"


def checkpoint_row(config_path, variant, checkpoint_root=None):
    config = load_config(config_path, variant)
    checkpoint_path = checkpoint_path_for(config, variant, checkpoint_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "variant": variant,
        "score": float(checkpoint.get("best_macro_f1", -1.0)),
        "precision": 0.0,
        "checkpoint": str(checkpoint_path),
        "config": config,
        "source_epoch": checkpoint.get("epoch"),
    }


def valid_label_row(
    config_path,
    variant,
    validation_root,
    label_name,
    checkpoint_root=None,
):
    config = load_config(config_path, variant)
    threshold_path = Path(validation_root) / variant / "per_label_thresholds_valid.json"
    if not threshold_path.exists():
        raise FileNotFoundError(threshold_path)
    selection = json.loads(threshold_path.read_text(encoding="utf-8"))
    label_row = next(
        row for row in selection["per_label"] if row["label_name"] == label_name
    )
    checkpoint_path = checkpoint_path_for(config, variant, checkpoint_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "variant": variant,
        "score": float(label_row["best_valid_f1"]),
        "precision": float(label_row["best_valid_precision"]),
        "average_precision": float(label_row.get("average_precision") or 0.0),
        "threshold": float(label_row["best_threshold"]),
        "selection_method": label_row.get("selection_method"),
        "checkpoint": str(checkpoint_path),
        "config": config,
        "source_epoch": checkpoint.get("epoch"),
    }


def main():
    args = parse_args()
    config_path = Path(args.config)
    if args.selection == "valid_label" and not args.label_name:
        raise ValueError("valid_label selection requires --label_name")
    rows = []
    for variant in args.variants:
        if args.selection == "checkpoint":
            rows.append(checkpoint_row(config_path, variant, args.checkpoint_root))
        else:
            rows.append(
                valid_label_row(
                    config_path,
                    variant,
                    args.validation_root,
                    args.label_name,
                    args.checkpoint_root,
                )
            )
    if args.selection == "valid_label":
        top_score = max(row["score"] for row in rows)
        eligible = [row for row in rows if top_score - row["score"] < 0.005]
        winner = max(
            eligible,
            key=lambda row: (
                row.get("average_precision", 0.0),
                row["precision"],
            ),
        )
    else:
        winner = max(rows, key=lambda row: (row["score"], row["precision"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = output_dir / "best_selection.pt"
    shutil.copy2(winner["checkpoint"], selected_checkpoint)
    (output_dir / "selected_config.yaml").write_text(
        yaml.safe_dump(winner["config"], sort_keys=False),
        encoding="utf-8",
    )
    serializable_rows = [
        {key: value for key, value in row.items() if key != "config"}
        for row in rows
    ]
    report = {
        "selection": args.selection,
        "label_name": args.label_name,
        "winner": {key: value for key, value in winner.items() if key != "config"},
        "selected_checkpoint": str(selected_checkpoint),
        "candidates": serializable_rows,
    }
    prefix = Path(args.report_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "DIVE ClassicNet-ESCORT checkpoint selection",
        "",
        f"selection: {args.selection}",
        f"label_name: {args.label_name}",
        f"winner: {winner['variant']}",
        f"winner_score: {winner['score']:.6f}",
        f"selected_checkpoint: {selected_checkpoint}",
        "",
        "Candidates:",
    ]
    for row in serializable_rows:
        lines.append(
            f"- {row['variant']}: score={row['score']:.6f} "
            f"precision={row['precision']:.6f} epoch={row['source_epoch']}"
        )
    prefix.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] selected {winner['variant']} -> {selected_checkpoint}")


if __name__ == "__main__":
    main()
