"""Validation threshold selection and locked final evaluation for Source-Main6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from metrics import compute_multilabel_metrics_from_probs
from solidity_graph_dataset import SolidityGraphDataset, collate_solidity_graph
from solidity_source_v2_dataset import SoliditySourceV2Dataset, collate_source_v2
from solidity_graphcodebert_model import SolidityGraphCodeBERTMultiSlotMIL
from train_solidity_graphcodebert import evaluate, move

ROOT = Path(__file__).resolve().parents[1]


def config_for(path, variant):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")); output = dict(payload["common"]); output.update(payload["variants"][variant]); return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--variant", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--split", required=True, choices=["valid", "test"]); parser.add_argument("--threshold-file"); parser.add_argument("--threshold-search", action="store_true"); parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args(); config = config_for(args.config, args.variant); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = ROOT / config["graph_cache_dir"] / f"{args.split}.pt"
    schema = torch.load(cache_path, map_location="cpu").get("schema")
    dataset_type, collate = (SoliditySourceV2Dataset, collate_source_v2) if schema == "solidity_source_windows_v2" else (SolidityGraphDataset, collate_solidity_graph)
    dataset = dataset_type(cache_path)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate)
    model = SolidityGraphCodeBERTMultiSlotMIL(config); model.apply_lora(config); checkpoint = torch.load(args.checkpoint, map_location="cpu"); model.load_state_dict(checkpoint["model_state_dict"]); model.to(device)
    result_dir = ROOT / config["result_dir"]
    if args.threshold_search:
        raw = evaluate(model, loader, device, [0.5] * int(config["num_labels"])); candidates = list(config["threshold_candidates"]); thresholds = []
        for label in range(int(config["num_labels"])):
            scores = [(float(value), compute_multilabel_metrics_from_probs(raw["labels"][:, [label]], raw["probs"][:, [label]], [value])["recognition_macro_f1"]) for value in candidates]
            thresholds.append(max(scores, key=lambda item: item[1])[0])
        (result_dir / "per_label_thresholds_valid.json").write_text(json.dumps({"split": "valid", "thresholds": thresholds, "label_names": config["label_names"]}, indent=2) + "\n", encoding="utf-8")
    elif args.threshold_file:
        thresholds = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))["thresholds"]
    else:
        thresholds = [0.5] * int(config["num_labels"])
    result = evaluate(model, loader, device, thresholds); report = {"split": args.split, "thresholds": thresholds, "metrics": result["metrics"], "loss": result["loss"], "checkpoint": args.checkpoint, "test_labels_read": args.split == "test"}
    result_dir.mkdir(parents=True, exist_ok=True); (result_dir / f"{args.split}_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.save_predictions:
        with (result_dir / f"{args.split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row, sample_id in enumerate(dataset.rows):
                handle.write(json.dumps({"id": sample_id["id"], "multi_true": result["labels"][row].astype(int).tolist(), "multi_prob": result["probs"][row].tolist(), "thresholds": thresholds}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
