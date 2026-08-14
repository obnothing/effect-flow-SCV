"""Validation-only graph faithfulness checks for one CSDG candidate."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evm_opcode_graph_dataset import OpcodeGraphSequenceDataset, collate_opcode_graph, move_graph_batch  # noqa: E402
from evm_opcode_graph_residual_mil import OpcodeGraphResidualMIL, load_opcode_graph_state  # noqa: E402
from train_main6_opcode_csdg import thresholds_and_metrics  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path, variant):
    payload = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    config = dict(payload["common"])
    config.update(payload["variants"][variant])
    return config


def evaluate(model, loader, device, thresholds):
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_graph_batch(batch, device)
            output = model(batch["sequence_features"], batch["sequence_mask"], batch["node_features"], batch["node_mask"], batch["edge_index"], batch["edge_type"], node_local_features=batch["node_local_features"], node_local_offsets=batch["node_local_offsets"], node_type=batch["node_type"])
            logits.append(output["recognition_logits"].cpu())
            labels.append(batch["multi_labels"].cpu())
    values = torch.cat(logits).numpy()
    targets = torch.cat(labels).numpy()
    return thresholds_and_metrics(values, targets, thresholds)


def shuffled_edges(batch, generator):
    edges = []
    for edge_index, node_features in zip(batch["edge_index"], batch["node_features"]):
        current = edge_index.clone()
        if current.shape[1] > 1:
            permutation = torch.randperm(current.shape[1], generator=generator)
            current[1] = current[1][permutation]
        edges.append(current)
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.variant)
    checkpoint_path = resolve(config["checkpoint_dir"]) / "best_macro_f1.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = OpcodeGraphResidualMIL(config)
    load_opcode_graph_state(model, checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = OpcodeGraphSequenceDataset(resolve(config["graph_cache_dir"]) / "valid.pt", resolve(config["sequence_feature_dir"]) / "valid.pt", config["label_names"])
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)), collate_fn=collate_opcode_graph)
    thresholds = checkpoint["metrics"]["thresholds"]
    full = evaluate(model, loader, device, thresholds)
    original_use_edges = model.use_graph_edges
    model.use_graph_edges = False
    no_edges = evaluate(model, loader, device, thresholds)
    model.use_graph_edges = original_use_edges
    random_generator = torch.Generator().manual_seed(42)
    shuffled_logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            moved = move_graph_batch(batch, device)
            moved["edge_index"] = shuffled_edges(moved, random_generator)
            output = model(moved["sequence_features"], moved["sequence_mask"], moved["node_features"], moved["node_mask"], moved["edge_index"], moved["edge_type"], node_local_features=moved["node_local_features"], node_local_offsets=moved["node_local_offsets"], node_type=moved["node_type"])
            shuffled_logits.append(output["recognition_logits"].cpu())
            labels.append(moved["multi_labels"].cpu())
    shuffled = thresholds_and_metrics(torch.cat(shuffled_logits).numpy(), torch.cat(labels).numpy(), thresholds)
    report = {
        "variant": args.variant,
        "split": "valid",
        "test_labels_read": False,
        "full_macro_f1": full["macro_f1"],
        "no_edge_macro_f1": no_edges["macro_f1"],
        "edge_shuffle_macro_f1": shuffled["macro_f1"],
        "edge_removal_macro_drop": full["macro_f1"] - no_edges["macro_f1"],
        "edge_shuffle_macro_drop": full["macro_f1"] - shuffled["macro_f1"],
        "full_metrics": full,
        "no_edge_metrics": no_edges,
        "edge_shuffle_metrics": shuffled,
    }
    output = resolve(config["result_dir"]) / "graph_ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
