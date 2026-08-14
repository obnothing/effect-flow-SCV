"""Validate Opcode-CSDG graph/sequence cache alignment and integrity."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def check_split(graph_path, sequence_path, label_width):
    graph = torch.load(resolve(graph_path), map_location="cpu")
    sequence = torch.load(resolve(sequence_path), map_location="cpu")
    sidecar_path = resolve(graph_path).with_suffix(".manifest.json")
    if not sidecar_path.exists():
        raise ValueError(f"Missing graph cache manifest: {sidecar_path}")
    digest = hashlib.sha256(resolve(graph_path).read_bytes()).hexdigest()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("cache_file_sha256") != digest:
        raise ValueError(f"Graph cache hash mismatch: {graph_path}")
    if graph.get("schema") not in {"main6_opcode_csdg_v1", "main6_opcode_csdg_v2"}:
        raise ValueError(f"Unsupported graph schema: {graph.get('schema')}")
    if graph["ids"] != sequence["ids"]:
        raise ValueError("Graph and sequence IDs are not exactly aligned")
    labels = graph["multi_labels"]
    if labels.ndim != 2 or labels.shape[1] != label_width:
        raise ValueError("Graph labels do not have Main-6 width")
    if sequence["features"].shape[0] != len(graph["ids"]):
        raise ValueError("Sequence and graph sample counts differ")
    node_count = 0
    edge_count = 0
    for sample_id, record in zip(graph["ids"], graph["records"]):
        features = record["node_features"]
        mask = record["node_mask"].bool()
        edges = record["edge_index"].long()
        types = record["edge_type"].long()
        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError(f"Invalid node feature shape for {sample_id}")
        if mask.shape != features.shape[:1]:
            raise ValueError(f"Invalid node mask shape for {sample_id}")
        if features.shape[0] and not bool(mask.all()):
            raise ValueError(f"Incomplete token feature coverage for {sample_id}")
        if edges.shape[0] != 2 or edges.shape[1] != types.shape[0]:
            raise ValueError(f"Invalid edge shape for {sample_id}")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= features.shape[0]):
            raise ValueError(f"Out-of-range edge for {sample_id}")
        if types.numel() and (int(types.min()) < 0 or int(types.max()) > 5):
            raise ValueError(f"Invalid edge type for {sample_id}")
        if torch.isnan(features.float()).any() or torch.isinf(features.float()).any():
            raise ValueError(f"NaN/Inf node feature for {sample_id}")
        local = record.get("node_local_features")
        offsets = record.get("node_local_offsets")
        if (local is None) != (offsets is None):
            raise ValueError(f"Incomplete local-node cache for {sample_id}")
        if local is not None and (offsets.numel() != features.shape[0] + 1 or int(offsets[-1]) != local.shape[0]):
            raise ValueError(f"Invalid local-node offsets for {sample_id}")
        node_count += int(mask.sum())
        edge_count += int(types.numel())
    return {"samples": len(graph["ids"]), "valid_nodes": node_count, "edges": edge_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--splits", nargs="+", choices=["train", "valid"], default=["train", "valid"])
    args = parser.parse_args()
    payload = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    config = {**payload["common"], **(payload.get("variants", {}).get(args.variant, {}) if args.variant else {})}
    result = {"route": config["route_name"], "test_checked": False, "splits": {}}
    for split in args.splits:
        result["splits"][split] = check_split(
            Path(config["graph_cache_dir"]) / f"{split}.pt",
            Path(config["sequence_feature_dir"]) / f"{split}.pt",
            len(config["label_names"]),
        )
    output = resolve(config["report_dir"]) / "graph_cache_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
