"""Validation-only CRER evidence similarity by label co-occurrence state.

This diagnostic does not train a model and never reads test data.  It is used
to separate shared evidence caused by multi-label contracts from model-level
representation collapse.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crer_classifier import CRERClassifier  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def load_valid(config):
    payload = torch.load(resolve(config["feature_dir"]) / "valid.pt", map_location="cpu")
    features = payload["features"].float()
    mask = payload["chunk_mask"].bool()
    labels = payload["multi_labels"].float()
    if (~mask).all(dim=1).any() or not torch.isfinite(features).all():
        raise ValueError("valid cache contains an empty sample or NaN/Inf")
    return features, mask, labels


def build_model(config, checkpoint):
    variant = str(checkpoint.get("variant", ""))
    model = CRERClassifier(
        feature_dim=int(config["feature_dim"]),
        num_labels=int(config["num_labels"]),
        max_chunks=int(config["max_chunks"]),
        chunk_view_index=int(config.get("chunk_view_index", 1)),
        hidden_dim=int(config["hidden_dim"]),
        num_heads=int(config["num_heads"]),
        shared_encoder_layers=int(config["shared_encoder_layers"]),
        evidence_blocks=1 if variant in {"A4_single_block", "A6_minimal"} else int(config["evidence_blocks"]),
        dropout=0.0,
        temperature=float(config["temperature"]),
        use_label_queries=variant != "A5_shared_query",
        sparsity_target=float(config["sparsity_target"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def infer(model, features, mask, device, batch_size):
    loader = DataLoader(TensorDataset(features, mask), batch_size=int(batch_size), shuffle=False)
    evidence, scores, weights = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_features, batch_mask in loader:
            output = model(batch_features.to(device), batch_mask.to(device), return_diagnostics=True)
            evidence.append(output["evidence_representation"].float().cpu())
            scores.append(output["routing_scores"].float().cpu())
            weights.append(output["routing_weights"].float().cpu())
    return torch.cat(evidence), torch.cat(scores), torch.cat(weights)


def safe_mean(values):
    return float(np.mean(values)) if values else None


def pair_rows(labels, evidence, scores, label_names, top_k):
    normalized = torch.nn.functional.normalize(evidence, dim=-1)
    pair_rows = []
    for left in range(labels.shape[1]):
        for right in range(left + 1, labels.shape[1]):
            cosine = (normalized[:, left] * normalized[:, right]).sum(dim=-1).numpy()
            left_top = torch.topk(scores[:, :, left], k=min(int(top_k), scores.shape[1]), dim=1).indices
            right_top = torch.topk(scores[:, :, right], k=min(int(top_k), scores.shape[1]), dim=1).indices
            overlaps = []
            for row in range(labels.shape[0]):
                a = set(left_top[row].tolist())
                b = set(right_top[row].tolist())
                overlaps.append(len(a & b) / max(1, len(a | b)))
            states = {
                "both_positive": (labels[:, left] > 0.5) & (labels[:, right] > 0.5),
                "left_only_positive": (labels[:, left] > 0.5) & (labels[:, right] <= 0.5),
                "right_only_positive": (labels[:, left] <= 0.5) & (labels[:, right] > 0.5),
                "both_negative": (labels[:, left] <= 0.5) & (labels[:, right] <= 0.5),
            }
            row = {
                "label_left": label_names[left],
                "label_right": label_names[right],
                "top_k": int(top_k),
                "all_cosine": float(cosine.mean()),
                "all_topk_routing_jaccard": float(np.mean(overlaps)),
            }
            for state, selector in states.items():
                selected_cosine = cosine[selector.numpy()]
                selected_overlap = np.asarray(overlaps)[selector.numpy()]
                row[state + "_count"] = int(selector.sum())
                row[state + "_cosine"] = safe_mean(selected_cosine.tolist())
                row[state + "_topk_routing_jaccard"] = safe_mean(selected_overlap.tolist())
            pair_rows.append(row)
    return pair_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_crer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/crer_main6/label_cooccurrence.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(resolve(args.checkpoint), map_location="cpu")
    if checkpoint.get("test_checked", True):
        raise ValueError("diagnostic checkpoint must be marked test_checked=false")
    features, mask, labels = load_valid(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, checkpoint).to(device)
    evidence, scores, weights = infer(model, features, mask, device, args.batch_size)
    rows = pair_rows(labels, evidence, scores, config["label_names"], args.top_k)
    counts = labels.sum(dim=1).numpy()
    report = {
        "route": "DIVE Main6 Standalone CRER label co-occurrence diagnosis",
        "dataset": "DIVE Main6 random split",
        "checkpoint": str(resolve(args.checkpoint)),
        "validation_only": True,
        "test_checked": False,
        "evidence_ground_truth": False,
        "label_positive_rate": dict(zip(config["label_names"], labels.float().mean(dim=0).tolist())),
        "multi_label_contract_ratio": float((counts >= 2).mean()),
        "mean_label_count": float(counts.mean()),
        "pairs": rows,
        "interpretation": {
            "both_positive": "shared evidence is compatible with real multi-label co-occurrence",
            "single_positive": "tests whether evidence remains similar without label co-occurrence",
            "both_negative": "high similarity here is more consistent with model-level shared representation",
            "warning": "cosine and routing overlap are diagnostic representation measures, not local evidence ground truth",
        },
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = list(rows[0]) if rows else []
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "pairs": len(rows), "multi_label_contract_ratio": report["multi_label_contract_ratio"], "test_checked": False}, indent=2))


if __name__ == "__main__":
    main()
