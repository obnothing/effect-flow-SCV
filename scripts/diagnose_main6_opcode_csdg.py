"""Validation-only diagnostics for an Opcode-CSDG residual candidate."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evm_control_stack_graph import EDGE_TYPES  # noqa: E402
from evm_opcode_graph_dataset import OpcodeGraphSequenceDataset, collate_opcode_graph, move_graph_batch  # noqa: E402
from evm_opcode_graph_residual_mil import OpcodeGraphResidualMIL  # noqa: E402
from train_main6_opcode_csdg import thresholds_and_metrics  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path, variant):
    payload = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    config = dict(payload["common"])
    config.update(payload["variants"][variant])
    config["variant"] = variant
    return config


def metrics_at_thresholds(logits, labels, thresholds):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    thresholds = np.asarray(thresholds, dtype=np.float64)
    predictions = probabilities >= thresholds[None, :]
    return {
        "thresholds": thresholds.tolist(),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "per_label_f1": f1_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_label_average_precision": [
            float(average_precision_score(labels[:, index], probabilities[:, index]))
            for index in range(labels.shape[1])
        ],
    }


def predict(model, loader, device, collect_attention=False):
    sequence_logits, graph_logits, final_logits, labels = [], [], [], []
    attention_summary = defaultdict(lambda: defaultdict(float))
    attention_samples = defaultdict(int)
    with torch.no_grad():
        for batch in loader:
            moved = move_graph_batch(batch, device)
            output = model(
                moved["sequence_features"], moved["sequence_mask"], moved["node_features"],
                moved["node_mask"], moved["edge_index"], moved["edge_type"],
                return_attention=collect_attention,
            )
            sequence_logits.append(output["sequence_logits"].cpu())
            graph_logits.append(output["graph_logits"].cpu())
            final_logits.append(output["recognition_logits"].cpu())
            labels.append(moved["multi_labels"].cpu())
            if not collect_attention:
                continue
            for item in output["graph_attention"]:
                weights = item["node_attention"].float().cpu()
                slots = item["slot_weights"].float().cpu()
                if weights.shape[-1] == 0:
                    continue
                combined = torch.einsum("ls,lsv->lv", slots, weights)
                for label_id, values in enumerate(combined):
                    values = values[values > 0]
                    if values.numel() == 0:
                        continue
                    values = values / values.sum().clamp_min(1e-12)
                    entropy = float(-(values * values.clamp_min(1e-12).log()).sum().item())
                    count = int(values.numel())
                    denom = float(np.log(count)) if count > 1 else 1.0
                    summary = attention_summary[label_id]
                    summary["node_count"] += count
                    summary["entropy"] += entropy
                    summary["normalized_entropy"] += entropy / denom if count > 1 else 0.0
                    summary["effective_nodes"] += float(np.exp(entropy))
                    for top_k in (1, 5, 10):
                        summary[f"top_{top_k}_mass"] += float(values.topk(min(top_k, count)).values.sum().item())
                    attention_samples[label_id] += 1
    concentration = {}
    for label_id, summary in attention_summary.items():
        count = max(1, attention_samples[label_id])
        concentration[str(label_id)] = {
            "samples": attention_samples[label_id],
            "mean_nodes": summary["node_count"] / count,
            "mean_entropy": summary["entropy"] / count,
            "mean_normalized_entropy": summary["normalized_entropy"] / count,
            "mean_effective_nodes": summary["effective_nodes"] / count,
            "mean_top_1_mass": summary["top_1_mass"] / count,
            "mean_top_5_mass": summary["top_5_mass"] / count,
            "mean_top_10_mass": summary["top_10_mass"] / count,
        }
    return {
        "sequence_logits": torch.cat(sequence_logits).numpy(),
        "graph_logits": torch.cat(graph_logits).numpy(),
        "final_logits": torch.cat(final_logits).numpy(),
        "labels": torch.cat(labels).numpy(),
        "attention_concentration": concentration,
    }


def error_complementarity(sequence_logits, graph_logits, labels, sequence_thresholds, graph_thresholds, label_names):
    sequence_predictions = 1.0 / (1.0 + np.exp(-np.clip(sequence_logits, -40, 40))) >= np.asarray(sequence_thresholds)
    graph_predictions = 1.0 / (1.0 + np.exp(-np.clip(graph_logits, -40, 40))) >= np.asarray(graph_thresholds)
    targets = labels.astype(bool)
    sequence_correct = sequence_predictions == targets
    graph_correct = graph_predictions == targets
    aggregate = {
        "units": int(targets.size),
        "sequence_correct_graph_wrong": int((sequence_correct & ~graph_correct).sum()),
        "sequence_wrong_graph_correct": int((~sequence_correct & graph_correct).sum()),
        "both_wrong": int((~sequence_correct & ~graph_correct).sum()),
        "both_correct": int((sequence_correct & graph_correct).sum()),
    }
    per_label = {}
    for index, label_name in enumerate(label_names):
        seq = sequence_correct[:, index]
        graph = graph_correct[:, index]
        per_label[label_name] = {
            "sequence_correct_graph_wrong": int((seq & ~graph).sum()),
            "sequence_wrong_graph_correct": int((~seq & graph).sum()),
            "both_wrong": int((~seq & ~graph).sum()),
            "both_correct": int((seq & graph).sum()),
        }
    return {"aggregate_label_decisions": aggregate, "per_label": per_label}


def residual_summary(sequence_logits, graph_logits, gate, label_names):
    residual = graph_logits * gate[None, :]
    result = {}
    for index, label_name in enumerate(label_names):
        values = np.abs(residual[:, index])
        sequence = np.abs(sequence_logits[:, index])
        result[label_name] = {
            "gate_tanh_alpha": float(gate[index]),
            "mean_abs_residual_logit": float(values.mean()),
            "median_abs_residual_logit": float(np.median(values)),
            "p95_abs_residual_logit": float(np.percentile(values, 95)),
            "mean_abs_residual_to_sequence_ratio": float(values.mean() / max(1e-8, sequence.mean())),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; keep 0 for ragged graph-cache diagnostics on low-FD systems.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.variant)
    graph_dir = resolve(config["graph_cache_dir"])
    sequence_dir = resolve(config["sequence_feature_dir"])
    if (graph_dir / "test.pt").exists() or (sequence_dir / "test.pt").exists():
        raise RuntimeError("Refusing CSDG diagnostics after a test cache exists")
    checkpoint_path = resolve(config["checkpoint_dir"]) / "best_macro_f1.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = OpcodeGraphResidualMIL(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = OpcodeGraphSequenceDataset(graph_dir / "valid.pt", sequence_dir / "valid.pt", config["label_names"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        collate_fn=collate_opcode_graph,
    )

    full = predict(model, loader, device, collect_attention=True)
    labels = full["labels"]
    final_thresholds = checkpoint["metrics"]["thresholds"]
    sequence_metrics = thresholds_and_metrics(full["sequence_logits"], labels, config["thresholds"])
    graph_metrics = thresholds_and_metrics(full["graph_logits"], labels, config["thresholds"])
    final_metrics = metrics_at_thresholds(full["final_logits"], labels, final_thresholds)
    gate = model.alpha.detach().cpu().tanh().numpy()

    original_mask = set(model.graph_edge_type_mask)
    edge_name_by_id = {value: name for name, value in EDGE_TYPES.items()}
    edge_removal = {}
    for edge_type in sorted(original_mask):
        model.graph_edge_type_mask = original_mask - {edge_type}
        removed = predict(model, loader, device)
        removed_metrics = metrics_at_thresholds(removed["final_logits"], labels, final_thresholds)
        edge_removal[edge_name_by_id[edge_type]] = {
            "edge_type_id": edge_type,
            "macro_f1": removed_metrics["macro_f1"],
            "micro_f1": removed_metrics["micro_f1"],
            "macro_f1_drop": final_metrics["macro_f1"] - removed_metrics["macro_f1"],
            "micro_f1_drop": final_metrics["micro_f1"] - removed_metrics["micro_f1"],
        }
    model.graph_edge_type_mask = original_mask

    report = {
        "route": config["route_name"],
        "variant": args.variant,
        "split": "valid",
        "test_labels_read": False,
        "checkpoint": str(checkpoint_path),
        "fixed_final_metrics": final_metrics,
        "sequence_independent_metrics": sequence_metrics,
        "graph_independent_metrics": graph_metrics,
        "error_complementarity": error_complementarity(
            full["sequence_logits"], full["graph_logits"], labels,
            sequence_metrics["thresholds"], graph_metrics["thresholds"], config["label_names"],
        ),
        "residual_summary": residual_summary(
            full["sequence_logits"], full["graph_logits"], gate, config["label_names"],
        ),
        "edge_type_removal": edge_removal,
        "attention_concentration": {
            config["label_names"][int(label_id)]: value
            for label_id, value in full["attention_concentration"].items()
        },
        "pooling_comparison_status": {
            "mean_pooling": "available in the current graph cache",
            "block_local_attention": "requires a separately trained token-level pooling encoder and a new train/valid cache; not inferred post hoc from block means",
        },
    }
    output = resolve(args.output) if args.output else resolve(config["result_dir"]) / "valid_graph_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
