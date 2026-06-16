import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from chunk_feature_dataset import ChunkFeatureDataset
from evm_chunk_mil_model import EVMChunkMILClassifier
from metrics import compute_metrics, sigmoid
from train_chunk_mil import collate_batch, compute_pos_weight_from_feature_cache, label_table


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EVM chunk MIL classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--threshold", default="auto")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def feature_path(config, split):
    return Path(config["feature_dir"]) / f"{split}.pt"


def make_loader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def load_model(config, checkpoint, device):
    model = EVMChunkMILClassifier(config).to(device)
    if config.get("use_pos_weight", False):
        pos_weight, _ = compute_pos_weight_from_feature_cache(config)
        model.set_recognition_pos_weight(pos_weight.to(device))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


@torch.no_grad()
def collect_predictions(model, loader, device):
    losses = []
    ids = []
    metadata = []
    detection_logits = []
    recognition_logits = []
    chunk_logits = []
    chunk_masks = []
    binary_labels = []
    multi_labels = []
    for batch in tqdm(loader, desc="eval", leave=False):
        inputs = {
            "chunk_features": batch["chunk_features"].to(device),
            "chunk_mask": batch["chunk_mask"].to(device),
            "binary_label": batch["binary_label"].to(device),
            "multi_labels": batch["multi_labels"].to(device),
        }
        outputs = model(**inputs)
        losses.append(float(outputs["loss"].detach().cpu().item()))
        ids.extend(batch["id"])
        metadata.extend(batch["metadata"])
        detection_logits.append(outputs["detection_logits"].detach().cpu())
        recognition_logits.append(outputs["recognition_logits"].detach().cpu())
        chunk_logits.append(outputs["chunk_logits"].detach().cpu())
        chunk_masks.append(inputs["chunk_mask"].detach().cpu())
        binary_labels.append(inputs["binary_label"].detach().cpu())
        multi_labels.append(inputs["multi_labels"].detach().cpu())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "ids": ids,
        "metadata": metadata,
        "detection_logits": torch.cat(detection_logits).numpy(),
        "recognition_logits": torch.cat(recognition_logits).numpy(),
        "chunk_logits": torch.cat(chunk_logits),
        "chunk_mask": torch.cat(chunk_masks),
        "binary_labels": torch.cat(binary_labels).numpy(),
        "multi_labels": torch.cat(multi_labels).numpy(),
    }


def write_top_chunks(path, config, predictions, threshold):
    label_names = config.get("label_names", [f"label_{idx}" for idx in range(10)])
    probs = sigmoid(predictions["recognition_logits"])
    pred_labels = (probs >= threshold).astype(int)
    true_labels = predictions["multi_labels"].astype(int)
    chunk_logits = predictions["chunk_logits"]
    chunk_mask = predictions["chunk_mask"].bool()
    top_k = int(config.get("top_k", 2))
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row_idx, contract_id in enumerate(predictions["ids"]):
            label_ids = sorted(
                set(np.where(pred_labels[row_idx] == 1)[0].tolist())
                | set(np.where(true_labels[row_idx] == 1)[0].tolist())
            )
            real_count = int(chunk_mask[row_idx].sum().item())
            if not label_ids:
                continue
            for label_id in label_ids:
                scores = chunk_logits[row_idx, :, label_id].clone()
                scores[~chunk_mask[row_idx]] = -1e9
                k = min(top_k, real_count)
                top_scores, top_indices = torch.topk(scores, k=k)
                record = {
                    "id": contract_id,
                    "predicted_labels": [
                        label_names[idx] for idx in np.where(pred_labels[row_idx] == 1)[0]
                    ],
                    "true_labels": [
                        label_names[idx] for idx in np.where(true_labels[row_idx] == 1)[0]
                    ],
                    "label_name": label_names[label_id],
                    "label_id": int(label_id),
                    "contract_label_probability": float(probs[row_idx, label_id]),
                    "top_chunk_indices": [int(v) for v in top_indices.tolist()],
                    "top_chunk_scores": [float(v) for v in top_scores.tolist()],
                    "num_chunks_kept": real_count,
                    "metadata": predictions["metadata"][row_idx],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(prefix, report):
    ensure_dir(prefix.parent)
    json_path = prefix.with_suffix(".json")
    txt_path = prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    lines = ["EVM chunk MIL strict evaluation report", ""]
    for key, value in report.items():
        if key == "per_label":
            lines.append("Per-label metrics:")
            for row in value:
                lines.append(
                    f"{row['label_id']} | {row['label_name']} | "
                    f"precision={row['precision']:.6f} recall={row['recall']:.6f} "
                    f"f1={row['f1']:.6f} support={row['support']} "
                    f"predicted={row['predicted_positive_count']}"
                )
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {json_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChunkFeatureDataset(feature_path(config, args.split), seed=config.get("seed", 42))
    loader = make_loader(dataset, config)
    model, checkpoint = load_model(config, args.checkpoint, device)
    if args.threshold == "auto":
        threshold = checkpoint.get("best_threshold_by_macro_f1") or config.get("threshold", 0.5)
        threshold_source = "checkpoint_best_threshold_by_macro_f1"
    else:
        threshold = float(args.threshold)
        threshold_source = "cli"
    threshold = float(threshold)
    predictions = collect_predictions(model, loader, device)
    metrics = compute_metrics(
        predictions["detection_logits"],
        predictions["binary_labels"],
        predictions["recognition_logits"],
        predictions["multi_labels"],
        threshold=threshold,
        scan_thresholds=config.get("thresholds", [threshold]),
    )
    result_dir = Path(config["result_dir"])
    top_chunks_path = result_dir / f"top_chunks_{args.split}.jsonl"
    write_top_chunks(top_chunks_path, config, predictions, threshold)
    baseline = config.get("baseline_comparison", {})
    report = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "model_type": "evm_chunk_mil",
        "encoder_frozen": True,
        "feature_dir": config["feature_dir"],
        "evaluated_samples": len(dataset),
        "loss": predictions["loss"],
        "detection_precision": metrics["detection_precision"],
        "detection_recall": metrics["detection_recall"],
        "detection_f1": metrics["detection_f1"],
        "recognition_micro_precision": metrics["recognition_micro_precision"],
        "recognition_micro_recall": metrics["recognition_micro_recall"],
        "recognition_micro_f1": metrics["recognition_micro_f1"],
        "recognition_macro_precision": metrics["recognition_macro_precision"],
        "recognition_macro_recall": metrics["recognition_macro_recall"],
        "recognition_macro_f1": metrics["recognition_macro_f1"],
        "predicted_positive_total": metrics["predicted_positive_total"],
        "per_label_predicted_positive_count": metrics["per_label_predicted_positive_count"],
        "per_label_true_positive_count": metrics["per_label_true_positive_count"],
        "per_label_mean_pred_prob": metrics["per_label_mean_pred_prob"],
        "per_label": label_table(config, metrics),
        "threshold_scan": metrics.get("threshold_scan", {}),
        "top_chunks_path": str(top_chunks_path),
        "baseline_comparison": baseline,
        "macro_f1_change_over_evm_bert_first512": (
            metrics["recognition_macro_f1"] - baseline.get("evm_bert_first512_macro_f1", 0.0)
        ),
        "micro_f1_change_over_evm_bert_first512": (
            metrics["recognition_micro_f1"] - baseline.get("evm_bert_first512_micro_f1", 0.0)
        ),
        "detection_f1_change_over_evm_bert_first512": (
            metrics["detection_f1"] - baseline.get("evm_bert_first512_detection_f1", 0.0)
        ),
        "is_transductive_pretraining": bool(config.get("is_transductive_pretraining", True)),
    }
    prefix = result_dir / f"{args.split}_best_macro_metrics"
    write_report(prefix, report)


if __name__ == "__main__":
    main()

