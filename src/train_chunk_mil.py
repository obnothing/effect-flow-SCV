import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from chunk_feature_dataset import build_chunk_feature_datasets
from evm_chunk_mil_model import EVMChunkMILClassifier
from metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train EVM-BERT chunk feature MIL head.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def compute_pos_weight_from_feature_cache(config):
    path = Path(config["feature_dir"]) / "train.pt"
    payload = torch.load(path, map_location="cpu")
    labels = payload["multi_labels"].float()
    total = labels.shape[0]
    positives = labels.sum(dim=0)
    negatives = total - positives
    weights = []
    rows = []
    max_weight = float(config.get("max_pos_weight", 5.0))
    for idx in range(labels.shape[1]):
        pos = float(positives[idx].item())
        neg = float(negatives[idx].item())
        if pos <= 0:
            weight = 1.0
            raw_ratio = None
            sqrt_ratio = None
        else:
            raw_ratio = neg / pos
            sqrt_ratio = math.sqrt(raw_ratio)
            if config.get("pos_weight_mode", "sqrt_ratio") == "ratio":
                value = raw_ratio
            else:
                value = sqrt_ratio
            weight = min(max(value, 1.0), max_weight)
        weights.append(weight)
        rows.append(
            {
                "label_id": idx,
                "label_name": config.get("label_names", [f"label_{idx}"] * 10)[idx],
                "positive_count": int(pos),
                "negative_count": int(neg),
                "raw_ratio": raw_ratio,
                "sqrt_ratio": sqrt_ratio,
                "final_pos_weight": weight,
            }
        )
    return torch.tensor(weights, dtype=torch.float32), rows


def best_threshold_from_scan(threshold_scan, metric_name):
    if not threshold_scan:
        return None, None
    key, row = max(
        threshold_scan.items(),
        key=lambda item: (item[1].get(metric_name, 0.0), float(item[0])),
    )
    return float(key), row.get(metric_name)


def collate_batch(batch):
    return {
        "id": [item["id"] for item in batch],
        "chunk_features": torch.stack([item["chunk_features"] for item in batch]),
        "chunk_mask": torch.stack([item["chunk_mask"] for item in batch]),
        "binary_label": torch.stack([item["binary_label"] for item in batch]),
        "multi_labels": torch.stack([item["multi_labels"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def make_loader(dataset, config, shuffle):
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def move_batch(batch, device):
    return {
        "chunk_features": batch["chunk_features"].to(device),
        "chunk_mask": batch["chunk_mask"].to(device),
        "binary_label": batch["binary_label"].to(device),
        "multi_labels": batch["multi_labels"].to(device),
    }


def train_one_epoch(model, loader, optimizer, config, device):
    model.train()
    total_loss = 0.0
    steps = 0
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        model_inputs = move_batch(batch, device)
        outputs = model(**model_inputs)
        loss = outputs["loss"] / grad_accum
        loss.backward()
        if step % grad_accum == 0:
            if config.get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["gradient_clip_norm"]),
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(outputs["loss"].detach().cpu().item())
        steps += 1
    if steps % grad_accum != 0:
        if config.get("gradient_clip_norm"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["gradient_clip_norm"]),
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, config, device):
    model.eval()
    losses = []
    detection_logits = []
    recognition_logits = []
    binary_labels = []
    multi_labels = []
    for batch in tqdm(loader, desc="valid", leave=False):
        model_inputs = move_batch(batch, device)
        outputs = model(**model_inputs)
        losses.append(float(outputs["loss"].detach().cpu().item()))
        detection_logits.append(outputs["detection_logits"].detach().cpu())
        recognition_logits.append(outputs["recognition_logits"].detach().cpu())
        binary_labels.append(model_inputs["binary_label"].detach().cpu())
        multi_labels.append(model_inputs["multi_labels"].detach().cpu())
    detection_logits = torch.cat(detection_logits).numpy()
    recognition_logits = torch.cat(recognition_logits).numpy()
    binary_labels = torch.cat(binary_labels).numpy()
    multi_labels = torch.cat(multi_labels).numpy()
    metrics = compute_metrics(
        detection_logits,
        binary_labels,
        recognition_logits,
        multi_labels,
        threshold=float(config.get("threshold", 0.5)),
        scan_thresholds=config.get("thresholds", [0.5]),
    )
    return float(np.mean(losses)) if losses else 0.0, metrics


def checkpoint_payload(model, optimizer, epoch, config, valid_loss, metrics):
    micro_threshold, micro_f1 = best_threshold_from_scan(
        metrics.get("threshold_scan", {}),
        "micro_f1",
    )
    macro_threshold, macro_f1 = best_threshold_from_scan(
        metrics.get("threshold_scan", {}),
        "macro_f1",
    )
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "valid_loss": valid_loss,
        "threshold_scan": metrics.get("threshold_scan", {}),
        "best_threshold_by_micro_f1": micro_threshold,
        "best_threshold_by_macro_f1": macro_threshold,
        "best_micro_f1": micro_f1,
        "best_macro_f1": macro_f1,
        "recognition_micro_f1": metrics.get("recognition_micro_f1"),
        "recognition_macro_f1": metrics.get("recognition_macro_f1"),
        "encoder_frozen": True,
        "feature_dir": config["feature_dir"],
        "is_transductive_pretraining": bool(config.get("is_transductive_pretraining", True)),
        "metrics": metrics,
    }


def write_epoch_history(path_json, path_txt, history):
    ensure_dir(Path(path_json).parent)
    Path(path_json).write_text(json.dumps(history, indent=2, default=json_default), encoding="utf-8")
    lines = ["EVM chunk MIL epoch history", ""]
    for row in history:
        lines.append(
            "epoch {epoch}/{epochs} train_loss={train_loss:.6f} "
            "valid_loss={valid_loss:.6f} micro_f1={micro:.6f} macro_f1={macro:.6f} "
            "predicted_positive_total={pred}".format(
                epoch=row["epoch"],
                epochs=row["epochs"],
                train_loss=row["train_loss"],
                valid_loss=row["valid_loss"],
                micro=row["recognition_micro_f1"],
                macro=row["recognition_macro_f1"],
                pred=row["predicted_positive_total"],
            )
        )
        lines.append(f"per_label_f1: {[round(v, 6) for v in row['per_label_f1']]}")
        for threshold, metrics in row["threshold_scan"].items():
            lines.append(
                f"threshold={threshold}: micro_f1={metrics['micro_f1']:.6f} "
                f"macro_f1={metrics['macro_f1']:.6f} "
                f"predicted_positive_total={metrics['predicted_positive_total']}"
            )
        lines.append("")
    Path(path_txt).write_text("\n".join(lines), encoding="utf-8")


def write_summary(path_json, path_txt, summary):
    ensure_dir(Path(path_json).parent)
    Path(path_json).write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    lines = ["EVM chunk MIL training report", ""]
    for key, value in summary.items():
        if key == "per_label_table":
            lines.append("per_label_table:")
            for row in value:
                lines.append(
                    f"- {row['label_name']}: precision={row['precision']:.6f} "
                    f"recall={row['recall']:.6f} f1={row['f1']:.6f} "
                    f"support={row['support']}"
                )
        else:
            lines.append(f"{key}: {value}")
    Path(path_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")


def label_table(config, metrics):
    names = config.get("label_names", [f"label_{idx}" for idx in range(10)])
    return [
        {
            "label_id": idx,
            "label_name": names[idx],
            "precision": metrics["per_label_precision"][idx],
            "recall": metrics["per_label_recall"][idx],
            "f1": metrics["per_label_f1"][idx],
            "support": metrics["per_label_support"][idx],
            "predicted_positive_count": metrics["per_label_predicted_positive_count"][idx],
            "true_positive_count": metrics["per_label_true_positive_count"][idx],
        }
        for idx in range(len(names))
    ]


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    ensure_dir(config["checkpoint_dir"])
    ensure_dir(config["result_dir"])
    ensure_dir(config["report_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = build_chunk_feature_datasets(config)
    train_loader = make_loader(datasets["train"], config, shuffle=True)
    valid_loader = make_loader(datasets["valid"], config, shuffle=False)
    model = EVMChunkMILClassifier(config).to(device)
    pos_weight_rows = None
    if config.get("use_pos_weight", False):
        pos_weight, pos_weight_rows = compute_pos_weight_from_feature_cache(config)
        model.set_recognition_pos_weight(pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    checkpoint_dir = Path(config["checkpoint_dir"])
    result_dir = Path(config["result_dir"])
    best_loss = float("inf")
    best_micro = -1.0
    best_macro = -1.0
    best_loss_epoch = None
    best_micro_epoch = None
    best_macro_epoch = None
    patience = int(config.get("early_stopping_patience", 5))
    patience_counter = 0
    history = []
    best_metrics = None
    best_valid_loss_for_macro = None

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
        valid_loss, metrics = evaluate(model, valid_loader, config, device)
        micro_threshold, best_epoch_micro = best_threshold_from_scan(
            metrics.get("threshold_scan", {}),
            "micro_f1",
        )
        macro_threshold, best_epoch_macro = best_threshold_from_scan(
            metrics.get("threshold_scan", {}),
            "macro_f1",
        )
        record = {
            "epoch": epoch,
            "epochs": int(config["epochs"]),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "recognition_micro_f1": metrics["recognition_micro_f1"],
            "recognition_macro_f1": metrics["recognition_macro_f1"],
            "best_threshold_by_micro_f1": micro_threshold,
            "best_micro_f1": best_epoch_micro,
            "best_threshold_by_macro_f1": macro_threshold,
            "best_macro_f1": best_epoch_macro,
            "detection_f1": metrics["detection_f1"],
            "predicted_positive_total": metrics["predicted_positive_total"],
            "per_label_f1": metrics["per_label_f1"],
            "per_label_precision": metrics["per_label_precision"],
            "per_label_recall": metrics["per_label_recall"],
            "per_label_support": metrics["per_label_support"],
            "threshold_scan": metrics["threshold_scan"],
        }
        history.append(record)
        payload = checkpoint_payload(model, optimizer, epoch, config, valid_loss, metrics)
        torch.save(payload, checkpoint_dir / "last.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_loss_epoch = epoch
            torch.save(payload, checkpoint_dir / "best_loss.pt")
        if best_epoch_micro is not None and best_epoch_micro > best_micro:
            best_micro = best_epoch_micro
            best_micro_epoch = epoch
            torch.save(payload, checkpoint_dir / "best_micro_f1.pt")
        if best_epoch_macro is not None and best_epoch_macro > best_macro:
            best_macro = best_epoch_macro
            best_macro_epoch = epoch
            best_metrics = metrics
            best_valid_loss_for_macro = valid_loss
            torch.save(payload, checkpoint_dir / "best_macro_f1.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"epoch {epoch}/{config['epochs']} train_loss={train_loss:.6f} "
            f"valid_loss={valid_loss:.6f} micro_f1={best_epoch_micro:.6f} "
            f"macro_f1={best_epoch_macro:.6f} pred={metrics['predicted_positive_total']}"
        )
        if patience_counter >= patience:
            print(f"[INFO] early stopping at epoch {epoch}")
            break

    write_epoch_history(
        result_dir / "epoch_history.json",
        result_dir / "epoch_history.txt",
        history,
    )
    write_epoch_history(
        Path(config["report_dir"]) / f"{config['experiment_name']}_epoch_history.json",
        Path(config["report_dir"]) / f"{config['experiment_name']}_epoch_history.txt",
        history,
    )
    best_metrics = best_metrics or metrics
    baseline = config.get("baseline_comparison", {})
    summary = {
        "experiment_name": config["experiment_name"],
        "model_type": "evm_chunk_mil",
        "encoder_frozen": True,
        "feature_dir": config["feature_dir"],
        "feature_pooling": config.get("feature_pooling"),
        "max_chunks": config["max_chunks"],
        "feature_dim": config["feature_dim"],
        "hidden_dim": config["hidden_dim"],
        "recognition_aggregation": config["recognition_aggregation"],
        "top_k": int(config.get("top_k", 2)),
        "use_pos_weight": config.get("use_pos_weight", False),
        "pos_weight_rows": pos_weight_rows,
        "best_loss_epoch": best_loss_epoch,
        "best_loss_value": best_loss,
        "best_micro_f1_epoch": best_micro_epoch,
        "best_micro_f1_value": best_micro,
        "best_macro_f1_epoch": best_macro_epoch,
        "best_macro_f1_value": best_macro,
        "best_macro_f1_threshold": best_threshold_from_scan(
            best_metrics.get("threshold_scan", {}),
            "macro_f1",
        )[0],
        "valid_loss_at_best_macro": best_valid_loss_for_macro,
        "valid_micro_f1": best_metrics["recognition_micro_f1"],
        "valid_macro_f1": best_metrics["recognition_macro_f1"],
        "detection_f1": best_metrics["detection_f1"],
        "predicted_positive_total": best_metrics["predicted_positive_total"],
        "baseline_comparison": baseline,
        "improves_evm_bert_first512_macro_f1": (
            best_macro > baseline.get("evm_bert_first512_macro_f1", float("inf"))
        ),
        "is_transductive_pretraining": bool(config.get("is_transductive_pretraining", True)),
        "train_samples": len(datasets["train"]),
        "valid_samples": len(datasets["valid"]),
        "per_label_table": label_table(config, best_metrics),
    }
    write_summary(
        result_dir / "checkpoint_summary.json",
        result_dir / "checkpoint_summary.txt",
        summary,
    )
    write_summary(
        Path(config["report_dir"]) / f"{config['experiment_name']}_report.json",
        Path(config["report_dir"]) / f"{config['experiment_name']}_report.txt",
        summary,
    )


if __name__ == "__main__":
    main()
