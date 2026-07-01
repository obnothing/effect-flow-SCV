import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from chunk_feature_dataset import build_chunk_feature_datasets
from evm_chunk_mil_model import (
    EVEFMVDV2SideEvidenceMIL,
    EVMChunkMILClassifier,
    EffectFlowGuidedChunkMIL,
)
from metrics import compute_metrics


EFFECT_FLOW_MODEL_TYPES = {
    "effect_flow_guided_chunk_mil",
    "evef_mvd_v2_side_evidence_mil",
}


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
                "label_name": config.get(
                    "label_names",
                    [f"label_{label_idx}" for label_idx in range(labels.shape[1])],
                )[idx],
                "positive_count": int(pos),
                "negative_count": int(neg),
                "raw_ratio": raw_ratio,
                "sqrt_ratio": sqrt_ratio,
                "final_pos_weight": weight,
            }
        )
    return torch.tensor(weights, dtype=torch.float32), rows


def build_label_balanced_sampler(dataset, config):
    labels = dataset.multi_labels[dataset.indices].float()
    positives = labels.sum(dim=0)
    positive_nonzero = positives[positives > 0]
    if positive_nonzero.numel() == 0:
        raise ValueError("label-balanced sampler requires at least one positive label")

    max_pos = positive_nonzero.max()
    label_weights = torch.ones_like(positives)
    valid_labels = positives > 0
    label_weights[valid_labels] = torch.sqrt(max_pos / positives[valid_labels])
    label_weights = label_weights.clamp(max=float(config.get("sampler_max_sample_weight", 10.0)))

    positive_mask = labels > 0
    weighted_labels = positive_mask.float() * label_weights.unsqueeze(0)
    sample_weights = weighted_labels.max(dim=1).values
    sample_weights = torch.where(
        positive_mask.any(dim=1),
        sample_weights,
        torch.ones_like(sample_weights),
    )
    sample_weights = sample_weights.clamp(
        min=1e-6,
        max=float(config.get("sampler_max_sample_weight", 10.0)),
    )
    num_samples = int(config.get("sampler_num_samples") or len(dataset))
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 42)))
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    label_names = config.get(
        "label_names",
        [f"label_{idx}" for idx in range(labels.shape[1])],
    )
    stats = {
        "train_sampler": "label_balanced",
        "sampler_num_samples": num_samples,
        "sampler_max_sample_weight": float(config.get("sampler_max_sample_weight", 10.0)),
        "sampler_label_positive_counts": [
            int(value) for value in positives.tolist()
        ],
        "sampler_label_weights": [
            {
                "label_id": idx,
                "label_name": label_names[idx],
                "positive_count": int(positives[idx].item()),
                "weight": float(label_weights[idx].item()),
            }
            for idx in range(labels.shape[1])
        ],
        "sampler_sample_weight_min": float(sample_weights.min().item()),
        "sampler_sample_weight_mean": float(sample_weights.mean().item()),
        "sampler_sample_weight_max": float(sample_weights.max().item()),
    }
    return sampler, stats


def best_threshold_from_scan(threshold_scan, metric_name):
    if not threshold_scan:
        return None, None
    key, row = max(
        threshold_scan.items(),
        key=lambda item: (item[1].get(metric_name, 0.0), float(item[0])),
    )
    return float(key), row.get(metric_name)


def collate_batch(batch):
    collated = {
        "id": [item["id"] for item in batch],
        "chunk_features": torch.stack([item["chunk_features"] for item in batch]),
        "chunk_mask": torch.stack([item["chunk_mask"] for item in batch]),
        "binary_label": torch.stack([item["binary_label"] for item in batch]),
        "multi_labels": torch.stack([item["multi_labels"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }
    if "efpp_probs" in batch[0]:
        collated["efpp_probs"] = torch.stack([item["efpp_probs"] for item in batch])
        collated["etp_distribution"] = torch.stack(
            [item["etp_distribution"] for item in batch]
        )
        collated["relation_distribution"] = torch.stack(
            [item["relation_distribution"] for item in batch]
        )
        collated["vulnerability_evidence_probs"] = torch.stack(
            [item["vulnerability_evidence_probs"] for item in batch]
        )
        collated["template_match_scores"] = torch.stack(
            [item["template_match_scores"] for item in batch]
        )
        collated["chunk_vulnerability_evidence"] = torch.stack(
            [item["chunk_vulnerability_evidence"] for item in batch]
        )
        collated["vulnerability_template_matches"] = torch.stack(
            [item["vulnerability_template_matches"] for item in batch]
        )
        collated["active_vulnerability_label_mask"] = torch.stack(
            [item["active_vulnerability_label_mask"] for item in batch]
        )
    return collated


def make_loader(dataset, config, shuffle, sampler=None):
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def move_batch(batch, device):
    moved = {
        "chunk_features": batch["chunk_features"].to(device),
        "chunk_mask": batch["chunk_mask"].to(device),
        "binary_label": batch["binary_label"].to(device),
        "multi_labels": batch["multi_labels"].to(device),
    }
    if "efpp_probs" in batch:
        moved["efpp_probs"] = batch["efpp_probs"].to(device)
        moved["etp_distribution"] = batch["etp_distribution"].to(device)
        moved["relation_distribution"] = batch["relation_distribution"].to(device)
        moved["vulnerability_evidence_probs"] = batch["vulnerability_evidence_probs"].to(device)
        moved["template_match_scores"] = batch["template_match_scores"].to(device)
        moved["chunk_vulnerability_evidence"] = batch["chunk_vulnerability_evidence"].to(device)
        moved["vulnerability_template_matches"] = batch["vulnerability_template_matches"].to(device)
        moved["active_vulnerability_label_mask"] = batch["active_vulnerability_label_mask"].to(device)
    return moved


def build_model(config):
    model_type = config.get("model_type", "evm_chunk_mil")
    if model_type == "effect_flow_guided_chunk_mil":
        return EffectFlowGuidedChunkMIL(config)
    if model_type == "evef_mvd_v2_side_evidence_mil":
        return EVEFMVDV2SideEvidenceMIL(config)
    if model_type == "evm_chunk_mil":
        return EVMChunkMILClassifier(config)
    raise ValueError(f"Unsupported chunk MIL model_type: {model_type}")


def set_model_epoch(model, epoch):
    target = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(target, "set_training_epoch"):
        target.set_training_epoch(epoch)


def cuda_peak_memory_mb(device):
    if device.type != "cuda":
        return 0.0, 0.0
    torch.cuda.synchronize(device)
    return (
        torch.cuda.max_memory_allocated(device) / (1024**2),
        torch.cuda.max_memory_reserved(device) / (1024**2),
    )


def train_one_epoch(model, loader, optimizer, config, device):
    model.train()
    total_loss = 0.0
    steps = 0
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        model_inputs = move_batch(batch, device)
        outputs = model(**model_inputs)
        batch_loss = outputs["loss"].mean()
        loss = batch_loss / grad_accum
        loss.backward()
        if step % grad_accum == 0:
            if config.get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["gradient_clip_norm"]),
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(batch_loss.detach().cpu().item())
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
    evaluation_model = model.module if isinstance(model, nn.DataParallel) else model
    evaluation_model.eval()
    losses = []
    detection_logits = []
    recognition_logits = []
    binary_labels = []
    multi_labels = []
    for batch in tqdm(loader, desc="valid", leave=False):
        model_inputs = move_batch(batch, device)
        outputs = evaluation_model(**model_inputs)
        losses.append(float(outputs["loss"].mean().detach().cpu().item()))
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
    unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
    return {
        "epoch": epoch,
        "model_state_dict": unwrapped_model.state_dict(),
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
        "semantic_feature_dir": config.get("semantic_feature_dir"),
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
            "predicted_positive_total={pred} max_memory_allocated_mb={alloc:.2f} "
            "max_memory_reserved_mb={reserved:.2f}".format(
                epoch=row["epoch"],
                epochs=row["epochs"],
                train_loss=row["train_loss"],
                valid_loss=row["valid_loss"],
                micro=row["recognition_micro_f1"],
                macro=row["recognition_macro_f1"],
                pred=row["predicted_positive_total"],
                alloc=float(row.get("max_memory_allocated_mb", 0.0)),
                reserved=float(row.get("max_memory_reserved_mb", 0.0)),
            )
        )
        lines.append(
            f"per_label_precision: {[round(v, 6) for v in row['per_label_precision']]}"
        )
        lines.append(
            f"per_label_recall: {[round(v, 6) for v in row['per_label_recall']]}"
        )
        lines.append(
            f"per_label_accuracy: {[round(v, 6) for v in row['per_label_accuracy']]}"
        )
        lines.append(
            f"per_label_number_vulnerability: {row['per_label_number_vulnerability']}"
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
                    f"recall={row['recall']:.6f} accuracy={row['accuracy']:.6f} "
                    f"f1={row['f1']:.6f} "
                    f"support={row['support']}"
                )
        else:
            lines.append(f"{key}: {value}")
    Path(path_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")


def label_table(config, metrics):
    names = config.get(
        "label_names",
        [f"label_{idx}" for idx in range(int(config.get("num_labels", 10)))],
    )
    return [
        {
            "label_id": idx,
            "label_name": names[idx],
            "precision": metrics["per_label_precision"][idx],
            "recall": metrics["per_label_recall"][idx],
            "accuracy": metrics["per_label_accuracy"][idx],
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
    train_sampler_stats = {
        "train_sampler": config.get("train_sampler", "shuffle"),
    }
    train_sampler = None
    if config.get("train_sampler") == "label_balanced":
        train_sampler, train_sampler_stats = build_label_balanced_sampler(
            datasets["train"],
            config,
        )
    elif config.get("train_sampler") not in (None, "", "shuffle"):
        raise ValueError(f"Unsupported train_sampler: {config.get('train_sampler')}")
    train_loader = make_loader(
        datasets["train"],
        config,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    valid_loader = make_loader(datasets["valid"], config, shuffle=False)
    model = build_model(config).to(device)
    if config.get("recognition_loss_type") == "asl" and config.get("use_pos_weight", False):
        raise ValueError("ASL experiments must set use_pos_weight=false")
    pos_weight_rows = None
    if config.get("use_pos_weight", False):
        pos_weight, pos_weight_rows = compute_pos_weight_from_feature_cache(config)
        model.set_recognition_pos_weight(pos_weight.to(device))
    use_data_parallel = bool(config.get("use_data_parallel", False))
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if use_data_parallel:
        requested_devices = [
            int(device_id)
            for device_id in config.get(
                "data_parallel_device_ids",
                list(range(available_gpus)),
            )
        ]
        if len(requested_devices) < 2 or available_gpus < 2:
            raise RuntimeError(
                "use_data_parallel=true requires at least two visible CUDA devices"
            )
        if max(requested_devices) >= available_gpus:
            raise ValueError(
                f"data_parallel_device_ids={requested_devices} but only "
                f"{available_gpus} CUDA devices are visible"
            )
        model = nn.DataParallel(model, device_ids=requested_devices)
        print(
            f"[INFO] DataParallel enabled: devices={requested_devices} "
            f"total_batch_size={config['batch_size']} "
            f"approx_batch_per_gpu={int(config['batch_size']) // len(requested_devices)}"
        )
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
        set_model_epoch(model, epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
        set_model_epoch(model, epoch)
        valid_loss, metrics = evaluate(model, valid_loader, config, device)
        max_memory_allocated_mb, max_memory_reserved_mb = cuda_peak_memory_mb(device)
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
            "per_label_accuracy": metrics["per_label_accuracy"],
            "per_label_support": metrics["per_label_support"],
            "per_label_number_vulnerability": metrics["per_label_support"],
            "threshold_scan": metrics["threshold_scan"],
            "max_memory_allocated_mb": max_memory_allocated_mb,
            "max_memory_reserved_mb": max_memory_reserved_mb,
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
            f"macro_f1={best_epoch_macro:.6f} pred={metrics['predicted_positive_total']} "
            f"max_memory_allocated_mb={max_memory_allocated_mb:.2f} "
            f"max_memory_reserved_mb={max_memory_reserved_mb:.2f}"
        )
        print(
            f"per_label_precision: "
            f"{[round(v, 6) for v in metrics['per_label_precision']]}"
        )
        print(
            f"per_label_recall: "
            f"{[round(v, 6) for v in metrics['per_label_recall']]}"
        )
        print(
            f"per_label_accuracy: "
            f"{[round(v, 6) for v in metrics['per_label_accuracy']]}"
        )
        print(f"per_label_number_vulnerability: {metrics['per_label_support']}")
        print(f"per_label_f1: {[round(v, 6) for v in metrics['per_label_f1']]}")
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
    max_allocated_mb = max(
        [float(row.get("max_memory_allocated_mb", 0.0)) for row in history] or [0.0]
    )
    max_reserved_mb = max(
        [float(row.get("max_memory_reserved_mb", 0.0)) for row in history] or [0.0]
    )
    batch_size = int(config["batch_size"])
    max_chunks = int(config["max_chunks"])
    summary = {
        "experiment_name": config["experiment_name"],
        "model_type": config.get("model_type", "evm_chunk_mil"),
        "ablation_dataset": config.get("ablation_dataset"),
        "ablation_variant": config.get("ablation_variant"),
        "side_evidence_enabled": bool(config.get("side_evidence_enabled", False)),
        "coefficient_scale": config.get("coefficient_scale"),
        "encoder_frozen": True,
        "feature_dir": config["feature_dir"],
        "semantic_feature_dir": config.get("semantic_feature_dir"),
        "use_effect_flow_semantics": config.get("model_type") in EFFECT_FLOW_MODEL_TYPES,
        "effect_flow_semantic_mode": config.get("effect_flow_semantic_mode"),
        "efpp_dim": int(config.get("efpp_dim", 0)),
        "etp_dim": int(config.get("etp_dim", 0)),
        "relation_dim": int(config.get("relation_dim", 0)),
        "semantic_projection_dim": int(config.get("semantic_projection_dim", 0)),
        "semantic_fusion": config.get("semantic_fusion"),
        "semantic_dropout": float(config.get("semantic_dropout", 0.0)),
        "lambda_gate": float(config.get("lambda_gate", 0.0)),
        "risk_evidence_weight": config.get("risk_evidence_weight"),
        "missing_check_evidence_weight": config.get("missing_check_evidence_weight"),
        "protective_evidence_weight": config.get("protective_evidence_weight"),
        "effect_type_evidence_weight": config.get("effect_type_evidence_weight"),
        "relation_evidence_weight": config.get("relation_evidence_weight"),
        "behavior_weight_path": config.get("behavior_weight_path"),
        "use_weighted_behavior_scoring": bool(config.get("behavior_weight_path")),
        "beta_reliable_init": config.get("beta_reliable_init"),
        "gamma_reliable_init": config.get("gamma_reliable_init"),
        "warmup_epochs_neural_only": int(config.get("warmup_epochs_neural_only", 0)),
        "enable_reliable_semantic_epoch": int(
            config.get("enable_reliable_semantic_epoch", 0)
        ),
        "use_weak_semantic_features": bool(
            config.get("use_weak_semantic_features", False)
        ),
        "feature_pooling": config.get("feature_pooling"),
        "max_chunks": max_chunks,
        "feature_dim": config["feature_dim"],
        "hidden_dim": config["hidden_dim"],
        "use_chunk_context": bool(config.get("use_chunk_context", False)),
        "chunk_context_num_layers": int(config.get("chunk_context_num_layers", 0)),
        "chunk_context_num_heads": int(config.get("chunk_context_num_heads", 0)),
        "chunk_context_dropout": float(config.get("chunk_context_dropout", 0.0)),
        "use_data_parallel": use_data_parallel,
        "single_gpu_training": not use_data_parallel,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_device_count": available_gpus,
        "batch_size": batch_size,
        "total_batch_size": batch_size,
        "effective_chunks_per_batch": batch_size * max_chunks,
        "num_workers": int(config.get("num_workers", 0)),
        "gradient_accumulation_steps": int(
            config.get("gradient_accumulation_steps", 1)
        ),
        "batch_size_fallback_due_to_oom": bool(
            config.get("batch_size_fallback_due_to_oom", False)
        ),
        "max_memory_allocated_mb": max_allocated_mb,
        "max_memory_reserved_mb": max_reserved_mb,
        "approx_batch_per_gpu": (
            batch_size // len(requested_devices)
            if use_data_parallel
            else batch_size
        ),
        "recognition_aggregation": config["recognition_aggregation"],
        "top_k": int(config.get("top_k", 2)),
        "use_pos_weight": config.get("use_pos_weight", False),
        "pos_weight_rows": pos_weight_rows,
        "recognition_loss_type": config.get("recognition_loss_type", "bce"),
        "asl_gamma_neg": config.get("asl_gamma_neg"),
        "asl_gamma_pos": config.get("asl_gamma_pos"),
        "asl_clip": config.get("asl_clip"),
        "asl_eps": config.get("asl_eps"),
        **train_sampler_stats,
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
    final_model = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(final_model, "gate_values"):
        summary["gate_values"] = final_model.gate_values()
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
