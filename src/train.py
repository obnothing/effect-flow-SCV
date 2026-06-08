import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import build_datasets
from evm_dataset import build_evm_chunk_datasets
from evm_model import EVMChunkCorrelaScan
from metrics import compute_metrics
from model import CorrelaScan
from utils import ensure_dir, get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage-1 CorrelaScan.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def normalize_training_config(config):
    typed_config = dict(config)
    int_keys = [
        "seed",
        "num_labels",
        "max_len",
        "batch_size",
        "epochs",
        "num_workers",
        "debug_num_train_samples",
        "debug_num_valid_samples",
        "early_stopping_patience",
        "gradient_accumulation_steps",
        "chunk_size",
        "chunk_stride",
        "max_chunks",
        "embedding_dim",
        "bigru_hidden_size",
        "num_transformer_layers",
        "num_attention_heads",
    ]
    float_keys = ["learning_rate", "threshold", "gradient_clip_norm", "dropout"]

    for key in int_keys:
        if typed_config.get(key) is not None:
            typed_config[key] = int(typed_config[key])
    for key in float_keys:
        if typed_config.get(key) is not None:
            typed_config[key] = float(typed_config[key])
    return typed_config


def get_thresholds(config):
    thresholds = config.get("thresholds", [0.1, 0.2, 0.3, 0.4, 0.5])
    return [float(threshold) for threshold in thresholds]


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    gradient_clip_norm=None,
    gradient_accumulation_steps=1,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="train", leave=False), start=1):
        batch = move_batch_to_device(batch, device)

        outputs = model(**batch)
        loss = outputs["loss"]
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item()

        should_step = (
            step % gradient_accumulation_steps == 0
            or step == len(dataloader)
        )
        if should_step:
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold, scan_thresholds):
    model.eval()
    total_loss = 0.0
    detection_logits = []
    binary_labels = []
    recognition_logits = []
    multi_labels = []

    for batch in tqdm(dataloader, desc="valid", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        total_loss += outputs["loss"].item()
        detection_logits.append(outputs["detection_logits"].detach().cpu().numpy())
        binary_labels.append(batch["binary_label"].detach().cpu().numpy())
        recognition_logits.append(outputs["recognition_logits"].detach().cpu().numpy())
        multi_labels.append(batch["multi_labels"].detach().cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(detection_logits),
        np.concatenate(binary_labels),
        np.concatenate(recognition_logits),
        np.concatenate(multi_labels),
        threshold=threshold,
        scan_thresholds=scan_thresholds,
    )
    return total_loss / max(len(dataloader), 1), metrics


def cuda_memory_stats():
    if not torch.cuda.is_available():
        return {
            "max_memory_allocated_mb": 0.0,
            "max_memory_reserved_mb": 0.0,
        }
    return {
        "max_memory_allocated_mb": (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        ),
        "max_memory_reserved_mb": (
            torch.cuda.max_memory_reserved() / (1024 ** 2)
        ),
    }


@torch.no_grad()
def inspect_first_batch(model, dataloader, device):
    model.eval()
    batch = next(iter(dataloader))
    device_batch = move_batch_to_device(batch, device)
    outputs = model(**device_batch)
    batch_shapes = {
        key: list(value.shape)
        for key, value in batch.items()
        if hasattr(value, "shape")
    }
    return {
        "batch_shapes": batch_shapes,
        "logits_shapes": {
            "detection_logits": list(outputs["detection_logits"].shape),
            "recognition_logits": list(outputs["recognition_logits"].shape),
        },
    }


def print_run_info(config, device, datasets, model):
    trainable, total = count_parameters(model)
    model_type = config.get("model_type", "codebert")
    cuda_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    lines = [
        f"model_type: {model_type}",
        f"model_name: {config.get('model_name')}",
        f"device: {device}",
        f"torch version: {torch.__version__}",
        f"cuda available: {torch.cuda.is_available()}",
        f"cuda device name: {cuda_name}",
        f"train samples: {len(datasets['train'])}",
        f"valid samples: {len(datasets['valid'])}",
        f"batch_size: {config['batch_size']}",
        "gradient_accumulation_steps: "
        f"{config.get('gradient_accumulation_steps', 1)}",
        "effective_batch_size: "
        f"{config['batch_size'] * config.get('gradient_accumulation_steps', 1)}",
        f"max_len: {config.get('max_len')}",
        f"chunk_size: {config.get('chunk_size')}",
        f"max_chunks: {config.get('max_chunks')}",
        f"freeze_encoder: {config.get('freeze_encoder', False)}",
        f"trainable parameters / total parameters: {trainable} / {total}",
    ]
    for line in lines:
        print(f"[INFO] {line}")


def write_sanity_report(path, report):
    ensure_dir(path.parent)
    lines = ["CorrelaScan sanity training report", ""]
    for key, value in report.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"- {sub_key}: {sub_value}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(f"{key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_epoch_history_report(path, history):
    ensure_dir(path.parent)
    lines = ["CorrelaScan epoch history", ""]
    for record in history:
        lines.append(
            "epoch "
            f"{record['epoch']}/{record['total_epochs']} "
            f"train_loss={record['train_loss']:.6f} "
            f"valid_loss={record['valid_loss']:.6f} "
            f"is_best={record['is_best']} "
            f"best_epoch={record['best_epoch']} "
            f"patience_counter={record['patience_counter']}"
        )
        lines.append(
            "metrics: "
            f"detection_accuracy={record['detection_accuracy']:.6f} "
            f"recognition_micro_f1={record['recognition_micro_f1']:.6f} "
            f"recognition_macro_f1={record['recognition_macro_f1']:.6f} "
            f"predicted_positive_total={record['predicted_positive_total']}"
        )
        lines.append(
            "memory_mb: "
            f"allocated={record['max_memory_allocated_mb']:.2f} "
            f"reserved={record['max_memory_reserved_mb']:.2f}"
        )
        lines.append(
            "per_label_predicted_positive_count: "
            f"{record['per_label_predicted_positive_count']}"
        )
        lines.append(
            "per_label_true_positive_count: "
            f"{record['per_label_true_positive_count']}"
        )
        lines.append(
            "per_label_mean_pred_prob: "
            f"{[round(value, 6) for value in record['per_label_mean_pred_prob']]}"
        )
        lines.append(
            "per_label_accuracy: "
            f"{[round(value, 6) for value in record['per_label_accuracy']]}"
        )
        lines.append(
            "per_label_precision: "
            f"{[round(value, 6) for value in record['per_label_precision']]}"
        )
        lines.append(
            "per_label_recall: "
            f"{[round(value, 6) for value in record['per_label_recall']]}"
        )
        lines.append(
            "per_label_f1: "
            f"{[round(value, 6) for value in record['per_label_f1']]}"
        )
        for threshold, threshold_metrics in record["threshold_scan"].items():
            lines.append(
                f"threshold={threshold}: "
                f"micro_f1={threshold_metrics['micro_f1']:.6f} "
                f"macro_f1={threshold_metrics['macro_f1']:.6f} "
                "predicted_positive_total="
                f"{threshold_metrics['predicted_positive_total']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def get_report_stem(config):
    checkpoint_name = Path(config["checkpoint_dir"]).name
    if checkpoint_name == "sanity":
        return "sanity_train_report"
    if checkpoint_name.startswith("sanity_"):
        return f"{checkpoint_name}_report"
    return f"{checkpoint_name}_report"


def get_sanity_report_paths(config, report_dir):
    report_stem = get_report_stem(config)
    return report_dir / f"{report_stem}.txt", report_dir / f"{report_stem}.json"


def get_epoch_history_paths(config, report_dir):
    report_stem = get_report_stem(config)
    if report_stem.endswith("_report"):
        history_stem = report_stem[: -len("_report")] + "_epoch_history"
    else:
        history_stem = f"{report_stem}_epoch_history"
    return (
        report_dir / f"{history_stem}.txt",
        report_dir / f"{history_stem}.json",
    )


def build_training_components(config):
    if config.get("model_type") == "evm_chunk":
        datasets, tokenizer = build_evm_chunk_datasets(config)
        model = EVMChunkCorrelaScan(
            config,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
        return datasets, tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        local_files_only=config.get("local_files_only", True),
    )
    datasets = build_datasets(
        config["data_dir"],
        tokenizer,
        config["max_len"],
        config["num_labels"],
        debug_num_train_samples=config.get("debug_num_train_samples"),
        debug_num_valid_samples=config.get("debug_num_valid_samples"),
        seed=config.get("seed", 42),
        use_mlsmote_train=config.get("use_mlsmote_train", False),
    )
    return datasets, tokenizer, CorrelaScan(config)


def save_tokenizer_artifact(config, tokenizer, checkpoint_dir):
    tokenizer_save_dir = checkpoint_dir / "tokenizer"
    ensure_dir(tokenizer_save_dir)
    if config.get("model_type") == "evm_chunk":
        vocab_path = Path(config["vocab_path"])
        target_path = tokenizer_save_dir / "evm_vocab.json"
        target_path.write_text(vocab_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        tokenizer.save_pretrained(tokenizer_save_dir)
    return tokenizer_save_dir


def maybe_plot_training_history(config, history_json_path):
    if not history_json_path.exists():
        return False
    script_path = Path("scripts/plot_training_results.py")
    if not script_path.exists():
        return False
    command = [
        sys.executable,
        str(script_path),
        "--history",
        str(history_json_path),
        "--config",
        config.get("_config_path", ""),
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except Exception as exc:
        print(f"[WARN] failed to plot training history: {exc}")
        print("[WARN] install matplotlib or run scripts/plot_training_results.py manually.")
        return False


def main():
    args = parse_args()
    config = normalize_training_config(load_config(args.config))
    config["_config_path"] = args.config
    set_seed(config["seed"])

    device = get_device()
    tokenizer_loaded = False
    model_loaded = False
    datasets, tokenizer, model = build_training_components(config)
    tokenizer_loaded = True

    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        datasets["valid"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    model = model.to(device)
    model_loaded = True
    print_run_info(config, device, datasets, model)
    trainable_parameters, total_parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config["learning_rate"],
    )

    checkpoint_dir = Path(config["checkpoint_dir"])
    ensure_dir(checkpoint_dir)
    ensure_dir(Path(config.get("log_dir", "logs")))
    ensure_dir(Path(config.get("result_dir", "results")))
    report_dir = Path(config.get("report_dir", "data/reports"))
    ensure_dir(report_dir)
    history_txt_path, history_json_path = get_epoch_history_paths(config, report_dir)

    tokenizer_save_dir = save_tokenizer_artifact(config, tokenizer, checkpoint_dir)

    first_batch_info = inspect_first_batch(model, valid_loader, device)
    best_valid_loss = float("inf")
    best_metrics = {}
    best_train_loss = None
    best_epoch = None
    best_checkpoint_path = checkpoint_dir / "best.pt"
    threshold = float(config.get("threshold", 0.5))
    scan_thresholds = get_thresholds(config)
    gradient_clip_norm = config.get("gradient_clip_norm")
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    last_train_loss = None
    last_valid_loss = None
    last_metrics = {}
    last_memory_stats = {}
    patience = config.get("early_stopping_patience")
    patience_counter = 0
    epoch_history = []
    early_stopped = False
    stopped_epoch = None

    for epoch in range(1, config["epochs"] + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gradient_clip_norm=gradient_clip_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        valid_loss, metrics = evaluate(
            model, valid_loader, device, threshold, scan_thresholds
        )
        last_memory_stats = cuda_memory_stats()
        last_train_loss = train_loss
        last_valid_loss = valid_loss
        last_metrics = metrics
        print(
            f"epoch {epoch}/{config['epochs']} "
            f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"detection_accuracy={metrics['detection_accuracy']:.6f} "
            f"recognition_micro_f1={metrics['recognition_micro_f1']:.6f} "
            f"recognition_macro_f1={metrics['recognition_macro_f1']:.6f} "
            f"predicted_positive_total={metrics['predicted_positive_total']} "
            "max_memory_allocated_mb="
            f"{last_memory_stats['max_memory_allocated_mb']:.2f} "
            "max_memory_reserved_mb="
            f"{last_memory_stats['max_memory_reserved_mb']:.2f}"
        )
        print(
            "[INFO] per-label predicted_positive_count: "
            f"{metrics['per_label_predicted_positive_count']}"
        )
        print(
            "[INFO] per-label true_positive_count: "
            f"{metrics['per_label_true_positive_count']}"
        )
        print(
            "[INFO] per-label mean_pred_prob: "
            f"{[round(value, 6) for value in metrics['per_label_mean_pred_prob']]}"
        )
        print(
            "[INFO] per-label accuracy: "
            f"{[round(value, 6) for value in metrics['per_label_accuracy']]}"
        )
        print(
            "[INFO] per-label precision: "
            f"{[round(value, 6) for value in metrics['per_label_precision']]}"
        )
        print(
            "[INFO] per-label recall: "
            f"{[round(value, 6) for value in metrics['per_label_recall']]}"
        )
        for scan_threshold, scan_metrics in metrics["threshold_scan"].items():
            print(
                f"[INFO] threshold={scan_threshold} "
                f"micro_f1={scan_metrics['micro_f1']:.6f} "
                f"macro_f1={scan_metrics['macro_f1']:.6f} "
                "predicted_positive_total="
                f"{scan_metrics['predicted_positive_total']}"
            )

        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_train_loss = train_loss
            best_metrics = metrics
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_loss": valid_loss,
                    "metrics": metrics,
                    "config": config,
                },
                best_checkpoint_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_record = {
            "epoch": epoch,
            "total_epochs": config["epochs"],
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "is_best": is_best,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "patience_counter": patience_counter,
            "detection_accuracy": metrics["detection_accuracy"],
            "recognition_micro_f1": metrics["recognition_micro_f1"],
            "recognition_macro_f1": metrics["recognition_macro_f1"],
            "predicted_positive_total": metrics["predicted_positive_total"],
            "per_label_predicted_positive_count": metrics[
                "per_label_predicted_positive_count"
            ],
            "per_label_true_positive_count": metrics[
                "per_label_true_positive_count"
            ],
            "per_label_mean_pred_prob": metrics["per_label_mean_pred_prob"],
            "per_label_accuracy": metrics["per_label_accuracy"],
            "per_label_precision": metrics["per_label_precision"],
            "per_label_recall": metrics["per_label_recall"],
            "per_label_f1": metrics["per_label_f1"],
            "threshold_scan": metrics["threshold_scan"],
            "max_memory_allocated_mb": last_memory_stats[
                "max_memory_allocated_mb"
            ],
            "max_memory_reserved_mb": last_memory_stats[
                "max_memory_reserved_mb"
            ],
        }
        epoch_history.append(epoch_record)
        write_epoch_history_report(history_txt_path, epoch_history)
        history_json_path.write_text(
            json.dumps(epoch_history, indent=2), encoding="utf-8"
        )
        (
            Path(config.get("result_dir", "results")) / "epoch_history.json"
        ).write_text(json.dumps(epoch_history, indent=2), encoding="utf-8")

        if not is_best and patience is not None and patience_counter >= patience:
            early_stopped = True
            stopped_epoch = epoch
            print(f"[INFO] early stopping at epoch {epoch}")
            break

    cuda_device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    warnings = []
    if config.get("use_mlsmote_train", False):
        warnings.append(
            "MLSMOTE-compatible text oversampling duplicates real opcode samples; "
            "it does not synthesize opcode sequences by linear interpolation."
        )
    if config.get("model_type") == "evm_chunk":
        max_covered_tokens = config["chunk_size"] + config["chunk_stride"] * (
            config["max_chunks"] - 1
        )
        warnings.append(
            "EVM chunk baseline is trained from scratch with an EVM opcode-aware "
            "vocabulary; it does not reuse CodeBERT pretrained embeddings."
        )
        warnings.append(
            f"Each contract is capped at {max_covered_tokens} EVM tokenizer tokens; "
            "longer contracts are truncated."
        )
    elif config.get("freeze_encoder", False):
        warnings.append(
            "Encoder is frozen; this run checks classifier-head training only."
        )
    else:
        warnings.append("Encoder is unfrozen; monitor GPU memory and loss stability.")
    threshold_scan = best_metrics.get("threshold_scan", {})
    threshold_key = str(threshold)
    default_threshold_f1 = threshold_scan.get(threshold_key, {}).get("micro_f1")
    lower_threshold_has_f1 = any(
        float(key) < threshold and value.get("micro_f1", 0.0) > 0.0
        for key, value in threshold_scan.items()
    )
    if default_threshold_f1 == 0.0 and lower_threshold_has_f1:
        warnings.append(
            "threshold=0.5 produced zero F1 while a lower threshold produced "
            "non-zero F1; threshold=0.5 may be too high."
        )

    sanity_report = {
        "loaded_model": model_loaded,
        "loaded_local_model": (
            model_loaded if config.get("model_type") != "evm_chunk" else "not_applicable"
        ),
        "loaded_tokenizer": tokenizer_loaded,
        "model_type": config.get("model_type", "codebert"),
        "tokenizer_type": config.get("tokenizer_type", "transformers"),
        "model_name": config.get("model_name"),
        "vocab_path": config.get("vocab_path"),
        "vocab_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        "train_samples": len(datasets["train"]),
        "valid_samples": len(datasets["valid"]),
        "completed_epochs": len(epoch_history),
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "early_stopping_patience": patience,
        "best_epoch": best_epoch,
        "use_mlsmote_train": config.get("use_mlsmote_train", False),
        "freeze_encoder": config.get("freeze_encoder", False),
        "max_len": config.get("max_len"),
        "chunk_size": config.get("chunk_size"),
        "chunk_stride": config.get("chunk_stride"),
        "max_chunks": config.get("max_chunks"),
        "max_covered_tokens": (
            config["chunk_size"] + config["chunk_stride"] * (config["max_chunks"] - 1)
            if config.get("model_type") == "evm_chunk"
            else None
        ),
        "chunk_pooling": config.get("chunk_pooling"),
        "encoder_type": config.get("encoder_type"),
        "embedding_dim": config.get("embedding_dim"),
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": config["batch_size"] * gradient_accumulation_steps,
        "learning_rate": config["learning_rate"],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": cuda_device_name,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_ratio": (
            trainable_parameters / total_parameters if total_parameters else 0.0
        ),
        "batch_shapes": first_batch_info["batch_shapes"],
        "logits_shapes": first_batch_info["logits_shapes"],
        "train_loss": best_train_loss,
        "valid_loss": best_valid_loss,
        "metric_source": "best_checkpoint_epoch",
        "last_train_loss": last_train_loss,
        "last_valid_loss": last_valid_loss,
        "last_detection_accuracy": last_metrics.get("detection_accuracy"),
        "last_recognition_micro_f1": last_metrics.get("recognition_micro_f1"),
        "last_recognition_macro_f1": last_metrics.get("recognition_macro_f1"),
        "detection_accuracy": best_metrics.get("detection_accuracy"),
        "recognition_micro_f1": best_metrics.get("recognition_micro_f1"),
        "recognition_macro_f1": best_metrics.get("recognition_macro_f1"),
        "predicted_positive_total": best_metrics.get("predicted_positive_total"),
        "per_label_predicted_positive_count": best_metrics.get(
            "per_label_predicted_positive_count"
        ),
        "per_label_true_positive_count": best_metrics.get(
            "per_label_true_positive_count"
        ),
        "per_label_mean_pred_prob": best_metrics.get("per_label_mean_pred_prob"),
        "per_label_accuracy": best_metrics.get("per_label_accuracy"),
        "per_label_precision": best_metrics.get("per_label_precision"),
        "per_label_recall": best_metrics.get("per_label_recall"),
        "per_label_f1": best_metrics.get("per_label_f1"),
        "threshold_scan": threshold_scan,
        "max_memory_allocated_mb": last_memory_stats.get("max_memory_allocated_mb"),
        "max_memory_reserved_mb": last_memory_stats.get("max_memory_reserved_mb"),
        "checkpoint_saved": best_checkpoint_path.exists(),
        "checkpoint_path": str(best_checkpoint_path),
        "tokenizer_saved": tokenizer_save_dir.exists(),
        "tokenizer_path": str(tokenizer_save_dir),
        "epoch_history_saved": history_txt_path.exists() and history_json_path.exists(),
        "epoch_history_txt_path": str(history_txt_path),
        "epoch_history_json_path": str(history_json_path),
        "plots_generated": maybe_plot_training_history(config, history_json_path),
        "can_enter_full_training": (
            best_checkpoint_path.exists()
            and last_train_loss is not None
            and last_valid_loss is not None
        ),
        "can_enter_full_evm_chunk_training": (
            config.get("model_type") == "evm_chunk"
            and best_checkpoint_path.exists()
            and last_train_loss is not None
            and last_valid_loss is not None
        ),
        "warnings": warnings,
    }
    report_txt_path, report_json_path = get_sanity_report_paths(config, report_dir)
    write_sanity_report(report_txt_path, sanity_report)
    report_json_path.write_text(
        json.dumps(sanity_report, indent=2), encoding="utf-8"
    )
    (Path(config.get("result_dir", "results")) / "sanity_metrics.json").write_text(
        json.dumps(sanity_report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
