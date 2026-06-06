import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import build_datasets
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
    ]
    float_keys = ["learning_rate", "threshold"]

    for key in int_keys:
        if typed_config.get(key) is not None:
            typed_config[key] = int(typed_config[key])
    for key in float_keys:
        if typed_config.get(key) is not None:
            typed_config[key] = float(typed_config[key])
    return typed_config


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold):
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
    )
    return total_loss / max(len(dataloader), 1), metrics


@torch.no_grad()
def inspect_first_batch(model, dataloader, device):
    model.eval()
    batch = next(iter(dataloader))
    device_batch = move_batch_to_device(batch, device)
    outputs = model(**device_batch)
    return {
        "input_ids_shape": list(batch["input_ids"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "binary_label_shape": list(batch["binary_label"].shape),
        "multi_labels_shape": list(batch["multi_labels"].shape),
        "detection_logits_shape": list(outputs["detection_logits"].shape),
        "recognition_logits_shape": list(outputs["recognition_logits"].shape),
    }


def print_run_info(config, device, datasets, model):
    trainable, total = count_parameters(model)
    cuda_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    lines = [
        f"model_name: {config['model_name']}",
        f"device: {device}",
        f"torch version: {torch.__version__}",
        f"cuda available: {torch.cuda.is_available()}",
        f"cuda device name: {cuda_name}",
        f"train samples: {len(datasets['train'])}",
        f"valid samples: {len(datasets['valid'])}",
        f"batch_size: {config['batch_size']}",
        f"max_len: {config['max_len']}",
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
        else:
            lines.append(f"{key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    config = normalize_training_config(load_config(args.config))
    set_seed(config["seed"])

    device = get_device()
    local_files_only = config.get("local_files_only", True)
    tokenizer_loaded = False
    model_loaded = False
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        local_files_only=local_files_only,
    )
    tokenizer_loaded = True
    datasets = build_datasets(
        config["data_dir"],
        tokenizer,
        config["max_len"],
        config["num_labels"],
        debug_num_train_samples=config.get("debug_num_train_samples"),
        debug_num_valid_samples=config.get("debug_num_valid_samples"),
        seed=config.get("seed", 42),
    )

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

    model = CorrelaScan(config).to(device)
    model_loaded = True
    print_run_info(config, device, datasets, model)
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

    tokenizer_save_dir = checkpoint_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_save_dir)

    first_batch_shapes = inspect_first_batch(model, valid_loader, device)
    best_valid_loss = float("inf")
    best_metrics = {}
    best_checkpoint_path = checkpoint_dir / "best.pt"
    threshold = float(config.get("threshold", 0.5))
    last_train_loss = None
    last_valid_loss = None

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss, metrics = evaluate(model, valid_loader, device, threshold)
        last_train_loss = train_loss
        last_valid_loss = valid_loss
        print(
            f"epoch {epoch}/{config['epochs']} "
            f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"detection_accuracy={metrics['detection_accuracy']:.6f} "
            f"recognition_micro_f1={metrics['recognition_micro_f1']:.6f} "
            f"recognition_macro_f1={metrics['recognition_macro_f1']:.6f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_metrics = metrics
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

    sanity_report = {
        "loaded_local_model": model_loaded,
        "loaded_tokenizer": tokenizer_loaded,
        "model_name": config["model_name"],
        "train_samples": len(datasets["train"]),
        "valid_samples": len(datasets["valid"]),
        "batch_shapes": first_batch_shapes,
        "train_loss": last_train_loss,
        "valid_loss": last_valid_loss,
        "detection_accuracy": best_metrics.get("detection_accuracy"),
        "recognition_micro_f1": best_metrics.get("recognition_micro_f1"),
        "recognition_macro_f1": best_metrics.get("recognition_macro_f1"),
        "checkpoint_saved": best_checkpoint_path.exists(),
        "checkpoint_path": str(best_checkpoint_path),
        "tokenizer_saved": tokenizer_save_dir.exists(),
        "tokenizer_path": str(tokenizer_save_dir),
        "can_enter_next_stage_mlsmote": (
            "yes, if this sanity run completes successfully on the server"
        ),
    }
    write_sanity_report(report_dir / "sanity_train_report.txt", sanity_report)
    (Path(config.get("result_dir", "results")) / "sanity_metrics.json").write_text(
        json.dumps(sanity_report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
