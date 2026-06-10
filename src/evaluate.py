import argparse
import json
from pathlib import Path

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
from train import normalize_training_config
from utils import ensure_dir, get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Strict checkpoint evaluation.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        default="auto",
        help="Classification threshold or 'auto'. Auto uses checkpoint macro-F1 threshold.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix. Defaults to result_dir/<split>_best_macro_metrics.",
    )
    return parser.parse_args()


def build_components(config):
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
        seed=config.get("seed", 42),
        use_mlsmote_train=config.get("use_mlsmote_train", False),
    )
    return datasets, tokenizer, CorrelaScan(config)


def resolve_threshold(raw_threshold, checkpoint):
    if raw_threshold != "auto":
        return float(raw_threshold), "cli"
    threshold = checkpoint.get("best_threshold_by_macro_f1")
    if threshold is None:
        threshold = 0.5
        source = "default_0.5"
    else:
        source = "checkpoint_best_threshold_by_macro_f1"
    return float(threshold), source


@torch.no_grad()
def run_evaluation(model, dataloader, device, threshold):
    model.eval()
    total_loss = 0.0
    detection_logits = []
    binary_labels = []
    recognition_logits = []
    multi_labels = []

    for batch in tqdm(dataloader, desc="evaluate", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
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
        scan_thresholds=None,
    )
    return total_loss / max(len(dataloader), 1), metrics


def label_table(config, metrics):
    names = config.get("label_names") or [f"label_{idx}" for idx in range(config["num_labels"])]
    rows = []
    for idx, name in enumerate(names):
        rows.append(
            {
                "label_id": idx,
                "label_name": name,
                "precision": metrics["per_label_precision"][idx],
                "recall": metrics["per_label_recall"][idx],
                "f1": metrics["per_label_f1"][idx],
                "support": metrics["per_label_support"][idx],
                "predicted_positive_count": metrics[
                    "per_label_predicted_positive_count"
                ][idx],
                "true_positive_count": metrics["per_label_true_positive_count"][idx],
                "mean_pred_prob": metrics["per_label_mean_pred_prob"][idx],
            }
        )
    return rows


def write_text_report(path, report):
    ensure_dir(path.parent)
    lines = ["Strict evaluation report", ""]
    scalar_keys = [
        "split",
        "checkpoint",
        "checkpoint_epoch",
        "threshold",
        "threshold_source",
        "loss",
        "detection_accuracy",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "recognition_micro_precision",
        "recognition_micro_recall",
        "recognition_micro_f1",
        "recognition_macro_precision",
        "recognition_macro_recall",
        "recognition_macro_f1",
        "predicted_positive_total",
    ]
    for key in scalar_keys:
        lines.append(f"{key}: {report.get(key)}")
    lines.append("")
    lines.append("Per-label metrics:")
    lines.append("id | label | precision | recall | f1 | support | predicted | true | mean_prob")
    for row in report["per_label"]:
        lines.append(
            f"{row['label_id']} | {row['label_name']} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | "
            f"{row['support']} | {row['predicted_positive_count']} | "
            f"{row['true_positive_count']} | {row['mean_pred_prob']:.6f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_output_paths(config, args):
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        prefix = Path(config.get("result_dir", "results")) / f"{args.split}_best_macro_metrics"
    ensure_dir(prefix.parent)
    return prefix.with_suffix(".json"), prefix.with_suffix(".txt")


def main():
    args = parse_args()
    config = normalize_training_config(load_config(args.config))
    config["_config_path"] = args.config
    set_seed(config.get("seed", 42))

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    threshold, threshold_source = resolve_threshold(args.threshold, checkpoint)

    datasets, tokenizer, model = build_components(config)
    dataset = datasets[args.split]
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )

    device = get_device()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    loss, metrics = run_evaluation(model, dataloader, device, threshold)

    report = {
        "split": args.split,
        "samples": len(dataset),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "loss": loss,
        "detection_accuracy": metrics["detection_accuracy"],
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
        "per_label_predicted_positive_count": metrics[
            "per_label_predicted_positive_count"
        ],
        "per_label_true_positive_count": metrics["per_label_true_positive_count"],
        "per_label_mean_pred_prob": metrics["per_label_mean_pred_prob"],
        "per_label": label_table(config, metrics),
    }

    json_path, txt_path = resolve_output_paths(config, args)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_text_report(txt_path, report)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {txt_path}")


if __name__ == "__main__":
    main()
