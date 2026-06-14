import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import build_datasets
from evm_bert_chunk_correlascan import EVMBertChunkCorrelaScan
from evm_bert_chunk_dataset import build_evm_bert_chunk_datasets
from evm_bert_classification_dataset import build_evm_bert_datasets
from evm_bert_correlascan import EVMBertCorrelaScan
from evm_dataset import build_evm_chunk_datasets
from evm_model import EVMChunkCorrelaScan
from metrics import (
    binary_detection_metrics,
    compute_metrics,
    compute_multilabel_metrics_from_probs,
    select_per_label_thresholds,
    sigmoid,
)
from model import CorrelaScan
from train import maybe_apply_recognition_pos_weight, normalize_training_config
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
        "--output_prefix",
        dest="output_prefix",
        default=None,
        help="Output file prefix. Defaults to result_dir/<split>_best_macro_metrics.",
    )
    parser.add_argument(
        "--threshold_search",
        default=None,
        choices=["per_label"],
        help="Search validation thresholds. Use per_label with --split valid.",
    )
    parser.add_argument(
        "--threshold_file",
        default=None,
        help="Validation-selected per-label threshold JSON.",
    )
    parser.add_argument(
        "--global_threshold",
        type=float,
        default=None,
        help="Global threshold baseline for comparison and support=0 fallback.",
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

    if config.get("model_type") == "evm_bert":
        datasets, tokenizer = build_evm_bert_datasets(config)
        model = EVMBertCorrelaScan(
            config,
            pad_token_id=tokenizer.pad_token_id,
        )
        return datasets, tokenizer, model

    if config.get("model_type") == "evm_bert_chunk":
        datasets, tokenizer = build_evm_bert_chunk_datasets(config)
        model = EVMBertChunkCorrelaScan(
            config,
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


def default_threshold_grid():
    return [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]


@torch.no_grad()
def collect_predictions(model, dataloader, device, fp16=False, bf16=False):
    model.eval()
    total_loss = 0.0
    detection_logits = []
    binary_labels = []
    recognition_logits = []
    multi_labels = []
    use_amp = (fp16 or bf16) and torch.cuda.is_available()
    autocast_dtype = torch.bfloat16 if bf16 else torch.float16

    for batch in tqdm(dataloader, desc="evaluate", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
            outputs = model(**batch)
        total_loss += outputs["loss"].item()
        detection_logits.append(outputs["detection_logits"].detach().cpu().numpy())
        binary_labels.append(batch["binary_label"].detach().cpu().numpy())
        recognition_logits.append(outputs["recognition_logits"].detach().cpu().numpy())
        multi_labels.append(batch["multi_labels"].detach().cpu().numpy())

    detection_logits = np.concatenate(detection_logits)
    binary_labels = np.concatenate(binary_labels)
    recognition_logits = np.concatenate(recognition_logits)
    multi_labels = np.concatenate(multi_labels)
    loss = total_loss / max(len(dataloader), 1)
    return {
        "loss": loss,
        "detection_logits": detection_logits,
        "binary_labels": binary_labels,
        "recognition_logits": recognition_logits,
        "recognition_probs": sigmoid(recognition_logits),
        "multi_labels": multi_labels,
        "evaluated_samples": int(len(binary_labels)),
    }


def run_global_evaluation(predictions, threshold):
    metrics = compute_metrics(
        predictions["detection_logits"],
        predictions["binary_labels"],
        predictions["recognition_logits"],
        predictions["multi_labels"],
        threshold=threshold,
        scan_thresholds=None,
    )
    return predictions["loss"], metrics


def run_per_label_evaluation(predictions, thresholds, detection_threshold):
    recognition_metrics = compute_multilabel_metrics_from_probs(
        predictions["multi_labels"],
        predictions["recognition_probs"],
        thresholds,
    )
    detection_metrics = binary_detection_metrics(
        predictions["detection_logits"],
        predictions["binary_labels"],
        threshold=detection_threshold,
    )
    metrics = {}
    metrics.update(detection_metrics)
    metrics.update(recognition_metrics)
    return predictions["loss"], metrics


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


def load_threshold_file(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "thresholds" in data:
        thresholds = data["thresholds"]
    else:
        thresholds = [row["best_threshold"] for row in data["per_label"]]
    return data, [float(value) for value in thresholds]


def threshold_output_paths(config):
    result_dir = Path(config.get("result_dir", "results"))
    return (
        result_dir / "per_label_thresholds_valid.json",
        result_dir / "per_label_thresholds_valid.txt",
    )


def write_threshold_report(path, report):
    ensure_dir(path.parent)
    lines = ["Validation-selected per-label thresholds", ""]
    lines.append(f"checkpoint: {report['checkpoint']}")
    lines.append(f"checkpoint_epoch: {report['checkpoint_epoch']}")
    lines.append(f"split: {report['split']}")
    lines.append(f"global_threshold_fallback: {report['global_threshold_fallback']}")
    lines.append("")
    lines.append(
        "label_name | support | best_threshold | valid_precision | "
        "valid_recall | valid_f1 | predicted_positive_count"
    )
    for row in report["per_label"]:
        lines.append(
            f"{row['label_name']} | {row['support']} | "
            f"{row['best_threshold']:.2f} | "
            f"{row['best_valid_precision']:.6f} | "
            f"{row['best_valid_recall']:.6f} | "
            f"{row['best_valid_f1']:.6f} | "
            f"{row['predicted_positive_count_at_best_threshold']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_text_report(path, report):
    ensure_dir(path.parent)
    lines = ["Strict evaluation report", ""]
    scalar_keys = [
        "split",
        "checkpoint",
        "checkpoint_epoch",
        "threshold",
        "threshold_source",
        "threshold_mode",
        "threshold_file",
        "model_type",
        "hf_model_path",
        "is_transductive_pretraining",
        "evaluated_samples",
        "expected_samples",
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
        "micro_f1_change_over_global_threshold",
        "macro_f1_improvement_over_global_threshold",
        "reference_baseline_name",
        "baseline_micro_f1",
        "baseline_macro_f1",
        "baseline_detection_f1",
        "micro_f1_change_over_reference_baseline",
        "macro_f1_change_over_reference_baseline",
        "detection_f1_change_over_reference_baseline",
        "micro_f1_improvement_over_unweighted_baseline",
        "macro_f1_improvement_over_unweighted_baseline",
        "detection_f1_improvement_over_baseline",
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
    if report.get("per_label_thresholds"):
        lines.append("")
        lines.append("Per-label thresholds:")
        for row in report["per_label_thresholds"]:
            lines.append(
                f"{row['label_id']} | {row['label_name']} | "
                f"{row['best_threshold']:.2f}"
            )
    if report.get("global_threshold_baseline"):
        lines.append("")
        lines.append("Global threshold baseline:")
        for key, value in report["global_threshold_baseline"].items():
            lines.append(f"{key}: {value}")
    if report.get("small_label_f1_changes"):
        lines.append("")
        lines.append("Small-label F1 changes vs unweighted baseline:")
        lines.append("label | baseline_f1 | current_f1 | diff")
        for row in report["small_label_f1_changes"]:
            lines.append(
                f"{row['label_name']} | {row['baseline_f1']:.6f} | "
                f"{row['current_f1']:.6f} | {row['diff']:.6f}"
            )
    if report.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_output_paths(config, args):
    result_dir = Path(config.get("result_dir", "results"))
    if args.threshold_file:
        prefix = Path(config.get("result_dir", "results")) / f"{args.split}_per_label_threshold_metrics"
    elif args.output_prefix:
        prefix = Path(args.output_prefix)
        if prefix.parent == Path("."):
            prefix = result_dir / prefix
    else:
        prefix = result_dir / f"{args.split}_best_macro_metrics"
    ensure_dir(prefix.parent)
    return prefix.with_suffix(".json"), prefix.with_suffix(".txt")


def build_evaluation_warnings(metrics, expected_samples, evaluated_samples):
    warnings = []
    if evaluated_samples != expected_samples:
        warnings.append(
            "evaluated_samples does not match dataset size: "
            f"{evaluated_samples} vs {expected_samples}"
        )
    for row in metrics["per_label"]:
        predicted = row["predicted_positive_count"]
        support = row["support"]
        if predicted == 0:
            warnings.append(f"{row['label_name']}: predicted all-zero.")
        if support > 0 and predicted > 3 * support and predicted - support > 50:
            warnings.append(
                f"{row['label_name']}: predicted positives may be too many "
                f"({predicted} vs support {support})."
            )
    if metrics.get("model_type") in {"evm_bert", "evm_bert_chunk"}:
        warnings.append(
            "EVM-BERT uses full BJUT unlabeled transductive pretraining; "
            "this result is not a strict inductive baseline."
        )
    return warnings


def build_baseline_comparison(config, report):
    comparison = {
        "reference_baseline_name": config.get(
            "reference_baseline_name",
            "CodeBERT-base weighted BCE global threshold baseline",
        ),
        "baseline_micro_f1": config.get("baseline_micro_f1"),
        "baseline_macro_f1": config.get("baseline_macro_f1"),
        "baseline_detection_f1": config.get("baseline_detection_f1"),
        "micro_f1_change_over_reference_baseline": None,
        "macro_f1_change_over_reference_baseline": None,
        "detection_f1_change_over_reference_baseline": None,
        "micro_f1_improvement_over_unweighted_baseline": None,
        "macro_f1_improvement_over_unweighted_baseline": None,
        "detection_f1_improvement_over_baseline": None,
        "small_label_f1_changes": [],
    }
    if comparison["baseline_micro_f1"] is not None:
        comparison["micro_f1_change_over_reference_baseline"] = (
            report["recognition_micro_f1"] - comparison["baseline_micro_f1"]
        )
        comparison["micro_f1_improvement_over_unweighted_baseline"] = comparison[
            "micro_f1_change_over_reference_baseline"
        ]
    if comparison["baseline_macro_f1"] is not None:
        comparison["macro_f1_change_over_reference_baseline"] = (
            report["recognition_macro_f1"] - comparison["baseline_macro_f1"]
        )
        comparison["macro_f1_improvement_over_unweighted_baseline"] = comparison[
            "macro_f1_change_over_reference_baseline"
        ]
    if comparison["baseline_detection_f1"] is not None:
        comparison["detection_f1_change_over_reference_baseline"] = (
            report["detection_f1"] - comparison["baseline_detection_f1"]
        )
        comparison["detection_f1_improvement_over_baseline"] = comparison[
            "detection_f1_change_over_reference_baseline"
        ]

    baseline_per_label = config.get("baseline_per_label_f1", {}) or {}
    small_label_names = config.get("small_label_names", []) or []
    rows_by_name = {row["label_name"]: row for row in report["per_label"]}
    for label_name in small_label_names:
        if label_name not in baseline_per_label or label_name not in rows_by_name:
            continue
        baseline_f1 = float(baseline_per_label[label_name])
        current_f1 = float(rows_by_name[label_name]["f1"])
        comparison["small_label_f1_changes"].append(
            {
                "label_name": label_name,
                "baseline_f1": baseline_f1,
                "current_f1": current_f1,
                "diff": current_f1 - baseline_f1,
            }
        )
    return comparison


def resolve_global_threshold(args, config, checkpoint_threshold):
    if args.global_threshold is not None:
        return float(args.global_threshold)
    if config.get("threshold") is not None:
        return float(config["threshold"])
    return float(checkpoint_threshold if checkpoint_threshold is not None else 0.5)


def main():
    args = parse_args()
    config = normalize_training_config(load_config(args.config))
    config["_config_path"] = args.config
    set_seed(config.get("seed", 42))

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    threshold, threshold_source = resolve_threshold(args.threshold, checkpoint)
    global_threshold = resolve_global_threshold(args, config, threshold)

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
    maybe_apply_recognition_pos_weight(model, config, device, is_main=True)
    fp16 = bool(config.get("fp16", False)) and torch.cuda.is_available()
    bf16 = bool(config.get("bf16", False)) and torch.cuda.is_available()
    predictions = collect_predictions(model, dataloader, device, fp16=fp16, bf16=bf16)

    if args.threshold_search == "per_label":
        if args.split != "valid":
            raise ValueError("--threshold_search per_label must be run on --split valid.")
        threshold_grid = default_threshold_grid()
        selection = select_per_label_thresholds(
            predictions["multi_labels"],
            predictions["recognition_probs"],
            threshold_grid,
            label_names=config.get("label_names"),
            global_threshold=global_threshold,
        )
        threshold_report = {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "split": args.split,
            "evaluated_samples": predictions["evaluated_samples"],
            "expected_samples": len(dataset),
            "global_threshold_fallback": global_threshold,
            "threshold_grid": threshold_grid,
            "thresholds": selection["thresholds"],
            "per_label": selection["per_label"],
            "warnings": (
                []
                if predictions["evaluated_samples"] == len(dataset)
                else [
                    "evaluated_samples does not match validation dataset size: "
                    f"{predictions['evaluated_samples']} vs {len(dataset)}"
                ]
            ),
        }
        json_path, txt_path = threshold_output_paths(config)
        ensure_dir(json_path.parent)
        json_path.write_text(
            json.dumps(threshold_report, indent=2), encoding="utf-8"
        )
        write_threshold_report(txt_path, threshold_report)
        print(f"[OK] wrote {json_path}")
        print(f"[OK] wrote {txt_path}")
        return

    threshold_mode = "global"
    threshold_file_data = None
    per_label_threshold_rows = None
    global_baseline = None
    macro_improvement = None
    micro_change = None
    if args.threshold_file:
        threshold_file_data, per_label_thresholds = load_threshold_file(
            args.threshold_file
        )
        threshold_mode = "per_label"
        threshold_source = "validation set"
        threshold = None
        loss, metrics = run_per_label_evaluation(
            predictions,
            per_label_thresholds,
            detection_threshold=global_threshold,
        )
        _, global_metrics = run_global_evaluation(
            predictions,
            threshold=global_threshold,
        )
        global_baseline = {
            "threshold": global_threshold,
            "recognition_micro_precision": global_metrics[
                "recognition_micro_precision"
            ],
            "recognition_micro_recall": global_metrics["recognition_micro_recall"],
            "recognition_micro_f1": global_metrics["recognition_micro_f1"],
            "recognition_macro_precision": global_metrics[
                "recognition_macro_precision"
            ],
            "recognition_macro_recall": global_metrics["recognition_macro_recall"],
            "recognition_macro_f1": global_metrics["recognition_macro_f1"],
            "predicted_positive_total": global_metrics["predicted_positive_total"],
        }
        macro_improvement = (
            metrics["recognition_macro_f1"]
            - global_baseline["recognition_macro_f1"]
        )
        micro_change = (
            metrics["recognition_micro_f1"]
            - global_baseline["recognition_micro_f1"]
        )
        per_label_threshold_rows = threshold_file_data.get("per_label", [])
    else:
        loss, metrics = run_global_evaluation(predictions, threshold)

    report = {
        "split": args.split,
        "samples": len(dataset),
        "evaluated_samples": predictions["evaluated_samples"],
        "expected_samples": len(dataset),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_mode": threshold_mode,
        "threshold_file": args.threshold_file,
        "model_type": config.get("model_type", "codebert"),
        "hf_model_path": config.get("hf_model_path"),
        "is_transductive_pretraining": (
            True
            if config.get("model_type") in {"evm_bert", "evm_bert_chunk"}
            else None
        ),
        "fp16": fp16,
        "bf16": bf16,
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
        "per_label_thresholds": per_label_threshold_rows,
        "global_threshold_baseline": global_baseline,
        "micro_f1_change_over_global_threshold": micro_change,
        "macro_f1_improvement_over_global_threshold": macro_improvement,
    }
    report.update(build_baseline_comparison(config, report))
    report["warnings"] = build_evaluation_warnings(
        report,
        expected_samples=len(dataset),
        evaluated_samples=predictions["evaluated_samples"],
    )

    json_path, txt_path = resolve_output_paths(config, args)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_text_report(txt_path, report)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {txt_path}")


if __name__ == "__main__":
    main()
