"""Validation-only analysis for the process01 Main-6 M0 experiments."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import compute_multilabel_metrics_from_probs, select_per_label_thresholds  # noqa: E402
from train_chunk_mil import build_model, load_config  # noqa: E402
from chunk_feature_dataset import build_chunk_feature_datasets  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    p = int(y.sum())
    n = len(y) - p
    if not p or not n:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y == 1].sum() - p * (p + 1) / 2.0) / (p * n))


def average_precision(y, score):
    y = np.asarray(y).astype(int)
    p = int(y.sum())
    if not p:
        return None
    ranked = y[np.argsort(-np.asarray(score), kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(y) + 1)
    return float((precision * ranked).sum() / p)


def metric(labels, probs, thresholds):
    item = compute_multilabel_metrics_from_probs(labels, probs, thresholds)
    return {
        "macro_f1": float(item["recognition_macro_f1"]),
        "micro_f1": float(item["recognition_micro_f1"]),
        "per_label_f1": [float(x) for x in item["per_label_f1"]],
        "per_label_precision": [float(x) for x in item["per_label_precision"]],
        "per_label_recall": [float(x) for x in item["per_label_recall"]],
    }


def evaluate(config, variant, checkpoint_path, device, batch_size):
    datasets = build_chunk_feature_datasets(config, required_splits=("valid",))
    dataset = datasets["valid"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    labels, logits, views, chunks, slots = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            features = batch["chunk_features"].to(device)
            mask = batch["chunk_mask"].to(device)
            output = model(features, mask, return_attention=True)
            labels.append(batch["multi_labels"].float().cpu())
            logits.append(output["recognition_logits"].float().cpu())
            views.append(output["view_attention"].float().cpu())
            chunks.append(output["slot_chunk_attention"].float().cpu())
            slots.append(output["slot_weights"].float().cpu())
    labels = torch.cat(labels).numpy().astype(int)
    probabilities = torch.sigmoid(torch.cat(logits)).numpy()
    threshold_report = select_per_label_thresholds(
        labels, probabilities, config["thresholds"], config["label_names"], global_threshold=0.2
    )
    thresholds = threshold_report["thresholds"]
    counts = labels.sum(axis=1)
    cardinality = {}
    for name, selected in (("single", counts == 1), ("multi_2_3", (counts >= 2) & (counts <= 3)), ("multi_4_plus", counts >= 4)):
        cardinality[name] = {"contracts": int(selected.sum()), "metrics": metric(labels[selected], probabilities[selected], thresholds) if selected.any() else None}
    view_attention = torch.cat(views).mean(dim=(0, 1, 2)).tolist()
    slot_attention = torch.cat(slots).mean(dim=(0, 1)).tolist()
    chunk_attention = torch.cat(chunks).mean(dim=(0, 1, 2)).tolist()
    per_label_score = []
    for idx, name in enumerate(config["label_names"]):
        per_label_score.append({
            "label": name,
            "roc_auc": auc(labels[:, idx], probabilities[:, idx]),
            "average_precision": average_precision(labels[:, idx], probabilities[:, idx]),
        })
    return {
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(payload.get("epoch", -1)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "fixed": metric(labels, probabilities, [0.5] * labels.shape[1]),
        "tuned": metric(labels, probabilities, thresholds),
        "thresholds": [float(x) for x in thresholds],
        "cardinality": cardinality,
        "view_attention_mean": view_attention,
        "slot_weights_mean": slot_attention,
        "slot_chunk_attention_mean": chunk_attention,
        "score_diagnostics": per_label_score,
        "dos": per_label_score[4],
        "test_checked": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_process01_090.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/main6_process01_090_mlm8/process01_analysis.json")
    args = parser.parse_args()
    config = load_config(args.config, "mlm8_slot3")
    if config.get("allow_test"):
        raise ValueError("process01 analysis is validation-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = ["mlm8_slot1", "mlm8_slot2", "mlm8_slot3", "mlm8_slot4", "shared_query_slot3", "shared_view_slot3"]
    reports = []
    missing_checkpoints = []
    for variant in variants:
        variant_config = load_config(args.config, variant)
        checkpoint = resolve(variant_config["checkpoint_dir"]) / "best_macro_f1.pt"
        if not checkpoint.exists():
            missing_checkpoints.append(str(checkpoint))
            continue
        reports.append(evaluate(variant_config, variant, checkpoint, device, args.batch_size))
    if not reports:
        checked = "\n  ".join(missing_checkpoints)
        raise FileNotFoundError(
            "No process01 checkpoints were found. Checked:\n  " + checked
        )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"route": "DIVE Main6 process01 M0 analysis", "dataset": "DIVE_main6_opcode_process01", "validation_only": True, "test_checked": False, "variants": reports, "missing_checkpoints": missing_checkpoints}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = []
    for item in reports:
        rows.append({"variant": item["variant"], "fixed_macro_f1": item["fixed"]["macro_f1"], "tuned_macro_f1": item["tuned"]["macro_f1"], "tuned_micro_f1": item["tuned"]["micro_f1"], "total_params": item["total_params"], "dos_auc": item["dos"]["roc_auc"], "dos_ap": item["dos"]["average_precision"]})
    with output.with_name("model_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "variants": len(reports), "device": str(device), "test_checked": False}, indent=2))


if __name__ == "__main__":
    main()
