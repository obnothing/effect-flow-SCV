"""Validation-only calibration and decision-boundary diagnostics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from execution_aware_dataset import ExecutionAwareDataset  # noqa: E402
from execution_aware_mil_model import ExecutionAwareMIL  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def brier(probs, labels):
    return np.mean((probs - labels) ** 2, axis=0)


def ece(probs, labels, bins=10):
    result = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for label in range(labels.shape[1]):
        total = 0.0
        for left, right in zip(edges[:-1], edges[1:]):
            selected = (probs[:, label] >= left) & (
                probs[:, label] < right if right < 1.0 else probs[:, label] <= right
            )
            if selected.any():
                total += float(selected.mean()) * abs(
                    float(probs[selected, label].mean())
                    - float(labels[selected, label].mean())
                )
        result.append(total)
    return np.asarray(result)


def threshold_report(probs, labels, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)
    pred = probs >= thresholds[None, :]
    return {
        "thresholds": thresholds.tolist(),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, pred, average="micro", zero_division=0)),
        "per_label_f1": f1_score(labels, pred, average=None, zero_division=0).tolist(),
        "tp": (pred & labels.astype(bool)).sum(0).astype(int).tolist(),
        "fp": (pred & ~labels.astype(bool)).sum(0).astype(int).tolist(),
        "fn": ((~pred) & labels.astype(bool)).sum(0).astype(int).tolist(),
    }


def probability_report(probs, labels):
    positives = []
    negatives = []
    for label in range(labels.shape[1]):
        pos = labels[:, label] > 0.5
        neg = ~pos
        positives.append({
            "count": int(pos.sum()),
            "mean": float(probs[pos, label].mean()) if pos.any() else None,
            "p10": float(np.percentile(probs[pos, label], 10)) if pos.any() else None,
            "p50": float(np.percentile(probs[pos, label], 50)) if pos.any() else None,
            "p90": float(np.percentile(probs[pos, label], 90)) if pos.any() else None,
        })
        negatives.append({
            "count": int(neg.sum()),
            "mean": float(probs[neg, label].mean()) if neg.any() else None,
            "p10": float(np.percentile(probs[neg, label], 10)) if neg.any() else None,
            "p50": float(np.percentile(probs[neg, label], 50)) if neg.any() else None,
            "p90": float(np.percentile(probs[neg, label], 90)) if neg.any() else None,
        })
    return {"positive": positives, "negative": negatives}


def evaluate_checkpoint(model, checkpoint, loader, device, labels):
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["chunk_features"].to(device),
                batch["execution_features"].to(device),
                batch["chunk_mask"].to(device),
            )
            logits.append(output["recognition_logits"].float().cpu())
    logits = torch.cat(logits).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    tuned = []
    candidates = np.arange(0.05, 1.0, 0.05)
    for label in range(labels.shape[1]):
        scores = [
            (f1_score(labels[:, label], probs[:, label] >= threshold, zero_division=0), threshold)
            for threshold in candidates
        ]
        tuned.append(max(scores, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1])
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "valid_loss_unweighted_bce": float(
            torch.nn.functional.binary_cross_entropy(
                torch.from_numpy(probs), torch.from_numpy(labels)
            )
        ),
        "pr_auc": [float(average_precision_score(labels[:, i], probs[:, i])) for i in range(labels.shape[1])],
        "fixed_threshold_0_5": threshold_report(probs, labels, [0.5] * labels.shape[1]),
        "reoptimized_valid_threshold": threshold_report(probs, labels, tuned),
        "brier_score": brier(probs, labels).tolist(),
        "ece_10_bins": ece(probs, labels).tolist(),
        "probability_distribution": probability_report(probs, labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_execution_aware_mil.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    result_dir = resolve(config["result_dir"])
    checkpoint_dir = resolve(config["checkpoint_dir"])
    semantic = resolve(config["semantic_feature_dir"])
    execution = resolve(config["execution_feature_dir"])
    common = dict(
        num_labels=config["num_labels"],
        expected_num_views=config["num_views"],
        source_label_names=config["label_names"],
        label_names=config["label_names"],
    )
    dataset = ExecutionAwareDataset(semantic / "valid.pt", execution / "valid.pt", **common)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]))
    labels = dataset.semantic.multi_labels.numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExecutionAwareMIL(config).to(device)
    checkpoints = [checkpoint_dir / name for name in ("best_macro_f1.pt", "last.pt") if (checkpoint_dir / name).exists()]
    reports = [evaluate_checkpoint(model, path, loader, device, labels) for path in checkpoints]
    history_path = result_dir / "epoch_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    threshold_history = [
        {"epoch": item.get("epoch"), "thresholds": item.get("thresholds"), "macro_f1": item.get("macro_f1"), "valid_loss": item.get("valid_loss")}
        for item in history
    ]
    output = {
        "route": config["route_name"],
        "split": "valid",
        "test_labels_read": False,
        "labels": config["label_names"],
        "checkpoint_reports": reports,
        "threshold_history": threshold_history,
        "interpretation": {
            "fixed_0_5_vs_reoptimized": "Compare these to distinguish probability drift from decision-boundary changes.",
            "brier_and_ece": "Lower is better; increases indicate calibration degradation.",
            "pr_auc": "Ranking quality is threshold-independent; compare across checkpoints.",
        },
    }
    output_path = result_dir / "valid_calibration_diagnostics.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
