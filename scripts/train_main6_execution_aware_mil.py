"""Train the isolated opcode execution-aware MIL route on train/valid only."""

import argparse
import json
import os
import random
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from execution_aware_dataset import ExecutionAwareDataset  # noqa: E402
from execution_aware_mil_model import ExecutionAwareMIL  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(logits, labels, candidates):
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    thresholds = []
    for label in range(labels.shape[1]):
        values = [(f1_score(labels[:, label], probs[:, label] >= t, zero_division=0), float(t)) for t in candidates]
        thresholds.append(max(values, key=lambda x: (x[0], -abs(x[1] - 0.5)))[1])
    pred = probs >= np.asarray(thresholds)[None, :]
    return {"thresholds": thresholds, "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)), "micro_f1": float(f1_score(labels, pred, average="micro", zero_division=0)), "macro_precision": float(precision_score(labels, pred, average="macro", zero_division=0)), "macro_recall": float(recall_score(labels, pred, average="macro", zero_division=0)), "per_label_f1": f1_score(labels, pred, average=None, zero_division=0).tolist(), "per_label_average_precision": [float(average_precision_score(labels[:, i], probs[:, i])) for i in range(labels.shape[1])], "tp": (pred & labels.astype(bool)).sum(0).astype(int).tolist(), "fp": (pred & ~labels.astype(bool)).sum(0).astype(int).tolist(), "fn": ((~pred) & labels.astype(bool)).sum(0).astype(int).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_execution_aware_mil.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    seed_all(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic = resolve(config["semantic_feature_dir"])
    execution = resolve(config["execution_feature_dir"])
    common = dict(num_labels=config["num_labels"], expected_num_views=config["num_views"], source_label_names=config["label_names"], label_names=config["label_names"], exclude_augmented_ids=False)
    train_set = ExecutionAwareDataset(semantic / "train.pt", execution / "train.pt", **common)
    valid_set = ExecutionAwareDataset(semantic / "valid.pt", execution / "valid.pt", **common)
    loader = DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_set, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda")
    model = ExecutionAwareMIL(config).to(device)
    labels = train_set.semantic.multi_labels
    positive = labels.sum(0).clamp_min(1)
    pos_weight = torch.sqrt((labels.shape[0] - positive) / positive).clamp(1, float(config["max_pos_weight"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    total_epochs = int(config["epochs"])
    warmup = int(config["scheduler_warmup_epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / max(1, warmup) if epoch < warmup else 0.5 * (1 + np.cos(np.pi * min(1, (epoch - warmup) / max(1, total_epochs - warmup)))))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["fp16"]) and device.type == "cuda")
    checkpoint_dir, result_dir = resolve(config["checkpoint_dir"]), resolve(config["result_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (result_dir / "pos_weight.json").write_text(json.dumps({"values": pos_weight.cpu().tolist(), "source": "train_only_sqrt_ratio"}, indent=2), encoding="utf-8")
    best, patience, history = -1.0, 0, []
    for epoch in range(1, total_epochs + 1):
        model.train(); train_total = 0.0; optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            features = batch["chunk_features"].to(device); execution_features = batch["execution_features"].to(device); mask = batch["chunk_mask"].to(device); target = batch["multi_labels"].to(device)
            with torch.cuda.amp.autocast(enabled=bool(config["fp16"]) and device.type == "cuda"):
                output = model(features, execution_features, mask)
                loss = nn.functional.binary_cross_entropy_with_logits(output["recognition_logits"], target, pos_weight=pos_weight) + float(config["diversity_loss_weight"]) * output["diversity_loss"]
                loss = loss / int(config["gradient_accumulation_steps"])
            scaler.scale(loss).backward()
            if (step + 1) % int(config["gradient_accumulation_steps"]) == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            train_total += float(loss.detach()) * int(config["gradient_accumulation_steps"]) * target.shape[0]
        model.eval(); val_total = 0.0; logits, labels_out, gates = [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                output = model(batch["chunk_features"].to(device), batch["execution_features"].to(device), batch["chunk_mask"].to(device))
                target = batch["multi_labels"].to(device); val_total += float(nn.functional.binary_cross_entropy_with_logits(output["recognition_logits"], target, pos_weight=pos_weight)) * target.shape[0]; logits.append(output["recognition_logits"].cpu()); labels_out.append(target.cpu()); gates.append(output["execution_gate"].mean((0, 1)).cpu())
        valid_logits, valid_labels = torch.cat(logits).numpy(), torch.cat(labels_out).numpy(); report = metrics(valid_logits, valid_labels, config["thresholds"]); report["mean_execution_gate"] = torch.cat(gates).mean(0).tolist()
        record = {"epoch": epoch, "train_loss": train_total / len(train_set), "valid_loss": val_total / len(valid_set), "learning_rate": optimizer.param_groups[0]["lr"], **report}; history.append(record); (result_dir / "epoch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        payload = {"schema": "main6_opcode_execution_aware_mil_v1", "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "metrics": report}; torch.save(payload, checkpoint_dir / "last.pt")
        if report["macro_f1"] > best:
            best, patience = report["macro_f1"], 0; torch.save(payload, checkpoint_dir / "best_macro_f1.pt")
        else:
            patience += 1
        print(f"[execution_aware] epoch={epoch} train_loss={record['train_loss']:.6f} valid_loss={record['valid_loss']:.6f} macro_f1={report['macro_f1']:.6f} micro_f1={report['micro_f1']:.6f}", flush=True)
        scheduler.step()
        if patience >= int(config["early_stopping_patience"]): break
    checkpoint_hash = hashlib.sha256((checkpoint_dir / "best_macro_f1.pt").read_bytes()).hexdigest()
    best_record = max(history, key=lambda item: item["macro_f1"])
    summary = {"route": config["route_name"], "selection_source": "validation_only", "test_labels_read": False, "valid_macro_f1": best, "valid_micro_f1": best_record["micro_f1"], "valid_metrics": best_record, "checkpoint": str(checkpoint_dir / "best_macro_f1.pt"), "checkpoint_sha256": checkpoint_hash, "feature_cache": str(execution), "semantic_cache": str(semantic)}
    (result_dir / "valid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
