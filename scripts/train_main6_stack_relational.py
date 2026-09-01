"""Train the isolated stack-relational input route on train/valid only."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from stack_aware_mil_model import StackAwareMLM8MIL  # noqa: E402
from stack_relation_dataset import StackRelationDataset, collate_stack_relation  # noqa: E402
from stack_relation_feature_dataset import StackRelationFeatureDataset, collate_stack_relation_features  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metric_report(logits, labels, candidates):
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    thresholds = []
    for label in range(labels.shape[1]):
        values = [(f1_score(labels[:, label], probs[:, label] >= threshold, zero_division=0), float(threshold)) for threshold in candidates]
        thresholds.append(max(values, key=lambda x: (x[0], -abs(x[1] - 0.5)))[1])
    def score(threshold_values):
        pred = probs >= np.asarray(threshold_values)[None, :]
        return {
            "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(labels, pred, average="micro", zero_division=0)),
            "per_label_f1": f1_score(labels, pred, average=None, zero_division=0).tolist(),
            "tp": (pred & labels.astype(bool)).sum(0).astype(int).tolist(),
            "fp": (pred & ~labels.astype(bool)).sum(0).astype(int).tolist(),
            "fn": ((~pred) & labels.astype(bool)).sum(0).astype(int).tolist(),
        }
    tuned = score(thresholds)
    fixed = score([0.5] * labels.shape[1])
    return {
        "thresholds": thresholds,
        **tuned,
        "fixed_thresholds": [0.5] * labels.shape[1],
        "fixed_macro_f1": fixed["macro_f1"],
        "fixed_micro_f1": fixed["micro_f1"],
        "fixed_per_label_f1": fixed["per_label_f1"],
        "per_label_average_precision": [float(average_precision_score(labels[:, i], probs[:, i])) for i in range(labels.shape[1])],
    }


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_model(model, batch, use_cached):
    """Call the matching interface for cached MIL or live encoder mode."""
    if use_cached:
        # Encoder caches are stored as fp16 to reduce disk/RAM usage.  The
        # cached MIL path also runs during validation without autocast, while
        # its LayerNorm parameters are fp32; normalize the input dtype here.
        chunk_features = batch["chunk_features"].float()
        return model(
            chunk_features,
            batch["chunk_mask"],
            multi_labels=None,
        )
    return model(batch)


def evaluate(model, loader, device, pos_weight, config):
    model.eval(); losses = []; logits = []; labels = []
    use_cached = bool(config.get("use_cached_encoder_features", False))
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = forward_model(model, batch, use_cached)
            target = batch["multi_labels"]
            losses.append(float(nn.functional.binary_cross_entropy_with_logits(output["recognition_logits"], target, pos_weight=pos_weight)))
            logits.append(output["recognition_logits"].float().cpu()); labels.append(target.float().cpu())
    all_logits, all_labels = torch.cat(logits).numpy(), torch.cat(labels).numpy()
    report = metric_report(all_logits, all_labels, config["thresholds"])
    report["loss"] = float(np.mean(losses))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_relational.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    seed_all(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dir = resolve(config["feature_dir"])
    use_cached = bool(config.get("use_cached_encoder_features", False))
    common = {"num_labels": 6, "label_names": config["label_names"]}
    if use_cached:
        cache_dir = resolve(config["encoder_feature_dir"])
        train_set = StackRelationFeatureDataset(cache_dir / "train.pt", **common)
        valid_set = StackRelationFeatureDataset(cache_dir / "valid.pt", **common)
        loader = DataLoader(train_set, batch_size=int(config.get("cached_batch_size", 16)), shuffle=True, num_workers=int(config["num_workers"]), collate_fn=collate_stack_relation_features, pin_memory=device.type == "cuda")
        valid_loader = DataLoader(valid_set, batch_size=int(config.get("cached_batch_size", 16)), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate_stack_relation_features, pin_memory=device.type == "cuda")
        from evm_chunk_mil_model import MLM8ViewMultiSlotMIL
        model = MLM8ViewMultiSlotMIL(config).to(device)
    else:
        train_set = StackRelationDataset(feature_dir / "train.pt", **common)
        valid_set = StackRelationDataset(feature_dir / "valid.pt", **common)
        loader = DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]), collate_fn=collate_stack_relation, pin_memory=device.type == "cuda")
        valid_loader = DataLoader(valid_set, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate_stack_relation, pin_memory=device.type == "cuda")
        model = StackAwareMLM8MIL(config).to(device)
        # The base language model stays frozen, but stack injection parameters
        # remain trainable in the legacy live-encoder diagnostic mode.
        for layer in model.encoder.base.bert.encoder.layer:
            layer.attention.self.relation_scale.requires_grad = True
    labels = train_set.multi_labels
    positive = labels.sum(0).clamp_min(1)
    pos_weight = torch.sqrt((labels.shape[0] - positive) / positive).clamp(1, float(config["max_pos_weight"])).to(device)
    learning_rate = float(config.get("cached_learning_rate", config["learning_rate"])) if use_cached else float(config["learning_rate"])
    weight_decay = float(config.get("cached_weight_decay", config["weight_decay"])) if use_cached else float(config["weight_decay"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    epochs = int(config.get("cached_epochs", config["epochs"])) if use_cached else int(config["epochs"])
    warmup = int(config["scheduler_warmup_epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / max(1, warmup) if epoch < warmup else 0.5 * (1 + np.cos(np.pi * min(1, (epoch - warmup) / max(1, epochs - warmup)))))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["fp16"]) and device.type == "cuda")
    checkpoint_dir = resolve(config.get("cached_checkpoint_dir", config["checkpoint_dir"])) if use_cached else resolve(config["checkpoint_dir"])
    result_dir = resolve(config.get("cached_result_dir", config["result_dir"])) if use_cached else resolve(config["result_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (result_dir / "pos_weight.json").write_text(json.dumps({"values": pos_weight.cpu().tolist(), "source": "train_only_sqrt_ratio"}, indent=2), encoding="utf-8")
    selection_metric = str(config.get("selection_metric", "macro_f1"))
    patience_limit = int(config.get("cached_early_stopping_patience", config["early_stopping_patience"])) if use_cached else int(config["early_stopping_patience"])
    best, patience, history, best_record = -1.0, 0, [], None
    for epoch in range(1, epochs + 1):
        model.train(); running = 0.0; seen = 0; optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(loader, 1):
            batch = move_batch(raw_batch, device)
            with torch.cuda.amp.autocast(enabled=bool(config["fp16"]) and device.type == "cuda"):
                output = forward_model(model, batch, use_cached)
                loss = nn.functional.binary_cross_entropy_with_logits(output["recognition_logits"], batch["multi_labels"], pos_weight=pos_weight)
                scaled = loss / int(config["gradient_accumulation_steps"])
            scaler.scale(scaled).backward()
            running += float(loss.detach()) * batch["multi_labels"].shape[0]; seen += batch["multi_labels"].shape[0]
            if step % int(config["gradient_accumulation_steps"]) == 0 or step == len(loader):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        valid = evaluate(model, valid_loader, device, pos_weight, config)
        record = {"epoch": epoch, "train_loss": running / max(1, seen), "valid_loss": valid.pop("loss"), "learning_rate": optimizer.param_groups[0]["lr"], **valid}
        history.append(record); (result_dir / "epoch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        payload = {"schema": "main6_opcode_stack_relational_v2", "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "metrics": record}
        torch.save(payload, checkpoint_dir / "last.pt")
        current_score = float(record[selection_metric])
        if current_score > best:
            best, patience, best_record = current_score, 0, record; torch.save(payload, checkpoint_dir / "best_macro_f1.pt")
        else:
            patience += 1
        print(f"[stack_relational] epoch={epoch} train_loss={record['train_loss']:.6f} valid_loss={record['valid_loss']:.6f} macro_f1={record['macro_f1']:.6f} fixed_macro_f1={record['fixed_macro_f1']:.6f} micro_f1={record['micro_f1']:.6f}", flush=True)
        scheduler.step()
        if patience >= patience_limit: break
    checkpoint = checkpoint_dir / "best_macro_f1.pt"
    summary = {"route": config["route_name"], "selection_source": "validation_only", "selection_metric": selection_metric, "test_labels_read": False, "valid_macro_f1": float(best_record["macro_f1"]), "valid_micro_f1": float(best_record["micro_f1"]), "fixed_valid_macro_f1": float(best_record["fixed_macro_f1"]), "best_epoch": int(best_record["epoch"]), "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "feature_cache": str(resolve(config["encoder_feature_dir"]) if use_cached else feature_dir), "baseline_summary": config["baseline_summary"]}
    (result_dir / "valid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
