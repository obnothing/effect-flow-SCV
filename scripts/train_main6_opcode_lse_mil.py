"""Train the isolated opcode Label-Specific Evidence Routing MIL route."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chunk_feature_dataset import ChunkFeatureDataset  # noqa: E402
from lse_mil_model import LabelSpecificEvidenceRoutingMIL  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path, variant):
    payload = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    config = dict(payload["common"])
    config.update(payload["variants"][variant])
    config["view_indices"] = [int(value) for value in config.get("view_indices", list(range(config.get("num_views", 8))))]
    config["num_views"] = len(config["view_indices"])
    config["variant"] = variant
    return config


def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return world, int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))


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
    return {
        "thresholds": thresholds,
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, pred, average="macro", zero_division=0)),
        "per_label_f1": f1_score(labels, pred, average=None, zero_division=0).tolist(),
        "per_label_average_precision": [float(average_precision_score(labels[:, i], probs[:, i])) for i in range(labels.shape[1])],
        "tp": (pred & labels.astype(bool)).sum(axis=0).astype(int).tolist(),
        "fp": (pred & ~labels.astype(bool)).sum(axis=0).astype(int).tolist(),
        "fn": ((~pred) & labels.astype(bool)).sum(axis=0).astype(int).tolist(),
    }


def run_epoch(model, loader, device, pos_weight, config, optimizer=None, scaler=None, train=False):
    model.train(train)
    if not train:
        model.eval()
    accumulation = int(config["gradient_accumulation_steps"])
    if train:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    logits, labels = [], []
    for step, batch in enumerate(loader):
        features = batch["chunk_features"].to(device, non_blocking=True)
        mask = batch["chunk_mask"].to(device, non_blocking=True)
        target = batch["multi_labels"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=bool(config.get("fp16", True)) and device.type == "cuda"):
                output = model(features, mask, multi_labels=target)
                bce = nn.functional.binary_cross_entropy_with_logits(
                    output["recognition_logits"], target, pos_weight=pos_weight
                )
                loss = bce + float(config["diversity_loss_weight"]) * output["diversity_loss"]
                scaled = loss / accumulation
        if train:
            scaler.scale(scaled).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().item()) * target.shape[0]
        logits.append(output["recognition_logits"].detach().float().cpu())
        labels.append(target.detach().float().cpu())
    return total_loss / max(1, len(loader.dataset)), torch.cat(logits).numpy(), torch.cat(labels).numpy()


def gather_arrays(logits, labels, world):
    # The validation loader intentionally has no DistributedSampler, so every
    # rank evaluates the complete validation split. Avoid all_gather_object:
    # object collectives use NCCL for their internal metadata exchange and can
    # deadlock on clusters where the NCCL key-value store is fragile.
    return logits, labels


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_lse_mil.yaml")
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    world, rank, local_rank = init_dist()
    config = load_config(args.config, args.variant)
    seed_all(int(config["seed"]) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    feature_dir = resolve(config["feature_dir"])
    train_set = ChunkFeatureDataset(feature_dir / "train.pt", num_labels=config["num_labels"], expected_num_views=config["num_views"], source_label_names=config["label_names"], label_names=config["label_names"], exclude_augmented_ids=True, view_indices=config["view_indices"])
    valid_set = ChunkFeatureDataset(feature_dir / "valid.pt", num_labels=config["num_labels"], expected_num_views=config["num_views"], source_label_names=config["label_names"], label_names=config["label_names"], view_indices=config["view_indices"])
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=int(config["seed"])) if world > 1 else None
    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=train_sampler is None, sampler=train_sampler, num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_set, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda")
    model = LabelSpecificEvidenceRoutingMIL(config).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)
    labels = train_set.multi_labels
    pos = labels.sum(dim=0).clamp_min(1.0)
    pos_weight = torch.sqrt((labels.shape[0] - pos) / pos).clamp(1.0, float(config["max_pos_weight"])).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    total_epochs = int(config["epochs"])
    warmup = int(config["scheduler_warmup_epochs"])
    base_lr = float(config["learning_rate"])
    min_lr = float(config["min_learning_rate"])
    def schedule(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + np.cos(np.pi * min(1, max(0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("fp16", True)) and device.type == "cuda")
    checkpoint_dir = resolve(config["checkpoint_dir"])
    result_dir = resolve(config["result_dir"])
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        save_json(result_dir / "config.json", config)
        save_json(result_dir / "pos_weight.json", {"values": pos_weight.cpu().tolist(), "source": "train_only_sqrt_ratio"})
    best = -1.0
    best_payload = None
    history = []
    patience = 0
    for epoch in range(1, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, _, _ = run_epoch(model, train_loader, device, pos_weight, config, optimizer, scaler, True)
        valid_loss, local_logits, local_labels = run_epoch(model, valid_loader, device, pos_weight, config, train=False)
        valid_logits, valid_labels = gather_arrays(local_logits, local_labels, world)
        if rank == 0:
            report = metrics(valid_logits, valid_labels, config["thresholds"])
            record = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, **report, "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(record)
            save_json(result_dir / "epoch_history.json", history)
            payload = {"schema": "main6_opcode_lse_mil_checkpoint_v1", "epoch": epoch, "model_state_dict": (model.module if world > 1 else model).state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "metrics": report}
            torch.save(payload, checkpoint_dir / "last.pt")
            if report["macro_f1"] > best:
                best = report["macro_f1"]
                best_payload = payload
                torch.save(payload, checkpoint_dir / "best_macro_f1.pt")
                patience = 0
            else:
                patience += 1
            print(f"[{args.variant}] epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} macro_f1={report['macro_f1']:.6f} micro_f1={report['micro_f1']:.6f}", flush=True)
        if world > 1:
            stop = torch.tensor([int(rank == 0 and patience >= int(config["early_stopping_patience"]))], device=device)
            dist.broadcast(stop, src=0)
            if bool(stop.item()):
                break
        elif patience >= int(config["early_stopping_patience"]):
            break
        scheduler.step()
    if rank == 0 and best_payload is not None:
        summary = {"route": config["route_name"], "variant": args.variant, "selection_source": "validation_only", "test_labels_read": False, "best_epoch": best_payload["epoch"], "valid_macro_f1": best_payload["metrics"]["macro_f1"], "valid_micro_f1": best_payload["metrics"]["micro_f1"], "valid_metrics": best_payload["metrics"], "checkpoint": str(checkpoint_dir / "best_macro_f1.pt"), "feature_cache": str(feature_dir), "view_indices": config["view_indices"], "num_views": config["num_views"], "evidence_slots": config["evidence_slots"]}
        save_json(result_dir / "valid_summary.json", summary)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
