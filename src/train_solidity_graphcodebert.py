"""DDP training for the audited DIVE Source-Main6 GraphCodeBERT route."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from metrics import compute_multilabel_metrics_from_probs, sigmoid
from solidity_graph_dataset import SolidityGraphDataset, collate_solidity_graph
from solidity_graphcodebert_model import SolidityGraphCodeBERTMultiSlotMIL, multilabel_supervised_contrastive, symmetric_kl

ROOT = Path(__file__).resolve().parents[1]


def load_config(path, variant):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = dict(payload["common"])
    config.update(payload["variants"][variant])
    config["variant"] = variant
    return config


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return int(os.environ["RANK"]), world_size, torch.device("cuda", local_rank)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def pos_weight(dataset, mode, max_weight):
    labels = torch.stack([row["multi_labels"] for row in dataset.rows])
    positives, total = labels.sum(dim=0), float(labels.shape[0])
    ratio = (total - positives) / positives.clamp_min(1)
    values = ratio if mode == "ratio" else torch.sqrt(ratio)
    return values.clamp(min=1, max=max_weight)


def evaluate(model, loader, device, thresholds):
    model.eval(); logits, labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            output = model(**{key: batch[key] for key in ("input_ids", "token_mask", "graph_mask", "unit_mask")})
            logits.append(output["logits"].cpu()); labels.append(batch["multi_labels"].cpu())
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(output["logits"], batch["multi_labels"]).item())
    logits, labels = torch.cat(logits).numpy(), torch.cat(labels).numpy()
    probs = sigmoid(logits)
    metrics = compute_multilabel_metrics_from_probs(labels, probs, thresholds)
    metrics["mean_pr_auc"] = float(np.mean([
        average_precision_score(labels[:, index], probs[:, index])
        for index in range(labels.shape[1]) if len(np.unique(labels[:, index])) > 1
    ]))
    return {"loss": float(np.mean(losses)), "logits": logits, "labels": labels, "probs": probs, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--variant", required=True)
    args = parser.parse_args(); config = load_config(args.config, args.variant)
    rank, world_size, device = setup_distributed(); set_seed(int(config["seed"]) + rank)
    cache_dir = ROOT / config["graph_cache_dir"]
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("Refusing validation training because a locked test source cache exists")
    train_set, valid_set = SolidityGraphDataset(cache_dir / "train.pt"), SolidityGraphDataset(cache_dir / "valid.pt")
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=int(config["seed"])) if world_size > 1 else None
    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]), sampler=sampler, shuffle=sampler is None,
                              num_workers=int(config["num_workers"]), pin_memory=True, collate_fn=collate_solidity_graph)
    valid_loader = DataLoader(valid_set, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), pin_memory=True, collate_fn=collate_solidity_graph)
    model = SolidityGraphCodeBERTMultiSlotMIL(config); model.apply_lora(config); model.to(device)
    wrapped = DDP(model, device_ids=[device.index]) if world_size > 1 else model
    weights = pos_weight(train_set, config["pos_weight_mode"], float(config["max_pos_weight"])).to(device)
    optimizer = torch.optim.AdamW([item for item in wrapped.parameters() if item.requires_grad], lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    warmup, total = int(config["warmup_epochs"]), int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(1.0, (epoch + 1) / max(1, warmup)) if epoch < warmup else max(float(config["min_learning_rate"]) / float(config["learning_rate"]), 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, total - warmup)))) )
    checkpoint_dir, result_dir = ROOT / config["checkpoint_dir"], ROOT / config["result_dir"]
    if rank == 0: checkpoint_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1: dist.barrier()
    best, patience, history = -1.0, 0, []
    for epoch in range(1, total + 1):
        if sampler: sampler.set_epoch(epoch)
        wrapped.train(); optimizer.zero_grad(set_to_none=True); train_losses = []
        for step, batch in enumerate(train_loader, 1):
            batch = move(batch, device); kwargs = {key: batch[key] for key in ("input_ids", "token_mask", "graph_mask", "unit_mask")}
            first = wrapped(**kwargs); bce = torch.nn.functional.binary_cross_entropy_with_logits(first["logits"], batch["multi_labels"], pos_weight=weights)
            contrast = multilabel_supervised_contrastive(first["projection"], batch["multi_labels"], config["contrastive_temperature"]) if config["use_contrastive"] else bce * 0.0
            rdrop = bce * 0.0
            if config["use_rdrop"]:
                second = wrapped(**kwargs); rdrop = symmetric_kl(first["logits"], second["logits"])
            loss = bce + float(config["contrastive_weight"]) * contrast + float(config["rdrop_weight"]) * rdrop
            (loss / int(config["gradient_accumulation_steps"])).backward(); train_losses.append(float(loss.detach()))
            if step % int(config["gradient_accumulation_steps"]) == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            result = evaluate(model, valid_loader, device, [0.5] * int(config["num_labels"]))
            metric = float(result["metrics"]["recognition_macro_f1"])
            record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "valid_loss": result["loss"], "valid_macro_f1": metric, "valid_micro_f1": result["metrics"]["recognition_micro_f1"], "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(record); print(json.dumps(record))
            if metric > best:
                best, patience = metric, 0
                torch.save({"epoch": epoch, "config": config, "model_state_dict": model.state_dict(), "best_metrics": result["metrics"]}, checkpoint_dir / "best_macro_f1.pt")
            else: patience += 1
            (result_dir / "epoch_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        scheduler.step()
        stop = torch.tensor([patience >= int(config["early_stopping_patience"])] if rank == 0 else [False], device=device)
        if world_size > 1: dist.broadcast(stop, 0)
        if bool(stop.item()): break
    if rank == 0:
        checkpoint = torch.load(checkpoint_dir / "best_macro_f1.pt", map_location="cpu")
        summary = {"experiment_name": f"dive_source_main6_{args.variant}", "variant": args.variant, "best_macro_f1_value": float(checkpoint["best_metrics"]["recognition_macro_f1"]), "best_micro_f1_value": float(checkpoint["best_metrics"]["recognition_micro_f1"]), "mean_pr_auc": float(checkpoint["best_metrics"]["mean_pr_auc"]), "best_macro_f1_epoch": int(checkpoint["epoch"]), "selection_source": "validation_only", "test_labels_read": False, "per_label_f1": checkpoint["best_metrics"]["per_label_f1"]}
        (result_dir / "checkpoint_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if world_size > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
