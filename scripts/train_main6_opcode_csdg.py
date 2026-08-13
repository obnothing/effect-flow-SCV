"""Train one validation-only Opcode-CSDG residual candidate."""

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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evm_opcode_graph_dataset import OpcodeGraphSequenceDataset, collate_opcode_graph, move_graph_batch  # noqa: E402
from evm_opcode_graph_residual_mil import OpcodeGraphResidualMIL  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path, variant):
    payload = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    config = dict(payload.get("common", {}))
    config.update(payload.get("variants", {}).get(variant, {}))
    config["variant"] = variant
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return world, int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))


def is_main(rank):
    return rank == 0


def compute_pos_weight(dataset, maximum):
    labels = dataset.multi_labels
    positive = labels.sum(dim=0).clamp_min(1.0)
    negative = labels.shape[0] - positive
    return torch.sqrt(negative / positive).clamp(min=1.0, max=float(maximum))


def thresholds_and_metrics(logits, labels, candidates):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    thresholds = []
    for label_id in range(labels.shape[1]):
        best = (-1.0, 0.5)
        for threshold in candidates:
            score = f1_score(labels[:, label_id], probabilities[:, label_id] >= threshold, zero_division=0)
            if score > best[0]:
                best = (float(score), float(threshold))
        thresholds.append(best[1])
    predictions = probabilities >= np.asarray(thresholds)[None, :]
    per_f1 = f1_score(labels, predictions, average=None, zero_division=0).tolist()
    return {
        "thresholds": thresholds,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "per_label_f1": per_f1,
        "per_label_precision": precision_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_label_recall": recall_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_label_support": labels.sum(axis=0).astype(int).tolist(),
        "per_label_average_precision": [float(average_precision_score(labels[:, i], probabilities[:, i])) for i in range(labels.shape[1])],
        "tp": (predictions & labels.astype(bool)).sum(axis=0).astype(int).tolist(),
        "fp": (predictions & ~labels.astype(bool)).sum(axis=0).astype(int).tolist(),
        "fn": ((~predictions) & labels.astype(bool)).sum(axis=0).astype(int).tolist(),
    }


def run_epoch(model, loader, optimizer, scaler, device, pos_weight, config, train, world, epoch, collect_logits=False):
    model.train(train)
    if not train:
        model.eval()
    total_loss = 0.0
    total_count = 0
    logits_list, labels_list = [], []
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = move_graph_batch(batch, device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=bool(config.get("fp16", True)) and device.type == "cuda"):
                output = model(
                    batch["sequence_features"], batch["sequence_mask"], batch["node_features"],
                    batch["node_mask"], batch["edge_index"], batch["edge_type"], return_attention=False,
                )
                final_loss = nn.functional.binary_cross_entropy_with_logits(
                    output["recognition_logits"], batch["multi_labels"], pos_weight=pos_weight
                )
                graph_loss = nn.functional.binary_cross_entropy_with_logits(
                    output["graph_logits"], batch["multi_labels"], pos_weight=pos_weight
                )
                if train and epoch <= int(config.get("warmup_graph_epochs", 0)):
                    loss = graph_loss
                else:
                    loss = final_loss + float(config.get("graph_auxiliary_loss_weight", 0.2)) * graph_loss
                scaled_loss = loss / accumulation
        if train:
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().item()) * batch["multi_labels"].shape[0]
        total_count += batch["multi_labels"].shape[0]
        if collect_logits:
            logits_list.append(output["recognition_logits"].detach().cpu())
            labels_list.append(batch["multi_labels"].detach().cpu())
    loss_stats = torch.tensor([total_loss, float(total_count)], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
    loss_value = float((loss_stats[0] / loss_stats[1].clamp_min(1.0)).item())
    if not collect_logits:
        return loss_value, None, None
    return loss_value, torch.cat(logits_list, dim=0).numpy(), torch.cat(labels_list, dim=0).numpy()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    world, rank, local_rank = init_distributed()
    config = load_config(args.config, args.variant)
    seed_everything(int(config.get("seed", 42)) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_graph = resolve(config["graph_cache_dir"]) / "train.pt"
    valid_graph = resolve(config["graph_cache_dir"]) / "valid.pt"
    train_sequence = resolve(config["sequence_feature_dir"]) / "train.pt"
    valid_sequence = resolve(config["sequence_feature_dir"]) / "valid.pt"
    train_set = OpcodeGraphSequenceDataset(train_graph, train_sequence, config["label_names"])
    valid_set = OpcodeGraphSequenceDataset(valid_graph, valid_sequence, config["label_names"])
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=int(config.get("seed", 42))) if world > 1 else None
    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=train_sampler is None, sampler=train_sampler, num_workers=int(config.get("num_workers", 0)), pin_memory=device.type == "cuda", collate_fn=collate_opcode_graph)
    valid_loader = DataLoader(valid_set, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)), pin_memory=device.type == "cuda", collate_fn=collate_opcode_graph)
    model = OpcodeGraphResidualMIL(config).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable graph parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.01)))
    warmup_epochs = int(config.get("scheduler_warmup_epochs", 0))
    total_epochs = int(config["epochs"])
    min_lr = float(config.get("min_learning_rate", 1e-6))
    base_lr = float(config["learning_rate"])
    def lr_lambda(step):
        if warmup_epochs > 0 and step < warmup_epochs:
            return float(step + 1) / float(warmup_epochs)
        progress = (step - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))
        return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("fp16", True)) and device.type == "cuda")
    pos_weight = compute_pos_weight(train_set, config.get("max_pos_weight", 5.0)).to(device)
    checkpoint_dir = resolve(config["checkpoint_dir"])
    result_dir = resolve(config["result_dir"])
    if is_main(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        save_json(result_dir / "config.json", config)
        save_json(result_dir / "pos_weight.json", {"values": pos_weight.cpu().tolist(), "source": "train_only_sqrt_ratio"})
    best_macro = -1.0
    best_payload = None
    history = []
    patience = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, scaler, device, pos_weight, config, True, world, epoch)
        valid_loss, valid_logits, valid_labels = run_epoch(model, valid_loader, optimizer, scaler, device, pos_weight, config, False, world, epoch, collect_logits=is_main(rank))
        if is_main(rank):
            metrics = thresholds_and_metrics(valid_logits, valid_labels, config["thresholds"])
            record = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, **metrics, "learning_rate": optimizer.param_groups[0]["lr"], "residual_gate": (model.module if world > 1 else model).alpha.detach().cpu().tanh().tolist()}
            history.append(record)
            save_json(result_dir / "epoch_history.json", history)
            payload = {
                "schema": "main6_opcode_csdg_checkpoint_v1",
                "epoch": epoch,
                "model_state_dict": (model.module if world > 1 else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "metrics": metrics,
            }
            torch.save(payload, checkpoint_dir / "last.pt")
            if metrics["macro_f1"] > best_macro:
                best_macro = metrics["macro_f1"]
                best_payload = payload
                torch.save(payload, checkpoint_dir / "best_macro_f1.pt")
                patience = 0
            else:
                patience += 1
            print(f"[{config['variant']}] epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} macro_f1={metrics['macro_f1']:.6f} micro_f1={metrics['micro_f1']:.6f}")
        if world > 1:
            stop = torch.tensor([int(is_main(rank) and patience >= int(config["early_stopping_patience"]))], device=device, dtype=torch.int32)
            dist.broadcast(stop, src=0)
            if bool(stop.item()):
                break
        elif patience >= int(config["early_stopping_patience"]):
            break
        scheduler.step()
    if is_main(rank):
        if best_payload is None:
            raise RuntimeError("No validation checkpoint was produced")
        summary = {
            "route": config["route_name"],
            "variant": config["variant"],
            "selection_source": "validation_only",
            "test_labels_read": False,
            "best_epoch": best_payload["epoch"],
            "valid_macro_f1": best_payload["metrics"]["macro_f1"],
            "valid_micro_f1": best_payload["metrics"]["micro_f1"],
            "valid_metrics": best_payload["metrics"],
            "checkpoint": str(checkpoint_dir / "best_macro_f1.pt"),
            "sequence_checkpoint": config["sequence_checkpoint"],
            "residual_gate": (model.module if world > 1 else model).alpha.detach().cpu().tanh().tolist(),
        }
        save_json(result_dir / "valid_summary.json", summary)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
