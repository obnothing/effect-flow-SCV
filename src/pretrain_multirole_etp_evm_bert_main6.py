"""DDP entry point for train-only MLM + 17-role ETP pretraining."""

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from effect_flow_pretraining_dataset import BalancedDistributedBatchSampler
from effect_flow_schema import EFFECT_TYPES
from evm_tokenizer import EVMOpcodeTokenizer
from multirole_etp_pretraining import (
    MultiRoleETPDataset,
    MultiRoleETPModel,
    build_index,
    role_pos_weights,
    split_by_contract,
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


@torch.no_grad()
def validate(model, loader, device, role_count):
    model.eval()
    total = mom_total = etp_total = 0.0
    logits = []
    targets = []
    masks = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        total += float(outputs["loss"].item())
        mom_total += float(outputs["mom_loss"].item())
        etp_total += float(outputs["etp_loss"].item())
        logits.append(outputs["etp_logits"].cpu())
        targets.append(batch["etp_targets"].cpu())
        masks.append(batch["etp_mask"].cpu())
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    mask = torch.cat(masks).bool()
    probs = torch.sigmoid(logits)[mask]
    truth = targets[mask].bool()
    thresholds = []
    per_role = []
    for role in range(role_count):
        best = None
        for threshold in np.linspace(0.1, 0.9, 17):
            pred = probs[:, role] >= float(threshold)
            tp = int((pred & truth[:, role]).sum())
            fp = int((pred & ~truth[:, role]).sum())
            fn = int((~pred & truth[:, role]).sum())
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            candidate = (f1, precision, recall, -abs(float(threshold) - 0.5), float(threshold))
            if best is None or candidate > best:
                best = candidate
        thresholds.append(best[-1])
        per_role.append({"name": EFFECT_TYPES[role], "f1": best[0], "precision": best[1], "recall": best[2], "threshold": best[-1]})
    macro = float(np.mean([item["f1"] for item in per_role]))
    return {"loss": total / max(1, len(loader)), "mom_loss": mom_total / max(1, len(loader)), "etp_loss": etp_total / max(1, len(loader)), "etp_macro_f1": macro, "etp_thresholds": thresholds, "per_role": per_role}


def main():
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rank, world, local_rank = setup()
    set_seed(int(config["seed"]) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    source_rows = []
    all_counts = [0] * len(EFFECT_TYPES)
    valid_rows = []
    for name, path in config["train_corpora"].items():
        offsets, ids, counts = build_index(resolve(path))
        for index, count in enumerate(counts):
            all_counts[index] += count
        train_offsets, valid_offsets = split_by_contract(offsets, ids, config["internal_valid_ratio"], config["seed"])
        source_rows.append({"name": name, "path": str(resolve(path)), "offsets": train_offsets})
        valid_rows.append({"name": name, "path": str(resolve(path)), "offsets": valid_offsets})
    train_dataset = MultiRoleETPDataset(source_rows, tokenizer, config["mlm_probability"], config["seed"])
    valid_dataset = MultiRoleETPDataset(valid_rows, tokenizer, config["mlm_probability"], config["seed"])
    sampler = BalancedDistributedBatchSampler(
        [len(row["offsets"]) for row in source_rows],
        config["batch_size"],
        config["steps_per_epoch"],
        config["gradient_accumulation_steps"],
        config["seed"], rank, world,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=int(config["num_workers"]), pin_memory=True, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=int(config["eval_batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate)
    model = MultiRoleETPModel(resolve(config["base_hf_model_path"]), len(tokenizer), len(EFFECT_TYPES), role_pos_weights(all_counts, config["max_etp_pos_weight"])).to(device)
    if bool(config.get("gradient_checkpointing", True)):
        model.bert.gradient_checkpointing_enable()
    wrapped = DDP(model, device_ids=[local_rank]) if world > 1 else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    total_steps = int(config["epochs"]) * int(config["steps_per_epoch"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * float(config["warmup_ratio"])), total_steps)
    checkpoint_dir = resolve(config["checkpoint_dir"])
    result_dir = resolve(config["result_dir"])
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    history = []
    best_score = float("inf")
    best_report = None
    for epoch in range(1, int(config["epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch}"), 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = wrapped(**batch)
            (outputs["loss"] / int(config["gradient_accumulation_steps"])).backward()
            if step % int(config["gradient_accumulation_steps"]) == 0:
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), float(config["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(outputs["loss"].detach().item())
        if world > 1:
            dist.barrier()
        if rank == 0:
            report = validate(model, valid_loader, device, len(EFFECT_TYPES))
            score = report["loss"] - 0.01 * report["etp_macro_f1"]
            row = {"epoch": epoch, "train_loss": total_loss / max(1, len(train_loader)), **report}
            history.append(row)
            if score < best_score:
                best_score = score
                best_report = row
                model.save(checkpoint_dir, {"effect_type_names": EFFECT_TYPES, "etp_thresholds": report["etp_thresholds"], "seed": int(config["seed"]), "train_only": True})
        if world > 1:
            dist.barrier()
    if rank == 0:
        (result_dir / "pretraining_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        manifest = {"config": config, "effect_type_names": EFFECT_TYPES, "best": best_report, "contains_downstream_labels": False, "internal_holdout_policy": "contract_id", "seed": int(config["seed"])}
        (result_dir / "pretraining_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {checkpoint_dir / 'hf_model'}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
