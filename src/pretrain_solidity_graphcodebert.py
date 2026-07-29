"""Train-only Solidity domain-adaptive MLM for GraphCodeBERT."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

from solidity_graph_dataset import SolidityGraphDataset, build_graph_attention
from solidity_source_v2_dataset import SoliditySourceV2Dataset

ROOT = Path(__file__).resolve().parents[1]


class UnitMLMDataset(Dataset):
    def __init__(self, cache_path, max_windows_per_contract=None):
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("schema") == "solidity_source_windows_v2":
            self.units = []
            for row in SoliditySourceV2Dataset(cache_path).rows:
                count = row["input_ids"].shape[0]
                limit = min(count, int(max_windows_per_contract or count))
                indices = sorted({round(i * (count - 1) / max(1, limit - 1)) for i in range(limit)})
                self.units.extend((row["input_ids"][unit], row["token_mask"][unit], None, None, None, None) for unit in indices)
        else:
            rows = SolidityGraphDataset(cache_path).rows
            self.units = [(row["input_ids"][unit], row["token_mask"][unit], row["code_ends"][unit],
                           row["node_to_code"][unit], row["edge_src"][unit], row["edge_dst"][unit])
                          for row in rows for unit in range(row["input_ids"].shape[0]) if row["unit_mask"][unit]]
    def __len__(self): return len(self.units)
    def __getitem__(self, index): return self.units[index]


def collate(batch):
    ids, mask, code_ends, nodes, edge_src, edge_dst = zip(*batch)
    token_mask = torch.stack(mask)
    if code_ends[0] is None:
        return {"input_ids": torch.stack(ids), "token_mask": token_mask, "attention_mask": token_mask}
    return {"input_ids": torch.stack(ids), "token_mask": token_mask,
            "attention_mask": build_graph_attention(token_mask.unsqueeze(1), torch.stack(code_ends).unsqueeze(1), torch.stack(nodes).unsqueeze(1), torch.stack(edge_src).unsqueeze(1), torch.stack(edge_dst).unsqueeze(1)).squeeze(1)}


def mask_tokens(input_ids, attention_mask, tokenizer, probability):
    labels = input_ids.clone(); candidates = attention_mask.bool()
    for special in tokenizer.all_special_ids:
        candidates &= input_ids.ne(special)
    selected = torch.bernoulli(torch.full(labels.shape, probability, device=labels.device)).bool() & candidates
    labels[~selected] = -100
    replace = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & selected
    input_ids = input_ids.clone(); input_ids[replace] = tokenizer.mask_token_id
    random_replace = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & selected & ~replace
    input_ids[random_replace] = torch.randint(len(tokenizer), labels.shape, device=labels.device)[random_replace]
    return input_ids, labels


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    asset_manifest = ROOT / config["base_model_asset_manifest"]
    if not asset_manifest.exists() or json.loads(asset_manifest.read_text(encoding="utf-8")).get("revision") != config["base_model_revision"]:
        raise RuntimeError("Pinned GraphCodeBERT asset manifest is missing or has the wrong revision")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
    if world > 1: torch.cuda.set_device(device); dist.init_process_group("nccl")
    seed = int(config["seed"]) + rank; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    base = ROOT / config["base_model_path"]; dataset = UnitMLMDataset(ROOT / config["graph_cache_dir"] / "train.pt", config.get("dapt_max_windows_per_contract"))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=int(config["seed"])) if world > 1 else None
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), sampler=sampler, shuffle=sampler is None, num_workers=int(config["num_workers"]), pin_memory=True, collate_fn=collate)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True); model = AutoModelForMaskedLM.from_pretrained(base, local_files_only=True).to(device)
    wrapped = DDP(model, device_ids=[device.index]) if world > 1 else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    total_steps = max(1, (len(loader) * int(config["epochs"])) // int(config["gradient_accumulation_steps"])); scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * float(config["warmup_ratio"])), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["fp16"]) and device.type == "cuda"); losses = []; optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(config["epochs"]) + 1):
        if sampler: sampler.set_epoch(epoch)
        wrapped.train()
        for step, batch in enumerate(loader, 1):
            ids, token_mask, graph_mask = batch["input_ids"].to(device), batch["token_mask"].to(device), batch["attention_mask"].to(device)
            ids, labels = mask_tokens(ids, token_mask, tokenizer, float(config["mlm_probability"]))
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = wrapped(input_ids=ids, attention_mask=graph_mask.long(), labels=labels).loss
            scaler.scale(loss / int(config["gradient_accumulation_steps"])).backward(); losses.append(float(loss.detach()))
            if step % int(config["gradient_accumulation_steps"]) == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        if rank == 0: print(json.dumps({"epoch": epoch, "train_mlm_loss": float(np.mean(losses[-len(loader):]))}))
    if rank == 0:
        output = ROOT / config["checkpoint_dir"] / "hf_model"; output.mkdir(parents=True, exist_ok=True); (model.module if world > 1 else model).save_pretrained(output); tokenizer.save_pretrained(output)
        report_dir = ROOT / config["report_dir"]; report_dir.mkdir(parents=True, exist_ok=True); (report_dir / "source_main6_dapt_report.json").write_text(json.dumps({"train_only": True, "test_labels_read": False, "epochs": config["epochs"], "dapt_max_windows_per_contract": config.get("dapt_max_windows_per_contract"), "dapt_window_samples": len(dataset), "final_train_mlm_loss": float(np.mean(losses[-len(loader):]))}, indent=2) + "\n", encoding="utf-8")
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
