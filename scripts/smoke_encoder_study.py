"""One-epoch GPU smoke test on fixed train/valid subsets for every encoder."""

import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import run_polarity_queries as base

ROOT = base.ROOT
CONFIGS = (
    ROOT / "configs/encoder_study/e1_local256.yaml",
    ROOT / "configs/encoder_study/e1_local512.yaml",
    ROOT / "configs/encoder_study/e2_bigru_local.yaml",
    ROOT / "configs/encoder_study/e3_bigru_block_global.yaml",
)


def subset(dataset, count, seed):
    indices = random.Random(seed).sample(range(len(dataset)), count)
    return torch.utils.data.Subset(dataset, sorted(indices))


def loader(dataset, tokenizer, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda items: base.collate_light_label(items, tokenizer.pad_token_id))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the encoder smoke test")
    reports = []
    for config_path in CONFIGS:
        config = base.load_config(config_path)
        base.set_seed(config["seed"])
        train, valid = base.datasets(config)
        tokenizer = base.EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
        model = base.build_model("P11", config, len(tokenizer), tokenizer.pad_token_id).cuda()
        optimizer = base.optimizer_for(model, config)
        scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
        weight = base.compute_weights(config, train).cuda()
        positive, negative, soft = base.auxiliary_settings(config, train)
        train_loader = loader(subset(train, 256, 42), tokenizer, 2)
        valid_loader = loader(subset(valid, 128, 43), tokenizer, 2)
        torch.cuda.reset_peak_memory_stats(); start = time.perf_counter(); model.train()
        first_shape = None
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if first_shape is None:
                with torch.no_grad():
                    first_shape = list(model.encode_tokens(batch["input_ids"].cuda(), batch["lengths"], batch["mask"].cuda()).shape)
            with torch.autocast("cuda", dtype=torch.float16, enabled=config["amp"]):
                output = base.forward(model, "P11", batch, "cuda")
                loss, _, _ = base.loss_terms(output, batch["labels"].cuda(), weight, config["auxiliary_weight"],
                    config["positive_auxiliary_multiplier"], config["negative_auxiliary_multiplier"], positive, negative, soft)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite smoke loss for {config['encoder_type']}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError(f"Nonfinite smoke gradient for {config['encoder_type']}")
            scaler.step(optimizer); scaler.update()
        model.eval(); valid_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                with torch.autocast("cuda", dtype=torch.float16, enabled=config["amp"]):
                    output = base.forward(model, "P11", batch, "cuda")
                    loss, _, _ = base.loss_terms(output, batch["labels"].cuda(), weight, config["auxiliary_weight"],
                        config["positive_auxiliary_multiplier"], config["negative_auxiliary_multiplier"], positive, negative, soft)
                valid_losses.append(float(loss))
        report = {
            "config": str(config_path.relative_to(ROOT)), "encoder_type": config["encoder_type"],
            "train_samples": 256, "valid_samples": 128, "epochs": 1, "physical_batch": 2,
            "first_hidden_shape": first_shape, "valid_loss": sum(valid_losses) / len(valid_losses),
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            "seconds": time.perf_counter() - start, "amp": config["amp"], "finite": True,
            "test_checked": False,
        }
        reports.append(report); print(json.dumps(report), flush=True)
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    out = ROOT / "reports/encoder_study"; out.mkdir(parents=True, exist_ok=True)
    base.atomic_json(out / "smoke_test.json", reports)
    lines = ["# Encoder Study Smoke Test", "", "256 train / 128 valid, one epoch, validation only, test_checked=false.", "",
             "| Encoder | Hidden shape | Valid loss | Peak MB | Seconds |", "|---|---|---:|---:|---:|"]
    for row in reports:
        lines.append(f"| {row['encoder_type']} | {row['first_hidden_shape']} | {row['valid_loss']:.6f} | {row['peak_memory_mb']:.1f} | {row['seconds']:.1f} |")
    (out / "smoke_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
