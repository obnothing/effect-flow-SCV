"""Train only the Stack-Adapter and Q/V LoRA on train-only MLM chunks."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from stack_adapter_bert import StackAdapterBertForMaskedLM  # noqa: E402
from stack_relation_mlm_dataset import StackRelationMLMDataset, collate_stack_relation_mlm  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_adapter.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    seed_all(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    dataset = StackRelationMLMDataset(resolve(config["relation_cache_dir"]) / "train.pt", seed=int(config["seed"]))
    dataset.special_ids = {tokenizer.vocab[token] for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")}
    dataset.set_random_token_ids(tokenizer.vocab.values(), tokenizer.vocab["[MASK]"])
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("mlm_batch_size", 1)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_stack_relation_mlm,
        pin_memory=device.type == "cuda",
    )
    model = StackAdapterBertForMaskedLM.from_pretrained(
        config["hf_model_path"], local_files_only=True,
        stack_hidden_dim=int(config.get("stack_adapter_hidden_dim", 256)),
        stack_layers=int(config.get("stack_adapter_layers", 1)),
        lora_rank=int(config.get("lora_rank", 8)),
        lora_alpha=float(config.get("lora_alpha", 16.0)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        fusion_init=float(config.get("fusion_init", 0.05)),
    ).to(device)
    model.freeze_base()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=float(config.get("mlm_learning_rate", 1e-4)), weight_decay=float(config.get("mlm_weight_decay", 0.01)))
    use_bf16 = bool(config.get("mlm_bf16", True)) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = bool(config.get("mlm_fp16", False)) and device.type == "cuda" and not use_bf16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    history = []
    output_dir = resolve(config["checkpoint_dir"])
    result_dir = resolve(config["result_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(config.get("mlm_epochs", 3)) + 1):
        dataset.set_epoch(epoch)
        model.train()
        total, count, skipped = 0.0, 0, 0
        for raw_batch in loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            if int(batch["labels"].ne(-100).sum()) == 0:
                skipped += 1
                continue
            if use_bf16:
                context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                context = torch.cuda.amp.autocast(enabled=use_fp16)
            with context:
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stack-Adapter MLM loss at epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * batch["input_ids"].shape[0]
            count += batch["input_ids"].shape[0]
        record = {"epoch": epoch, "train_mlm_loss": total / max(1, count), "skipped_empty_target_batches": skipped, "bf16": use_bf16}
        history.append(record)
        print(f"[stack_adapter_mlm] epoch={epoch} train_mlm_loss={record['train_mlm_loss']:.6f}", flush=True)
    torch.save({"schema": "main6_opcode_stack_adapter_mlm_v1", "model_state_dict": model.state_dict(), "config": config, "history": history, "train_only": True}, output_dir / "stack_adapter_mlm.pt")
    (result_dir / "stack_adapter_mlm_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
