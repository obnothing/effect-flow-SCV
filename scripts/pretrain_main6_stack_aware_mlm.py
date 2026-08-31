"""Train-only stack-aware MLM continuation for the isolated route."""

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stack_aware_bert import StackAwareBertForMaskedLM  # noqa: E402
from stack_relation_mlm_dataset import StackRelationMLMDataset, collate_stack_relation_mlm  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_stack_relational.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))["common"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    dataset = StackRelationMLMDataset(resolve(config["feature_dir"]) / "train.pt", seed=int(config["seed"]))
    dataset.special_ids = {tokenizer.vocab[token] for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")}
    dataset.set_random_token_ids(tokenizer.vocab.values(), tokenizer.vocab["[MASK]"])
    loader = DataLoader(dataset, batch_size=int(config.get("mlm_batch_size", 2)), shuffle=True, num_workers=int(config.get("num_workers", 0)), collate_fn=collate_stack_relation_mlm, pin_memory=device.type == "cuda")
    model = StackAwareBertForMaskedLM.from_pretrained(config["hf_model_path"], local_files_only=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("mlm_learning_rate", 1e-5)), weight_decay=0.01)
    # MLM is the only stage that backpropagates through the full BERT stack.
    # Prefer BF16 on supported GPUs: it has FP32-like exponent range without
    # the memory cost of FP32. FP16 remains opt-in for older accelerators.
    mlm_bf16 = bool(config.get("mlm_bf16", True)) and device.type == "cuda"
    mlm_fp16 = bool(config.get("mlm_fp16", False)) and device.type == "cuda" and not mlm_bf16
    scaler = torch.cuda.amp.GradScaler(enabled=mlm_fp16)
    epochs = int(config.get("mlm_epochs", 3)); history = []
    checkpoint_dir = resolve(config["checkpoint_dir"]); result_dir = resolve(config["result_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    output = checkpoint_dir / "stack_aware_mlm.pt"
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch); model.train(); total = 0.0; count = 0
        skipped_empty = 0
        for raw_batch in loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            target_count = int(batch["labels"].ne(-100).sum().item())
            if target_count == 0:
                skipped_empty += 1
                continue
            if mlm_bf16:
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                autocast_context = torch.cuda.amp.autocast(enabled=mlm_fp16)
            with autocast_context:
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                finite = {name: bool(torch.isfinite(value).all().item()) for name, value in model.named_parameters() if value.requires_grad}
                raise FloatingPointError(f"non-finite Stack-Aware MLM loss at epoch={epoch}, targets={target_count}, finite_parameters={finite}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * batch["input_ids"].shape[0]; count += batch["input_ids"].shape[0]
        record = {"epoch": epoch, "train_mlm_loss": total / max(1, count), "skipped_empty_target_batches": skipped_empty}; history.append(record)
        print(f"[stack_aware_mlm] epoch={epoch} train_mlm_loss={record['train_mlm_loss']:.6f}", flush=True)
    torch.save({"schema": "main6_opcode_stack_relational_mlm_v1", "model_state_dict": model.state_dict(), "config": config, "history": history, "train_only": True}, output)
    (result_dir / "stack_aware_mlm_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
