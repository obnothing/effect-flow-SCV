import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evm_bert_model import build_evm_bert_mlm_model, load_evm_vocab_size  # noqa: E402
from evm_pretrain_dataset import build_evm_mlm_dataset  # noqa: E402

TRANSDUCTIVE_WARNING = (
    "This pretraining corpus uses opcode inputs from train/valid/test without "
    "labels. It is a transductive self-supervised setting."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain EVM-BERT-base with MLM.")
    parser.add_argument(
        "--config",
        default="configs/pretrain_evm_bert_base_full_bjut.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_config(path):
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in [
        "vocab_path",
        "pretrain_corpus_path",
        "checkpoint_dir",
        "result_dir",
        "report_dir",
        "log_dir",
    ]:
        if key in config:
            config[key] = str(resolve_project_path(config[key]))
    config["_config_path"] = str(config_path)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP pretraining requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return distributed, rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def main_print(rank, message):
    if is_main_process(rank):
        print(message)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def count_parameters(model):
    return sum(param.numel() for param in unwrap_model(model).parameters())


def cuda_memory_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    return allocated, reserved


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, loss, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "mlm_loss": loss,
            "config": config,
        },
        path,
    )


def write_history(result_dir, history):
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "pretrain_epoch_history.json"
    txt_path = result_dir / "pretrain_epoch_history.txt"
    json_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["EVM-BERT MLM pretraining epoch history", ""]
    for row in history:
        lines.append(
            "epoch {epoch}/{epochs} train_mlm_loss={train_mlm_loss:.6f} "
            "global_step={global_step} lr={learning_rate:.8g} "
            "max_memory_allocated_mb={max_memory_allocated_mb:.2f} "
            "max_memory_reserved_mb={max_memory_reserved_mb:.2f}".format(**row)
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(config, report):
    report_dir = Path(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = config["experiment_name"]
    txt_path = report_dir / f"{experiment_name}_report.txt"
    json_path = report_dir / f"{experiment_name}_report.json"
    lines = [
        "EVM-BERT-base MLM pretraining report",
        "",
        f"experiment_name: {report['experiment_name']}",
        f"is_transductive_pretraining: {report['is_transductive_pretraining']}",
        f"use_labels: {report['use_labels']}",
        f"transductive_warning: {report['transductive_warning']}",
        f"corpus_path: {report['corpus_path']}",
        f"total samples: {report['total_samples']}",
        f"vocab_size: {report['vocab_size']}",
        f"model_scale: {report['model_scale']}",
        f"parameter_count: {report['parameter_count']}",
        f"hidden_size: {report['hidden_size']}",
        f"num_hidden_layers: {report['num_hidden_layers']}",
        f"num_attention_heads: {report['num_attention_heads']}",
        f"intermediate_size: {report['intermediate_size']}",
        f"max_len: {report['max_len']}",
        f"batch_size: {report['batch_size']}",
        f"gradient_accumulation_steps: {report['gradient_accumulation_steps']}",
        f"per_gpu_effective_batch_size: {report['per_gpu_effective_batch_size']}",
        f"effective_batch_size: {report['effective_batch_size']}",
        f"global_effective_batch_size: {report['global_effective_batch_size']}",
        f"distributed: {report['distributed']}",
        f"world_size: {report['world_size']}",
        f"learning_rate: {report['learning_rate']}",
        f"warmup_ratio: {report['warmup_ratio']}",
        f"epochs: {report['epochs']}",
        f"completed_epochs: {report['completed_epochs']}",
        f"final_mlm_loss: {report['final_mlm_loss']}",
        f"best_mlm_loss: {report['best_mlm_loss']}",
        f"best_epoch: {report['best_epoch']}",
        f"checkpoint_saved: {report['checkpoint_saved']}",
        f"hf_model_saved: {report['hf_model_saved']}",
        f"best_mlm_loss_path: {report['best_mlm_loss_path']}",
        f"last_checkpoint_path: {report['last_checkpoint_path']}",
        f"hf_model_path: {report['hf_model_path']}",
        f"cuda_device_name: {report['cuda_device_name']}",
        f"max_memory_allocated_mb: {report['max_memory_allocated_mb']}",
        f"max_memory_reserved_mb: {report['max_memory_reserved_mb']}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {project_relative(txt_path)}")
    print(f"[OK] wrote {project_relative(json_path)}")


def main():
    args = parse_args()
    config = load_config(args.config)
    distributed, rank, local_rank, world_size = setup_distributed()
    seed = int(config.get("seed", 42))
    set_seed(seed + rank)

    checkpoint_dir = Path(config["checkpoint_dir"])
    result_dir = Path(config["result_dir"])
    log_dir = Path(config["log_dir"])
    if is_main_process(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    vocab_copy_path = checkpoint_dir / "evm_vocab.json"
    if is_main_process(rank):
        shutil.copy2(config["vocab_path"], vocab_copy_path)

    dataset, tokenizer = build_evm_mlm_dataset(config)
    vocab_size = load_evm_vocab_size(config["vocab_path"])
    model = build_evm_bert_mlm_model(config, vocab_size)

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    batch_size = int(config.get("batch_size", 16))
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    epochs = int(config.get("epochs", 10))
    num_workers = int(config.get("num_workers", 0))
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        if distributed
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if len(dataloader) == 0:
        raise ValueError("Pretraining dataloader is empty.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.999))),
        eps=float(config.get("adam_epsilon", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.06)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    fp16 = bool(config.get("fp16", False)) and torch.cuda.is_available()
    bf16 = bool(config.get("bf16", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=fp16 and not bf16)
    autocast_dtype = torch.bfloat16 if bf16 else torch.float16
    logging_steps = int(config.get("logging_steps", 100))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    main_print(rank, f"[INFO] experiment_name: {config['experiment_name']}")
    main_print(rank, f"[INFO] corpus: {project_relative(config['pretrain_corpus_path'])}")
    main_print(rank, "[INFO] tokenizer: EVM opcode-aware vocab")
    main_print(rank, f"[INFO] vocab_size: {vocab_size}")
    main_print(rank, "[INFO] model: BertForMaskedLM from scratch")
    main_print(rank, f"[INFO] parameters: {count_parameters(model)}")
    main_print(rank, f"[INFO] device: {device}")
    main_print(rank, f"[INFO] distributed: {distributed}")
    main_print(rank, f"[INFO] world_size: {world_size}")
    main_print(rank, f"[INFO] total MLM chunks: {len(dataset)}")
    main_print(rank, f"[INFO] per_gpu_batch_size: {batch_size}")
    main_print(rank, f"[INFO] gradient_accumulation_steps: {grad_accum_steps}")
    main_print(rank, f"[INFO] per_gpu_effective_batch_size: {batch_size * grad_accum_steps}")
    main_print(
        rank,
        f"[INFO] global_effective_batch_size: {batch_size * grad_accum_steps * world_size}",
    )
    main_print(rank, f"[INFO] {TRANSDUCTIVE_WARNING}")

    best_loss = None
    best_epoch = None
    global_step = 0
    history = []

    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen_samples = 0
        progress = tqdm(
            dataloader,
            desc=f"pretrain epoch {epoch}/{epochs}",
            disable=not is_main_process(rank),
        )
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=(fp16 or bf16), dtype=autocast_dtype):
                outputs = model(**batch)
                loss = outputs.loss
                scaled_loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            batch_size_actual = batch["input_ids"].size(0)
            running_loss += float(loss.detach().cpu()) * batch_size_actual
            seen_samples += batch_size_actual

            should_step = (step % grad_accum_steps == 0) or (step == len(dataloader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % logging_steps == 0:
                    allocated, reserved = cuda_memory_mb()
                    lr = scheduler.get_last_lr()[0]
                    current_loss = running_loss / max(1, seen_samples)
                    main_print(
                        rank,
                        "[INFO] global_step={} epoch={} mlm_loss={:.6f} "
                        "lr={:.8g} max_memory_allocated_mb={:.2f} "
                        "max_memory_reserved_mb={:.2f}".format(
                            global_step,
                            epoch,
                            current_loss,
                            lr,
                            allocated,
                            reserved,
                        ),
                    )
            progress.set_postfix(loss=float(loss.detach().cpu()))

        loss_stats = torch.tensor(
            [running_loss, float(seen_samples)], dtype=torch.float64, device=device
        )
        if distributed:
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        epoch_loss = float(loss_stats[0].item() / max(1.0, loss_stats[1].item()))
        allocated, reserved = cuda_memory_mb()
        lr = scheduler.get_last_lr()[0]
        epoch_row = {
            "epoch": epoch,
            "epochs": epochs,
            "train_mlm_loss": epoch_loss,
            "global_step": global_step,
            "learning_rate": lr,
            "max_memory_allocated_mb": allocated,
            "max_memory_reserved_mb": reserved,
        }
        history.append(epoch_row)
        main_print(
            rank,
            "epoch {}/{} train_mlm_loss={:.6f} global_step={} lr={:.8g} "
            "max_memory_allocated_mb={:.2f} max_memory_reserved_mb={:.2f}".format(
                epoch,
                epochs,
                epoch_loss,
                global_step,
                lr,
                allocated,
                reserved,
            ),
        )

        if is_main_process(rank):
            last_path = checkpoint_dir / "last.pt"
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                epoch_loss,
                config,
            )
            if config.get("save_every_epoch", True):
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    epoch_loss,
                    config,
                )

        is_best = best_loss is None or epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            best_epoch = epoch
            if is_main_process(rank):
                best_path = checkpoint_dir / "best_mlm_loss.pt"
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    epoch_loss,
                    config,
                )
                hf_model_dir = checkpoint_dir / "hf_model"
                unwrap_model(model).save_pretrained(hf_model_dir)
                shutil.copy2(config["vocab_path"], hf_model_dir / "evm_vocab.json")

        if is_main_process(rank):
            write_history(result_dir, history)
        if distributed:
            dist.barrier()

    final_loss = history[-1]["train_mlm_loss"]
    max_allocated, max_reserved = cuda_memory_mb()
    bert_config = unwrap_model(model).config
    report = {
        "experiment_name": config["experiment_name"],
        "is_transductive_pretraining": bool(
            config.get("is_transductive_pretraining", True)
        ),
        "use_labels": False,
        "transductive_warning": TRANSDUCTIVE_WARNING,
        "corpus_path": project_relative(config["pretrain_corpus_path"]),
        "total_samples": len(dataset),
        "vocab_size": vocab_size,
        "model_scale": "BERT-base architecture from scratch with EVM vocab",
        "parameter_count": count_parameters(model),
        "hidden_size": bert_config.hidden_size,
        "num_hidden_layers": bert_config.num_hidden_layers,
        "num_attention_heads": bert_config.num_attention_heads,
        "intermediate_size": bert_config.intermediate_size,
        "max_len": int(config.get("max_len", 512)),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "per_gpu_effective_batch_size": batch_size * grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps * world_size,
        "global_effective_batch_size": batch_size * grad_accum_steps * world_size,
        "distributed": distributed,
        "world_size": world_size,
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.06)),
        "epochs": epochs,
        "completed_epochs": len(history),
        "final_mlm_loss": final_loss,
        "best_mlm_loss": best_loss,
        "best_epoch": best_epoch,
        "checkpoint_saved": (checkpoint_dir / "best_mlm_loss.pt").exists()
        and (checkpoint_dir / "last.pt").exists(),
        "hf_model_saved": (checkpoint_dir / "hf_model" / "config.json").exists(),
        "best_mlm_loss_path": project_relative(checkpoint_dir / "best_mlm_loss.pt"),
        "last_checkpoint_path": project_relative(checkpoint_dir / "last.pt"),
        "hf_model_path": project_relative(checkpoint_dir / "hf_model"),
        "vocab_copy_path": project_relative(vocab_copy_path),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
        "max_memory_allocated_mb": max_allocated,
        "max_memory_reserved_mb": max_reserved,
    }
    if is_main_process(rank):
        write_report(config, report)
        print(f"[OK] best_mlm_loss: {best_loss}")
        print(f"[OK] hf_model: {report['hf_model_path']}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
