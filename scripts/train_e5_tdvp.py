"""Train one isolated E5 TDVP control/candidate on train/valid only."""

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e5_tdvp.dictionary import collect_train_contexts, save_dictionary, build_dictionary, sha256_file, sha256_ids  # noqa: E402
from e5_tdvp.model import E5TDVPModel  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LengthBucketBatchSampler, LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet  # noqa: E402
from metrics import (compute_multilabel_metrics_from_probs, derived_detection_metrics_from_multilabel_probs,
                     select_per_label_thresholds)  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    config = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if config.get("base_config"):
        base = yaml.safe_load(resolve(config["base_config"]).read_text(encoding="utf-8"))
        base.update(config)
        config = base
    runtime = resolve(config.get("runtime_path", "results/light_label/resolved_runtime.json"))
    if runtime.exists():
        config.update(json.loads(runtime.read_text(encoding="utf-8")))
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config, split):
    path = resolve(config["cache_dir"]) / f"{split}_max{config['max_len']}.pt"
    return LightLabelDataset(path, runtime_max_len=config["max_len"])


def make_loader(data, config, pad_id, shuffle):
    if shuffle:
        sampler = LengthBucketBatchSampler(data, int(config["batch_size"]), int(config["seed"]))
        return DataLoader(data, batch_sampler=sampler, num_workers=int(config["num_workers"]),
                          collate_fn=partial(collate_light_label, pad_id=pad_id), pin_memory=torch.cuda.is_available())
    return DataLoader(data, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]),
                      collate_fn=partial(collate_light_label, pad_id=pad_id), pin_memory=torch.cuda.is_available())


def pos_weight(data, config):
    labels = data.labels[data.indices]
    positive = labels.sum(0)
    negative = len(labels) - positive
    if config.get("pos_weight_mode") == "sqrt_ratio":
        value = torch.sqrt(negative / positive.clamp_min(1))
    else:
        value = negative / positive.clamp_min(1)
    return value.clamp(1.0, float(config["max_pos_weight"]))


def metric_pack(config, labels, logits):
    probabilities = torch.sigmoid(logits).numpy()
    targets = labels.numpy().astype(int)
    selected = select_per_label_thresholds(targets, probabilities, config["thresholds"], config["label_names"], global_threshold=0.2)
    fixed = compute_multilabel_metrics_from_probs(targets, probabilities, 0.5)
    tuned = compute_multilabel_metrics_from_probs(targets, probabilities, selected["thresholds"])
    detection = derived_detection_metrics_from_multilabel_probs(targets, probabilities, selected["thresholds"])

    def pack(value):
        return {"macro_f1": float(value["recognition_macro_f1"]), "micro_f1": float(value["recognition_micro_f1"]),
                "macro_precision": float(value["recognition_macro_precision"]), "macro_recall": float(value["recognition_macro_recall"]),
                "per_label_f1": [float(x) for x in value["per_label_f1"]],
                "per_label_precision": [float(x) for x in value["per_label_precision"]],
                "per_label_recall": [float(x) for x in value["per_label_recall"]]}

    return {"fixed": pack(fixed), "tuned": pack(tuned), "thresholds": [float(x) for x in selected["thresholds"]],
            "detection_f1": float(detection["detection_f1"])}


def make_model(config, tokenizer):
    args = (len(tokenizer), tokenizer.pad_token_id, config["embedding_dim"], config["gru_hidden_size"],
            config["num_labels"], config["bidirectional"])
    if config["variant"] == "d0_b2":
        return LabelGuidedOpcodeNet("b2_label_attention", *args, local_radius=config.get("local_radius", 8))
    return E5TDVPModel(config["variant"], *args, local_radius=config.get("local_radius", 8),
                       dictionary_size=config.get("dictionary_size", 16), joint_dim=config.get("joint_dim", 128))


def forward_loss(model, batch, device, weight, amp, **kwargs):
    inputs = batch["input_ids"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        output = model(inputs, batch["lengths"], mask, **kwargs)
        loss = F.binary_cross_entropy_with_logits(output["logits"], labels, pos_weight=weight)
    return output, loss, labels


@torch.no_grad()
def evaluate(model, loader, device, weight, amp, **kwargs):
    model.eval()
    logits, labels, losses = [], [], []
    for batch in loader:
        output, loss, target = forward_loss(model, batch, device, weight, amp, **kwargs)
        logits.append(output["logits"].float().cpu())
        labels.append(target.cpu())
        losses.append(float(loss))
    return {"logits": torch.cat(logits), "labels": torch.cat(labels), "loss": float(np.mean(losses))}


def load_initial(model, path):
    if not path:
        return {"enabled": False, "path": None, "missing_keys": [], "unexpected_keys": []}
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    return {"enabled": True, "path": str(path), "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


def train_one(config, seed, init_checkpoint=None):
    if config.get("allow_test"):
        raise ValueError("test remains locked")
    config = dict(config)
    config["seed"] = int(seed)
    set_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    train_data = build_dataset(config, "train")
    valid_data = build_dataset(config, "valid")
    train_loader = make_loader(train_data, config, tokenizer.pad_token_id, True)
    valid_loader = make_loader(valid_data, config, tokenizer.pad_token_id, False)
    model = make_model(config, tokenizer).to(device)
    initial = load_initial(model, init_checkpoint or config.get("init_checkpoint"))
    amp = device.type == "cuda" and bool(config.get("amp", True))
    dictionary_meta = None
    dictionary_path = None
    if config["variant"] != "d0_b2":
        contexts = collect_train_contexts(model, train_data, config, tokenizer, device, amp)
        centroids, global_context, assignments, dictionary_meta = build_dictionary(
            contexts, config["variant"], int(config.get("dictionary_size", 16)), int(seed)
        )
        model.set_dictionary(centroids)
        model.set_global_context(global_context)
        dictionary_path = resolve(config["dictionary_root"]) / f"{config['variant']}_seed{seed}.pt"
        dictionary_meta = save_dictionary(
            dictionary_path, centroids, global_context, assignments, dictionary_meta,
            train_data, resolve(config["data_dir"]) / "train.jsonl"
        )
        print(json.dumps({"dictionary": str(dictionary_path), "metadata": dictionary_meta}, indent=2), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scaler_enabled = device.type == "cuda" and bool(config.get("amp", True))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    weight = pos_weight(train_data, config).to(device) if config.get("weighted_bce") else None
    result_dir = resolve(config["result_root"]) / config["variant"] / f"seed_{seed}"
    checkpoint_dir = resolve(config["checkpoint_root"]) / config["variant"] / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"variant": config["variant"], "seed": seed, "device": str(device),
                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                      "max_len": config["max_len"], "batch_size": config["batch_size"],
                      "params": sum(p.numel() for p in model.parameters()), "test_checked": False}, indent=2), flush=True)
    best_score, best_payload, stale, history = -1.0, None, 0, []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        start = time.perf_counter()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, batch in enumerate(train_loader, 1):
            _, loss, _ = forward_loss(model, batch, device, weight, amp)
            scaled = loss / int(config["gradient_accumulation_steps"])
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            losses.append(float(loss.detach()))
            if step % int(config["gradient_accumulation_steps"]) == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        epoch_seconds = time.perf_counter() - start
        valid = evaluate(model, valid_loader, device, weight, amp)
        metrics = metric_pack(config, valid["labels"], valid["logits"])
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": valid["loss"],
                  "metrics": metrics, "epoch_seconds": epoch_seconds,
                  "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0}
        history.append(record)
        score = metrics["tuned"]["macro_f1"]
        print(f"[{config['variant']}] seed={seed} epoch={epoch} train={record['train_loss']:.6f} valid={record['valid_loss']:.6f} fixed_macro={metrics['fixed']['macro_f1']:.6f} tuned_macro={score:.6f}", flush=True)
        payload = {"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                   "config": config, "epoch": epoch, "metrics": metrics, "dictionary_meta": dictionary_meta,
                   "dictionary_path": str(dictionary_path) if dictionary_path else None, "test_checked": False}
        torch.save(payload, checkpoint_dir / "last.pt")
        if score > best_score:
            best_score, best_payload, stale = score, payload, 0
            torch.save(payload, checkpoint_dir / "best.pt")
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    if best_payload is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    final = evaluate(model, valid_loader, device, weight, amp)
    final_metrics = metric_pack(config, final["labels"], final["logits"])
    train_file = resolve(config["data_dir"]) / "train.jsonl"
    valid_file = resolve(config["data_dir"]) / "valid.jsonl"
    summary = {"route": "DIVE Main6 process01 E5 TDVP final validation",
               "dataset": "DIVE_main6_opcode_process01", "protocol": config.get("protocol", "random"),
               "variant": config["variant"], "seed": int(seed), "max_len": int(config["max_len"]),
               "metrics": final_metrics, "best_epoch": int(best_payload["epoch"]), "history": history,
               "total_params": sum(p.numel() for p in model.parameters()),
               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "peak_memory_mb": max(item["peak_memory_mb"] for item in history),
               "mean_epoch_seconds": float(np.mean([item["epoch_seconds"] for item in history])),
               "initial_checkpoint": initial, "dictionary_meta": dictionary_meta,
               "split_metadata": {"train_file_sha256": sha256_file(train_file), "valid_file_sha256": sha256_file(valid_file),
                                  "train_ids_sha256": sha256_ids(train_data.ids), "valid_ids_sha256": sha256_ids(valid_data.ids)},
               "test_checked": False}
    (result_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save({"labels": final["labels"], "logits": final["logits"], "thresholds": final_metrics["thresholds"], "test_checked": False}, result_dir / "valid_predictions.pt")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.variant:
        config["variant"] = args.variant
    seed = int(args.seed if args.seed is not None else config["seed"])
    print(json.dumps(train_one(config, seed, args.init_checkpoint), indent=2), flush=True)


if __name__ == "__main__":
    main()

