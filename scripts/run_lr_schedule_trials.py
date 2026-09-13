"""Audited 45-epoch LR, scheduler, and regularization trials for process01."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "results/light_label/lr_schedule_trials_v1"
CHECKPOINT_ROOT = ROOT / "checkpoints/light_label/lr_schedule_trials_v1"
HISTORICAL_T0 = 0.8222974476700604
ETA_MIN = 0.0001
PROTOCOL_VERSION = "lr_schedule_v1_fixed_accumulation"

sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LightLabelDataset, collate_light_label, LengthBucketBatchSampler  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet, validate_model_config  # noqa: E402
from metrics import (  # noqa: E402
    compute_multilabel_metrics_from_probs,
    derived_detection_metrics_from_multilabel_probs,
    select_per_label_thresholds,
)
from torch.utils.data import DataLoader  # noqa: E402


TRIALS = {
    "R0_fixed_lr": {"trial_group": "initial_lr", "learning_rate": 0.001},
    "R1_lr_0007": {"trial_group": "initial_lr", "learning_rate": 0.0007},
    "R2_lr_0015": {"trial_group": "initial_lr", "learning_rate": 0.0015},
    "S0_cosine_immediate": {"trial_group": "scheduler", "scheduler": "cosine_immediate"},
    "S1_cosine_delayed20": {"trial_group": "scheduler", "scheduler": "cosine_delayed20"},
    "S2_multistep_26_36": {"trial_group": "scheduler", "scheduler": "multistep_26_36"},
    "S3_plateau": {"trial_group": "scheduler", "scheduler": "plateau"},
    "G0_dropout005": {"trial_group": "regularization", "representation_dropout": 0.05},
    "G1_dropout010": {"trial_group": "regularization", "representation_dropout": 0.10},
    "G2_weight_decay0002": {"trial_group": "regularization", "weight_decay": 0.0002},
}


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def base_config():
    config = yaml.safe_load((ROOT / "configs/light_label/b2_label_attention.yaml").read_text(encoding="utf-8"))
    config.update({
        "embedding_dim": 128,
        "gru_hidden_size": 384,
        "gru_layers": 1,
        "bidirectional": True,
        "max_len": 8192,
        "batch_size": 64,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 45,
        "early_stopping_patience": 8,
        "seed": 42,
        "weighted_bce": True,
        "pos_weight_mode": "sqrt_ratio",
        "max_pos_weight": 5.0,
        "representation_dropout": 0.0,
        "scheduler": "none",
        "scheduler_min_lr": ETA_MIN,
        "scheduler_warmup_epochs": 20,
        "dos_weight_multiplier": 1.0,
        "test_checked": False,
    })
    if config.get("allow_test") or config["data_dir"] != "data/processed/DIVE_main6_opcode_process01":
        raise ValueError("Unexpected dataset or test unlocked")
    return config


def model_for(config, tokenizer, device):
    model = LabelGuidedOpcodeNet(
        config["variant"], len(tokenizer), tokenizer.pad_token_id,
        config["embedding_dim"], config["gru_hidden_size"], config["num_labels"],
        config["bidirectional"], gru_layers=config["gru_layers"],
        representation_dropout=config["representation_dropout"],
    ).to(device)
    validate_model_config(model, config)
    if model.representation_dropout.p != float(config["representation_dropout"]):
        raise RuntimeError("representation_dropout was not wired into the model")
    return model


def compute_weights(config, data):
    labels = data.labels[data.indices]
    positive = labels.sum(0)
    negative = len(labels) - positive
    if config["pos_weight_mode"] == "sqrt_ratio":
        value = torch.sqrt(negative / positive.clamp_min(1))
    elif config["pos_weight_mode"] == "ratio":
        value = negative / positive.clamp_min(1)
    else:
        raise ValueError(f"Unsupported pos_weight_mode: {config['pos_weight_mode']}")
    value = value.clamp(1.0, float(config["max_pos_weight"]))
    value[config["label_names"].index("DoS")] *= float(config["dos_weight_multiplier"])
    return value.clamp(max=float(config["max_pos_weight"]))


def optimizer_for(config, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    group = optimizer.param_groups[0]
    if group["lr"] != float(config["learning_rate"]):
        raise RuntimeError("optimizer learning_rate does not match config")
    if group["weight_decay"] != float(config["weight_decay"]):
        raise RuntimeError("optimizer weight_decay does not match config")
    return optimizer


def scheduled_lr(config, epoch):
    initial = float(config["learning_rate"])
    minimum = float(config["scheduler_min_lr"])
    scheduler = config["scheduler"]
    if scheduler == "none":
        return initial
    if scheduler == "cosine_immediate":
        progress = (epoch - 1) / max(int(config["epochs"]) - 1, 1)
        return minimum + 0.5 * (initial - minimum) * (1.0 + np.cos(np.pi * progress))
    if scheduler == "cosine_delayed20":
        warmup = int(config["scheduler_warmup_epochs"])
        if epoch <= warmup:
            return initial
        progress = (epoch - warmup - 1) / max(int(config["epochs"]) - warmup - 1, 1)
        return minimum + 0.5 * (initial - minimum) * (1.0 + np.cos(np.pi * progress))
    if scheduler == "multistep_26_36":
        factor = 1.0 if epoch < 26 else 0.5 if epoch < 36 else 0.25
        return max(minimum, initial * factor)
    if scheduler == "plateau":
        return initial
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def set_optimizer_lr(optimizer, value):
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def update_plateau(config, optimizer, state, valid_loss):
    if config["scheduler"] != "plateau":
        return state
    threshold = 0.001
    best_valid = state.get("best_valid_loss")
    if best_valid is None or valid_loss < best_valid * (1.0 - threshold):
        state["best_valid_loss"] = float(valid_loss)
        state["bad_epochs"] = 0
    else:
        state["bad_epochs"] = int(state.get("bad_epochs", 0)) + 1
        if state["bad_epochs"] >= 3:
            state["current_lr"] = max(float(config["scheduler_min_lr"]), float(state["current_lr"]) * 0.5)
            state["bad_epochs"] = 0
            set_optimizer_lr(optimizer, state["current_lr"])
    return state


def metric_pack(config, labels, logits):
    probabilities = torch.sigmoid(logits).numpy()
    targets = labels.numpy().astype(int)
    selected = select_per_label_thresholds(targets, probabilities, config["thresholds"], config["label_names"], global_threshold=0.2)
    fixed = compute_multilabel_metrics_from_probs(targets, probabilities, 0.5)
    tuned = compute_multilabel_metrics_from_probs(targets, probabilities, selected["thresholds"])
    detection = derived_detection_metrics_from_multilabel_probs(targets, probabilities, selected["thresholds"])

    def pack(value):
        return {
            "macro_f1": float(value["recognition_macro_f1"]),
            "micro_f1": float(value["recognition_micro_f1"]),
            "macro_precision": float(value["recognition_macro_precision"]),
            "macro_recall": float(value["recognition_macro_recall"]),
            "per_label_f1": [float(x) for x in value["per_label_f1"]],
            "per_label_precision": [float(x) for x in value["per_label_precision"]],
            "per_label_recall": [float(x) for x in value["per_label_recall"]],
        }
    return {"fixed": pack(fixed), "tuned": pack(tuned),
            "thresholds": [float(x) for x in selected["thresholds"]],
            "detection_f1": float(detection["detection_f1"])}


@torch.no_grad()
def evaluate(model, loader, device, weight, amp):
    model.eval()
    logits, labels, losses = [], [], []
    for batch in loader:
        inputs = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            output = model(inputs, batch["lengths"], mask)
            loss = F.binary_cross_entropy_with_logits(output["logits"], batch["labels"].to(device), pos_weight=weight)
        logits.append(output["logits"].float().cpu())
        labels.append(batch["labels"].cpu())
        losses.append(float(loss))
    return {"logits": torch.cat(logits), "labels": torch.cat(labels), "loss": float(np.mean(losses))}


def preflight(config, tokenizer, train, weight, device):
    # Probe the longest real batch. The first dataset batch can be much shorter
    # and would under-estimate the memory needed by the actual sampler.
    longest = sorted(range(len(train)), key=train.sequence_length, reverse=True)[:int(config["batch_size"])]
    batch = collate_light_label([train[index] for index in longest], tokenizer.pad_token_id)
    model = optimizer = output = loss = None
    try:
        model = model_for(config, tokenizer, device)
        optimizer = optimizer_for(config, model)
        torch.cuda.reset_peak_memory_stats(device)
        inputs = batch["input_ids"].to(device)
        mask = batch["mask"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            output = model(inputs, batch["lengths"], mask)
            loss = F.binary_cross_entropy_with_logits(output["logits"].float(), batch["labels"].to(device), pos_weight=weight)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite preflight loss")
        loss.backward()
        optimizer.step()
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        return {"batch_size": int(config["batch_size"]), "peak_memory_mb": peak, "status": "pass"}
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("fixed batch preflight OOM; pause queue for manual resource review") from exc
    finally:
        del model, optimizer, output, loss
        torch.cuda.empty_cache()


def run_trial(name, requested, tokenizer, train, valid, provenance, smoke=False):
    run_name = "smoke" if smoke else "full"
    folder = ARTIFACT_ROOT / name / run_name
    checkpoint = CHECKPOINT_ROOT / name / run_name
    signature = hashlib.sha256(json.dumps({"config": requested, "provenance": provenance,
                                           "protocol": PROTOCOL_VERSION, "smoke": smoke}, sort_keys=True).encode()).hexdigest()
    existing = folder / "metrics.json"
    if existing.exists():
        result = json.loads(existing.read_text(encoding="utf-8"))
        if result.get("signature") != signature:
            raise RuntimeError(f"Existing result signature differs for {name}")
        return result
    device = torch.device("cuda")
    set_seed(int(requested["seed"]))
    weights = compute_weights(requested, train).to(device)
    preflight_info = preflight(requested, tokenizer, train, weights, device)
    config = dict(requested, batch_size=int(requested["batch_size"]),
                  gradient_accumulation_steps=int(requested["gradient_accumulation_steps"]))
    model = model_for(config, tokenizer, device)
    optimizer = optimizer_for(config, model)
    amp = bool(config["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    schedule_state = {"current_lr": float(config["learning_rate"]), "best_valid_loss": None, "bad_epochs": 0}
    audit = {
        "signature": signature, "protocol": PROTOCOL_VERSION, "requested_config": dict(requested),
        "effective_config": dict(config), "provenance": provenance,
        "parameter_shapes": {key: list(value.shape) for key, value in model.named_parameters()},
        "params": int(sum(p.numel() for p in model.parameters())),
        "pos_weight": [float(x) for x in weights.detach().cpu()],
        "preflight": preflight_info, "test_checked": False,
    }
    atomic_json(folder / "status.json", {"status": "running", "trial": name,
                                          "signature": signature, "test_checked": False})
    atomic_json(folder / "audit.json", audit)
    print(json.dumps({"trial": name, "effective_config": config, "params": audit["params"],
                      "pos_weight": audit["pos_weight"], "test_checked": False}), flush=True)
    loader = DataLoader(train, batch_sampler=LengthBucketBatchSampler(train, int(config["batch_size"]), int(config["seed"])),
                        num_workers=0, collate_fn=lambda items: collate_light_label(items, tokenizer.pad_token_id),
                        pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0,
                               collate_fn=lambda items: collate_light_label(items, tokenizer.pad_token_id), pin_memory=True)
    history, best, stale, start_epoch = [], -1.0, 0, 1
    last = checkpoint / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device)
        if state["signature"] != signature:
            raise RuntimeError(f"Resume signature differs for {name}")
        if state["batch_size"] != int(config["batch_size"]):
            raise RuntimeError(f"Resume batch differs for {name}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        schedule_state = state["schedule_state"]
        history, best, stale, start_epoch = state["history"], state["best"], state["stale"], state["epoch"] + 1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        if config["scheduler"] == "plateau":
            set_optimizer_lr(optimizer, schedule_state["current_lr"])
        else:
            set_optimizer_lr(optimizer, scheduled_lr(config, epoch))
        model.train()
        loader.batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        losses, sample_count = [], 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        for step, batch in enumerate(loader, 1):
            inputs = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                output = model(inputs, batch["lengths"], mask)
                loss = F.binary_cross_entropy_with_logits(output["logits"], target, pos_weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at {name} epoch {epoch}")
            scaler.scale(loss / int(config["gradient_accumulation_steps"])).backward()
            losses.append(float(loss.detach()))
            sample_count += len(batch["labels"])
            if step % int(config["gradient_accumulation_steps"]) == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(f"[{name}] epoch={epoch} step={step}/{len(loader)} loss={float(loss):.6f}", flush=True)
        valid_output = evaluate(model, valid_loader, device, weights, amp)
        metrics = metric_pack(config, valid_output["labels"], valid_output["logits"])
        score = metrics["tuned"]["macro_f1"]
        current_lr = float(optimizer.param_groups[0]["lr"])
        schedule_state = update_plateau(config, optimizer, schedule_state, valid_output["loss"])
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": valid_output["loss"],
            "metrics": metrics, "lr": current_lr,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 2**20),
        }
        history.append(record)
        if score > best:
            best, stale = score, 0
            atomic_save(checkpoint / "best.pt", {"model_state_dict": model.state_dict(), "config": config,
                                                  "epoch": epoch, "metrics": metrics, "signature": signature})
        else:
            stale += 1
        atomic_save(checkpoint / "last.pt", {
            "signature": signature, "batch_size": int(config["batch_size"]), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "schedule_state": schedule_state,
            "history": history, "best": best, "stale": stale, "epoch": epoch,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        })
        atomic_json(folder / "history.json", history)
        print(f"[{name}] epoch={epoch} train={record['train_loss']:.6f} valid={record['valid_loss']:.6f} "
              f"lr={current_lr:.8f} macro={score:.6f}", flush=True)
        if stale >= int(config["early_stopping_patience"]):
            break
    saved = torch.load(checkpoint / "best.pt", map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    final_output = evaluate(model, valid_loader, device, weights, amp)
    final_metrics = metric_pack(config, final_output["labels"], final_output["logits"])
    best_record = next(row for row in history if row["epoch"] == saved["epoch"])
    tail = history[-5:]
    result = dict(audit, trial=name, trial_group=requested.get("trial_group", ""),
                  metrics=final_metrics, best_epoch=int(saved["epoch"]), history=history,
                  train_loss_at_best=best_record["train_loss"], valid_loss_at_best=best_record["valid_loss"],
                  loss_gap_at_best=best_record["valid_loss"] - best_record["train_loss"],
                  final_macro_f1=history[-1]["metrics"]["tuned"]["macro_f1"],
                  last5_macro_mean=float(np.mean([row["metrics"]["tuned"]["macro_f1"] for row in tail])),
                  last5_macro_std=float(np.std([row["metrics"]["tuned"]["macro_f1"] for row in tail])),
                  mean_epoch_seconds=float(np.mean([row["epoch_seconds"] for row in history])),
                  peak_memory_mb=max(row["peak_memory_mb"] for row in history),
                  test_checked=False)
    atomic_save(folder / "valid_predictions.pt", {"labels": final_output["labels"], "logits": final_output["logits"], "test_checked": False})
    atomic_json(folder / "metrics.json", result)
    atomic_json(folder / "status.json", {"status": "completed", "trial": name,
                                          "signature": signature, "best_epoch": int(saved["epoch"]),
                                          "test_checked": False})
    return result


def write_summary(rows):
    if not rows:
        return
    root = ARTIFACT_ROOT
    rows = sorted(rows, key=lambda row: list(TRIALS).index(row["trial"]))
    t0 = rows[0]["macro_f1"]
    for row in rows:
        row["delta_vs_R0"] = row["macro_f1"] - t0
        row["decision"] = "candidate" if row["delta_vs_R0"] >= 0.01 else "below_adoption_threshold"
    atomic_json(root / "summary.json", rows)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# LR schedule trials (validation only)", "", "Dataset: DIVE_main6_opcode_process01; seed=42; test_checked=false.", "",
             "| Trial | Macro-F1 | Delta vs R0 | Last-5 mean | Last-5 std | Decision |", "|---|---:|---:|---:|---:|---|"]
    lines += [f"| {row['trial']} | {row['macro_f1']:.6f} | {row['delta_vs_R0']:+.6f} | {row['last5_macro_mean']:.6f} | {row['last5_macro_std']:.6f} | {row['decision']} |" for row in rows]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    config = base_config()
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
    provenance_paths = [Path(__file__), ROOT / "src/light_label_model.py", ROOT / "src/light_label_data.py",
                        ROOT / "src/metrics.py", ROOT / "configs/light_label/b2_label_attention.yaml", ROOT / config["vocab_path"],
                        ROOT / config["cache_dir"] / "train_max8192.pt", ROOT / config["cache_dir"] / "valid_max8192.pt",
                        ROOT / config["data_dir"] / "train.jsonl", ROOT / config["data_dir"] / "valid.jsonl"]
    provenance = {str(path.relative_to(ROOT)): digest(path) for path in provenance_paths}
    train_payload = torch.load(ROOT / config["cache_dir"] / "train_max8192.pt", map_location="cpu")
    valid_payload = torch.load(ROOT / config["cache_dir"] / "valid_max8192.pt", map_location="cpu")
    train = LightLabelDataset(ROOT / config["cache_dir"] / "train_max8192.pt", runtime_max_len=8192)
    valid = LightLabelDataset(ROOT / config["cache_dir"] / "valid_max8192.pt", runtime_max_len=8192)
    print(f"[setup] train={len(train)} valid={len(valid)} train_cache_rows={len(train_payload['ids'])} valid_cache_rows={len(valid_payload['ids'])} test_checked=false", flush=True)
    if args.preflight_only:
        device = torch.device("cuda")
        weights = compute_weights(config, train).to(device)
        print(json.dumps({"preflight": preflight(config, tokenizer, train, weights, device),
                          "params": sum(p.numel() for p in model_for(config, tokenizer, device).parameters()),
                          "test_checked": False}, indent=2), flush=True)
        return
    names = list(TRIALS)
    start_index = max(0, int(args.start_index))
    if start_index >= len(names):
        raise ValueError(f"start-index must be smaller than {len(names)}")
    rows = []
    summary = ARTIFACT_ROOT / ("smoke_summary.json" if args.smoke else "summary.json")
    if start_index and summary.exists():
        rows = json.loads(summary.read_text(encoding="utf-8"))
        if [row["trial"] for row in rows] != names[:start_index]:
            raise RuntimeError("Existing summary prefix does not match start-index")
    for index, (name, delta) in enumerate(TRIALS.items()):
        if index < start_index:
            continue
        requested = dict(config, **delta)
        result = run_trial(name, requested, tokenizer, train, valid, provenance, args.smoke)
        row = {
            "trial": name, "trial_group": result.get("trial_group", ""),
            "macro_f1": result["metrics"]["tuned"]["macro_f1"],
            "micro_f1": result["metrics"]["tuned"]["micro_f1"],
            "detection_f1": result["metrics"]["detection_f1"],
            "dos_f1": result["metrics"]["tuned"]["per_label_f1"][4],
            "best_epoch": result["best_epoch"], "params": result["params"],
            "peak_memory_mb": result["peak_memory_mb"], "last5_macro_mean": result["last5_macro_mean"],
            "last5_macro_std": result["last5_macro_std"], "test_checked": False,
        }
        rows.append(row)
        write_summary(rows)
        if not args.smoke and name == "R0_fixed_lr" and abs(row["macro_f1"] - HISTORICAL_T0) > 0.002:
            raise RuntimeError("R0 differs from historical hidden384 reference by >0.002; queue paused")
    write_summary(rows)
    print(f"[complete] {len(rows)}/{len(names)} trials; test_checked=false", flush=True)


if __name__ == "__main__":
    main()
