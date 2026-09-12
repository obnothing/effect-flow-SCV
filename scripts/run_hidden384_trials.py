"""Sequential, audited one-factor trials on the process01 B2 baseline."""

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
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "results/light_label/hidden384_trials_v3"
CHECKPOINT_ROOT = ROOT / "checkpoints/light_label/hidden384_trials_v3"
sys.path.insert(0, str(ROOT / "src"))
from light_label_model import LabelGuidedOpcodeNet, validate_model_config
from train_light_label_model import build_dataset, make_loader, evaluate, metric_pack, set_seed, pos_weight
from evm_tokenizer import EVMOpcodeTokenizer

TRIALS = {
    "T0_reference384": {},
    "T1_hidden512": {"gru_hidden_size": 512},
    "T2_hidden640": {"gru_hidden_size": 640},
    "T3_layers2": {"gru_layers": 2},
    "T4_embedding256": {"embedding_dim": 256},
    "T5_dropout": {"representation_dropout": 0.1},
    "T6_weight_decay": {"weight_decay": 0.0003},
    "T7_cosine": {"scheduler": "cosine"},
    "T8_ratio": {"pos_weight_mode": "ratio"},
    "T9_dos_weight": {"dos_weight_multiplier": 1.5},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def configuration():
    c = yaml.safe_load((ROOT / "configs/light_label/b2_label_attention.yaml").read_text(encoding="utf-8"))
    allowed = set("route_name variant data_dir vocab_path cache_dir checkpoint_dir result_dir report_dir label_names num_labels embedding_dim gru_hidden_size gru_layers bidirectional max_len batch_size gradient_accumulation_steps learning_rate weight_decay epochs early_stopping_patience seed amp weighted_bce pos_weight_mode max_pos_weight thresholds num_workers allow_test".split())
    if set(c) - allowed:
        raise ValueError(f"Unreviewed config fields: {set(c) - allowed}")
    c.update(embedding_dim=128, gru_hidden_size=384, gru_layers=1, bidirectional=True,
             max_len=8192, batch_size=64, gradient_accumulation_steps=4,
             learning_rate=0.001, weight_decay=0.0001, epochs=30, early_stopping_patience=5,
             seed=42, weighted_bce=True, pos_weight_mode="sqrt_ratio", max_pos_weight=5.0,
             representation_dropout=0.0, scheduler="none", dos_weight_multiplier=1.0)
    if c["allow_test"] or c["data_dir"] != "data/processed/DIVE_main6_opcode_process01":
        raise ValueError("Unexpected dataset or test unlocked")
    return c


def model_for(c, tokenizer, device):
    model = LabelGuidedOpcodeNet(c["variant"], len(tokenizer), tokenizer.pad_token_id,
        c["embedding_dim"], c["gru_hidden_size"], c["num_labels"], c["bidirectional"],
        gru_layers=c["gru_layers"], representation_dropout=c["representation_dropout"]).to(device)
    validate_model_config(model, c)
    assert model.representation_dropout.p == c["representation_dropout"]
    return model


def weights_for(c, data):
    if c["pos_weight_mode"] not in {"sqrt_ratio", "ratio"}:
        raise ValueError("Unsupported class weight mode")
    weights = pos_weight(data, c)
    weights[c["label_names"].index("DoS")] *= c["dos_weight_multiplier"]
    return weights.clamp(max=c["max_pos_weight"])


def optimizer_for(c, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    assert optimizer.param_groups[0]["lr"] == c["learning_rate"]
    assert optimizer.param_groups[0]["weight_decay"] == c["weight_decay"]
    if c["scheduler"] not in {"none", "cosine"}:
        raise ValueError("Unsupported scheduler")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0001) if c["scheduler"] == "cosine" else None
    return optimizer, scheduler


def preflight(c, tokenizer, device, weight):
    attempts = []
    for size in (64, 32, 16, 8, 4, 2, 1):
        model = optimizer = output = loss = ids = None
        try:
            model = model_for(c, tokenizer, device)
            optimizer, _ = optimizer_for(c, model)
            torch.cuda.reset_peak_memory_stats()
            ids = torch.full((size, c["max_len"]), tokenizer.unk_token_id, device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.float16, enabled=c["amp"]):
                output = model(ids, torch.full((size,), c["max_len"], dtype=torch.long), torch.ones_like(ids, dtype=torch.bool))
                loss = F.binary_cross_entropy_with_logits(output["logits"].float(), torch.zeros_like(output["logits"]), pos_weight=weight)
            loss.backward()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite preflight")
            optimizer.step()
            peak = torch.cuda.max_memory_allocated()
            # RNN workspaces and allocator fragmentation are not represented
            # reliably by the preflight peak alone. Keep an 8 GiB reserve so
            # the first real long sequence batch cannot exhaust the device.
            reserve = 8 * 2**30
            if peak + reserve > torch.cuda.get_device_properties(0).total_memory:
                raise torch.cuda.OutOfMemoryError("less than 8 GiB headroom")
            attempts.append({"batch_size": size, "peak_mb": peak / 2**20, "status": "pass"})
            return size, attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch_size": size, "status": "oom"})
        finally:
            model = optimizer = output = loss = ids = None
            torch.cuda.empty_cache()
    raise RuntimeError(f"Preflight failed: {attempts}")


def run_trial(name, c, tokenizer, train, valid, provenance, smoke=False):
    folder = ARTIFACT_ROOT / name / ("smoke" if smoke else "full")
    checkpoint = CHECKPOINT_ROOT / name / ("smoke" if smoke else "full")
    signature = hashlib.sha256(json.dumps({"config": c, "provenance": provenance, "smoke": smoke}, sort_keys=True).encode()).hexdigest()
    if (folder / "metrics.json").exists():
        result = json.loads((folder / "metrics.json").read_text())
        if result["signature"] != signature:
            raise ValueError(f"Existing result signature differs: {name}")
        return result
    device = torch.device("cuda")
    requested_config = dict(c)
    weights = weights_for(c, train).to(device)
    size, attempts = preflight(c, tokenizer, device, weights)
    c = dict(c, batch_size=size, gradient_accumulation_steps=256 // size)
    set_seed(c["seed"])
    model = model_for(c, tokenizer, device)
    optimizer, scheduler = optimizer_for(c, model)
    scaler = torch.cuda.amp.GradScaler(enabled=c["amp"])
    audit = {"signature": signature, "requested_config": requested_config, "effective_config": c, "provenance": provenance,
             "parameter_shapes": {k: list(v.shape) for k, v in model.named_parameters()},
             "params": sum(p.numel() for p in model.parameters()), "pos_weight": weights.tolist(),
             "preflight": attempts, "test_checked": False}
    atomic_json(folder / "audit.json", audit)
    print(json.dumps({"trial": name, "effective_config": c, "params": audit["params"], "pos_weight": audit["pos_weight"]}), flush=True)
    loader = make_loader(train, c, tokenizer.pad_token_id, True)
    valid_loader = make_loader(valid, c, tokenizer.pad_token_id, False)
    history, best, stale, start_epoch = [], -1.0, 0, 1
    if (checkpoint / "last.pt").exists():
        state = torch.load(checkpoint / "last.pt", map_location=device)
        if state["signature"] != signature or state["batch_size"] != size:
            raise ValueError("Resume configuration changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler: scheduler.load_state_dict(state["scheduler"])
        history, best, stale, start_epoch = state["history"], state["best"], state["stale"], state["epoch"] + 1
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    for epoch in range(start_epoch, (2 if smoke else c["epochs"]) + 1):
        if stale >= c["early_stopping_patience"]: break
        model.train(); loader.batch_sampler.set_epoch(epoch); optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter(); total_loss = 0.0; seen = 0
        torch.cuda.reset_peak_memory_stats()
        lr = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=c["amp"]):
                out = model(batch["input_ids"].to(device), batch["lengths"], batch["mask"].to(device))
                loss = F.binary_cross_entropy_with_logits(out["logits"], batch["labels"].to(device), pos_weight=weights)
            count = len(batch["labels"])
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            # Match the historical train_light_label_model protocol exactly:
            # every physical loss is divided by the fixed accumulation count.
            scaler.scale(loss / c["gradient_accumulation_steps"]).backward()
            seen += count; total_loss += float(loss.detach()) * count
            if (step + 1) % c["gradient_accumulation_steps"] == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 50 == 0:
                print(f"[{name}] epoch={epoch} step={step+1}/{len(loader)} loss={float(loss):.6f}", flush=True)
        output = evaluate(model, valid_loader, device, weights, c["amp"])
        metrics = metric_pack(c, output["labels"], output["logits"])
        score = metrics["tuned"]["macro_f1"]
        if scheduler: scheduler.step()
        record = {"epoch": epoch, "train_loss": total_loss / seen, "valid_loss": output["loss"], "metrics": metrics,
                  "lr": lr, "epoch_seconds": time.perf_counter()-start, "peak_memory_mb": torch.cuda.max_memory_allocated()/2**20}
        history.append(record)
        if score > best:
            best, stale = score, 0
            atomic_save(checkpoint / "best.pt", {"model_state_dict": model.state_dict(), "config": c, "epoch": epoch, "metrics": metrics, "signature": signature})
        else: stale += 1
        atomic_save(checkpoint / "last.pt", {"signature": signature, "batch_size": size, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
            "history": history, "best": best, "stale": stale, "epoch": epoch, "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
        atomic_json(folder / "history.json", history)
        print(f"[{name}] epoch={epoch} train={record['train_loss']:.6f} valid={output['loss']:.6f} macro={score:.6f}", flush=True)
    saved = torch.load(checkpoint / "best.pt", map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    output = evaluate(model, valid_loader, device, weights, c["amp"])
    metrics = metric_pack(c, output["labels"], output["logits"])
    result = dict(audit, trial=name, metrics=metrics, best_epoch=saved["epoch"], history=history,
        dos_ap=float(average_precision_score(output["labels"][:,4].numpy(), output["logits"][:,4].sigmoid().numpy())))
    atomic_save(folder / "valid_predictions.pt", {k:output[k] for k in ("ids", "labels", "logits")})
    atomic_json(folder / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT); torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    c = configuration()
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / c["vocab_path"])
    paths = [Path(__file__), ROOT/"src/light_label_model.py", ROOT/"src/train_light_label_model.py",
             ROOT/"src/light_label_data.py", ROOT/c["vocab_path"]]
    paths += [ROOT/c["cache_dir"]/f"{s}_max8192.pt" for s in ("train", "valid")]
    paths += [ROOT/c["data_dir"]/f"{s}.jsonl" for s in ("train", "valid")]
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    train, valid = build_dataset(c, "train", args.smoke), build_dataset(c, "valid", args.smoke)
    print(f"[setup] train={len(train)} valid={len(valid)} test_checked=false", flush=True)
    rows = []
    for name, delta in TRIALS.items():
        requested = dict(c, **delta)
        assert {k for k in requested if requested[k] != c[k]} == set(delta)
        result = run_trial(name, requested, tokenizer, train, valid, provenance, args.smoke)
        row = {"trial":name, "macro_f1": result["metrics"]["tuned"]["macro_f1"],
               "micro_f1": result["metrics"]["tuned"]["micro_f1"], "dos_ap": result["dos_ap"], "params":result["params"]}
        row["delta_vs_T0"] = row["macro_f1"] - (rows[0]["macro_f1"] if rows else row["macro_f1"])
        rows.append(row)
        atomic_json(ARTIFACT_ROOT/("smoke_summary.json" if args.smoke else "summary.json"), rows)
        if not args.smoke:
            root = ARTIFACT_ROOT
            with (root/"summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            (root/"summary.md").write_text("# Hidden384 trials (validation only)\n\n"
                "Dataset: DIVE_main6_opcode_process01. Seed: 42. test_checked=false.\n\n"
                "| Trial | Macro-F1 | Delta vs T0 | Decision |\n|---|---:|---:|---|\n" + "".join(
                    f"| {r['trial']} | {r['macro_f1']:.6f} | {r['delta_vs_T0']:+.6f} | "
                    f"{'candidate' if r['delta_vs_T0'] >= 0.01 else 'below adoption threshold'} |\n" for r in rows), encoding="utf-8")
        if not args.smoke and name.startswith("T0") and abs(row["macro_f1"] - 0.8222974476700604) > 0.002:
            raise RuntimeError("T0 differs from historical reference by >0.002; queue paused for audit")
    print("[complete] 10 trials; test_checked=false", flush=True)


if __name__ == "__main__":
    main()
