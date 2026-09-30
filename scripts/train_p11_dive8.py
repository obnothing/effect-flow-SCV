"""Full eight-label P11 supervision, isolated from Main-6 and LOVO experiments."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer
from light_label_data import LightLabelDataset, LengthBucketBatchSampler, collate_light_label
from metrics import select_per_label_thresholds
from polarity_query_model import build_model, loss_terms, tensor_hash

CONFIG = ROOT / "configs/p11_dive8_supervised/comparison.yaml"
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS",
          "Bad Randomness", "Front Running", "Time manipulation"]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp); os.replace(tmp, path)


def load_config(variant):
    c = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if c["allow_test"] is not False or c["label_names"] != LABELS or c["num_labels"] != 8:
        raise ValueError("DIVE-8 train/valid protocol mismatch")
    if c["scheduler"] != "none" or c["batch_size"] * c["gradient_accumulation_steps"] != c["effective_batch_size"]:
        raise ValueError("Training budget mismatch")
    c.update(c["models"][variant]); c["variant"] = variant
    return c


def prepare(c):
    sources = {}
    caches = {}
    for split, key in (("train", "train_cache"), ("valid", "valid_input_cache")):
        source = ROOT / c["data_dir"] / f"{split}.jsonl"
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        payload = torch.load(ROOT / c[key], map_location="cpu", weights_only=False)
        ids = [str(row["id"]) for row in rows]
        labels = torch.tensor([row["multi_labels"] for row in rows], dtype=torch.float32)
        if len(rows) != c[f"{split}_rows"] or ids != payload["ids"] or labels.shape[1] != 8:
            raise ValueError(f"{split} ID/count/label alignment mismatch")
        if len(set(ids)) != len(ids) or payload["max_len"] != c["max_len"]:
            raise ValueError(f"{split} duplicate IDs or wrong token budget")
        if split == "train":
            if not torch.equal(labels, payload["labels"]):
                raise ValueError("Train cache labels differ from full train split")
        else:
            payload = dict(payload); payload["labels"] = labels; payload["test_checked"] = False
            atomic_save(ROOT / c["valid_cache"], payload)
        sources[split] = {"source_sha256": file_hash(source), "rows": len(rows)}
        caches[split] = set(ids)
    if caches["train"] & caches["valid"]:
        raise ValueError("Train and valid IDs intersect")
    provenance = {"dataset": c["data_dir"], "label_names": LABELS, "sources": sources,
                  "train_cache_sha256": file_hash(ROOT / c["train_cache"]),
                  "valid_cache_sha256": file_hash(ROOT / c["valid_cache"]),
                  "vocab_sha256": file_hash(ROOT / c["vocab_path"]), "test_checked": False}
    atomic_json(ROOT / c["provenance_path"], provenance)
    print(json.dumps(provenance, indent=2), flush=True)


def datasets(c):
    provenance = json.loads((ROOT / c["provenance_path"]).read_text(encoding="utf-8"))
    for key in ("train_cache", "valid_cache", "vocab"):
        path = ROOT / c[key if key != "vocab" else "vocab_path"]
        if file_hash(path) != provenance[f"{key}_sha256"]:
            raise ValueError(f"{key} differs from audited source")
    train = LightLabelDataset(ROOT / c["train_cache"], c["max_len"])
    valid = LightLabelDataset(ROOT / c["valid_cache"], c["max_len"])
    for split, data in (("train", train), ("valid", valid)):
        if len(data) != c[f"{split}_rows"] or data.labels.shape != (len(data), 8):
            raise ValueError(f"{split} dimensions mismatch")
        if not torch.isfinite(data.labels).all() or not torch.all((data.labels == 0) | (data.labels == 1)):
            raise ValueError(f"{split} invalid labels")
        if len(set(data.ids)) != len(data.ids) or min(data.sequence_length(i) for i in range(len(data))) < 1:
            raise ValueError(f"{split} invalid IDs/empty sequences")
    if set(train.ids) & set(valid.ids):
        raise ValueError("Train/valid ID overlap")
    return train, valid, provenance


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)


def loss(output, target, weight, c):
    soft = {"positive_high": c["dos_soft_positive"], "positive_low": c["dos_soft_negative"],
            "negative_low": c["dos_soft_negative"], "negative_high": c["dos_soft_positive"]}
    return loss_terms(output, target, weight, c["auxiliary_weight"],
                      positive_label_multiplier=[1.0] * 8, negative_label_multiplier=[1.0] * 8,
                      dos_soft_targets=soft, dos_label_index=LABELS.index("DoS"))


def metrics(labels, logits, c):
    y = labels.numpy().astype(int); p = torch.sigmoid(logits).numpy()
    selected = select_per_label_thresholds(y, p, c["thresholds"], LABELS, global_threshold=0.2)
    def pack(thresholds):
        pred = p >= np.asarray(thresholds)
        pr, re, f1, support = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
        _, _, micro, _ = precision_recall_fscore_support(y, pred, average="micro", zero_division=0)
        _, _, detection, _ = precision_recall_fscore_support(y.any(1), pred.any(1), average="binary", zero_division=0)
        return {"macro_f1": float(f1.mean()), "micro_f1": float(micro),
                "detection_f1": float(detection),
                "macro_precision": float(pr.mean()), "macro_recall": float(re.mean()),
                "per_label_f1": f1.tolist(), "per_label_precision": pr.tolist(),
                "per_label_recall": re.tolist(), "support": support.tolist(),
                "tp": ((y == 1) & pred).sum(0).tolist(), "fp": ((y == 0) & pred).sum(0).tolist(),
                "fn": ((y == 1) & ~pred).sum(0).tolist(), "tn": ((y == 0) & ~pred).sum(0).tolist()}
    return {"fixed": pack(0.5), "tuned": pack(selected["thresholds"]),
            "thresholds": selected["thresholds"], "threshold_selection": selected["per_label"],
            "per_label_ap": [float(average_precision_score(y[:, i], p[:, i])) for i in range(8)],
            "per_label_roc_auc": [float(roc_auc_score(y[:, i], p[:, i])) if len(np.unique(y[:, i])) > 1 else None for i in range(8)]}


@torch.no_grad()
def evaluate(model, loader, weight, c):
    model.eval(); predictions=[]; targets=[]; ids=[]; energies=[]; total=0.0; n=0
    for batch in loader:
        target = batch["labels"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=c["amp"]):
            output = model(batch["input_ids"].cuda(non_blocking=True), batch["lengths"], batch["mask"].cuda(non_blocking=True))
            _, cls, _ = loss(output, target, weight, c)
        count=len(batch["ids"]); total += float(cls) * count; n += count
        predictions.append(output["logits"].float().cpu()); targets.append(batch["labels"])
        energies.append(output["energies"].float().cpu()); ids.extend(batch["ids"])
    logits=torch.cat(predictions); labels=torch.cat(targets)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite valid logits")
    return {"logits": logits, "labels": labels, "ids": ids, "energies": torch.cat(energies),
            "valid_loss": total / n, "metrics": metrics(labels, logits, c)}


def run(c, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a Slurm GPU allocation")
    train, valid, provenance = datasets(c)
    tok=EVMOpcodeTokenizer.from_vocab_file(ROOT / c["vocab_path"])
    seed_all(c["seed"]); model=build_model("P11", c, len(tok), tok.pad_token_id).cuda()
    root=ROOT / c["result_root"] / c["variant"] / f"seed_{c['seed']}"
    ckpt=ROOT / c["checkpoint_root"] / c["variant"] / f"seed_{c['seed']}"
    root.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    weight=((len(train)-train.labels.sum(0))/train.labels.sum(0).clamp_min(1)).pow(c["pos_weight_power"]).clamp(1, c["max_pos_weight"]).cuda()
    signature=hashlib.sha256(json.dumps({"config": c, "data": provenance}, sort_keys=True).encode()).hexdigest()
    audit={"config": c, "data": provenance, "signature": signature, "initialization_hash": tensor_hash(model.state_dict()),
           "source_sha256": {name: file_hash(ROOT / name) for name in
                             ("scripts/train_p11_dive8.py", "src/polarity_query_model.py", "src/light_label_model.py", "src/metrics.py")},
           "actual": {"embedding_dim": model.embedding.embedding_dim, "hidden_per_direction": model.encoder.hidden_size,
                      "bigru_output": model.output_dim, "queries": list(model.queries.shape),
                      "heads": model.cross_attention.num_heads, "head_dim": model.cross_attention.head_dim,
                      "kv_input": model.cross_attention.k_proj.in_features, "kv_output": model.cross_attention.k_proj.out_features},
           "parameters": sum(p.numel() for p in model.parameters()), "pos_weight": weight.tolist(),
           "gpu": torch.cuda.get_device_name(0), "cpu_threads": torch.get_num_threads(), "test_checked": False}
    print(json.dumps(audit["actual"], indent=2), flush=True)
    optimizer=torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    scaler=torch.amp.GradScaler("cuda", enabled=c["amp"])
    if smoke:
        start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        order=sorted(range(len(train)), key=train.sequence_length, reverse=True)
        batch=collate_light_label([train[i] for i in order[:c["batch_size"]]], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=c["amp"]):
            output=model(batch["input_ids"].cuda(), batch["lengths"], batch["mask"].cuda())
            value, _, _=loss(output, batch["labels"].cuda(), weight, c)
        if output["logits"].shape != (c["batch_size"], 8) or not torch.isfinite(value):
            raise RuntimeError("Smoke shape/finite check failed")
        scaler.scale(value).backward(); scaler.unscale_(optimizer)
        if model.queries.grad is None or not torch.isfinite(model.queries.grad).all():
            raise RuntimeError("Query gradient check failed")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=c["amp"]):
            probe=model(batch["input_ids"].cuda(), batch["lengths"], batch["mask"].cuda())
        metrics(batch["labels"], probe["logits"].float().cpu(), c)
        audit.update(smoke_loss=float(value.detach()), peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
                     smoke_seconds=time.perf_counter()-start, forward_backward_metrics="passed")
        atomic_json(root / "smoke.json", audit); print("[smoke] passed", flush=True); return
    atomic_json(root / "audit.json", audit)
    sampler=LengthBucketBatchSampler(train, c["batch_size"], c["seed"])
    loader=DataLoader(train, batch_sampler=sampler, num_workers=0, pin_memory=True,
                      collate_fn=partial(collate_light_label, pad_id=tok.pad_token_id))
    order=sorted(range(len(valid)), key=valid.sequence_length)
    vl=DataLoader(torch.utils.data.Subset(valid, order), batch_size=c["batch_size"], num_workers=0, pin_memory=True,
                  collate_fn=partial(collate_light_label, pad_id=tok.pad_token_id))
    history=[]; best=-1.0; stale=0; start_epoch=1
    if (ckpt / "last.pt").exists():
        saved=torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume configuration/data signature changed")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        history=saved["history"]; best=saved["best"]; stale=saved["stale"]; start_epoch=saved["epoch"]+1
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"[resume] {c['variant']} epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, c["epochs"]+1):
        if stale >= c["early_stopping_patience"]:
            break
        model.train(); sampler.set_epoch(epoch); optimizer.zero_grad(set_to_none=True)
        samples=0; cls_sum=0.0; pol_sum=0.0; group_samples=0; tic=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader, 1):
            n=len(batch["ids"])
            with torch.autocast("cuda", dtype=torch.float16, enabled=c["amp"]):
                output=model(batch["input_ids"].cuda(non_blocking=True), batch["lengths"], batch["mask"].cuda(non_blocking=True))
                total, cls, pol=loss(output, batch["labels"].cuda(non_blocking=True), weight, c)
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(total * n).backward(); group_samples+=n; samples+=n
            cls_sum+=float(cls.detach())*n; pol_sum+=float(pol.detach())*n
            if step % c["gradient_accumulation_steps"] == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); group_samples=0
            if step % 100 == 0:
                print(f"[{c['variant']}] epoch={epoch} step={step}/{len(loader)} cls={float(cls):.6f}", flush=True)
        output=evaluate(model, vl, weight, c); score=output["metrics"]["tuned"]["macro_f1"]
        row={"epoch": epoch, "train_loss": cls_sum/samples+c["auxiliary_weight"]*pol_sum/samples,
             "train_cls": cls_sum/samples, "train_polarity": pol_sum/samples, "valid_loss": output["valid_loss"],
             "fixed_macro_f1": output["metrics"]["fixed"]["macro_f1"], "tuned_macro_f1": score,
             "tuned_micro_f1": output["metrics"]["tuned"]["micro_f1"],
             "per_label_f1": output["metrics"]["tuned"]["per_label_f1"],
             "seconds": time.perf_counter()-tic, "peak_gpu_mib": torch.cuda.max_memory_allocated()/2**20}
        history.append(row)
        print(f"[{c['variant']}] epoch={epoch} train={row['train_loss']:.6f} valid={row['valid_loss']:.6f} fixed_macro={row['fixed_macro_f1']:.6f} tuned_macro={score:.6f} seconds={row['seconds']:.1f}", flush=True)
        if score > best:
            best=score; stale=0
            atomic_save(ckpt / "best.pt", {"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": output["metrics"], "signature": signature, "config": c, "test_checked": False})
        else:
            stale+=1
        atomic_save(ckpt / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "history": history, "best": best, "stale": stale, "signature": signature,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()})
        atomic_json(root / "history.json", history)
    saved=torch.load(ckpt / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(saved["model_state_dict"])
    output=evaluate(model, vl, weight, c)
    final={"route": c["route_name"], "variant": c["variant"], "best_epoch": saved["epoch"], **output["metrics"],
           "parameters": audit["parameters"], "actual": audit["actual"], "valid_loss": output["valid_loss"],
           "history": history, "test_checked": False}
    atomic_json(root / "metrics.json", final)
    atomic_save(root / "valid_predictions.pt", {"ids": output["ids"], "labels": output["labels"], "logits": output["logits"],
                "energies": output["energies"], "thresholds": output["metrics"]["thresholds"], "test_checked": False})
    with (root / "per_label.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["label", "f1", "precision", "recall", "ap", "roc_auc", "threshold", "tp", "fp", "fn", "tn"])
        for i, name in enumerate(LABELS):
            writer.writerow([name, final["tuned"]["per_label_f1"][i], final["tuned"]["per_label_precision"][i],
                             final["tuned"]["per_label_recall"][i], final["per_label_ap"][i], final["per_label_roc_auc"][i], final["thresholds"][i],
                             final["tuned"]["tp"][i], final["tuned"]["fp"][i], final["tuned"]["fn"][i], final["tuned"]["tn"][i]])
    print(json.dumps({"variant": c["variant"], "best_epoch": saved["epoch"], "tuned": final["tuned"], "test_checked": False}), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("command", choices=["prepare", "smoke", "train"])
    parser.add_argument("--variant", choices=["original", "e3"], default="original")
    args=parser.parse_args(); config=load_config(args.variant)
    if args.command == "prepare":
        prepare(config)
    else:
        run(config, smoke=args.command == "smoke")
