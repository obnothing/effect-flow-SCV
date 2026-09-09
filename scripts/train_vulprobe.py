"""Two-stage train/valid runner for VulProbe-V1."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import compute_multilabel_metrics_from_probs, derived_detection_metrics_from_multilabel_probs, select_per_label_thresholds  # noqa: E402
from vulprobe_dataset import VulProbeContractDataset, collate_contracts  # noqa: E402
from vulprobe_model import VulProbeModel  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dataset(config, split, limit=None):
    if split not in ("train", "valid"):
        raise ValueError("VulProbe training keeps test locked")
    return VulProbeContractDataset(
        resolve(config["data_dir"]) / f"{split}.jsonl", resolve(config["vocab_path"]),
        config["max_len"], config["chunk_stride"], config["max_chunks"], config["num_labels"], limit, config["seed"])


def loader(data, config, shuffle):
    return DataLoader(data, batch_size=int(config["contract_batch_size"]), shuffle=shuffle,
                      num_workers=int(config["num_workers"]), collate_fn=collate_contracts,
                      pin_memory=torch.cuda.is_available())


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def pos_weight(config):
    positive = torch.zeros(int(config["num_labels"]), dtype=torch.float32)
    total = 0
    with (resolve(config["data_dir"]) / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            labels = json.loads(line)["multi_labels"]
            if len(labels) != int(config["num_labels"]):
                raise ValueError("train row has unexpected label width")
            positive += torch.tensor(labels, dtype=torch.float32)
            total += 1
    negative = total - positive
    weight = torch.sqrt(negative / positive.clamp_min(1))
    return weight.clamp(1.0, float(config["max_pos_weight"]))


def metrics(config, labels, logits):
    probs = torch.sigmoid(logits).numpy(); target = labels.numpy().astype(int)
    tuned = select_per_label_thresholds(target, probs, config["thresholds"], config["label_names"], global_threshold=0.2)
    fixed_metrics = compute_multilabel_metrics_from_probs(target, probs, 0.5)
    tuned_metrics = compute_multilabel_metrics_from_probs(target, probs, tuned["thresholds"])
    detection = derived_detection_metrics_from_multilabel_probs(target, probs, tuned["thresholds"])
    def pack(item):
        return {"macro_f1": float(item["recognition_macro_f1"]), "micro_f1": float(item["recognition_micro_f1"]),
                "macro_precision": float(item["recognition_macro_precision"]), "macro_recall": float(item["recognition_macro_recall"]),
                "per_label_f1": [float(x) for x in item["per_label_f1"]], "per_label_precision": [float(x) for x in item["per_label_precision"]],
                "per_label_recall": [float(x) for x in item["per_label_recall"]]}
    return {"fixed": pack(fixed_metrics), "tuned": pack(tuned_metrics), "thresholds": [float(x) for x in tuned["thresholds"]],
            "detection_f1": float(detection["detection_f1"])}


@torch.no_grad()
def evaluate(model, data_loader, device, weight):
    model.eval(); logits=[]; labels=[]; ids=[]; chunks=[]; losses=[]
    for raw in data_loader:
        batch = move(raw, device)
        output = model(batch["input_ids"], batch["attention_mask"], batch["content_mask"], batch["chunk_mask"])
        loss = F.binary_cross_entropy_with_logits(output["logits"], batch["multi_labels"], pos_weight=weight)
        logits.append(output["logits"].float().cpu()); labels.append(batch["multi_labels"].cpu())
        ids.extend(raw["ids"]); chunks.extend(batch["chunk_mask"].sum(1).cpu().tolist()); losses.append(float(loss))
    return {"logits": torch.cat(logits), "labels": torch.cat(labels), "ids": ids, "chunks": chunks, "loss": float(np.mean(losses))}


def build_model(config, variant, device):
    return VulProbeModel(resolve(config["backbone_path"]), variant, config["num_labels"], config["tau"], config["num_heads"],
                         config.get("scorer", "shared"), config["encoder_chunk_batch"], config["interaction_dim"]).to(device)


def optimizer_for(model, config):
    backbone = [p for p in model.encoder.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    groups = [{"params": heads, "lr": float(config["head_learning_rate"])}]
    if backbone: groups.append({"params": backbone, "lr": float(config["backbone_learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))


def train_variant(config, variant, stage, smoke=False):
    set_seed(int(config["seed"])); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data=dataset(config,"train",config["smoke_train_samples"] if smoke else None)
    valid_data=dataset(config,"valid",config["smoke_valid_samples"] if smoke else None)
    train_loader=loader(train_data,config,True); valid_loader=loader(valid_data,config,False)
    model=build_model(config,variant,device)
    run_stage = stage + ("_smoke" if smoke else "")
    checkpoint_dir=resolve(config["checkpoint_root"])/variant/run_stage
    result_dir=resolve(config["result_root"])/variant/run_stage
    checkpoint_dir.mkdir(parents=True,exist_ok=True); result_dir.mkdir(parents=True,exist_ok=True)
    if stage == "joint":
        frozen_path=resolve(config["checkpoint_root"])/variant/"frozen"/"best.pt"
        if not frozen_path.exists(): raise FileNotFoundError(f"run frozen stage first: {frozen_path}")
        model.load_state_dict(torch.load(frozen_path,map_location="cpu")["model_state_dict"],strict=True)
    model.set_training_stage(stage,int(config["unfreeze_last_layers"]))
    optimizer=optimizer_for(model,config)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda" and bool(config["fp16"]))
    weight=pos_weight(config).to(device); accumulation=int(config["gradient_accumulation_steps"])
    epochs=1 if smoke else int(config["frozen_epochs"] if stage=="frozen" else config["joint_epochs"])
    best=-1.0; stale=0; history=[]; best_payload=None
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]; seen_chunks=0; started=time.perf_counter()
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        for step,raw in enumerate(train_loader,1):
            batch=move(raw,device)
            while True:
                try:
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        output=model(batch["input_ids"],batch["attention_mask"],batch["content_mask"],batch["chunk_mask"])
                        full_loss=F.binary_cross_entropy_with_logits(output["logits"],batch["multi_labels"],pos_weight=weight)
                        loss=full_loss/accumulation
                    scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    if model.encoder_chunk_batch<=1: raise
                    model.encoder_chunk_batch=max(1,model.encoder_chunk_batch//2)
                    print(f"[oom-fallback] encoder_chunk_batch={model.encoder_chunk_batch}",flush=True)
            losses.append(float(full_loss.detach())); seen_chunks+=int(batch["chunk_mask"].sum())
            if step%accumulation==0 or step==len(train_loader):
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(config["gradient_clip_norm"]))
                if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        elapsed=time.perf_counter()-started
        valid=evaluate(model,valid_loader,device,weight); score=metrics(config,valid["labels"],valid["logits"])
        record={"epoch":epoch,"train_loss":float(np.mean(losses)),"valid_loss":valid["loss"],**score,"epoch_seconds":elapsed,
                "contracts_per_second":len(train_data)/max(elapsed,1e-6),"chunks_per_second":seen_chunks/max(elapsed,1e-6),
                "peak_memory_mb":float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=="cuda" else 0.0,
                "encoder_chunk_batch":model.encoder_chunk_batch}
        history.append(record); current=score["tuned"]["macro_f1"]
        print(f"[{variant}/{stage}] epoch={epoch} train={record['train_loss']:.6f} valid={valid['loss']:.6f} fixed_macro={score['fixed']['macro_f1']:.6f} tuned_macro={current:.6f}",flush=True)
        payload={"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"variant":variant,"stage":stage,"epoch":epoch,
                 "metrics":score,"config":config,"test_checked":False,"encoder_chunk_batch":model.encoder_chunk_batch}
        if current>best: best=current; stale=0; best_payload=payload; torch.save(payload,checkpoint_dir/"best.pt")
        else: stale+=1
        torch.save(payload,checkpoint_dir/"last.pt")
        if stale>=int(config["early_stopping_patience"]): break
    model.load_state_dict(best_payload["model_state_dict"]); final=evaluate(model,valid_loader,device,weight); final_metrics=metrics(config,final["labels"],final["logits"])
    summary={"route":config["route_name"],"dataset":"DIVE_main6_opcode_process01","variant":variant,"stage":stage,"seed":int(config["seed"]),
             "validation_only":True,"test_checked":False,"best_epoch":int(best_payload["epoch"]),"metrics":final_metrics,"history":history,
             "total_params":sum(p.numel() for p in model.parameters()),"trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (result_dir/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    torch.save({"ids":final["ids"],"labels":final["labels"],"logits":final["logits"],"chunk_counts":final["chunks"],"thresholds":final_metrics["thresholds"],"test_checked":False},result_dir/"valid_predictions.pt")
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/vulprobe/v1.yaml")
    parser.add_argument("--variant",required=True,choices=["b0_shared_representation","b1_shared_probe","b2_label_probe","b3_probe_interaction"])
    parser.add_argument("--stage",required=True,choices=["frozen","joint"]); parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(); config=load_config(args.config)
    if config.get("allow_test"): raise ValueError("VulProbe-V1 keeps test locked")
    print(json.dumps(train_variant(config,args.variant,args.stage,args.smoke),indent=2),flush=True)


if __name__=="__main__": main()
