"""Windows-friendly training entrypoint for LabelGuidedOpcodeNet."""

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

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LengthBucketBatchSampler, LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet, validate_model_config  # noqa: E402
from light_extensions import LightExtensionNet  # noqa: E402
from light_label_runtime import merge_runtime_config  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs, derived_detection_metrics_from_multilabel_probs, select_per_label_thresholds  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    config = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if config.get("base_config"):
        base = yaml.safe_load(resolve(config["base_config"]).read_text(encoding="utf-8"))
        base.update(config)
        config = base
    resolved = resolve(config.get("runtime_path", "results/light_label/resolved_runtime.json"))
    return merge_runtime_config(config, resolved)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_dataset(config, split, smoke=False):
    path = resolve(config["cache_dir"]) / f"{split}_max{config['max_len']}.pt"
    base = LightLabelDataset(path, runtime_max_len=config["max_len"])
    if not smoke:
        return base
    count = 256 if split == "train" else 128
    rng = random.Random(int(config["seed"]) + (0 if split == "train" else 1))
    return LightLabelDataset(path, runtime_max_len=config["max_len"], indices=sorted(rng.sample(range(len(base)), count)))


def make_loader(data, config, pad_id, shuffle):
    if shuffle:
        sampler = LengthBucketBatchSampler(data, int(config["batch_size"]), int(config["seed"]))
        return DataLoader(data, batch_sampler=sampler, num_workers=int(config["num_workers"]),
                          collate_fn=partial(collate_light_label, pad_id=pad_id), pin_memory=torch.cuda.is_available())
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return DataLoader(data, batch_size=int(config["batch_size"]), shuffle=shuffle, num_workers=int(config["num_workers"]),
                      collate_fn=partial(collate_light_label, pad_id=pad_id), pin_memory=torch.cuda.is_available(), generator=generator)


def pos_weight(data, config):
    labels = data.labels[data.indices]
    positive = labels.sum(0); negative = len(labels) - positive
    if config.get("pos_weight_mode") == "sqrt_ratio":
        value = torch.sqrt(negative / positive.clamp_min(1))
    else:
        value = negative / positive.clamp_min(1)
    return value.clamp(1.0, float(config["max_pos_weight"]))


def metric_pack(config, labels, logits):
    probabilities = torch.sigmoid(logits).numpy(); targets = labels.numpy().astype(int)
    selected = select_per_label_thresholds(targets, probabilities, config["thresholds"], config["label_names"], global_threshold=0.2)
    fixed = compute_multilabel_metrics_from_probs(targets, probabilities, 0.5)
    tuned = compute_multilabel_metrics_from_probs(targets, probabilities, selected["thresholds"])
    detection = derived_detection_metrics_from_multilabel_probs(targets, probabilities, selected["thresholds"])
    def pack(value):
        return {"macro_f1": float(value["recognition_macro_f1"]), "micro_f1": float(value["recognition_micro_f1"]),
                "macro_precision": float(value["recognition_macro_precision"]), "macro_recall": float(value["recognition_macro_recall"]),
                "per_label_f1": [float(x) for x in value["per_label_f1"]], "per_label_precision": [float(x) for x in value["per_label_precision"]],
                "per_label_recall": [float(x) for x in value["per_label_recall"]]}
    return {"fixed": pack(fixed), "tuned": pack(tuned), "thresholds": [float(x) for x in selected["thresholds"]], "detection_f1": float(detection["detection_f1"])}


def forward_loss(model, batch, device, weight, amp):
    inputs = batch["input_ids"].to(device, non_blocking=True); lengths = batch["lengths"]
    mask = batch["mask"].to(device, non_blocking=True); labels = batch["labels"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        output = model(inputs, lengths, mask)
        loss = F.binary_cross_entropy_with_logits(output["logits"], labels, pos_weight=weight)
        if "consistency_loss" in output:
            loss = loss + float(getattr(model, "consistency_lambda", 0.01)) * output["consistency_loss"]
    return output, loss, labels


@torch.no_grad()
def evaluate(model, loader, device, weight, amp):
    model.eval(); logits=[]; labels=[]; ids=[]; original=[]; losses=[]; batch_times=[]
    for batch in loader:
        start=time.perf_counter(); output, loss, target=forward_loss(model,batch,device,weight,amp); batch_times.append(time.perf_counter()-start)
        logits.append(output["logits"].float().cpu()); labels.append(target.cpu()); ids.extend(batch["ids"]); original.extend(batch["original_lengths"].tolist()); losses.append(float(loss))
    return {"logits":torch.cat(logits),"labels":torch.cat(labels),"ids":ids,"original_lengths":original,
            "loss":float(np.mean(losses)),"inference_seconds_per_batch":float(np.mean(batch_times))}


def train(config, smoke=False):
    if config.get("allow_test"):
        raise ValueError("test remains locked")
    set_seed(int(config["seed"])); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    train_data=build_dataset(config,"train",smoke); valid_data=build_dataset(config,"valid",smoke)
    train_loader=make_loader(train_data,config,tokenizer.pad_token_id,True); valid_loader=make_loader(valid_data,config,tokenizer.pad_token_id,False)
    if str(config["variant"]).startswith("e"):
        model=LightExtensionNet(
            config["variant"], len(tokenizer), tokenizer.pad_token_id,
            config["embedding_dim"], config["gru_hidden_size"], config["num_labels"],
            config["bidirectional"], local_radius=config.get("local_radius", 8),
            gru_layers=config.get("gru_layers", 1),
            erase_mass=config.get("erase_mass", 0.30), propagation_k=config.get("propagation_k", 4),
            segment_kappa=config.get("segment_kappa", 1.0), segment_gap=config.get("segment_gap", 2),
            segment_min_len=config.get("segment_min_len", 2), segment_max=config.get("segment_max", 8),
            lambda_consistency=config.get("lambda_consistency", 0.01),
            confounder_clusters=config.get("confounder_clusters", 16),
            propagation_chunk_size=config.get("propagation_chunk_size", 512),
        ).to(device)
    else:
        model=LabelGuidedOpcodeNet(config["variant"],len(tokenizer),tokenizer.pad_token_id,config["embedding_dim"],config["gru_hidden_size"],config["num_labels"],config["bidirectional"],config.get("local_radius",8),config.get("gru_layers",1)).to(device)
    validate_model_config(model, config)
    init_checkpoint = config.get("init_checkpoint")
    init_info = {"enabled": False, "path": None, "missing_keys": [], "unexpected_keys": []}
    if init_checkpoint:
        init_path = resolve(init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        init_payload = torch.load(init_path, map_location="cpu")
        loaded = model.load_state_dict(init_payload["model_state_dict"], strict=False)
        init_info = {"enabled": True, "path": str(init_path), "missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys)}
        print(json.dumps({"init_checkpoint": str(init_path), "missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys)}, indent=2), flush=True)
    amp=device.type=="cuda" and bool(config["amp"])
    dictionary_info=None
    if str(config["variant"]) == "e5_tdvp":
        model.eval(); contexts=[]
        with torch.no_grad():
            for batch in train_loader:
                input_ids=batch["input_ids"].to(device, non_blocking=True); mask=batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    hidden=model.encode(input_ids,batch["lengths"])
                values=(hidden.float()*mask.unsqueeze(-1).to(hidden.dtype)).sum(1)/batch["lengths"].to(device).float().clamp_min(1).unsqueeze(-1)
                contexts.append(values.cpu())
        from sklearn.cluster import KMeans
        matrix=torch.cat(contexts).numpy(); cluster_count=min(int(config.get("confounder_clusters",16)),len(matrix))
        clustering=KMeans(n_clusters=cluster_count,random_state=int(config["seed"]),n_init=10).fit(matrix)
        model.set_confounder_dictionary(torch.tensor(clustering.cluster_centers_,dtype=torch.float32))
        dictionary_info={"clusters":cluster_count,"samples":len(matrix),"cluster_sizes":np.bincount(clustering.labels_,minlength=cluster_count).tolist()}
        print(json.dumps({"tdvp_dictionary":dictionary_info},indent=2),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["learning_rate"]),weight_decay=float(config["weight_decay"]))
    if float(optimizer.param_groups[0]["lr"]) != float(config["learning_rate"]):
        raise RuntimeError("optimizer learning_rate does not match config")
    if float(optimizer.param_groups[0]["weight_decay"]) != float(config["weight_decay"]):
        raise RuntimeError("optimizer weight_decay does not match config")
    scaler_enabled = device.type == "cuda" and bool(config["amp"])
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    weight=pos_weight(train_data,config).to(device) if config.get("weighted_bce") else None
    epochs=2 if smoke else int(config["epochs"]); accumulation=int(config["gradient_accumulation_steps"])
    run_name="smoke" if smoke else "full"; result_dir=resolve(config["result_dir"])/run_name; checkpoint_dir=resolve(config["checkpoint_dir"])/run_name
    result_dir.mkdir(parents=True,exist_ok=True); checkpoint_dir.mkdir(parents=True,exist_ok=True)
    effective_config = {key: config[key] for key in (
        "variant", "embedding_dim", "gru_hidden_size", "gru_layers", "bidirectional", "max_len",
        "batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "weighted_bce", "pos_weight_mode", "max_pos_weight", "epochs", "early_stopping_patience", "seed"
    ) if key in config}
    print(json.dumps({"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"vram_gib":torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else 0,
                      "torch":torch.__version__,"cuda":torch.version.cuda,"effective_config":effective_config,"params":sum(p.numel() for p in model.parameters())},indent=2),flush=True)
    best=-1.0; best_payload=None; stale=0; history=[]
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]; query_gradient_rows=[]; start=time.perf_counter()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        for step,batch in enumerate(train_loader,1):
            output,loss,_=forward_loss(model,batch,device,weight,amp); scaled=loss/accumulation
            scaler.scale(scaled).backward() if scaler.is_enabled() else scaled.backward()
            if config["variant"]=="b2_label_attention" and model.label_queries.grad is not None:
                query_gradient_rows.append(model.label_queries.grad.detach().float().norm(dim=1).cpu())
            losses.append(float(loss.detach()))
            if step%accumulation==0 or step==len(train_loader):
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        elapsed=time.perf_counter()-start; valid=evaluate(model,valid_loader,device,weight,amp); metrics=metric_pack(config,valid["labels"],valid["logits"])
        query_grad=torch.stack(query_gradient_rows).mean(0).tolist() if query_gradient_rows else None
        record={"epoch":epoch,"train_loss":float(np.mean(losses)),"valid_loss":valid["loss"],"metrics":metrics,"epoch_seconds":elapsed,
                "peak_memory_mb":float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=="cuda" else 0,"query_gradient_norms":query_grad}
        history.append(record); score=metrics["tuned"]["macro_f1"]
        print(f"[{config['variant']}] epoch={epoch} train={record['train_loss']:.6f} valid={valid['loss']:.6f} fixed_macro={metrics['fixed']['macro_f1']:.6f} tuned_macro={score:.6f}",flush=True)
        payload={"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":config,"epoch":epoch,"metrics":metrics,"test_checked":False,"dictionary_info":dictionary_info}
        torch.save(payload,checkpoint_dir/"last.pt")
        if score>best: best=score; stale=0; best_payload=payload; torch.save(payload,checkpoint_dir/"best.pt")
        else: stale+=1
        if stale>=int(config["early_stopping_patience"]): break
    model.load_state_dict(best_payload["model_state_dict"]); final=evaluate(model,valid_loader,device,weight,amp); final_metrics=metric_pack(config,final["labels"],final["logits"])
    summary={"route":config["route_name"],"dataset":"DIVE_main6_opcode_process01","variant":config["variant"],"seed":config["seed"],"run":run_name,
             "actual_config":effective_config,
             "pos_weight":None if weight is None else [float(x) for x in weight.detach().cpu()],
             "metrics":final_metrics,"best_epoch":best_payload["epoch"],"total_params":sum(p.numel() for p in model.parameters()),"trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad),
             "peak_memory_mb":max(x["peak_memory_mb"] for x in history),"mean_epoch_seconds":float(np.mean([x["epoch_seconds"] for x in history])),
             "inference_seconds_per_batch":final["inference_seconds_per_batch"],"history":history,"init_checkpoint":init_info,"test_checked":False}
    (result_dir/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    torch.save({"ids":final["ids"],"labels":final["labels"],"logits":final["logits"],"original_lengths":final["original_lengths"],"thresholds":final_metrics["thresholds"],"test_checked":False},result_dir/"valid_predictions.pt")
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--init-checkpoint",default=None); args=parser.parse_args()
    config=load_config(args.config)
    if args.init_checkpoint:
        config["init_checkpoint"]=args.init_checkpoint
    print(json.dumps(train(config,args.smoke),indent=2),flush=True)


if __name__=="__main__": main()
