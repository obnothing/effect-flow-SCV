"""Train C1/C2 label-decoupled objectives on the existing B2 architecture."""

import argparse
import hashlib
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from label_decoupled_prototype import LabelDecoupledPrototype, pairwise_supervised_contrast  # noqa: E402
from light_label_data import LengthBucketBatchSampler, LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet, validate_model_config  # noqa: E402
from light_label_runtime import merge_runtime_config  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs, derived_detection_metrics_from_multilabel_probs, select_per_label_thresholds  # noqa: E402


def resolve(value):
    path=Path(value); return path if path.is_absolute() else ROOT/path


def load_config(path):
    config=yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    base=yaml.safe_load(resolve(config["base_config"]).read_text(encoding="utf-8"))
    base.update(config)
    resolved = resolve("results/light_label/resolved_runtime.json")
    return merge_runtime_config(base, resolved)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metric_pack(config, labels, logits):
    probabilities=torch.sigmoid(logits).numpy(); targets=labels.numpy().astype(int)
    selected=select_per_label_thresholds(targets,probabilities,config["thresholds"],config["label_names"],global_threshold=0.2)
    fixed=compute_multilabel_metrics_from_probs(targets,probabilities,0.5); tuned=compute_multilabel_metrics_from_probs(targets,probabilities,selected["thresholds"])
    detection=derived_detection_metrics_from_multilabel_probs(targets,probabilities,selected["thresholds"])
    def pack(value): return {"macro_f1":float(value["recognition_macro_f1"]),"micro_f1":float(value["recognition_micro_f1"]),"macro_precision":float(value["recognition_macro_precision"]),"macro_recall":float(value["recognition_macro_recall"]),"per_label_f1":[float(x) for x in value["per_label_f1"]],"per_label_precision":[float(x) for x in value["per_label_precision"]],"per_label_recall":[float(x) for x in value["per_label_recall"]]}
    return {"fixed":pack(fixed),"tuned":pack(tuned),"thresholds":[float(x) for x in selected["thresholds"]],"detection_f1":float(detection["detection_f1"])}


def make_loader(data, config, pad_id, shuffle):
    if shuffle:
        sampler=LengthBucketBatchSampler(data,int(config["batch_size"]),int(config["seed"]))
        return DataLoader(data,batch_sampler=sampler,num_workers=0,collate_fn=partial(collate_light_label,pad_id=pad_id),pin_memory=torch.cuda.is_available())
    return DataLoader(data,batch_size=int(config["batch_size"]),shuffle=False,num_workers=0,collate_fn=partial(collate_light_label,pad_id=pad_id),pin_memory=torch.cuda.is_available())


def pos_weight(data,config):
    labels=data.labels[data.indices]; positive=labels.sum(0); negative=len(labels)-positive
    value=torch.sqrt(negative/positive.clamp_min(1)) if config.get("pos_weight_mode")=="sqrt_ratio" else negative/positive.clamp_min(1)
    return value.clamp(1.0,float(config["max_pos_weight"]))


@torch.no_grad()
def evaluate(model,loader,device,weight,amp,proto=None):
    model.eval(); logits=[]; labels=[]; losses=[]; ids=[]; original=[]
    for batch in loader:
        input_ids=batch["input_ids"].to(device); mask=batch["mask"].to(device); target=batch["labels"].to(device)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp): output=model(input_ids,batch["lengths"],mask)
        losses.append(float(F.binary_cross_entropy_with_logits(output["logits"],target,pos_weight=weight))); logits.append(output["logits"].float().cpu()); labels.append(target.cpu()); ids.extend(batch["ids"]); original.extend(batch["original_lengths"].tolist())
    return {"logits":torch.cat(logits),"labels":torch.cat(labels),"loss":float(np.mean(losses)),"ids":ids,"original_lengths":original}


def checksum(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def run(config, smoke=False):
    seed_all(int(config["seed"])); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); tokenizer=EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    cache_dir=resolve(config["cache_dir"]); train_cache=cache_dir/f"train_max{config['max_len']}.pt"; valid_cache=cache_dir/f"valid_max{config['max_len']}.pt"
    if not train_cache.exists() or not valid_cache.exists(): raise FileNotFoundError("prepare light label cache first")
    train=LightLabelDataset(train_cache,runtime_max_len=config["max_len"]); valid=LightLabelDataset(valid_cache,runtime_max_len=config["max_len"])
    if smoke:
        rng=random.Random(int(config["seed"])); train=LightLabelDataset(train_cache,runtime_max_len=config["max_len"],indices=sorted(rng.sample(range(len(train)),256))); valid=LightLabelDataset(valid_cache,runtime_max_len=config["max_len"],indices=sorted(rng.sample(range(len(valid)),128)))
    train_loader=make_loader(train,config,tokenizer.pad_token_id,True); valid_loader=make_loader(valid,config,tokenizer.pad_token_id,False)
    model=LabelGuidedOpcodeNet("b2_label_attention",len(tokenizer),tokenizer.pad_token_id,config["embedding_dim"],config["gru_hidden_size"],config["num_labels"],config["bidirectional"],config.get("local_radius",8),config.get("gru_layers",1)).to(device)
    validate_model_config(model, config)
    proto=LabelDecoupledPrototype(config["num_labels"],model.output_dim,config["prototype_momentum"],config["prototype_temperature"]).to(device) if config["contrast_mode"]=="prototype" else None
    optimizer=torch.optim.AdamW(list(model.parameters())+(list(proto.parameters()) if proto else []),lr=float(config["learning_rate"]),weight_decay=float(config["weight_decay"]))
    if float(optimizer.param_groups[0]["lr"]) != float(config["learning_rate"]): raise RuntimeError("optimizer learning_rate does not match config")
    if float(optimizer.param_groups[0]["weight_decay"]) != float(config["weight_decay"]): raise RuntimeError("optimizer weight_decay does not match config")
    amp=device.type=="cuda" and bool(config["amp"]); scaler=torch.amp.GradScaler("cuda",enabled=amp); weight=pos_weight(train,config).to(device) if config.get("weighted_bce", True) else None; epochs=2 if smoke else int(config["epochs"]); accumulation=int(config["gradient_accumulation_steps"])
    run_name="smoke" if smoke else "full"; result_dir=resolve(config["prototype_result_dir"])/run_name; checkpoint_dir=resolve(config["prototype_checkpoint_dir"])/run_name; result_dir.mkdir(parents=True,exist_ok=True); checkpoint_dir.mkdir(parents=True,exist_ok=True)
    effective_config={key:config[key] for key in ("embedding_dim","gru_hidden_size","gru_layers","max_len","batch_size","gradient_accumulation_steps","learning_rate","weight_decay","weighted_bce","pos_weight_mode","max_pos_weight") if key in config}
    print(json.dumps({"effective_config":effective_config,"params":sum(p.numel() for p in model.parameters()),"pos_weight":[float(x) for x in weight.detach().cpu()],"test_checked":False},indent=2),flush=True)
    best=-1.; best_payload=None; stale=0; history=[]; start_total=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]; proto_losses=[]; pair_losses=[]; epoch_start=time.perf_counter()
        if hasattr(train_loader.batch_sampler,"set_epoch"): train_loader.batch_sampler.set_epoch(epoch)
        for step,batch in enumerate(train_loader,1):
            input_ids=batch["input_ids"].to(device); mask=batch["mask"].to(device); target=batch["labels"].to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                output=model(input_ids,batch["lengths"],mask); cls=F.binary_cross_entropy_with_logits(output["logits"],target,pos_weight=weight)
                extra=output["logits"].sum()*0.0
                if config["contrast_mode"]=="pairwise": extra=float(config["pairwise_lambda"])*pairwise_supervised_contrast(output["representations"],target,config["prototype_temperature"])
                elif config["contrast_mode"]=="prototype":
                    proto.update(output["representations"],target); proto_value,active=proto.prototype_loss(output["representations"],target); extra=float(config["prototype_lambda"])*proto_value if epoch>=2 else output["logits"].sum()*0.0; proto_losses.append(float(proto_value.detach()))
                total=cls+extra
            scaler.scale(total/accumulation).backward() if scaler.is_enabled() else (total/accumulation).backward(); losses.append(float(cls.detach())); pair_losses.append(float(extra.detach()))
            if step % 10 == 0 or step == len(train_loader):
                print(f"[{config['contrast_mode']}] epoch={epoch} step={step}/{len(train_loader)} cls={float(cls.detach()):.6f} extra={float(extra.detach()):.6f}",flush=True)
            if step%accumulation==0 or step==len(train_loader):
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        valid_out=evaluate(model,valid_loader,device,weight,amp,proto); score=metric_pack(config,valid_out["labels"],valid_out["logits"]); elapsed=time.perf_counter()-epoch_start
        record={"epoch":epoch,"train_cls_loss":float(np.mean(losses)),"train_extra_loss":float(np.mean(pair_losses)),"train_proto_loss":float(np.mean(proto_losses)) if proto_losses else 0.0,"valid_loss":valid_out["loss"],"metrics":score,"epoch_seconds":elapsed,"peak_memory_mb":float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=="cuda" else 0.0}
        if proto: record["prototype_initialized"]=proto.initialized.cpu().tolist(); record["prototype_separation"]=proto.diagnostics(output["representations"],target)["prototype_separation"]
        history.append(record); current=score["tuned"]["macro_f1"]; print(f"[{config['contrast_mode']}] epoch={epoch} train_cls={record['train_cls_loss']:.6f} train_extra={record['train_extra_loss']:.6f} valid_loss={record['valid_loss']:.6f} tuned_macro={current:.6f}",flush=True)
        payload={"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"prototype_state_dict":({k:v.detach().cpu() for k,v in proto.state_dict().items()} if proto else None),"config":config,"epoch":epoch,"metrics":score,"test_checked":False}
        torch.save(payload,checkpoint_dir/"last.pt")
        if current>best: best=current; best_payload=payload; stale=0; torch.save(payload,checkpoint_dir/"best.pt")
        else: stale+=1
        if stale>=int(config["early_stopping_patience"]): break
    model.load_state_dict(best_payload["model_state_dict"]); final=evaluate(model,valid_loader,device,weight,amp,proto); final_metrics=metric_pack(config,final["labels"],final["logits"])
    summary={"route":config["route_name"],"dataset":"DIVE_main6_opcode_process01","variant":config["contrast_mode"],"seed":config["seed"],"run":run_name,"effective_config":effective_config,"pos_weight":[float(x) for x in weight.detach().cpu()],"metrics":final_metrics,"best_epoch":best_payload["epoch"],"total_params":sum(p.numel() for p in model.parameters())+(sum(p.numel() for p in proto.parameters()) if proto else 0),"trainable_params":sum(p.numel() for p in model.parameters()),"prototype_trainable_params":0,"peak_memory_mb":max(x["peak_memory_mb"] for x in history),"mean_epoch_seconds":float(np.mean([x["epoch_seconds"] for x in history])),"history":history,"test_checked":False,"b2_reference_checkpoint":str(resolve("results/light_label/b2_label_attention/full/metrics.json"))}
    (result_dir/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); torch.save({"ids":final["ids"],"labels":final["labels"],"logits":final["logits"],"original_lengths":final["original_lengths"],"thresholds":final_metrics["thresholds"],"test_checked":False},result_dir/"valid_predictions.pt")
    if proto: torch.save({"positive_prototypes":proto.positive_prototypes.cpu(),"negative_prototypes":proto.negative_prototypes.cpu(),"initialized":proto.initialized.cpu(),"test_checked":False},result_dir/"prototypes.pt")
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args(); config=load_config(args.config); print(json.dumps(run(config,args.smoke),indent=2),flush=True)


if __name__=="__main__": main()
