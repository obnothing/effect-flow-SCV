"""Train one strict-novel 7-label E3/P11 base model on train-only IDs."""

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
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evm_tokenizer import EVMOpcodeTokenizer
from light_label_data import LightLabelDataset, LengthBucketBatchSampler, collate_light_label
from polarity_query_model import build_model, loss_terms


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()


def macro_f1(labels, logits):
    pred=(1/(1+np.exp(-np.asarray(logits)))>=.5).astype(int)
    return float(f1_score(np.asarray(labels,dtype=int),pred,average="macro",zero_division=0))


@torch.no_grad()
def evaluate(model,loader,base_indices,device,amp):
    model.eval();logits=[];labels=[]
    for batch in loader:
        x=batch["input_ids"].to(device,non_blocking=True);m=batch["mask"].to(device,non_blocking=True)
        with torch.autocast("cuda",dtype=torch.float16,enabled=amp): out=model(x,batch["lengths"],m)
        logits.append(out["logits"].float().cpu().numpy());labels.append(batch["labels"][:,base_indices].numpy())
    return macro_f1(np.concatenate(labels),np.concatenate(logits))


def train_one(label,config,split,train_full,tokenizer):
    checkpoint_dir=ROOT/config["checkpoint_root"]/label.replace(" ","_")/"base_seed42"
    result_dir=ROOT/config["result_root"]/label.replace(" ","_")/"base_seed42"
    checkpoint_dir.mkdir(parents=True,exist_ok=True);result_dir.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load((ROOT/config["e3_base_config"]).read_text(encoding="utf-8"))
    held=base["label_names"].index(label);base_indices=[i for i in range(8) if i!=held]
    base["label_names"]=[base["label_names"][i] for i in base_indices];base["num_labels"]=7
    base["positive_auxiliary_label_multiplier"]=[1.0]*7;base["negative_auxiliary_label_multiplier"]=[1.0]*7
    idmap={str(v):i for i,v in enumerate(train_full.ids)}
    train_idx=[idmap[x] for x in split["base_train_ids"]];dev_idx=[idmap[x] for x in split["base_dev_ids"]]
    train=LightLabelDataset(ROOT/config["cache_dir"]/f"train_max{config['max_len']}.pt",runtime_max_len=config["max_len"],indices=train_idx)
    dev=LightLabelDataset(ROOT/config["cache_dir"]/f"train_max{config['max_len']}.pt",runtime_max_len=config["max_len"],indices=dev_idx)
    selected_labels=train.labels[train.indices]
    if any(float(selected_labels[i,held]) for i in range(len(train))):raise ValueError("held-out positives leaked into BaseTrain")
    batch=int(config["base_batch_size"]);accum=int(config["base_gradient_accumulation_steps"])
    sampler=LengthBucketBatchSampler(train,batch,config["base_seed"])
    train_loader=DataLoader(train,batch_sampler=sampler,num_workers=0,collate_fn=partial(collate_light_label,pad_id=tokenizer.pad_token_id),pin_memory=True)
    dev_order=sorted(range(len(dev)),key=dev.sequence_length)
    dev_loader=DataLoader(torch.utils.data.Subset(dev,dev_order),batch_size=batch,shuffle=False,num_workers=0,
        collate_fn=partial(collate_light_label,pad_id=tokenizer.pad_token_id),pin_memory=True)
    torch.set_num_threads(2);torch.manual_seed(config["base_seed"]);np.random.seed(config["base_seed"]);random.seed(config["base_seed"]);torch.cuda.manual_seed_all(config["base_seed"])
    model=build_model("P11",base,len(tokenizer),tokenizer.pad_token_id).cuda()
    y=selected_labels[:,base_indices];pos=y.sum(0).clamp_min(1);neg=len(y)-pos
    weight=(neg/pos).pow(float(config["pos_weight_power"])).clamp(1,float(config["max_pos_weight"])).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["base_learning_rate"],weight_decay=config["base_weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    signature=digest({"label":label,"split":split,"config":config,"base_labels":base["label_names"]})
    bestpath=checkpoint_dir/"best.pt";lastpath=checkpoint_dir/"last.pt";best=-1.;stale=0;start=1;history=[]
    if lastpath.exists():
        prev=torch.load(lastpath,map_location="cpu",weights_only=False)
        if prev.get("signature")==signature:
            model.load_state_dict(prev["model"]);optimizer.load_state_dict(prev["optimizer"]);scaler.load_state_dict(prev["scaler"])
            best,stale,start,history=prev["best"],prev["stale"],prev["epoch"]+1,prev["history"]
            random.setstate(prev["python_rng"]);np.random.set_state(prev["numpy_rng"]);torch.set_rng_state(prev["torch_rng"]);torch.cuda.set_rng_state_all(prev["cuda_rng"])
    for epoch in range(start,int(config["base_max_epochs"])+1):
        if stale>=int(config["base_early_stopping_patience"]):break
        model.train();sampler.set_epoch(epoch);optimizer.zero_grad(set_to_none=True);losses=[];tic=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for step,batchdata in enumerate(train_loader,1):
            x=batchdata["input_ids"].cuda(non_blocking=True);mask=batchdata["mask"].cuda(non_blocking=True);target=batchdata["labels"][:,base_indices].cuda(non_blocking=True)
            dosindex=base["label_names"].index("DoS") if "DoS" in base["label_names"] else None
            soft={"positive_high":base["dos_soft_positive"],"positive_low":base["dos_soft_negative"],"negative_low":base["dos_soft_negative"],"negative_high":base["dos_soft_positive"]} if dosindex is not None and base.get("dos_soft_targets") else None
            with torch.autocast("cuda",dtype=torch.float16):
                out=model(x,batchdata["lengths"],mask)
                loss=loss_terms(out,target,weight,base["auxiliary_weight"],positive_label_multiplier=[1.0]*7,
                    negative_label_multiplier=[1.0]*7,dos_soft_targets=soft,dos_label_index=dosindex)[0]
            if not torch.isfinite(loss):raise RuntimeError("nonfinite base loss")
            scaler.scale(loss/accum).backward();losses.append(float(loss.detach()))
            if step%accum==0 or step==len(train_loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if step%100==0:print(f"[base:{label}] epoch={epoch} step={step}/{len(train_loader)} loss={float(loss.detach()):.5f}",flush=True)
        score=evaluate(model,dev_loader,base_indices,torch.device("cuda"),True)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"base_dev_old_macro_f1_fixed05":score,
             "seconds":time.perf_counter()-tic,"peak_memory_mib":torch.cuda.max_memory_allocated()/2**20}
        history.append(row);print(f"[base:{label}] epoch={epoch} train={row['train_loss']:.5f} dev_macro_f1@.5={score:.5f} sec={row['seconds']:.1f}",flush=True)
        if score>best:
            best,stale=score,0
            torch.save({"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"label":label,
                "base_labels":base["label_names"],"epoch":epoch,"base_dev_macro_f1_fixed05":best,"signature":signature,"test_checked":False},bestpath)
        else:stale+=1
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,
            "best":best,"stale":stale,"history":history,"signature":signature,"python_rng":random.getstate(),
            "numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},lastpath)
        (result_dir/"base_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    bestdata=torch.load(bestpath,map_location="cpu",weights_only=False);model.load_state_dict(bestdata["model_state_dict"]);model.eval()
    audit={"label":label,"base_labels":base["label_names"],"base_train_count":len(train),"base_dev_count":len(dev),
        "heldout_positive_in_base_train":0,"best_epoch":bestdata["epoch"],"base_dev_macro_f1_fixed05":bestdata["base_dev_macro_f1_fixed05"],
        "total_parameters":sum(p.numel() for p in model.parameters()),"optimizer":{"name":"AdamW","lr":config["base_learning_rate"],"weight_decay":config["base_weight_decay"]},
        "physical_batch":batch,"gradient_accumulation":accum,"effective_batch":batch*accum,"pos_weight":weight.tolist(),
        "best_checkpoint":str(bestpath.relative_to(ROOT)),"test_checked":False,"official_valid_used_for_checkpoint_selection":False}
    (result_dir/"base_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval()
    return model,base,base_indices,audit


if __name__ == "__main__":
    raise SystemExit("Import train_one from run_pilot.py; invoke the Pilot-0 orchestrator")
