"""Audited P0-P4 experiment queue; no test inputs or runtime config overrides."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer
from light_label_data import LightLabelDataset, collate_light_label
from polarity_query_model import VARIANTS, build_model, encoder_state, tensor_hash, loss_terms
from train_light_label_model import make_loader, metric_pack, set_seed, pos_weight
from metrics import compute_multilabel_metrics_from_probs
from run_hidden384_trials import atomic_json, atomic_save, digest

CONFIG = ROOT / "configs/light_label/polarity_queries.yaml"


def load_config(path=CONFIG):
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    allowed = set("route_name data_dir vocab_path cache_dir result_root checkpoint_root label_names num_labels embedding_dim gru_hidden_size gru_layers bidirectional attention_heads auxiliary_weight dos_weight_multiplier positive_auxiliary_multiplier negative_auxiliary_multiplier positive_auxiliary_label_multiplier negative_auxiliary_label_multiplier adaptive_auxiliary adaptive_auxiliary_mode dos_soft_targets dos_soft_positive dos_soft_negative max_len batch_size gradient_accumulation_steps learning_rate weight_decay epochs early_stopping_patience seed amp weighted_bce pos_weight_mode max_pos_weight thresholds num_workers allow_test".split())
    optional = {"dos_weight_multiplier", "positive_auxiliary_multiplier", "negative_auxiliary_multiplier",
                "positive_auxiliary_label_multiplier", "negative_auxiliary_label_multiplier",
                "adaptive_auxiliary", "adaptive_auxiliary_mode", "dos_soft_targets",
                "dos_soft_positive", "dos_soft_negative"}
    required = allowed - optional
    if not required.issubset(c) or set(c) - allowed:
        raise ValueError(f"Configuration keys mismatch: {set(c) ^ allowed}")
    c.setdefault("dos_weight_multiplier", 1.0)
    c.setdefault("positive_auxiliary_multiplier", 1.0)
    c.setdefault("negative_auxiliary_multiplier", 1.0)
    c.setdefault("positive_auxiliary_label_multiplier", [1.0] * 6)
    c.setdefault("negative_auxiliary_label_multiplier", [1.0] * 6)
    c.setdefault("adaptive_auxiliary", False)
    c.setdefault("adaptive_auxiliary_mode", "none")
    c.setdefault("dos_soft_targets", False)
    c.setdefault("dos_soft_positive", 0.8)
    c.setdefault("dos_soft_negative", 0.1)
    if len(c["positive_auxiliary_label_multiplier"]) != 6 or len(c["negative_auxiliary_label_multiplier"]) != 6:
        raise ValueError("Auxiliary label multipliers must contain six values")
    if not 0.0 < float(c["dos_soft_negative"]) < float(c["dos_soft_positive"]) < 1.0:
        raise ValueError("DoS soft targets must satisfy 0 < negative < positive < 1")
    for k,v in {"data_dir":"data/processed/DIVE_main6_opcode_process01", "num_labels":6,
                "gru_layers":1,"bidirectional":True,"allow_test":False}.items():
        if c[k] != v: raise ValueError(f"Protocol mismatch: {k}")
    return c


def provenance(c):
    names = ["scripts/run_polarity_queries.py", "src/polarity_query_model.py", "src/light_label_model.py",
             "src/light_label_data.py", "src/train_light_label_model.py", "src/metrics.py", "src/evm_tokenizer.py",
             "scripts/run_hidden384_trials.py", "configs/light_label/polarity_queries.yaml", c["vocab_path"]]
    names += [f"{c['cache_dir']}/{s}_max{c['max_len']}.pt" for s in ("train","valid")]
    names += [f"{c['data_dir']}/{s}.jsonl" for s in ("train","valid")]
    return {n:digest(ROOT/n) for n in names}


def signature(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def datasets(c):
    result = []
    for split in ("train","valid"):
        p = ROOT/c["cache_dir"]/f"{split}_max{c['max_len']}.pt"
        payload = torch.load(p,map_location="cpu")
        if payload["max_len"] != c["max_len"] or payload["labels"].shape[1] != 6:
            raise ValueError("Cache dimensions mismatch")
        if len(set(payload["ids"])) != len(payload["ids"]): raise ValueError("Duplicate IDs within split")
        if not torch.isfinite(payload["labels"]).all(): raise ValueError("Invalid labels")
        # Validate cache identity against the named split without touching test.
        with (ROOT/c["data_dir"]/f"{split}.jsonl").open(encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        if [str(x["id"]) for x in raw] != payload["ids"]: raise ValueError("Cache ID order mismatch")
        if not torch.equal(torch.tensor([x["multi_labels"] for x in raw]).float(), payload["labels"]):
            raise ValueError("Cache labels mismatch")
        result.append(LightLabelDataset(p,runtime_max_len=c["max_len"]))
    return result


def compute_weights(c, data):
    value = pos_weight(data, c).clone()
    dos_id = c["label_names"].index("DoS")
    value[dos_id] = value[dos_id] * float(c.get("dos_weight_multiplier", 1.0))
    return value.clamp(max=float(c["max_pos_weight"]))


def auxiliary_settings(c, data):
    positive = torch.tensor(c["positive_auxiliary_label_multiplier"], dtype=torch.float32)
    negative = torch.tensor(c["negative_auxiliary_label_multiplier"], dtype=torch.float32)
    if c.get("adaptive_auxiliary"):
        labels = data.labels[data.indices]
        counts = labels.sum(0).clamp_min(1)
        ratio = torch.sqrt((len(labels) - counts) / counts)
        positive = ratio / ratio.mean().clamp_min(1e-8)
        negative = torch.ones_like(positive)
    soft = None
    if c.get("dos_soft_targets"):
        soft = {"positive_high": float(c["dos_soft_positive"]),
                "positive_low": float(c["dos_soft_negative"]),
                "negative_low": float(c["dos_soft_negative"]),
                "negative_high": float(c["dos_soft_positive"])}
    return positive, negative, soft


def initialize(mode,c,tok,device="cpu"):
    set_seed(c["seed"])
    template = build_model("P0",c,len(tok),tok.pad_token_id)
    shared = encoder_state(template)
    del template
    set_seed(c["seed"])
    model = build_model(mode,c,len(tok),tok.pad_token_id)
    model.load_state_dict(shared,strict=False)
    if tensor_hash(encoder_state(model)) != tensor_hash(shared): raise ValueError("Encoder initialization mismatch")
    return model.to(device), tensor_hash(shared)


def optimizer_for(model,c):
    optimizer = torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    for g in optimizer.param_groups:
        if g["lr"] != c["learning_rate"] or g["weight_decay"] != c["weight_decay"]: raise ValueError("Optimizer mismatch")
    return optimizer


def forward(model,mode,batch,device,diagnostics=False):
    args = (batch["input_ids"].to(device),batch["lengths"],batch["mask"].to(device))
    return model(*args) if mode == "P0" else model(*args,diagnostics=diagnostics)


def probe(mode,c,tok,train,batch_size,threads):
    torch.set_num_threads(threads)
    model, _ = initialize(mode,c,tok,"cuda")
    optimizer = optimizer_for(model,c)
    scaler = torch.cuda.amp.GradScaler(enabled=c["amp"])
    order = sorted(range(len(train)),key=train.sequence_length,reverse=True)
    longest = collate_light_label([train[i] for i in order[:batch_size]],tok.pad_token_id)
    typical = collate_light_label([train[i] for i in order[len(order)//2:len(order)//2+batch_size]],tok.pad_token_id)
    weight = pos_weight(train,c).cuda()
    torch.cuda.reset_peak_memory_stats()
    times = []
    cpu_times = []
    # Alternate lengths to exercise allocator/workspace reuse, then time
    # repeated longest batches including data transfers and optimizer updates.
    for i in range(6):
        batch = typical if i == 1 else longest
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); start=time.perf_counter(); cpu_start=time.process_time()
        with torch.autocast("cuda",dtype=torch.float16,enabled=c["amp"]):
            # Training and resource sizing do not retain attention maps. They
            # are collected later with a one-contract diagnostic loader.
            out = forward(model,mode,batch,"cuda",diagnostics=False)
            pos_label,neg_label,soft=auxiliary_settings(c,train)
            loss,_,_ = loss_terms(out,batch["labels"].cuda(),weight,c["auxiliary_weight"] if mode in ("P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13") else 0,
                                  c["positive_auxiliary_multiplier"],c["negative_auxiliary_multiplier"],pos_label,neg_label,soft)
        if not torch.isfinite(loss): raise ValueError("Nonfinite probe")
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        if i >= 2: times.append(time.perf_counter()-start); cpu_times.append(time.process_time()-cpu_start)
        del out,loss
    result = {"variant":mode,"threads":threads,"batch_size":batch_size,
            "peak_allocated_mb":torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_mb":torch.cuda.max_memory_reserved()/2**20,
            "contracts_per_second":batch_size/statistics.median(times),
            "cpu_core_equivalents":sum(cpu_times)/sum(times)}
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def select_resources(c,tok,train,prov):
    path=ROOT/c["result_root"]/"resources.json"
    sig=signature({"config":c,"provenance":prov})
    if path.exists():
        saved=json.loads(path.read_text())
        if saved["signature"] != sig: raise ValueError("Resource audit signature changed")
        return saved
    rows=[]
    for batch in (64,32,16,8,4,2,1):
        success=True
        for mode in VARIANTS:
            try:
                r=probe(mode,c,tok,train,batch,2); rows.append(r)
                if r["peak_reserved_mb"] > torch.cuda.get_device_properties(0).total_memory/2**20 * .9:
                    success=False; break
            except torch.cuda.OutOfMemoryError:
                rows.append({"variant":mode,"batch_size":batch,"status":"oom"}); success=False; break
            finally: torch.cuda.empty_cache()
            print(f"[preflight] {mode} batch={batch} passed",flush=True)
        if success: break
    else: raise RuntimeError("No common batch fits; queue paused")
    bench=[probe("P4",c,tok,train,batch,n) for n in (2,4,8)]
    if bench[-1]["cpu_core_equivalents"] >= 6.4 and bench[-1]["contracts_per_second"] > 1.05*bench[1]["contracts_per_second"]:
        bench += [probe("P4",c,tok,train,batch,n) for n in (16,20) if n <= len(os.sched_getaffinity(0))]
    top=max(x["contracts_per_second"] for x in bench)
    selected=min(x["threads"] for x in bench if x["contracts_per_second"] >= top/1.05)
    saved={"signature":sig,"batch_size":batch,"gradient_accumulation_steps":256//batch,
           "threads":selected,"preflight":rows,"cpu_benchmark":bench,"test_checked":False}
    atomic_json(path,saved); print(json.dumps({"resources":saved}),flush=True)
    return saved


@torch.no_grad()
def evaluate(model,mode,loader,c,weight,diagnostics=False):
    model.eval(); logits=[]; targets=[]; ids=[]; losses=[]; energies=[]; measures=[]
    interventions={"zero_positive":[],"zero_negative":[]}
    start=time.perf_counter()
    for batch in loader:
        with torch.autocast("cuda",dtype=torch.float16,enabled=c["amp"]):
            out=forward(model,mode,batch,"cuda",diagnostics)
            _,cls,_=loss_terms(out,batch["labels"].cuda(),weight,0)
        logits.append(out["logits"].float().cpu()); targets.append(batch["labels"]); ids.extend(batch["ids"]); losses.append(float(cls))
        if diagnostics and mode in ("P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13"):
            z=out["representations"].float(); a=out["attention"].float()
            energies.append(out["energies"].float().cpu())
            mid=(a[:,:,0]+a[:,:,1])/2
            js=.5*((a[:,:,0]*(a[:,:,0].clamp_min(1e-10).log()-mid.clamp_min(1e-10).log())).sum(-1)
                   +(a[:,:,1]*(a[:,:,1].clamp_min(1e-10).log()-mid.clamp_min(1e-10).log())).sum(-1))
            measures.append(torch.stack([F.cosine_similarity(z[:,:,0],z[:,:,1],dim=-1),
                F.cosine_similarity(a[:,:,0],a[:,:,1],dim=-1),js],dim=-1).cpu())
            for key,p in (("zero_positive",0),("zero_negative",1)):
                altered=z.clone(); altered[:,:,p]=0
                interventions[key].append(model.score(altered)[0].cpu())
    torch.cuda.synchronize()
    result={"logits":torch.cat(logits),"labels":torch.cat(targets),"ids":ids,"loss":float(np.mean(losses)),
            "inference_seconds":time.perf_counter()-start}
    if energies:
        result.update(energies=torch.cat(energies),measures=torch.cat(measures),
                      interventions={k:torch.cat(v) for k,v in interventions.items()})
    return result


def diagnostic_report(output,metrics,c):
    if "energies" not in output: return {"applicable":False,"test_checked":False}
    e=output["energies"]; y=output["labels"]; diff=e[...,0]-e[...,1]; rows=[]
    for l,label in enumerate(c["label_names"]):
        row={"label":label}
        for state in (0,1):
            sel=y[:,l]==state
            row[str(state)]={"count":int(sel.sum()),"positive_score_mean":float(e[sel,l,0].mean()),
                "negative_score_mean":float(e[sel,l,1].mean()),"margin_mean":float(diff[sel,l].mean()),
                "positive_score_std":float(e[sel,l,0].std(unbiased=False)),
                "negative_score_std":float(e[sel,l,1].std(unbiased=False)),
                "margin_quantiles":torch.quantile(diff[sel,l],torch.tensor([0.1,0.5,0.9])).tolist(),
                "margin_std":float(diff[sel,l].std(unbiased=False)),
                "correct_margin_fraction":float(((diff[sel,l]>0) if state else (diff[sel,l]<0)).float().mean()),
                "representation_cosine":float(output["measures"][sel,l,0].mean()),
                "attention_cosine":float(output["measures"][sel,l,1].mean()),
                "attention_js":float(output["measures"][sel,l,2].mean())}
        rows.append(row)
    interventions={}
    for key,logits in output["interventions"].items():
        m=compute_multilabel_metrics_from_probs(y.numpy().astype(int),logits.sigmoid().numpy(),metrics["thresholds"])
        interventions[key]={"macro_f1":float(m["recognition_macro_f1"]),"micro_f1":float(m["recognition_micro_f1"]),
                            "per_label_f1":[float(x) for x in m["per_label_f1"]]}
    return {"score_definition":"s_l=e_plus_l-e_minus_l","per_label":rows,
            "frozen_threshold_interventions":interventions,"test_checked":False,
            "interpretation":"Contract-label weak supervision and model sensitivity; not local evidence ground truth."}


def train_one(mode,c,tok,train,valid,prov,resources):
    root=ROOT/c["result_root"]/mode; ckpt=ROOT/c["checkpoint_root"]/mode
    sig=signature({"config":c,"mode":mode,"provenance":prov,"resources":resources})
    if (root/"metrics.json").exists():
        r=json.loads((root/"metrics.json").read_text())
        if r["signature"]!=sig: raise ValueError("Existing result mismatch")
        return r
    torch.set_num_threads(resources["threads"])
    effective=dict(c,batch_size=resources["batch_size"],gradient_accumulation_steps=resources["gradient_accumulation_steps"])
    previous_audit = json.loads((root/"audit.json").read_text()) if (root/"audit.json").exists() else None
    model,encoder_hash=initialize(mode,effective,tok,"cuda")
    optimizer=optimizer_for(model,effective); scaler=torch.cuda.amp.GradScaler(enabled=c["amp"])
    weight=compute_weights(c, train).cuda(); aux=c["auxiliary_weight"] if mode in ("P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13") else 0.0
    pos_label,neg_label,soft=auxiliary_settings(c,train)
    audit={"signature":sig,"requested_config":c,"effective_config":effective,"variant":mode,
           "encoder_init_hash":encoder_hash,"model_init_hash":tensor_hash(model.state_dict()),
           "parameter_shapes":{k:list(v.shape) for k,v in model.named_parameters()},
           "params":sum(p.numel() for p in model.parameters()),"auxiliary_weight":aux,
           "positive_auxiliary_multiplier":c["positive_auxiliary_multiplier"],
           "negative_auxiliary_multiplier":c["negative_auxiliary_multiplier"],
           "positive_auxiliary_label_multiplier":pos_label.tolist(),"negative_auxiliary_label_multiplier":neg_label.tolist(),
           "dos_soft_targets":soft,"threads":torch.get_num_threads(),
           "pos_weight":weight.tolist(),"optimizer":{"lr":optimizer.param_groups[0]["lr"],"weight_decay":optimizer.param_groups[0]["weight_decay"]},
           "query_shape":list(model.queries.shape) if mode!="P0" else list(model.label_queries.shape),
           "attention_heads":model.cross_attention.num_heads if mode!="P0" else 1,"provenance":prov,"test_checked":False}
    atomic_json(root/"audit.json",audit)
    print(json.dumps({"start":mode,"encoder_init_hash":encoder_hash,"params":audit["params"],"query_shape":audit["query_shape"],"heads":audit["attention_heads"],"auxiliary_weight":aux,"batch":effective["batch_size"],"threads":resources["threads"]}),flush=True)
    loader=make_loader(train,effective,tok.pad_token_id,True); vl=make_loader(valid,effective,tok.pad_token_id,False)
    history=[]; best=-1.; stale=0; epoch_start=1
    if (ckpt/"last.pt").exists():
        saved=torch.load(ckpt/"last.pt",map_location="cpu")
        if saved["signature"]!=sig:
            same_config = previous_audit is not None and previous_audit.get("requested_config") == c
            old_provenance = previous_audit.get("provenance", {}) if previous_audit else {}
            unchanged_dependencies = all(old_provenance.get(key) == value for key, value in prov.items()
                                         if key != "scripts/run_polarity_queries.py")
            runner_only_change = (set(old_provenance) == set(prov)
                                  and old_provenance.get("scripts/run_polarity_queries.py") != prov.get("scripts/run_polarity_queries.py"))
            if not (same_config and unchanged_dependencies and runner_only_change):
                raise ValueError("Resume signature changed")
            # The only accepted migration is a runner-only change, such as a
            # missing import fix after the previous epoch checkpoint.
            saved["signature"] = sig
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        history,best,stale,epoch_start=saved["history"],saved["best"],saved["stale"],saved["epoch"]+1
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    for epoch in range(epoch_start,c["epochs"]+1):
        if stale>=c["early_stopping_patience"]: break
        atomic_json(root/"status.json",{"status":"running","epoch":epoch,"test_checked":False})
        model.train(); loader.batch_sampler.set_epoch(epoch); optimizer.zero_grad(set_to_none=True)
        start=time.perf_counter(); losses=[]; torch.cuda.reset_peak_memory_stats()
        for step,batch in enumerate(loader,1):
            with torch.autocast("cuda",dtype=torch.float16,enabled=c["amp"]):
                out=forward(model,mode,batch,"cuda")
                total,cls,pol=loss_terms(out,batch["labels"].cuda(),weight,aux,
                                         effective["positive_auxiliary_multiplier"],
                                         effective["negative_auxiliary_multiplier"],
                                         pos_label,neg_label,soft)
            if not torch.isfinite(total): raise RuntimeError("Nonfinite training loss")
            scaler.scale(total/effective["gradient_accumulation_steps"]).backward()
            if step%effective["gradient_accumulation_steps"]==0 or step==len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append([float(total.detach()),float(cls.detach()),float(pol.detach())])
            if step%50==0: print(f"[{mode}] epoch={epoch} step={step}/{len(loader)} cls={float(cls):.6f} polarity={float(pol):.6f}",flush=True)
            del out,total,cls,pol
        duration=time.perf_counter()-start
        output=evaluate(model,mode,vl,effective,weight); m=metric_pack(c,output["labels"],output["logits"])
        avg=np.mean(losses,axis=0)
        row={"epoch":epoch,"train_loss":float(avg[0]),"train_cls":float(avg[1]),"train_polarity":float(avg[2]),
             "valid_loss":output["loss"],"metrics":m,"train_seconds":duration,"contracts_per_second":len(train)/duration,
             "peak_memory_mb":torch.cuda.max_memory_allocated()/2**20}
        history.append(row); score=m["tuned"]["macro_f1"]
        if score>best:
            best,stale=score,0
            atomic_save(ckpt/"best.pt",{"model_state_dict":model.state_dict(),"config":effective,"mode":mode,"metrics":m,"epoch":epoch,"signature":sig})
        else: stale+=1
        atomic_save(ckpt/"last.pt",{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
            "history":history,"best":best,"stale":stale,"epoch":epoch,"signature":sig,
            "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()})
        atomic_json(root/"history.json",history)
        print(f"[{mode}] epoch={epoch} train={row['train_loss']:.6f} valid={row['valid_loss']:.6f} fixed_macro={m['fixed']['macro_f1']:.6f} tuned_macro={score:.6f}",flush=True)
    saved=torch.load(ckpt/"best.pt",map_location="cpu"); model.load_state_dict(saved["model_state_dict"])
    output=evaluate(model,mode,vl,effective,weight,False); m=metric_pack(c,output["labels"],output["logits"])
    diagnostic_loader=DataLoader(valid,batch_size=1,shuffle=False,num_workers=0,
        collate_fn=lambda items: collate_light_label(items,tok.pad_token_id),pin_memory=True)
    diagnostic_output=evaluate(model,mode,diagnostic_loader,effective,weight,True)
    atomic_json(root/"diagnostics.json",diagnostic_report(diagnostic_output,m,c))
    score_artifact = None
    if "energies" in diagnostic_output:
        score_artifact = root / "polarity_scores.pt"
        atomic_save(score_artifact, {"ids": diagnostic_output["ids"],
            "labels": diagnostic_output["labels"],
            "e_plus": diagnostic_output["energies"][..., 0],
            "e_minus": diagnostic_output["energies"][..., 1],
            "s": diagnostic_output["energies"][..., 0] - diagnostic_output["energies"][..., 1],
            "score_definition": "s_l=e_plus_l-e_minus_l",
            "test_checked": False})
    atomic_save(root/"valid_predictions.pt",{k:output[k] for k in ("ids","labels","logits")})
    r=dict(audit,metrics=m,best_epoch=saved["epoch"],history=history,inference_seconds=output["inference_seconds"],
           score_artifact=str(score_artifact.relative_to(ROOT)) if score_artifact else None)
    atomic_json(root/"metrics.json",r); atomic_json(root/"status.json",{"status":"completed","test_checked":False})
    return r


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default=str(CONFIG)); parser.add_argument("--resources-only",action="store_true")
    args=parser.parse_args(); os.chdir(ROOT)
    c=load_config(args.config)
    torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    prov=provenance(c); train,valid=datasets(c); tok=EVMOpcodeTokenizer.from_vocab_file(ROOT/c["vocab_path"])
    print(f"[setup] train={len(train)} valid={len(valid)} dataset={c['data_dir']} test_checked=false",flush=True)
    resources=select_resources(c,tok,train,prov)
    if args.resources_only:return
    rows=[]
    for mode in VARIANTS:
        try: result=train_one(mode,c,tok,train,valid,prov,resources)
        except Exception as exc:
            atomic_json(ROOT/c["result_root"]/mode/"status.json",{"status":"failed","error":str(exc),"test_checked":False})
            raise
        if rows and rows[0]["encoder_init_hash"]!=result["encoder_init_hash"]: raise ValueError("Unequal encoder initialization")
        rows.append({"variant":mode,"macro_f1":result["metrics"]["tuned"]["macro_f1"],"micro_f1":result["metrics"]["tuned"]["micro_f1"],
                     "per_label_f1":result["metrics"]["tuned"]["per_label_f1"],"params":result["params"],"encoder_init_hash":result["encoder_init_hash"],"test_checked":False})
        root=ROOT/c["result_root"]
        atomic_json(root/"summary.json",rows)
        with (root/"summary.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    gain=rows[-1]["macro_f1"]-rows[0]["macro_f1"]
    supported=gain>=.01 and (len(rows)<3 or rows[-1]["macro_f1"]>max(rows[1]["macro_f1"],rows[2]["macro_f1"]))
    atomic_json(root/"decision.json",{"delta_P4_P0":gain,"performance_gate_passed":supported,"test_checked":False,
        "note":"Inspect branch diagnostics before claiming polarity semantics; single-seed validation only."})
    (root/"summary.md").write_text("# Polarity queries\n\nprocess01; seed42; validation only; test_checked=false.\n\n"
        "| Variant | Macro-F1 | Micro-F1 | Delta vs P0 |\n|---|---:|---:|---:|\n"+"".join(
        f"| {r['variant']} | {r['macro_f1']:.6f} | {r['micro_f1']:.6f} | {r['macro_f1']-rows[0]['macro_f1']:+.6f} |\n" for r in rows)
        +f"\nPerformance gate passed: {supported}. Branch diagnostics are not ground-truth localization.\n",encoding="utf-8")
    print("[complete] P0-P4 finished; test_checked=false",flush=True)


if __name__=="__main__": main()
