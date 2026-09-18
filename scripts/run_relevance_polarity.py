"""Audited B1/B2 relevance-polarity architecture validation on process01."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evm_tokenizer import EVMOpcodeTokenizer
from light_label_data import LightLabelDataset, collate_light_label
from polarity_query_model import build_model as build_p11, encoder_state, loss_terms, tensor_hash
from relevance_polarity_query_model import build_relevance_polarity_model
from run_hidden384_trials import atomic_json, atomic_save, digest
from run_polarity_queries import auxiliary_settings, compute_weights, datasets, load_config
from train_light_label_model import make_loader, metric_pack, set_seed


CONFIGS = {
    "B1": ROOT / "configs/light_label/relevance_polarity/b1_positive_locator.yaml",
    "B2": ROOT / "configs/light_label/relevance_polarity/b2_dedicated_relevance.yaml",
}
REFERENCE_CONFIG = ROOT / "configs/light_label/relevance_polarity/a0_p11_reference.yaml"
REFERENCE_CHECKPOINT = ROOT / "checkpoints/light_label/polarity_queries_followup_p11/P11/best.pt"
REFERENCE_METRICS = ROOT / "results/light_label/polarity_queries_followup_p11/P11/metrics.json"
REPORT_ROOT = ROOT / "reports/relevance_polarity"
FROZEN = {
    "data_dir": "data/processed/DIVE_main6_opcode_process01",
    "num_labels": 6, "embedding_dim": 128, "gru_hidden_size": 384,
    "gru_layers": 1, "bidirectional": True, "attention_heads": 4,
    "auxiliary_weight": 0.1, "dos_weight_multiplier": 1.0,
    "dos_soft_targets": True, "dos_soft_positive": 0.8,
    "dos_soft_negative": 0.1, "max_len": 8192, "batch_size": 64,
    "gradient_accumulation_steps": 4, "learning_rate": 0.001,
    "weight_decay": 0.0001, "epochs": 30,
    "early_stopping_patience": 5, "seed": 42, "amp": True,
    "weighted_bce": True, "pos_weight_mode": "sqrt_ratio",
    "max_pos_weight": 5.0, "num_workers": 0, "allow_test": False,
}


def validate_frozen_config(config):
    for key, expected in FROZEN.items():
        if config.get(key) != expected:
            raise ValueError(f"frozen protocol mismatch for {key}: {config.get(key)!r} != {expected!r}")
    if config.get("scheduler", "none") != "none":
        raise ValueError("scheduler must remain none")


def config_provenance(path, config):
    names = [
        path, ROOT / "src/relevance_polarity_query_model.py",
        ROOT / "src/polarity_query_model.py", ROOT / "src/light_label_model.py",
        ROOT / "scripts/run_relevance_polarity.py",
        ROOT / config["cache_dir"] / "train_max8192.pt",
        ROOT / config["cache_dir"] / "valid_max8192.pt",
        ROOT / config["data_dir"] / "train.jsonl",
        ROOT / config["data_dir"] / "valid.jsonl",
        ROOT / config["vocab_path"],
    ]
    return {str(Path(name).resolve().relative_to(ROOT)): digest(name) for name in names}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def initialize(variant, config, tokenizer, device):
    set_seed(config["seed"])
    template = build_p11("P11", config, len(tokenizer), tokenizer.pad_token_id)
    shared_encoder = encoder_state(template)
    p11_params = sum(parameter.numel() for parameter in template.parameters())
    del template
    set_seed(config["seed"])
    model = build_relevance_polarity_model(
        variant, config, len(tokenizer), tokenizer.pad_token_id)
    model.load_state_dict(shared_encoder, strict=False)
    if tensor_hash(encoder_state(model)) != tensor_hash(shared_encoder):
        raise ValueError("encoder initialization differs from P11")
    return model.to(device), tensor_hash(shared_encoder), p11_params


def subset(dataset, count, seed):
    rng = random.Random(seed)
    dataset.indices = sorted(rng.sample(dataset.indices, min(count, len(dataset.indices))))
    return dataset


def loss_settings(config, train_data):
    return auxiliary_settings(config, train_data)


@torch.no_grad()
def evaluate(model, loader, config, device, pos_weight, settings, diagnostics=False, **overrides):
    model.eval(); logits=[]; labels=[]; ids=[]; losses=[]; energies=[]
    positive, negative, soft = settings
    start = time.perf_counter()
    for batch in loader:
        amp = bool(config["amp"] and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(batch["input_ids"].to(device), batch["lengths"],
                           batch["mask"].to(device), diagnostics=diagnostics, **overrides)
            total, _, _ = loss_terms(
                output, batch["labels"].to(device), pos_weight,
                config["auxiliary_weight"], config["positive_auxiliary_multiplier"],
                config["negative_auxiliary_multiplier"], positive, negative, soft)
        logits.append(output["logits"].float().cpu()); labels.append(batch["labels"])
        ids.extend(batch["ids"]); losses.append(float(total))
        if diagnostics:
            energies.append(output["energies"].float().cpu())
    if device.type == "cuda": torch.cuda.synchronize()
    result = {"logits": torch.cat(logits), "labels": torch.cat(labels), "ids": ids,
              "loss": float(np.mean(losses)), "inference_seconds": time.perf_counter() - start}
    if energies: result["energies"] = torch.cat(energies)
    return result


def train_one(variant, smoke=False):
    config_path = CONFIGS[variant]
    config = load_config(config_path); validate_frozen_config(config)
    train_data, valid_data = datasets(config)
    if smoke:
        subset(train_data, 512, 42); subset(valid_data, 256, 43)
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required")
    torch.set_num_threads(2)
    model, encoder_hash, p11_params = initialize(variant, config, tokenizer, device)
    params = sum(parameter.numel() for parameter in model.parameters())
    expected_delta = 0 if variant == "B1" else 4608
    if params - p11_params != expected_delta:
        raise ValueError(f"parameter fairness failed: {params} - {p11_params} != {expected_delta}")
    run_name = f"smoke_{variant}" if smoke else variant
    result_dir = ROOT / config["result_root"] / run_name
    checkpoint_dir = ROOT / config["checkpoint_root"] / run_name
    provenance = config_provenance(config_path, config)
    run_signature = signature({"variant": variant, "config": config,
                               "provenance": provenance, "smoke": smoke})
    result_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_data, config, tokenizer.pad_token_id, True)
    valid_loader = make_loader(valid_data, config, tokenizer.pad_token_id, False)
    pos_weight = compute_weights(config, train_data).to(device)
    settings = loss_settings(config, train_data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    epochs = 1 if smoke else config["epochs"]
    best=-1.0; stale=0; history=[]; start_epoch=1
    last_path = checkpoint_dir / "last.pt"
    if last_path.exists() and not smoke:
        saved = torch.load(last_path, map_location="cpu")
        if saved["signature"] != run_signature: raise ValueError("resume signature mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"]); history=saved["history"]
        best=saved["best"]; stale=saved["stale"]; start_epoch=saved["epoch"]+1
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
    audit = {"variant": variant, "requested_config": config, "params": params,
             "p11_params": p11_params, "parameter_delta_vs_p11": params-p11_params,
             "query_shapes": {"polarity": list(model.polarity_queries.shape),
                              "relevance": None if model.relevance_queries is None else list(model.relevance_queries.shape)},
             "encoder_init_hash": encoder_hash, "provenance": provenance,
             "optimizer": {"name":"AdamW", "lr":optimizer.param_groups[0]["lr"],
                           "weight_decay":optimizer.param_groups[0]["weight_decay"], "scheduler":"none"},
             "pos_weight":pos_weight.detach().cpu().tolist(), "test_checked":False}
    atomic_json(result_dir/"audit.json", audit)
    for epoch in range(start_epoch, epochs+1):
        if stale >= config["early_stopping_patience"]: break
        model.train(); train_loader.batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True); losses=[]; start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        positive, negative, soft = settings
        for step,batch in enumerate(train_loader,1):
            amp = bool(config["amp"])
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
                output=model(batch["input_ids"].to(device),batch["lengths"],batch["mask"].to(device))
                total,classification,polarity=loss_terms(
                    output,batch["labels"].to(device),pos_weight,config["auxiliary_weight"],
                    config["positive_auxiliary_multiplier"],config["negative_auxiliary_multiplier"],
                    positive,negative,soft)
            scaler.scale(total/config["gradient_accumulation_steps"]).backward()
            if step%config["gradient_accumulation_steps"]==0 or step==len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append([float(total.detach()),float(classification.detach()),float(polarity.detach())])
            if step%50==0:
                print(f"[{variant}] epoch={epoch} step={step}/{len(train_loader)} cls={float(classification):.6f} polarity={float(polarity):.6f}",flush=True)
        elapsed=time.perf_counter()-start
        valid=evaluate(model,valid_loader,config,device,pos_weight,settings)
        metrics=metric_pack(config,valid["labels"],valid["logits"]); average=np.mean(losses,axis=0)
        row={"epoch":epoch,"train_loss":float(average[0]),"train_cls":float(average[1]),
             "train_polarity":float(average[2]),"valid_loss":valid["loss"],"metrics":metrics,
             "epoch_seconds":elapsed,"peak_memory_mb":torch.cuda.max_memory_allocated()/2**20}
        history.append(row); score=metrics["tuned"]["macro_f1"]
        if score>best:
            best=score; stale=0
            atomic_save(checkpoint_dir/"best.pt",{"model_state_dict":model.state_dict(),"config":config,
                        "variant":variant,"epoch":epoch,"metrics":metrics,"signature":run_signature,"test_checked":False})
        else: stale+=1
        atomic_save(last_path,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                    "history":history,"best":best,"stale":stale,"epoch":epoch,"signature":run_signature,
                    "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all()})
        atomic_json(result_dir/"history.json",history)
        print(f"[{variant}] epoch={epoch} train={row['train_loss']:.6f} valid={row['valid_loss']:.6f} fixed_macro={metrics['fixed']['macro_f1']:.6f} tuned_macro={score:.6f}",flush=True)
    saved=torch.load(checkpoint_dir/"best.pt",map_location="cpu"); model.load_state_dict(saved["model_state_dict"])
    final=evaluate(model,valid_loader,config,device,pos_weight,settings,diagnostics=True)
    metrics=metric_pack(config,final["labels"],final["logits"])
    atomic_save(result_dir/"valid_predictions.pt",{"ids":final["ids"],"labels":final["labels"],
                "logits":final["logits"],"thresholds":metrics["thresholds"],"test_checked":False})
    atomic_save(result_dir/"polarity_scores.pt",{"ids":final["ids"],"labels":final["labels"],
                "e_plus":final["energies"][...,0],"e_minus":final["energies"][...,1],
                "s":final["energies"][...,0]-final["energies"][...,1],"test_checked":False})
    summary=dict(audit,metrics=metrics,best_epoch=saved["epoch"],history=history,
                 mean_epoch_seconds=float(np.mean([row["epoch_seconds"] for row in history])),
                 peak_memory_mb=max(row["peak_memory_mb"] for row in history),
                 inference_seconds=final["inference_seconds"])
    atomic_json(result_dir/"metrics.json",summary)
    return summary


@torch.no_grad()
def regression_check():
    config=load_config(REFERENCE_CONFIG); validate_frozen_config(config)
    train_data,valid_data=datasets(config)
    tokenizer=EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"])
    model=build_p11("P11",config,len(tokenizer),tokenizer.pad_token_id).cuda()
    checkpoint=torch.load(REFERENCE_CHECKPOINT,map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    loader=make_loader(valid_data,config,tokenizer.pad_token_id,False)
    pos_weight=compute_weights(config,train_data).cuda()
    from run_polarity_queries import evaluate as evaluate_p11
    output=evaluate_p11(model,"P11",loader,config,pos_weight,False)
    metrics=metric_pack(config,output["labels"],output["logits"])
    reference=json.loads(REFERENCE_METRICS.read_text(encoding="utf-8"))["metrics"]
    delta=abs(metrics["tuned"]["macro_f1"]-reference["tuned"]["macro_f1"])
    if delta>=1e-5: raise ValueError(f"P11 regression mismatch: {delta}")
    report={"checkpoint":str(REFERENCE_CHECKPOINT.relative_to(ROOT)),"recomputed":metrics,
            "reference":reference,"macro_f1_abs_delta":delta,"passed":True,"test_checked":False}
    REPORT_ROOT.mkdir(parents=True,exist_ok=True); atomic_json(REPORT_ROOT/"p11_regression.json",report)
    return report


def shape_smoke(variant):
    config=load_config(CONFIGS[variant]); validate_frozen_config(config)
    train_data,_=datasets(config); tokenizer=EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"])
    device=torch.device("cuda"); model,_,p11_params=initialize(variant,config,tokenizer,device)
    loader=make_loader(subset(train_data,64,100+len(variant)),config,tokenizer.pad_token_id,False)
    batch=next(iter(loader)); output=model(batch["input_ids"].cuda(),batch["lengths"],batch["mask"].cuda(),diagnostics=True)
    alpha=output["alpha_rel"]; valid=batch["mask"].cuda()
    alpha_error=float((alpha.sum(-1)-1).abs().max())
    padding_mass=float((alpha*(~valid)[:,None,None,:]).abs().max())
    expected={"H":[len(batch["ids"]),batch["input_ids"].shape[1],768],"alpha_rel":[len(batch["ids"]),6,4,batch["input_ids"].shape[1]],
              "z_rel":[len(batch["ids"]),6,4,192],"gates":[6,2,4,192],"representations":[len(batch["ids"]),6,2,768],
              "energies":[len(batch["ids"]),6,2],"logits":[len(batch["ids"]),6]}
    actual={"H":list(output["hidden_shape"]),"alpha_rel":list(alpha.shape),"z_rel":list(output["z_rel"].shape),
            "gates":list(output["gates"].shape),"representations":list(output["representations"].shape),
            "energies":list(output["energies"].shape),"logits":list(output["logits"].shape)}
    if actual!=expected or alpha_error>1e-5 or padding_mass>1e-7:
        raise ValueError({"actual":actual,"expected":expected,"alpha_error":alpha_error,"padding_mass":padding_mass})
    params=sum(p.numel() for p in model.parameters())
    return {"variant":variant,"actual_shapes":actual,"alpha_sum_max_error":alpha_error,
            "padding_attention_max":padding_mass,"params":params,"delta_vs_p11":params-p11_params,
            "shared_support":True,"test_checked":False}


def write_smoke_report(rows):
    REPORT_ROOT.mkdir(parents=True,exist_ok=True)
    lines=["# Relevance–Polarity Smoke Test","","Dataset: process01 train subset 512 / valid subset 256; seed 42; test locked.",""]
    for row in rows:
        lines.extend([f"## {row['variant']}","",f"- shapes: `{json.dumps(row['actual_shapes'])}`",
                      f"- alpha sum maximum error: `{row['alpha_sum_max_error']:.3e}`",
                      f"- maximum padding attention: `{row['padding_attention_max']:.3e}`",
                      f"- parameters: `{row['params']}`; delta vs P11: `{row['delta_vs_p11']}`",
                      "- positive and negative branches consume the same `alpha_rel` and `z_rel` by construction.",""])
    (REPORT_ROOT/"smoke_test.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("audit","smoke","train")); args=parser.parse_args()
    os.chdir(ROOT); REPORT_ROOT.mkdir(parents=True,exist_ok=True)
    if args.command=="audit":
        print(json.dumps(regression_check(),indent=2)); return
    if args.command=="smoke":
        rows=[]
        for variant in ("B1","B2"):
            rows.append(shape_smoke(variant)); train_one(variant,smoke=True)
        write_smoke_report(rows); print(json.dumps(rows,indent=2)); return
    rows=[train_one(variant,smoke=False) for variant in ("B1","B2")]
    print(json.dumps({row["variant"]:row["metrics"] for row in rows},indent=2))


if __name__=="__main__": main()
