"""Inference-only causal diagnosis of P11 positive/negative attention supports."""

import csv
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))

from evm_tokenizer import EVMOpcodeTokenizer
from metrics import compute_multilabel_metrics_from_probs
from polarity_query_model import build_model
from run_polarity_queries import datasets, load_config
from train_light_label_model import make_loader


CONFIG=ROOT/"configs/light_label/polarity_queries_followup_p11.yaml"
CHECKPOINT=ROOT/"checkpoints/light_label/polarity_queries_followup_p11/P11/best.pt"
METRICS=ROOT/"results/light_label/polarity_queries_followup_p11/P11/metrics.json"
OUTPUT=ROOT/"results/light_label/p11_causal_diagnosis"
REPORT=ROOT/"reports/p11_causal_diagnosis.md"
LABELS=["Reentrancy","Access Control","Arithmetic","Unchecked Return Values","DoS","Time manipulation"]


def metrics_at_thresholds(labels,logits,thresholds):
    raw=compute_multilabel_metrics_from_probs(labels.numpy().astype(int),torch.sigmoid(logits).numpy(),thresholds)
    return {"macro_f1":float(raw["recognition_macro_f1"]),"micro_f1":float(raw["recognition_micro_f1"]),
            "per_label_f1":[float(x) for x in raw["per_label_f1"]],
            "per_label_precision":[float(x) for x in raw["per_label_precision"]],
            "per_label_recall":[float(x) for x in raw["per_label_recall"]]}


def attention_forward(model,input_ids,lengths,mask,support_mode="normal",diagnostics=False):
    hidden=model.encode(input_ids,lengths); batch,tokens,_=hidden.shape
    heads=model.cross_attention.num_heads; head_dim=model.cross_attention.head_dim
    queries=model.queries.flatten(0,1).unsqueeze(0).expand(batch,-1,-1)
    q=model.cross_attention.q_proj(queries).view(batch,12,heads,head_dim).transpose(1,2)
    k=model.cross_attention.k_proj(hidden).view(batch,tokens,heads,head_dim).transpose(1,2)
    v=model.cross_attention.v_proj(hidden).view(batch,tokens,heads,head_dim).transpose(1,2)
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(head_dim)
    scores=scores.masked_fill(~mask[:,None,None,:],torch.finfo(scores.dtype).min)
    attention=torch.softmax(scores,dim=-1).view(batch,heads,6,2,tokens)
    if support_mode=="average":
        mean=attention.mean(3,keepdim=True); used=mean.expand(-1,-1,-1,2,-1)
    elif support_mode=="swap":
        used=attention.flip(3)
    elif support_mode=="normal":
        used=attention
    else:
        raise ValueError(support_mode)
    flattened=used.reshape(batch,heads,12,tokens)
    attended=torch.matmul(flattened,v).transpose(1,2).contiguous().view(batch,12,model.output_dim)
    evidence=model.cross_attention.out_proj(attended).view(batch,6,2,model.output_dim)
    logits,energies=model.score(evidence)
    result={"logits":logits,"energies":energies}
    if diagnostics: result["attention"]=attention.mean(1)
    return result


def symmetric_js(first,second):
    middle=.5*(first+second)
    return .5*((first*(first.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum(-1)
               +(second*(second.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum(-1))


def margin_statistics(labels,logits):
    rows=[]
    for index,name in enumerate(LABELS):
        positive=logits[labels[:,index]>0.5,index]; negative=logits[labels[:,index]<0.5,index]
        rows.append({"label":name,"positive_mean":float(positive.mean()),"positive_median":float(positive.median()),
                     "negative_mean":float(negative.mean()),"negative_median":float(negative.median()),
                     "mean_separation":float(positive.mean()-negative.mean())})
    return rows


@torch.no_grad()
def run():
    config=load_config(CONFIG)
    if config["allow_test"]: raise ValueError("test must remain locked")
    _,valid=datasets(config); tokenizer=EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"])
    loader=make_loader(valid,config,tokenizer.pad_token_id,False)
    model=build_model("P11",config,len(tokenizer),tokenizer.pad_token_id).cuda().eval()
    model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu")["model_state_dict"])
    thresholds=json.loads(METRICS.read_text(encoding="utf-8"))["metrics"]["thresholds"]
    mask_id=tokenizer.vocab[tokenizer.mask_token]
    storage={name:[] for name in ("normal","average_support","swapped_support","mask_positive_top5",
                                  "mask_negative_top5","mask_union","mask_random5","mask_random_union")}
    labels=[]; ids=[]; attention_cos=torch.zeros(6); attention_js=torch.zeros(6); overlap=torch.zeros(6); count=0
    start=time.perf_counter(); checked=False; row_offset=0
    for batch in loader:
        input_ids=batch["input_ids"].cuda(); mask=batch["mask"].cuda(); lengths=batch["lengths"]
        with torch.autocast("cuda",dtype=torch.float16,enabled=config["amp"]):
            standard=model(input_ids,lengths,mask,diagnostics=True)
            normal=attention_forward(model,input_ids,lengths,mask,"normal",True)
            average=attention_forward(model,input_ids,lengths,mask,"average")
            swapped=attention_forward(model,input_ids,lengths,mask,"swap")
        if not checked:
            difference=float((standard["logits"].float()-normal["logits"].float()).abs().max())
            if difference>1e-5: raise ValueError(f"manual P11 reconstruction mismatch: {difference}")
            checked=True
        storage["normal"].append(normal["logits"].float().cpu())
        storage["average_support"].append(average["logits"].float().cpu())
        storage["swapped_support"].append(swapped["logits"].float().cpu())
        labels.append(batch["labels"]); ids.extend(batch["ids"])
        attention=normal["attention"].float().cpu(); positive=attention[:,:,0]; negative=attention[:,:,1]
        attention_cos+=torch.nn.functional.cosine_similarity(positive,negative,dim=-1).sum(0)
        attention_js+=symmetric_js(positive,negative).sum(0)
        valid_cpu=batch["mask"]
        pos_top=positive.masked_fill(~valid_cpu[:,None,:],-1).topk(5,dim=-1).indices
        neg_top=negative.masked_fill(~valid_cpu[:,None,:],-1).topk(5,dim=-1).indices
        overlap+=(pos_top.unsqueeze(-1)==neg_top.unsqueeze(-2)).any(-1).float().sum(-1).sum(0)/5.0
        count+=len(batch["ids"])

        condition_logits={name:[] for name in ("mask_positive_top5","mask_negative_top5","mask_union","mask_random5","mask_random_union")}
        for label in range(6):
            variants={name:input_ids.clone() for name in condition_logits}
            for row in range(len(batch["ids"])):
                pos=pos_top[row,label].tolist(); neg=neg_top[row,label].tolist(); union=sorted(set(pos+neg))
                variants["mask_positive_top5"][row,pos]=mask_id
                variants["mask_negative_top5"][row,neg]=mask_id
                variants["mask_union"][row,union]=mask_id
                valid=torch.where(mask[row])[0].tolist(); rng=random.Random(42+(row_offset+row)*101+label)
                variants["mask_random5"][row,rng.sample(valid,min(5,len(valid)))]=mask_id
                variants["mask_random_union"][row,rng.sample(valid,min(len(union),len(valid)))]=mask_id
            for name,values in variants.items():
                with torch.autocast("cuda",dtype=torch.float16,enabled=config["amp"]):
                    logits=model(values,lengths,mask)["logits"][:,label]
                condition_logits[name].append(logits.float().cpu())
        for name,values in condition_logits.items(): storage[name].append(torch.stack(values,dim=1))
        row_offset+=len(batch["ids"])
    labels=torch.cat(labels); joined={name:torch.cat(values) for name,values in storage.items()}
    results={}
    for name,logits in joined.items():
        results[name]={"metrics":metrics_at_thresholds(labels,logits,thresholds),
                       "margin":margin_statistics(labels,logits)}
    normal_macro=results["normal"]["metrics"]["macro_f1"]
    for name in results: results[name]["macro_f1_delta_vs_normal"]=results[name]["metrics"]["macro_f1"]-normal_macro
    branch=[]
    normal_f1=results["normal"]["metrics"]["per_label_f1"]
    pos_f1=results["mask_positive_top5"]["metrics"]["per_label_f1"]
    neg_f1=results["mask_negative_top5"]["metrics"]["per_label_f1"]
    union_f1=results["mask_union"]["metrics"]["per_label_f1"]
    for index,label in enumerate(LABELS):
        pos_drop=normal_f1[index]-pos_f1[index]; neg_drop=normal_f1[index]-neg_f1[index]
        union_drop=normal_f1[index]-union_f1[index]
        branch.append({"label":label,"positive_top5_f1_drop":pos_drop,"negative_top5_f1_drop":neg_drop,
                       "union_f1_drop":union_drop,"union_excess_over_stronger_branch":union_drop-max(pos_drop,neg_drop),
                       "dominant_branch":"positive" if pos_drop>neg_drop else "negative"})
    report={"route":"P11 inference-only causal support diagnosis","dataset":"DIVE_main6_opcode_process01",
            "seed":42,"thresholds":thresholds,"conditions":results,"branch_complementarity":branch,
            "attention_alignment":{"positive_negative_cosine":(attention_cos/count).tolist(),
                                   "positive_negative_js":(attention_js/count).tolist(),
                                   "top5_overlap":(overlap/count).tolist()},
            "inference_seconds":time.perf_counter()-start,"test_checked":False}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"causal_diagnosis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (OUTPUT/"per_label_causal.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=["label","normal_f1"]+[name+"_f1" for name in results if name!="normal"]+["positive_top5_f1_drop","negative_top5_f1_drop","union_f1_drop","union_excess_over_stronger_branch","dominant_branch"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for index,item in enumerate(branch):
            row={"label":LABELS[index],"normal_f1":normal_f1[index],**item}
            for name in results:
                if name!="normal": row[name+"_f1"]=results[name]["metrics"]["per_label_f1"][index]
            writer.writerow(row)
    lines=["# P11 Causal Support Diagnosis","","Inference-only; process01 validation; seed 42; frozen P11 checkpoint and thresholds; test_checked=false.","",
           "| Condition | Macro-F1 | Delta vs normal | Micro-F1 |","|---|---:|---:|---:|"]
    for name,item in results.items(): lines.append(f"| {name} | {item['metrics']['macro_f1']:.6f} | {item['macro_f1_delta_vs_normal']:+.6f} | {item['metrics']['micro_f1']:.6f} |")
    lines.extend(["","## Per-label branch complementarity","","| Label | Positive Top-5 drop | Negative Top-5 drop | Union drop | Union excess | Dominant branch |","|---|---:|---:|---:|---:|---|"])
    for item in branch: lines.append(f"| {item['label']} | {item['positive_top5_f1_drop']:+.6f} | {item['negative_top5_f1_drop']:+.6f} | {item['union_f1_drop']:+.6f} | {item['union_excess_over_stronger_branch']:+.6f} | {item['dominant_branch']} |")
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


if __name__=="__main__": print(json.dumps(run(),indent=2))
