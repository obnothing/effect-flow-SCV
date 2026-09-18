"""Mechanism diagnostics D1-D8 for aligned relevance-polarity models."""

import csv
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))

from evm_tokenizer import EVMOpcodeTokenizer
from metrics import compute_multilabel_metrics_from_probs
from polarity_query_model import build_model as build_p11
from relevance_polarity_query_model import build_relevance_polarity_model
from run_polarity_queries import compute_weights, datasets, load_config
from run_relevance_polarity import CONFIGS, REFERENCE_CHECKPOINT, REFERENCE_METRICS, evaluate, loss_settings
from train_light_label_model import make_loader


RESULT_ROOT=ROOT/"results/light_label/relevance_polarity"
REPORT_ROOT=ROOT/"reports/relevance_polarity"
LABELS=["Reentrancy","Access Control","Arithmetic","Unchecked Return Values","DoS","Time manipulation"]


def frozen_metrics(labels, logits, thresholds):
    raw=compute_multilabel_metrics_from_probs(labels.numpy().astype(int),torch.sigmoid(logits).numpy(),thresholds)
    return {"macro_f1":float(raw["recognition_macro_f1"]),"micro_f1":float(raw["recognition_micro_f1"]),
            "per_label_f1":[float(x) for x in raw["per_label_f1"]],
            "per_label_precision":[float(x) for x in raw["per_label_precision"]],
            "per_label_recall":[float(x) for x in raw["per_label_recall"]]}


def load_variant(variant):
    config=load_config(CONFIGS[variant]); tokenizer=EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"])
    model=build_relevance_polarity_model(variant,config,len(tokenizer),tokenizer.pad_token_id).cuda()
    payload=torch.load(ROOT/config["checkpoint_root"]/variant/"best.pt",map_location="cpu")
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    train,valid=datasets(config); loader=make_loader(valid,config,tokenizer.pad_token_id,False)
    weight=compute_weights(config,train).cuda(); settings=loss_settings(config,train)
    metrics=json.loads((ROOT/config["result_root"]/variant/"metrics.json").read_text(encoding="utf-8"))
    return config,tokenizer,model,train,valid,loader,weight,settings,metrics


def symmetric_js(first,second):
    midpoint=.5*(first+second)
    return .5*((first*(first.clamp_min(1e-12).log()-midpoint.clamp_min(1e-12).log())).sum(-1)
               +(second*(second.clamp_min(1e-12).log()-midpoint.clamp_min(1e-12).log())).sum(-1))


def topk_overlap(first,second,k=5):
    a=first.topk(min(k,first.shape[-1]),dim=-1).indices
    b=second.topk(min(k,second.shape[-1]),dim=-1).indices
    return (a.unsqueeze(-1)==b.unsqueeze(-2)).any(-1).float().sum(-1)/float(k)


@torch.no_grad()
def intervention_metrics(model,loader,config,weight,settings,thresholds,**overrides):
    output=evaluate(model,loader,config,torch.device("cuda"),weight,settings,**overrides)
    return frozen_metrics(output["labels"],output["logits"],thresholds)


@torch.no_grad()
def aligned_distribution_diagnostics(model,loader):
    pair_cos=torch.zeros(6,6); pair_js=torch.zeros(6,6); count=0
    rep_cos_sum=torch.zeros(6); rep_count=0
    for batch in loader:
        out=model(batch["input_ids"].cuda(),batch["lengths"],batch["mask"].cuda(),diagnostics=True)
        alpha=out["alpha_rel"].float().mean(2).cpu()
        for left in range(6):
            for right in range(6):
                pair_cos[left,right]+=torch.nn.functional.cosine_similarity(alpha[:,left],alpha[:,right],dim=-1).sum()
                pair_js[left,right]+=symmetric_js(alpha[:,left],alpha[:,right]).sum()
        rep=out["representations"].float().cpu()
        rep_cos_sum+=torch.nn.functional.cosine_similarity(rep[:,:,0],rep[:,:,1],dim=-1).sum(0)
        count+=len(batch["ids"]); rep_count+=len(batch["ids"])
    gates=model(torch.tensor([[1]],device="cuda"),torch.tensor([1]),torch.tensor([[True]],device="cuda"),diagnostics=True)["gates"].float().cpu()
    gate_cos=torch.nn.functional.cosine_similarity(gates[:,0].flatten(1),gates[:,1].flatten(1),dim=-1)
    return {"label_pair_attention_cosine":(pair_cos/count).tolist(),"label_pair_attention_js":(pair_js/count).tolist(),
            "gate_positive_negative_cosine":gate_cos.tolist(),
            "representation_positive_negative_cosine":(rep_cos_sum/rep_count).tolist()}


@torch.no_grad()
def p11_alignment_diagnostics(config,tokenizer,valid):
    model=build_p11("P11",config,len(tokenizer),tokenizer.pad_token_id).cuda()
    model.load_state_dict(torch.load(REFERENCE_CHECKPOINT,map_location="cpu")["model_state_dict"]); model.eval()
    loader=make_loader(valid,config,tokenizer.pad_token_id,False)
    sums={name:torch.zeros(6) for name in ("cosine","js","top5_overlap")}; count=0
    for batch in loader:
        out=model(batch["input_ids"].cuda(),batch["lengths"],batch["mask"].cuda(),diagnostics=True)
        attention=out["attention"].float().cpu(); positive=attention[:,:,0]; negative=attention[:,:,1]
        sums["cosine"]+=torch.nn.functional.cosine_similarity(positive,negative,dim=-1).sum(0)
        sums["js"]+=symmetric_js(positive,negative).sum(0)
        sums["top5_overlap"]+=topk_overlap(positive,negative,5).sum(0); count+=len(batch["ids"])
    return {key:(value/count).tolist() for key,value in sums.items()}


@torch.no_grad()
def localization_perturbation(model,loader,tokenizer,thresholds):
    normal_logits=[]; top_logits=[]; random_logits=[]; labels=[]; row_offset=0
    mask_id=tokenizer.vocab[tokenizer.mask_token]
    for batch in loader:
        input_ids=batch["input_ids"].cuda(); mask=batch["mask"].cuda()
        out=model(input_ids,batch["lengths"],mask,diagnostics=True)
        normal_logits.append(out["logits"].float().cpu()); labels.append(batch["labels"])
        relevance=out["alpha_rel"].float().mean(2)
        top_by_label=[]; random_by_label=[]
        for label in range(6):
            ranked=relevance[:,label].masked_fill(~mask,-1).topk(5,dim=-1).indices
            top_ids=input_ids.clone(); random_ids=input_ids.clone()
            for row in range(len(batch["ids"])):
                top_ids[row,ranked[row]]=mask_id
                valid_positions=torch.where(mask[row])[0].tolist()
                rng=random.Random(42+(row_offset+row)*101+label)
                selected=rng.sample(valid_positions,min(5,len(valid_positions)))
                random_ids[row,selected]=mask_id
            top_by_label.append(model(top_ids,batch["lengths"],mask)["logits"][:,label].float().cpu())
            random_by_label.append(model(random_ids,batch["lengths"],mask)["logits"][:,label].float().cpu())
        top_logits.append(torch.stack(top_by_label,dim=1)); random_logits.append(torch.stack(random_by_label,dim=1))
        row_offset+=len(batch["ids"])
    labels=torch.cat(labels); normal=torch.cat(normal_logits); top=torch.cat(top_logits); random_values=torch.cat(random_logits)
    return {"normal":frozen_metrics(labels,normal,thresholds),"mask_top5":frozen_metrics(labels,top,thresholds),
            "mask_random5":frozen_metrics(labels,random_values,thresholds),
            "macro_drop_top5":frozen_metrics(labels,normal,thresholds)["macro_f1"]-frozen_metrics(labels,top,thresholds)["macro_f1"],
            "macro_drop_random5":frozen_metrics(labels,normal,thresholds)["macro_f1"]-frozen_metrics(labels,random_values,thresholds)["macro_f1"]}


def margin_report(path):
    payload=torch.load(path,map_location="cpu"); labels=payload["labels"]; margin=payload["s"]; rows=[]
    for label,name in enumerate(LABELS):
        positive=margin[labels[:,label]>0.5,label]; negative=margin[labels[:,label]<0.5,label]
        rows.append({"label":name,"positive_mean":float(positive.mean()),"positive_median":float(positive.median()),
                     "negative_mean":float(negative.mean()),"negative_median":float(negative.median()),
                     "mean_separation":float(positive.mean()-negative.mean())})
    return rows


def write_reports(report,comparison):
    REPORT_ROOT.mkdir(parents=True,exist_ok=True)
    lines=["# Relevance–Polarity Mechanism Diagnostics","","All results are validation-only decision-sensitivity diagnostics; they are not ground-truth vulnerability localization.",""]
    for key in ("D1_relevance_shuffle","D2_polarity_swap","D3_score_modes","D4_localization_perturbation"):
        lines.extend([f"## {key}","","```json",json.dumps(report[key],indent=2),"```",""])
    lines.extend(["## D5/D6 representation diagnostics","","```json",json.dumps(report["D5_D6"],indent=2),"```","",
                  "## D7 margins","","```json",json.dumps(report["D7_margin"],indent=2),"```","",
                  "## D8 P11 independent support","","```json",json.dumps(report["D8_p11_alignment"],indent=2),"```",""])
    (REPORT_ROOT/"mechanism_diagnostics.md").write_text("\n".join(lines),encoding="utf-8")
    (REPORT_ROOT/"final_report.md").write_text(comparison,encoding="utf-8")


def main():
    config,tokenizer,model,train,valid,loader,weight,settings,b2_metrics=load_variant("B2")
    thresholds=b2_metrics["metrics"]["thresholds"]
    normal=intervention_metrics(model,loader,config,weight,settings,thresholds)
    permutation=torch.tensor([1,2,3,4,5,0],device="cuda")
    shuffled=intervention_metrics(model,loader,config,weight,settings,thresholds,relevance_permutation=permutation)
    swapped=intervention_metrics(model,loader,config,weight,settings,thresholds,swap_polarities=True)
    positive=intervention_metrics(model,loader,config,weight,settings,thresholds,score_mode="positive_only")
    negative=intervention_metrics(model,loader,config,weight,settings,thresholds,score_mode="negative_only")
    distribution=aligned_distribution_diagnostics(model,loader)
    perturbation=localization_perturbation(model,loader,tokenizer,thresholds)
    p11_alignment=p11_alignment_diagnostics(config,tokenizer,valid)
    a0_margin=margin_report(ROOT/"results/light_label/polarity_queries_followup_p11/P11/polarity_scores.pt")
    b2_margin=margin_report(RESULT_ROOT/"B2/polarity_scores.pt")
    report={"D1_relevance_shuffle":{"normal":normal,"shuffled":shuffled,"macro_delta":shuffled["macro_f1"]-normal["macro_f1"]},
            "D2_polarity_swap":{"normal":normal,"swapped":swapped,"macro_delta":swapped["macro_f1"]-normal["macro_f1"]},
            "D3_score_modes":{"full":normal,"positive_only":positive,"negative_only":negative},
            "D4_localization_perturbation":perturbation,"D5_D6":distribution,
            "D7_margin":{"A0_P11":a0_margin,"B2":b2_margin},"D8_p11_alignment":p11_alignment,
            "test_checked":False}
    (RESULT_ROOT/"diagnostics").mkdir(parents=True,exist_ok=True)
    (RESULT_ROOT/"diagnostics/mechanism_diagnostics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

    a0=json.loads(REFERENCE_METRICS.read_text(encoding="utf-8")); b1=json.loads((RESULT_ROOT/"B1/metrics.json").read_text(encoding="utf-8"))
    rows=[]
    for variant,localization,support,item in (("A0 P11","q+ / q- independently","independent",a0),
                                              ("B1","q+","shared aligned",b1),("B2","dedicated q_rel","shared aligned",b2_metrics)):
        metrics=item["metrics"]; params=item.get("params",item.get("p11_params"))
        rows.append({"variant":variant,"localization":localization,"polarity_support":support,
                     "tuned_macro_f1":metrics["tuned"]["macro_f1"],"fixed_macro_f1":metrics["fixed"]["macro_f1"],
                     "micro_f1":metrics["tuned"]["micro_f1"],"detection_f1":metrics["detection_f1"],"params":params})
    with (RESULT_ROOT/"comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (RESULT_ROOT/"per_label_comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["label"]+[row["variant"] for row in rows])
        items=[a0,b1,b2_metrics]
        for index,label in enumerate(LABELS): writer.writerow([label]+[item["metrics"]["tuned"]["per_label_f1"][index] for item in items])
    delta=rows[2]["tuned_macro_f1"]-rows[0]["tuned_macro_f1"]
    b2_ge_b1=rows[2]["tuned_macro_f1"]>=rows[1]["tuned_macro_f1"]
    mechanisms=(report["D1_relevance_shuffle"]["macro_delta"]<0 and report["D2_polarity_swap"]["macro_delta"]<0
                and normal["macro_f1"]>positive["macro_f1"] and normal["macro_f1"]>negative["macro_f1"]
                and perturbation["macro_drop_top5"]>perturbation["macro_drop_random5"])
    if delta>=.003 and b2_ge_b1 and mechanisms: verdict="STRONGLY SUPPORTED"; next_step="Run seeds 43/44"
    elif abs(delta)<.003 and b2_ge_b1 and mechanisms: verdict="PARTIALLY SUPPORTED"; next_step="Retain for mechanism evidence; do not claim performance gain"
    elif rows[1]["tuned_macro_f1"]>rows[0]["tuned_macro_f1"] and rows[1]["tuned_macro_f1"]>=rows[2]["tuned_macro_f1"]:
        verdict="PARTIALLY SUPPORTED"; next_step="Keep aligned support but remove dedicated relevance locator"
    else: verdict="NOT SUPPORTED"; next_step="Keep P11"
    final={"comparison":rows,"delta_B2_vs_A0":delta,"verdict":verdict,"next_step":next_step,"test_checked":False}
    (RESULT_ROOT/"final_summary.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    table="\n".join("| {variant} | {localization} | {polarity_support} | {tuned_macro_f1:.6f} | {fixed_macro_f1:.6f} | {micro_f1:.6f} | {detection_f1:.6f} | {params} |".format(**row) for row in rows)
    comparison=f"""# Relevance–Polarity Final Report

Dataset: DIVE_main6_opcode_process01; seed 42; validation only; test_checked=false.

| Variant | Localization | Polarity Support | Tuned Macro-F1 | Fixed Macro-F1 | Micro-F1 | Detection-F1 | Params |
|---|---|---|---:|---:|---:|---:|---:|
{table}

Delta B2 vs A0: {delta:+.6f}

VERDICT: **{verdict}**

NEXT STEP: **{next_step}**
"""
    write_reports(report,comparison); print(json.dumps(final,indent=2))


if __name__=="__main__": main()
