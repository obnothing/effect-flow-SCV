"""Comprehensive validation-error diagnosis for the frozen P11 model."""

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))

from evm_tokenizer import EVMOpcodeTokenizer
from polarity_query_model import build_model
from run_polarity_queries import datasets, load_config
from train_light_label_model import make_loader


CONFIG=ROOT/"configs/light_label/polarity_queries_followup_p11.yaml"
CHECKPOINT=ROOT/"checkpoints/light_label/polarity_queries_followup_p11/P11/best.pt"
METRICS=ROOT/"results/light_label/polarity_queries_followup_p11/P11/metrics.json"
PREDICTIONS=ROOT/"results/light_label/polarity_queries_followup_p11/P11/valid_predictions.pt"
SCORES=ROOT/"results/light_label/polarity_queries_followup_p11/P11/polarity_scores.pt"
CAUSAL=ROOT/"results/light_label/p11_causal_diagnosis/causal_diagnosis.json"
OUTPUT=ROOT/"results/light_label/p11_error_diagnosis"
REPORT=ROOT/"reports/p11_error_diagnosis.md"
LABELS=["Reentrancy","Access Control","Arithmetic","Unchecked Return Values","DoS","Time manipulation"]


def scalar_stats(values):
    values=np.asarray(values,dtype=float)
    if not len(values): return {"count":0}
    return {"count":int(len(values)),"mean":float(values.mean()),"std":float(values.std()),
            "median":float(np.median(values)),"q10":float(np.quantile(values,.1)),"q90":float(np.quantile(values,.9))}


def ece_score(labels,probabilities,bins=10,adaptive=False):
    labels=np.asarray(labels,dtype=float); probabilities=np.asarray(probabilities,dtype=float)
    if adaptive:
        groups=np.array_split(np.argsort(probabilities),bins)
    else:
        edges=np.linspace(0,1,bins+1); groups=[]
        for index in range(bins):
            right=probabilities<=edges[index+1] if index==bins-1 else probabilities<edges[index+1]
            groups.append(np.where((probabilities>=edges[index])&right)[0])
    error=0.0; rows=[]
    for group in groups:
        if not len(group): continue
        confidence=float(probabilities[group].mean()); frequency=float(labels[group].mean())
        weight=len(group)/len(labels); error+=weight*abs(confidence-frequency)
        rows.append({"count":int(len(group)),"mean_probability":confidence,"positive_frequency":frequency})
    return float(error),rows


def decision_confidence(probabilities,predictions,thresholds):
    thresholds=np.asarray(thresholds).reshape(1,-1); probabilities=np.asarray(probabilities)
    positive=(probabilities-thresholds)/(1-thresholds).clip(1e-8)
    negative=(thresholds-probabilities)/thresholds.clip(1e-8)
    return np.where(predictions==1,positive,negative).clip(0,1)


def confusion(labels,predictions):
    rows=[]
    for index,name in enumerate(LABELS):
        y=labels[:,index]; p=predictions[:,index]
        tn=int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum())
        fn=int(((y==1)&(p==0)).sum()); tp=int(((y==1)&(p==1)).sum())
        rows.append({"label":name,"tn":tn,"fp":fp,"fn":fn,"tp":tp,
                     "fpr":fp/max(1,fp+tn),"fnr":fn/max(1,fn+tp)})
    return rows


def branch_patterns(e_plus,e_minus,indices,true_state):
    positive=e_plus[indices]>0; negative=e_minus[indices]>0
    if true_state==1:
        correct_positive=positive; correct_negative=~negative
    else:
        correct_positive=~positive; correct_negative=negative
    total=max(1,len(indices))
    return {"both_correct":float((correct_positive&correct_negative).sum()/total),
            "positive_wrong_only":float(((~correct_positive)&correct_negative).sum()/total),
            "negative_wrong_only":float((correct_positive&(~correct_negative)).sum()/total),
            "both_wrong":float(((~correct_positive)&(~correct_negative)).sum()/total)}


def cross_label_matrices(labels,predictions):
    fp=np.zeros((6,6)); fn=np.zeros((6,6)); counts_fp=np.zeros((6,6),dtype=int); counts_fn=np.zeros((6,6),dtype=int)
    for target in range(6):
        for context in range(6):
            fp_den=(labels[:,target]==0)&(labels[:,context]==1)
            fn_den=(labels[:,target]==1)&(labels[:,context]==1)
            counts_fp[target,context]=int(((predictions[:,target]==1)&fp_den).sum())
            counts_fn[target,context]=int(((predictions[:,target]==0)&fn_den).sum())
            fp[target,context]=counts_fp[target,context]/max(1,int(fp_den.sum()))
            fn[target,context]=counts_fn[target,context]/max(1,int(fn_den.sum()))
    return {"fp_rate_given_true_context":fp.tolist(),"fn_rate_given_true_cooccurrence":fn.tolist(),
            "fp_count":counts_fp.tolist(),"fn_count":counts_fn.tolist()}


def selective_risk(labels,predictions,confidence):
    flat_error=(labels!=predictions).reshape(-1); flat_confidence=confidence.reshape(-1)
    order=np.argsort(-flat_confidence); rows=[]
    for coverage in (1.0,.9,.8,.7,.5,.3):
        count=max(1,int(round(len(order)*coverage))); chosen=order[:count]
        rows.append({"coverage":coverage,"decisions":count,"error_rate":float(flat_error[chosen].mean()),
                     "mean_confidence":float(flat_confidence[chosen].mean())})
    return rows


def attention_statistics(ids):
    config=load_config(CONFIG); _,valid=datasets(config)
    tokenizer=EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"])
    model=build_model("P11",config,len(tokenizer),tokenizer.pad_token_id).cuda().eval()
    model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu")["model_state_dict"])
    loader=make_loader(valid,config,tokenizer.pad_token_id,False); collected_ids=[]
    measures={name:[] for name in ("positive_entropy","negative_entropy","positive_max","negative_max",
                                   "positive_top5_mass","negative_top5_mass","cosine","js","top5_overlap")}
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda",dtype=torch.float16,enabled=config["amp"]):
                output=model(batch["input_ids"].cuda(),batch["lengths"],batch["mask"].cuda(),diagnostics=True)
            attention=output["attention"].float().cpu(); pos=attention[:,:,0]; neg=attention[:,:,1]
            normalizer=batch["lengths"].float().clamp_min(2).log().unsqueeze(1)
            middle=.5*(pos+neg)
            values={"positive_entropy":-(pos*pos.clamp_min(1e-12).log()).sum(-1)/normalizer,
                    "negative_entropy":-(neg*neg.clamp_min(1e-12).log()).sum(-1)/normalizer,
                    "positive_max":pos.max(-1).values,"negative_max":neg.max(-1).values,
                    "positive_top5_mass":pos.topk(5,dim=-1).values.sum(-1),
                    "negative_top5_mass":neg.topk(5,dim=-1).values.sum(-1),
                    "cosine":torch.nn.functional.cosine_similarity(pos,neg,dim=-1),
                    "js":.5*((pos*(pos.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum(-1)
                             +(neg*(neg.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum(-1)),
                    "top5_overlap":(pos.topk(5,dim=-1).indices.unsqueeze(-1)==neg.topk(5,dim=-1).indices.unsqueeze(-2)).any(-1).float().sum(-1)/5.0}
            for name,value in values.items(): measures[name].append(value.numpy())
            collected_ids.extend(batch["ids"])
    if collected_ids!=ids: raise ValueError("attention diagnostic ID order mismatch")
    return {name:np.concatenate(value) for name,value in measures.items()}


def grouped_error(labels,predictions,lengths):
    rows=[]
    cardinality=labels.sum(1); contract_error=(labels!=predictions).mean(1)
    for name,mask in (("zero",cardinality==0),("one",cardinality==1),("two",cardinality==2),
                      ("three",cardinality==3),("four_plus",cardinality>=4)):
        rows.append({"group":"label_count_"+name,"count":int(mask.sum()),
                     "hamming_error":float(contract_error[mask].mean()) if mask.any() else None,
                     "exact_match":float((labels[mask]==predictions[mask]).all(1).mean()) if mask.any() else None})
    for name,low,high in (("1_1024",1,1024),("1025_2048",1025,2048),("2049_4096",2049,4096),("4097_8192",4097,8192)):
        mask=(lengths>=low)&(lengths<=high)
        rows.append({"group":"length_"+name,"count":int(mask.sum()),
                     "hamming_error":float(contract_error[mask].mean()) if mask.any() else None,
                     "exact_match":float((labels[mask]==predictions[mask]).all(1).mean()) if mask.any() else None})
    return rows


def write_matrix(path,matrix):
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["target\\context"]+LABELS)
        for name,row in zip(LABELS,matrix): writer.writerow([name]+row)


def main():
    metrics=json.loads(METRICS.read_text(encoding="utf-8"))["metrics"]
    predictions=torch.load(PREDICTIONS,map_location="cpu"); scores=torch.load(SCORES,map_location="cpu")
    if predictions["ids"]!=scores["ids"]: raise ValueError("prediction/score ID mismatch")
    ids=predictions["ids"]; labels=predictions["labels"].numpy().astype(int); logits=predictions["logits"].numpy()
    e_plus=scores["e_plus"].numpy(); e_minus=scores["e_minus"].numpy(); margin=e_plus-e_minus
    probabilities=1/(1+np.exp(-logits)); thresholds=np.asarray(metrics["thresholds"]); predicted=(probabilities>=thresholds).astype(int)
    confidence=decision_confidence(probabilities,predicted,thresholds)
    cache=torch.load(ROOT/"data/features/light_label_process01/valid_max8192.pt",map_location="cpu")
    if cache["ids"]!=ids: raise ValueError("cache ID order mismatch")
    effective_lengths=np.minimum(np.diff(cache["offsets"].numpy()),8192); original_lengths=cache["original_lengths"].numpy()
    attention=attention_statistics(ids)
    matrices=confusion(labels,predicted); cross=cross_label_matrices(labels,predicted)
    outcomes=[]; detailed={}
    names=(("TP",(labels==1)&(predicted==1),1),("FN",(labels==1)&(predicted==0),1),
           ("FP",(labels==0)&(predicted==1),0),("TN",(labels==0)&(predicted==0),0))
    for label_index,label in enumerate(LABELS):
        calibration_ece,bins=ece_score(labels[:,label_index],probabilities[:,label_index],10,False)
        adaptive_ece,adaptive_bins=ece_score(labels[:,label_index],probabilities[:,label_index],10,True)
        per_label={"calibration":{"brier":float(np.mean((probabilities[:,label_index]-labels[:,label_index])**2)),
                                  "ece":calibration_ece,"adaptive_ece":adaptive_ece,"bins":bins,"adaptive_bins":adaptive_bins}}
        for outcome,all_mask,true_state in names:
            index=np.where(all_mask[:,label_index])[0]
            row={"label":label,"outcome":outcome,"count":int(len(index)),
                 "e_plus":scalar_stats(e_plus[index,label_index]),"e_minus":scalar_stats(e_minus[index,label_index]),
                 "margin":scalar_stats(margin[index,label_index]),"probability":scalar_stats(probabilities[index,label_index]),
                 "decision_confidence":scalar_stats(confidence[index,label_index]),
                 "branch_patterns":branch_patterns(e_plus[:,label_index],e_minus[:,label_index],index,true_state)}
            for name,value in attention.items(): row[name]=scalar_stats(value[index,label_index])
            outcomes.append(row); per_label[outcome]=row
        detailed[label]=per_label
    candidates=[]
    quality=labels*probabilities+(1-labels)*(1-probabilities)
    for flat in np.argsort(quality,axis=None)[:100]:
        row,label=np.unravel_index(flat,quality.shape)
        candidates.append({"id":ids[row],"label":LABELS[label],"truth":int(labels[row,label]),"prediction":int(predicted[row,label]),
                           "probability":float(probabilities[row,label]),"threshold":float(thresholds[label]),
                           "label_quality":float(quality[row,label]),"e_plus":float(e_plus[row,label]),"e_minus":float(e_minus[row,label]),
                           "margin":float(margin[row,label]),"true_labels":"|".join(LABELS[i] for i in np.where(labels[row]==1)[0]),
                           "effective_length":int(effective_lengths[row]),"original_length":int(original_lengths[row])})
    report={"route":"P11 validation error diagnosis","dataset":"DIVE_main6_opcode_process01","seed":42,
            "thresholds":thresholds.tolist(),"confusion_matrices":matrices,"cross_label":cross,
            "per_label_diagnostics":detailed,"selective_risk":selective_risk(labels,predicted,confidence),
            "grouped_error":grouped_error(labels,predicted,effective_lengths),
            "high_confidence_error_count":int(((labels!=predicted)&(confidence>=.8)).sum()),
            "review_candidate_count":len(candidates),"test_checked":False}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"error_diagnosis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (OUTPUT/"polarity_error_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=["label","outcome","count","e_plus_mean","e_minus_mean","margin_mean","probability_mean","confidence_mean",
                "both_correct","positive_wrong_only","negative_wrong_only","both_wrong","positive_entropy","negative_entropy","attention_cosine","attention_js","top5_overlap"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in outcomes:
            writer.writerow({"label":row["label"],"outcome":row["outcome"],"count":row["count"],
                "e_plus_mean":row["e_plus"].get("mean"),"e_minus_mean":row["e_minus"].get("mean"),"margin_mean":row["margin"].get("mean"),
                "probability_mean":row["probability"].get("mean"),"confidence_mean":row["decision_confidence"].get("mean"),
                **row["branch_patterns"],"positive_entropy":row["positive_entropy"].get("mean"),
                "negative_entropy":row["negative_entropy"].get("mean"),"attention_cosine":row["cosine"].get("mean"),
                "attention_js":row["js"].get("mean"),"top5_overlap":row["top5_overlap"].get("mean")})
    with (OUTPUT/"label_review_candidates.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(candidates[0])); writer.writeheader(); writer.writerows(candidates)
    write_matrix(OUTPUT/"cross_label_fp_rates.csv",cross["fp_rate_given_true_context"])
    write_matrix(OUTPUT/"cross_label_fn_rates.csv",cross["fn_rate_given_true_cooccurrence"])
    with (OUTPUT/"per_label_confusion.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(matrices[0])); writer.writeheader(); writer.writerows(matrices)
    causal=json.loads(CAUSAL.read_text(encoding="utf-8")) if CAUSAL.exists() else None
    lines=["# P11 Error Diagnosis","","Validation-only analysis; frozen checkpoint and thresholds; test_checked=false.","",
           "## Per-label confusion matrices","","| Label | TN | FP | FN | TP | FPR | FNR |","|---|---:|---:|---:|---:|---:|---:|"]
    for row in matrices: lines.append(f"| {row['label']} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} | {row['fpr']:.4f} | {row['fnr']:.4f} |")
    lines.extend(["","## Error-polarity means","","| Label | TP e+ | TP e- | FN e+ | FN e- | TN e+ | TN e- | FP e+ | FP e- |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for label in LABELS:
        d=detailed[label]; lines.append(f"| {label} | {d['TP']['e_plus']['mean']:.3f} | {d['TP']['e_minus']['mean']:.3f} | {d['FN']['e_plus']['mean']:.3f} | {d['FN']['e_minus']['mean']:.3f} | {d['TN']['e_plus']['mean']:.3f} | {d['TN']['e_minus']['mean']:.3f} | {d['FP']['e_plus']['mean']:.3f} | {d['FP']['e_minus']['mean']:.3f} |")
    lines.extend(["","## Calibration","","| Label | Brier | ECE | Adaptive ECE |","|---|---:|---:|---:|"])
    for label in LABELS:
        c=detailed[label]["calibration"]; lines.append(f"| {label} | {c['brier']:.4f} | {c['ece']:.4f} | {c['adaptive_ece']:.4f} |")
    lines.extend(["","## Selective risk","","| Coverage | Error rate | Mean decision confidence |","|---:|---:|---:|"])
    for row in report["selective_risk"]: lines.append(f"| {row['coverage']:.1f} | {row['error_rate']:.4f} | {row['mean_confidence']:.4f} |")
    if causal:
        lines.extend(["","## Counterfactual attention evidence","",
                      f"Positive Top-5 masking Macro-F1 drop: `{causal['conditions']['mask_positive_top5']['macro_f1_delta_vs_normal']:.6f}`",
                      f"Negative Top-5 masking Macro-F1 drop: `{causal['conditions']['mask_negative_top5']['macro_f1_delta_vs_normal']:.6f}`",
                      f"Matched random-5 masking drop: `{causal['conditions']['mask_random5']['macro_f1_delta_vs_normal']:.6f}`",""])
    lines.extend(["## Interpretation boundary","","Low label-quality scores identify contracts for manual review; they do not prove label errors. Attention statistics are interpreted only together with counterfactual masking, not as standalone explanations.",""])
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"confusion":matrices,"selective_risk":report["selective_risk"],"high_confidence_errors":report["high_confidence_error_count"],"output":str(OUTPUT)},indent=2))


if __name__=="__main__": main()
