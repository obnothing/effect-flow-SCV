"""Core comparison and gated validation diagnostics for VulProbe-V1."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import compute_multilabel_metrics_from_probs  # noqa: E402
from vulprobe_dataset import VulProbeContractDataset, collate_contracts  # noqa: E402
from vulprobe_model import VulProbeModel  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metric(labels, probs, thresholds):
    value = compute_multilabel_metrics_from_probs(labels, probs, thresholds)
    return {"macro_f1": float(value["recognition_macro_f1"]), "micro_f1": float(value["recognition_micro_f1"]),
            "per_label_f1": [float(x) for x in value["per_label_f1"]]}


def bootstrap_difference(left, right, labels, left_thresholds, right_thresholds, rounds=2000, seed=42):
    rng = np.random.default_rng(seed); values=[]
    for _ in range(rounds):
        idx = rng.integers(0, len(labels), len(labels))
        values.append(metric(labels[idx], left[idx], left_thresholds)["macro_f1"] - metric(labels[idx], right[idx], right_thresholds)["macro_f1"])
    return {"mean": float(np.mean(values)), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))], "rounds": rounds}


def core_report(config):
    root = resolve(config["result_root"]); names=config["label_names"]; variants=[]; predictions={}; stage_results=[]
    report_variants = list(config["variants"])
    if (root / "b3_probe_interaction" / "joint" / "metrics.json").exists() or (root / "b3_probe_interaction" / "frozen" / "metrics.json").exists():
        report_variants.append("b3_probe_interaction")
    for variant in report_variants:
        for recorded_stage in ("frozen", "joint"):
            recorded_path = root / variant / recorded_stage / "metrics.json"
            if recorded_path.exists():
                recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
                stage_results.append({"variant": variant, "stage": recorded_stage, "fixed": recorded["metrics"]["fixed"],
                                      "tuned": recorded["metrics"]["tuned"], "detection_f1": recorded["metrics"]["detection_f1"],
                                      "best_epoch": recorded["best_epoch"]})
        stage = "joint" if (root/variant/"joint"/"metrics.json").exists() else "frozen"
        metrics_path=root/variant/stage/"metrics.json"; prediction_path=root/variant/stage/"valid_predictions.pt"
        if not metrics_path.exists() or not prediction_path.exists():
            continue
        report=json.loads(metrics_path.read_text(encoding="utf-8")); pred=torch.load(prediction_path,map_location="cpu")
        item={"variant":variant,"stage":stage,"fixed":report["metrics"]["fixed"],"tuned":report["metrics"]["tuned"],
              "detection_f1":report["metrics"]["detection_f1"],"thresholds":report["metrics"]["thresholds"],
              "total_params":report["total_params"],"trainable_params":report["trainable_params"],"best_epoch":report["best_epoch"],
              "peak_memory_mb":max([x["peak_memory_mb"] for x in report["history"]] or [0]),
              "epoch_seconds":float(np.mean([x["epoch_seconds"] for x in report["history"]])),"test_checked":False}
        labels=pred["labels"].numpy().astype(int); probs=torch.sigmoid(pred["logits"]).numpy(); counts=np.asarray(pred["chunk_counts"])
        item["length_stratified"]=[]
        for group,selected in (("1-4",counts<=4),("5-8",(counts>=5)&(counts<=8)),("9-16",(counts>=9)&(counts<=16)),("17-32",(counts>=17)&(counts<=32)),("33-64",counts>=33)):
            if selected.any(): item["length_stratified"].append({"group":group,"contracts":int(selected.sum()),**metric(labels[selected],probs[selected],pred["thresholds"])})
        variants.append(item); predictions[variant]={"labels":pred["labels"].numpy().astype(int),"probs":torch.sigmoid(pred["logits"]).numpy(),"thresholds":pred["thresholds"],"chunks":np.asarray(pred["chunk_counts"])}
    by_name={x["variant"]:x for x in variants}; comparisons={}
    if "b2_label_probe" in predictions:
        for baseline in ("b1_shared_probe","b0_shared_representation"):
            if baseline in predictions:
                a=predictions["b2_label_probe"]; b=predictions[baseline]
                comparisons[f"b2_minus_{baseline}"]=bootstrap_difference(a["probs"],b["probs"],a["labels"],a["thresholds"],b["thresholds"])
    b2=by_name.get("b2_label_probe"); b1=by_name.get("b1_shared_probe")
    gate=bool(b2 and b1 and b2["tuned"]["macro_f1"]>b1["tuned"]["macro_f1"])
    report={"route":config["route_name"],"dataset":"DIVE_main6_opcode_process01","validation_only":True,"test_checked":False,
            "m0_strong_reference":{"validation_macro_f1":config["m0_validation_macro_f1"]},"variants":variants,"stage_results":stage_results,"paired_bootstrap":comparisons,
            "phase_d_gate":{"passed":gate,"rule":"B2 tuned validation Macro-F1 > B1","b3_allowed":gate,"diagnostics_allowed":gate},
            "parameter_fairness":{"b2_minus_b1_total_params":(by_name["b2_label_probe"]["total_params"]-by_name["b1_shared_probe"]["total_params"]) if "b2_label_probe" in by_name and "b1_shared_probe" in by_name else None,
                                  "note":"B1 and B2 share projections, scorer and protocol; B2 adds five probe vectors."},
            "b4_note":"A reasonable no-probe label-specific linear classifier is B0, so B4 is not duplicated."}
    root.mkdir(parents=True,exist_ok=True); (root/"main_results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    rows=[]; per_label=[]; length_rows=[]
    for x in variants:
        delta_b0=x["tuned"]["macro_f1"]-by_name["b0_shared_representation"]["tuned"]["macro_f1"] if "b0_shared_representation" in by_name else None
        delta_b1=x["tuned"]["macro_f1"]-by_name["b1_shared_probe"]["tuned"]["macro_f1"] if "b1_shared_probe" in by_name else None
        rows.append({"variant":x["variant"],"stage":x["stage"],"fixed_macro_f1":x["fixed"]["macro_f1"],"tuned_macro_f1":x["tuned"]["macro_f1"],
                     "tuned_micro_f1":x["tuned"]["micro_f1"],"detection_f1":x["detection_f1"],"total_params":x["total_params"],"trainable_params":x["trainable_params"],
                     "peak_memory_mb":x["peak_memory_mb"],"mean_epoch_seconds":x["epoch_seconds"],"delta_vs_b0":delta_b0,"delta_vs_b1":delta_b1})
        for idx,name in enumerate(names): per_label.append({"variant":x["variant"],"label":name,"f1":x["tuned"]["per_label_f1"][idx]})
        if x["variant"] in ("b0_shared_representation","b2_label_probe"):
            for group in x["length_stratified"]:
                row={"variant":x["variant"],"group":group["group"],"contracts":group["contracts"],"macro_f1":group["macro_f1"],"micro_f1":group["micro_f1"]}
                row.update({f"f1_{name}":group["per_label_f1"][idx] for idx,name in enumerate(names)})
                length_rows.append(row)
    write_csv(root/"main_results.csv",rows); write_csv(root/"per_label.csv",per_label); write_csv(root/"length_stratified.csv",length_rows)
    lines=["# VulProbe-V1 Results","","Validation only; process01; seed 42; test locked.","","| Variant | Stage | Fixed Macro-F1 | Tuned Macro-F1 | Micro-F1 | Params |","|---|---|---:|---:|---:|---:|"]
    lines += [f"| {x['variant']} | {x['stage']} | {x['fixed']['macro_f1']:.6f} | {x['tuned']['macro_f1']:.6f} | {x['tuned']['micro_f1']:.6f} | {x['total_params']} |" for x in variants]
    lines += ["",f"Phase D gate: **{'PASS' if gate else 'STOP'}**.","",f"M0 validation Strong Reference: {config['m0_validation_macro_f1']:.6f}."]
    (resolve(config["report_root"])/"vulprobe_v1_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def cosine_matrix(values):
    values=torch.nn.functional.normalize(values.float(),dim=-1)
    return torch.einsum("nlh,nmh->nlm",values,values).mean(0).tolist()


def js_matrix(distributions):
    p=distributions.clamp_min(1e-12); rows=[]
    for left in range(p.shape[1]):
        row=[]
        for right in range(p.shape[1]):
            m=0.5*(p[:,left]+p[:,right])
            value=0.5*((p[:,left]*(p[:,left].log()-m.log())).sum(1)+(p[:,right]*(p[:,right].log()-m.log())).sum(1))
            row.append(float(value.mean()))
        rows.append(row)
    return rows


def topk_overlap(distributions, k):
    indices=distributions.topk(min(k,distributions.shape[-1]),dim=-1).indices; labels=indices.shape[1]; result=[]
    for left in range(labels):
        row=[]
        for right in range(labels):
            values=[]
            for n in range(indices.shape[0]):
                a=set(indices[n,left].tolist()); b=set(indices[n,right].tolist()); values.append(len(a&b)/max(len(a|b),1))
            row.append(float(np.mean(values)))
        result.append(row)
    return result


def diagnostics(config, max_contracts=None):
    root=resolve(config["result_root"]); core=json.loads((root/"main_results.json").read_text(encoding="utf-8"))
    if not core["phase_d_gate"]["passed"]:
        report={"status":"stopped_by_gate","reason":"B2 did not exceed B1 on tuned validation Macro-F1","hypothesis_verdict":"NOT SUPPORTED","recommendation":"Abandon VulProbe","test_checked":False}
        (root/"probe_diagnostics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        (resolve(config["report_root"])/"vulprobe_v1_diagnostics.md").write_text("# VulProbe-V1 Diagnostics\n\nPhase D stopped because B2 did not exceed B1.\n\nHypothesis verdict: **NOT SUPPORTED**.\n\nRecommendation: **Abandon VulProbe**.\n",encoding="utf-8")
        return report
    stage="joint" if (root/"b2_label_probe"/"joint"/"metrics.json").exists() else "frozen"
    checkpoint=torch.load(resolve(config["checkpoint_root"])/"b2_label_probe"/stage/"best.pt",map_location="cpu")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=VulProbeModel(resolve(config["backbone_path"]),"b2_label_probe",config["num_labels"],config["tau"],config["num_heads"],config["scorer"],config["encoder_chunk_batch"],config["interaction_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True); model.eval()
    data=VulProbeContractDataset(resolve(config["data_dir"])/"valid.jsonl",resolve(config["vocab_path"]),config["max_len"],config["chunk_stride"],config["max_chunks"],config["num_labels"],max_contracts,config["seed"])
    data_loader=DataLoader(data,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_contracts)
    vocab=json.loads(resolve(config["vocab_path"]).read_text(encoding="utf-8"))["token_to_id"]; mask_id=vocab["[MASK]"]
    labels=[]; normal=[]; shuffled=[]; removed=[]; random_removed=[]; chunk_counts=[]
    label_count=int(config["num_labels"])
    attention_cos_sum=torch.zeros(label_count,label_count)
    attention_js_sum=torch.zeros(label_count,label_count)
    overlap_sums={k:torch.zeros(label_count,label_count) for k in (1,3,5)}
    representation_sums={key:torch.zeros(label_count,label_count) for key in ("all","single","multi")}
    representation_counts={key:0 for key in representation_sums}
    attention_behavior=[[{"entropy":[],"maximum":[],"top5_mass":[],"chunk_score_mean":[],"chunk_score_max":[]} for _ in range(2)] for _ in range(config["num_labels"])]
    rng=random.Random(int(config["seed"])); permutation=[1,2,3,4,5,0]
    with torch.no_grad():
        for raw in data_loader:
            batch={k:v.to(device) for k,v in raw.items() if isinstance(v,torch.Tensor)}
            output=model(batch["input_ids"],batch["attention_mask"],batch["content_mask"],batch["chunk_mask"])
            shuffle=model(batch["input_ids"],batch["attention_mask"],batch["content_mask"],batch["chunk_mask"],probe_permutation=permutation)
            count=int(batch["chunk_mask"].sum()); token_attention=output["valid_token_attention"][:count]
            chunk_weight=output["chunk_weights"][0,:count].transpose(0,1).unsqueeze(-1)
            joint=(token_attention.permute(1,0,2)*chunk_weight).reshape(config["num_labels"],-1)
            joint=joint/joint.sum(1,keepdim=True).clamp_min(1e-12)
            joint_cpu=joint.float().cpu()
            normalized_attention=torch.nn.functional.normalize(joint_cpu,dim=-1)
            attention_cos_sum += normalized_attention @ normalized_attention.T
            for left in range(label_count):
                for right in range(label_count):
                    middle=0.5*(joint_cpu[left]+joint_cpu[right])
                    attention_js_sum[left,right] += 0.5*((joint_cpu[left]*(joint_cpu[left].clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum()+(joint_cpu[right]*(joint_cpu[right].clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum())
            for k in overlap_sums:
                indices=joint_cpu.topk(min(k,joint_cpu.shape[-1]),dim=-1).indices
                for left in range(label_count):
                    for right in range(label_count):
                        a=set(indices[left].tolist()); b=set(indices[right].tolist())
                        overlap_sums[k][left,right] += len(a&b)/max(len(a|b),1)
            reps=output["valid_probe_representation"][:count].permute(1,0,2)
            current_representation=(reps*chunk_weight).sum(1).float().cpu()
            top_inputs=[]; random_inputs=[]
            flat_content=batch["content_mask"][0,:count].reshape(-1)
            for label_id in range(config["num_labels"]):
                chosen=joint[label_id].masked_fill(~flat_content, -1).topk(min(5,int(flat_content.sum()))).indices.tolist()
                by_chunk={}
                for index in chosen: by_chunk.setdefault(index//config["max_len"],0); by_chunk[index//config["max_len"]]+=1
                random_choice=[]
                for chunk_id,number in by_chunk.items():
                    candidates=torch.nonzero(batch["content_mask"][0,chunk_id],as_tuple=False).flatten().tolist()
                    random_choice += [chunk_id*config["max_len"]+x for x in rng.sample(candidates,min(number,len(candidates)))]
                top=batch["input_ids"].clone(); rand=batch["input_ids"].clone()
                top.view(1,-1)[0,chosen]=mask_id; rand.view(1,-1)[0,random_choice]=mask_id
                top_inputs.append(top); random_inputs.append(rand)
            top_logits=[]; rand_logits=[]
            for label_id in range(config["num_labels"]):
                top_out=model(top_inputs[label_id],batch["attention_mask"],batch["content_mask"],batch["chunk_mask"])
                rand_out=model(random_inputs[label_id],batch["attention_mask"],batch["content_mask"],batch["chunk_mask"])
                top_logits.append(top_out["logits"][0,label_id]); rand_logits.append(rand_out["logits"][0,label_id])
            target=batch["multi_labels"][0].long()
            normalized_representation=torch.nn.functional.normalize(current_representation,dim=-1)
            representation_cosine=normalized_representation @ normalized_representation.T
            representation_sums["all"] += representation_cosine; representation_counts["all"] += 1
            cardinality=int(target.sum())
            representation_group="single" if cardinality==1 else "multi" if cardinality>=2 else None
            if representation_group:
                representation_sums[representation_group] += representation_cosine
                representation_counts[representation_group] += 1
            for label_id in range(config["num_labels"]):
                state=int(target[label_id]); distribution=joint[label_id]; bucket=attention_behavior[label_id][state]
                bucket["entropy"].append(float(-(distribution*distribution.clamp_min(1e-12).log()).sum()))
                bucket["maximum"].append(float(distribution.max())); bucket["top5_mass"].append(float(distribution.topk(min(5,len(distribution))).values.sum()))
                scores=output["chunk_logits"][0,:count,label_id]
                bucket["chunk_score_mean"].append(float(scores.mean())); bucket["chunk_score_max"].append(float(scores.max()))
            labels.append(target.cpu()); normal.append(output["logits"][0].cpu()); shuffled.append(shuffle["logits"][0].cpu())
            removed.append(torch.stack(top_logits).cpu()); random_removed.append(torch.stack(rand_logits).cpu()); chunk_counts.append(count)
    labels=torch.stack(labels); normal=torch.stack(normal); shuffled=torch.stack(shuffled); removed=torch.stack(removed); random_removed=torch.stack(random_removed)
    thresholds=checkpoint["metrics"]["thresholds"]
    normal_prob=torch.sigmoid(normal).numpy(); label_array=labels.numpy(); counts=np.asarray(chunk_counts)
    length_rows=[]
    for name,selector in (("1-4",counts<=4),("5-8",(counts>=5)&(counts<=8)),("9-16",(counts>=9)&(counts<=16)),("17-32",(counts>=17)&(counts<=32)),("33-64",counts>=33)):
        if selector.any():
            value=metric(label_array[selector],normal_prob[selector],thresholds); length_rows.append({"group":name,"contracts":int(selector.sum()),"macro_f1":value["macro_f1"],"micro_f1":value["micro_f1"],"per_label_f1":value["per_label_f1"]})
    behavior_rows=[]
    for label_id,label_name in enumerate(config["label_names"]):
        for state,state_name in ((0,"negative"),(1,"positive")):
            values=attention_behavior[label_id][state]
            behavior_rows.append({"label":label_name,"state":state_name,**{key:(float(np.mean(value)) if value else None) for key,value in values.items()}})
    top_prob=torch.sigmoid(removed).numpy(); random_prob=torch.sigmoid(random_removed).numpy()
    contract_total=max(len(labels),1)
    attention_report={
        "cosine":(attention_cos_sum/contract_total).tolist(),
        "js_divergence":(attention_js_sum/contract_total).tolist(),
        "top1_overlap":(overlap_sums[1]/contract_total).tolist(),
        "top3_overlap":(overlap_sums[3]/contract_total).tolist(),
        "top5_overlap":(overlap_sums[5]/contract_total).tolist(),
    }
    representation_report={
        "all_cosine":(representation_sums["all"]/max(representation_counts["all"],1)).tolist(),
        "single_label_cosine":(representation_sums["single"]/representation_counts["single"]).tolist() if representation_counts["single"] else None,
        "multi_label_cosine":(representation_sums["multi"]/representation_counts["multi"]).tolist() if representation_counts["multi"] else None,
    }
    report={"status":"complete","variant":"b2_label_probe","stage":stage,"contracts":len(labels),"validation_only":True,"test_checked":False,
            "attention":attention_report,
            "representation":representation_report,
            "interventions":{"normal":metric(label_array,normal_prob,thresholds),"label_shuffle":metric(label_array,torch.sigmoid(shuffled).numpy(),thresholds),
                             "top5_removed":metric(label_array,torch.sigmoid(removed).numpy(),thresholds),"random5_removed":metric(label_array,torch.sigmoid(random_removed).numpy(),thresholds),
                             "mean_logit_drop_top5":(normal-removed).mean(0).tolist(),"mean_logit_drop_random5":(normal-random_removed).mean(0).tolist(),
                             "bootstrap_normal_minus_top5":bootstrap_difference(normal_prob,top_prob,label_array,thresholds,thresholds,1000,config["seed"]),
                             "bootstrap_normal_minus_random5":bootstrap_difference(normal_prob,random_prob,label_array,thresholds,thresholds,1000,config["seed"])},
            "positive_negative_attention":behavior_rows,
            "length_stratified":length_rows,"warning":"Attention and removal measure decision relevance, not ground-truth vulnerability evidence."}
    offdiag=~torch.eye(config["num_labels"],dtype=torch.bool)
    attention_cosine=torch.tensor(report["attention"]["cosine"])[offdiag].mean().item()
    attention_js=torch.tensor(report["attention"]["js_divergence"])[offdiag].mean().item()
    normal_macro=report["interventions"]["normal"]["macro_f1"]
    shuffle_drop=normal_macro-report["interventions"]["label_shuffle"]["macro_f1"]
    top_drop=normal_macro-report["interventions"]["top5_removed"]["macro_f1"]
    random_drop=normal_macro-report["interventions"]["random5_removed"]["macro_f1"]
    core_by_name={x["variant"]:x for x in core["variants"]}
    b2_macro=core_by_name["b2_label_probe"]["tuned"]["macro_f1"]
    b0_macro=core_by_name.get("b0_shared_representation",{"tuned":{"macro_f1":-1}})["tuned"]["macro_f1"]
    signals={"b2_exceeds_b1":True,"b2_not_below_b0":b2_macro>=b0_macro,"label_shuffle_hurts":shuffle_drop>0,
             "top_removal_exceeds_random":top_drop>random_drop,"attention_specificity":attention_cosine<0.99 or attention_js>0.001}
    if all(signals.values()): verdict="SUPPORTED"; recommendation="Continue VulProbe"
    elif signals["b2_exceeds_b1"] and sum(bool(x) for x in signals.values())>=3: verdict="PARTIALLY SUPPORTED"; recommendation="Modify one key mechanism"
    else: verdict="NOT SUPPORTED"; recommendation="Abandon VulProbe"
    report["hypothesis_assessment"]={"signals":signals,"attention_offdiag_cosine":attention_cosine,"attention_offdiag_js":attention_js,
                                     "label_shuffle_macro_drop":shuffle_drop,"top5_removal_macro_drop":top_drop,"random5_removal_macro_drop":random_drop,
                                     "verdict":verdict,"recommendation":recommendation}
    (root/"probe_diagnostics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    diagnostic_length_rows=[]
    for item in length_rows:
        row={key:value for key,value in item.items() if key!="per_label_f1"}
        row.update({f"f1_{name}":item["per_label_f1"][idx] for idx,name in enumerate(config["label_names"])})
        diagnostic_length_rows.append(row)
    write_csv(root/"length_stratified_diagnostics.csv",diagnostic_length_rows)
    lines=["# VulProbe-V1 Diagnostics","","Validation-only decision diagnostics; these are not evidence ground truth.","",f"Contracts: {len(labels)}",f"Stage: {stage}","",f"Normal Macro-F1: {report['interventions']['normal']['macro_f1']:.6f}",f"Label-shuffle Macro-F1: {report['interventions']['label_shuffle']['macro_f1']:.6f}",f"Top-5 removal Macro-F1: {report['interventions']['top5_removed']['macro_f1']:.6f}",f"Random-5 removal Macro-F1: {report['interventions']['random5_removed']['macro_f1']:.6f}","",f"Hypothesis verdict: **{verdict}**.",f"Recommendation: **{recommendation}**."]
    (resolve(config["report_root"])/"vulprobe_v1_diagnostics.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["core","diagnostics"]); parser.add_argument("--config",default="configs/vulprobe/v1.yaml"); parser.add_argument("--max-contracts",type=int)
    args=parser.parse_args(); config=load_config(args.config)
    if config.get("allow_test"): raise ValueError("VulProbe diagnostics keep test locked")
    result=core_report(config) if args.stage=="core" else diagnostics(config,args.max_contracts)
    print(json.dumps({"stage":args.stage,"status":result.get("status","complete"),"test_checked":False},indent=2))


if __name__=="__main__": main()
