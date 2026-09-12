"""Validation-only spatial/topology audit of B2 label attention."""

import argparse
import csv
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet  # noqa: E402
from light_label_runtime import merge_runtime_config  # noqa: E402


def resolve(value):
    path=Path(value); return path if path.is_absolute() else ROOT/path


def summarize_positions(positions, length, radii=(4,8,16,32,64)):
    positions=np.sort(np.asarray(positions,dtype=np.int64)); k=len(positions)
    if k<2:
        return {"nn_distance":None,"median_nn_distance":None,"pair_distance":None,"span":0.0,"span_norm":0.0,**{f"cluster_ratio_{r}":0.0 for r in radii}}
    distances=np.abs(positions[:,None]-positions[None,:]); distances[distances==0]=np.iinfo(np.int64).max
    nearest=distances.min(axis=1)
    pair=distances[np.triu_indices(k,1)]
    result={"nn_distance":float(nearest.mean()),"median_nn_distance":float(np.median(nearest)),"pair_distance":float(pair.mean()),"span":float(positions[-1]-positions[0]),"span_norm":float((positions[-1]-positions[0])/max(length,1))}
    for radius in radii:
        result[f"cluster_ratio_{radius}"]=float((nearest<=radius).mean())
    return result


def random_summary(length,k,rng,repeats=100):
    rows=[]
    for _ in range(repeats):
        rows.append(summarize_positions(rng.sample(range(length),min(k,length)),length))
    keys=rows[0].keys()
    return {key:float(np.mean([row[key] for row in rows])) for key in keys if rows[0][key] is not None}


def mean_ci(values,rounds=1000,seed=42):
    values=np.asarray([x for x in values if x is not None],dtype=float)
    if not len(values): return {"mean":None,"ci95":None,"n":0}
    rng=np.random.default_rng(seed); means=[]
    for _ in range(rounds): means.append(float(values[rng.integers(0,len(values),len(values))].mean()))
    return {"mean":float(values.mean()),"ci95":[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))],"n":int(len(values))}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/light_label/b2_label_attention.yaml"); parser.add_argument("--batch-size",type=int,default=64); parser.add_argument("--random-repeats",type=int,default=100); args=parser.parse_args()
    config=yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")); runtime=resolve(config.get("runtime_path", "results/light_label/resolved_runtime.json")); config=merge_runtime_config(config, runtime)
    if config.get("allow_test"): raise ValueError("test is locked")
    tokenizer=EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"])); valid=LightLabelDataset(resolve(config["cache_dir"])/f"valid_max{config['max_len']}.pt",runtime_max_len=config["max_len"])
    loader=DataLoader(valid,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=partial(collate_light_label,pad_id=tokenizer.pad_token_id))
    checkpoint=torch.load(resolve("checkpoints/light_label/b2_label_attention/full/best.pt"),map_location="cpu")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=LabelGuidedOpcodeNet("b2_label_attention",len(tokenizer),tokenizer.pad_token_id,config["embedding_dim"],config["gru_hidden_size"],config["num_labels"],config["bidirectional"],config.get("local_radius",8),config.get("gru_layers",1)).to(device); model.load_state_dict(checkpoint["model_state_dict"],strict=True); model.eval()
    names=config["label_names"]; rng=random.Random(int(config["seed"])); rows=[]; labels=[]
    with torch.no_grad():
        for batch in loader:
            output=model(batch["input_ids"].to(device),batch["lengths"],batch["mask"].to(device)); attention=output["attention"].float().cpu(); target=batch["labels"].long(); lengths=batch["lengths"].tolist()
            for row in range(attention.shape[0]):
                for label_id,name in enumerate(names):
                    length=int(lengths[row]); distribution=attention[row,label_id,:length].numpy(); distribution=distribution/max(distribution.sum(),1e-12); entropy=float(-(distribution*np.log(np.maximum(distribution,1e-12))).sum()); normalized=entropy/max(np.log(max(length,2)),1e-12); ess=float(np.exp(entropy)); top_mass={}
                    observed={}; random_values={}
                    for k in (5,10,20):
                        top=min(k,length); positions=np.argsort(-distribution)[:top]; observed[k]=summarize_positions(positions,length); random_values[k]=random_summary(length,top,rng,args.random_repeats)
                        for key,value in list(observed[k].items()):
                            observed[k][f"random_{key}"]=random_values[k].get(key)
                            if value is not None and random_values[k].get(key) not in (None,0): observed[k][f"ratio_{key}"]=float(value/random_values[k][key])
                        observed[k]["top_mass"]=float(distribution[positions].sum())
                    rows.append({"label":name,"label_id":label_id,"positive":bool(target[row,label_id]),"length":length,"entropy":entropy,"normalized_entropy":normalized,"ess":ess,"ess_ratio":ess/max(length,1),"top5_mass":observed[5]["top_mass"],"top10_mass":observed[10]["top_mass"],"top20_mass":observed[20]["top_mass"],"top5":observed[5],"top10":observed[10],"top20":observed[20]})
                    labels.append(int(target[row,label_id]))
    primary=[]; per_label=[]
    for label_id,name in enumerate(names):
        for positive in (True,False):
            selected=[row for row in rows if row["label_id"]==label_id and row["positive"]==positive]
            if not selected: continue
            def avg(key): return float(np.mean([row[key] for row in selected]))
            k10=[row["top10"] for row in selected]; primary.append({"label":name,"positive":positive,"count":len(selected),"normalized_entropy":avg("normalized_entropy"),"ess_ratio":avg("ess_ratio"),"top5_mass":avg("top5_mass"),"top10_mass":avg("top10_mass"),"top20_mass":avg("top20_mass"),"nn_ratio":float(np.mean([r["ratio_nn_distance"] for r in k10])),"pair_distance_ratio":float(np.mean([r["ratio_pair_distance"] for r in k10])),"span_ratio":float(np.mean([r["ratio_span"] for r in k10])),"cluster_gain_8":float(np.mean([r["cluster_ratio_8"]/max(r["random_cluster_ratio_8"],1e-12) for r in k10])),"cluster_gain_16":float(np.mean([r["cluster_ratio_16"]/max(r["random_cluster_ratio_16"],1e-12) for r in k10])),"cluster_gain_32":float(np.mean([r["cluster_ratio_32"]/max(r["random_cluster_ratio_32"],1e-12) for r in k10]))})
        positives=[r for r in rows if r["label_id"]==label_id and r["positive"]]
        if positives:
            primary_values={key:[r[key] for r in positives] for key in ("top10_mass",)}
            ratios={key:[r["top10"][f"ratio_{key}"] for r in positives] for key in ("nn_distance","span","cluster_ratio_16")}
            per_label.append({"label":name,"positive_count":len(positives),"top10_mass_bootstrap":mean_ci(primary_values["top10_mass"]),"nn_ratio_bootstrap":mean_ci(ratios["nn_distance"]),"span_ratio_bootstrap":mean_ci(ratios["span"]),"cluster_gain_16_bootstrap":mean_ci(ratios["cluster_ratio_16"])})
    positive=[r for r in primary if r["positive"]]; top10_mean=float(np.mean([r["top10_mass"] for r in positive])); nn_mean=float(np.mean([r["nn_ratio"] for r in positive])); cluster_gain=float(np.mean([r["cluster_gain_16"] for r in positive])); ess_ratio=float(np.mean([r["ess_ratio"] for r in positive]))
    if nn_mean<0.75 and cluster_gain>1.15: decision="LOCAL"
    elif top10_mean>0.25 and ess_ratio<0.25 and abs(nn_mean-1.0)<0.25 and abs(cluster_gain-1.0)<0.25: decision="SPARSE"
    else: decision="AMBIGUOUS"
    report={"dataset":"DIVE_main6_opcode_process01","validation_only":True,"test_checked":False,"b2_checkpoint":str(resolve("checkpoints/light_label/b2_label_attention/full/best.pt")),"overall_positive_summary":{"top10_mass":top10_mean,"nn_ratio":nn_mean,"cluster_gain_16":cluster_gain,"ess_ratio":ess_ratio},"label_rows":primary,"bootstrap_positive_rows":per_label,"topology_decision":decision,"decision_rule":"LOCAL requires mean Top-10 NN ratio < 0.75 and ClusterGain16 > 1.15; SPARSE requires concentrated Top-10/ESS with random-like spatial ratios; otherwise AMBIGUOUS.","warning":"Attention topology is a model-mechanism diagnostic, not ground-truth evidence localization."}
    root=resolve("results/light_label/topology"); root.mkdir(parents=True,exist_ok=True); (root/"topology_metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    report_dir=resolve("reports/light_label_model/topology"); report_dir.mkdir(parents=True,exist_ok=True); lines=["# B2 Attention Topology Audit","",f"Dataset: `DIVE_main6_opcode_process01`.","Validation only; test remains locked.","",f"Overall positive Top-10 mass: {top10_mean:.6f}",f"Overall positive ESS ratio: {ess_ratio:.6f}",f"Overall positive NN ratio: {nn_mean:.6f}",f"Overall positive ClusterGain16: {cluster_gain:.6f}","",f"ATTENTION TOPOLOGY VERDICT: **{decision}**", "", "All spatial ratios use same-length random positions as the baseline."]
    (report_dir/"b2_attention_topology_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    with (root/"topology_per_label.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(primary[0]) if primary else ["label"]); writer.writeheader(); writer.writerows(primary)
    print(json.dumps({"decision":decision,"positive_count":len(positive),"test_checked":False},indent=2))


if __name__=="__main__": main()
