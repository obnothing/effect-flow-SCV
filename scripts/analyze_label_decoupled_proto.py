"""Validation-only report for C0/C1/C2 label-decoupled contrast experiments."""

import argparse
import hashlib
import json
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
from label_decoupled_prototype import LabelDecoupledPrototype  # noqa: E402
from light_label_data import LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet  # noqa: E402


def resolve(value):
    path=Path(value); return path if path.is_absolute() else ROOT/path


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/light_label/c2_b2_prototype.yaml"); args=parser.parse_args()
    cfg=yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")); base=yaml.safe_load(resolve(cfg["base_config"]).read_text(encoding="utf-8")); base.update(cfg); cfg=base
    if cfg.get("allow_test"): raise ValueError("test is locked")
    root=resolve("results/light_label/prototype"); report_root=resolve(cfg["prototype_report_dir"]); report_root.mkdir(parents=True,exist_ok=True)
    c0_path=resolve("results/light_label/b2_label_attention/full/metrics.json")
    c0=json.loads(c0_path.read_text(encoding="utf-8"))
    records=[{"variant":"C0 B2","source":"existing B2","metrics":c0["metrics"],"trainable_params":c0["trainable_params"],"total_params":c0["total_params"],"peak_memory_mb":c0["peak_memory_mb"],"mean_epoch_seconds":c0["mean_epoch_seconds"],"checkpoint_sha256":None}]
    for name,label in (("c1_b2_pairwise","C1 + Pairwise SupCon"),("c2_b2_prototype","C2 + Label-Decoupled Prototype")):
        path=root/name/"full"/"metrics.json"
        if path.exists():
            value=json.loads(path.read_text(encoding="utf-8")); checkpoint=root/name/"full"/"best.pt"
            records.append({"variant":label,"source":str(path),"metrics":value["metrics"],"trainable_params":value["trainable_params"],"total_params":value["total_params"],"peak_memory_mb":value["peak_memory_mb"],"mean_epoch_seconds":value["mean_epoch_seconds"],"checkpoint_sha256":sha256(checkpoint) if checkpoint.exists() else None})
    by_name={x["variant"]:x for x in records}; c0_macro=records[0]["metrics"]["tuned"]["macro_f1"]
    geometry=None; examples=[]
    c2_checkpoint=root/"c2_b2_prototype"/"full"/"best.pt"
    if c2_checkpoint.exists():
        tokenizer=EVMOpcodeTokenizer.from_vocab_file(resolve(cfg["vocab_path"])); valid=LightLabelDataset(resolve(cfg["cache_dir"])/"valid_max8192.pt",runtime_max_len=cfg["max_len"])
        loader=DataLoader(valid,batch_size=int(cfg["batch_size"]),shuffle=False,num_workers=0,collate_fn=partial(collate_light_label,pad_id=tokenizer.pad_token_id)); payload=torch.load(c2_checkpoint,map_location="cpu")
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=LabelGuidedOpcodeNet("b2_label_attention",len(tokenizer),tokenizer.pad_token_id,cfg["embedding_dim"],cfg["gru_hidden_size"],cfg["num_labels"],True).to(device); model.load_state_dict(payload["model_state_dict"],strict=True); model.eval()
        proto=LabelDecoupledPrototype(cfg["num_labels"],model.output_dim,cfg["prototype_momentum"],cfg["prototype_temperature"]); proto.load_state_dict(payload["prototype_state_dict"],strict=True); proto.eval()
        labels=[]; reps=[]; ids=[]
        with torch.no_grad():
            for batch in loader:
                out=model(batch["input_ids"].to(device),batch["lengths"],batch["mask"].to(device)); labels.append(batch["labels"]); reps.append(out["representations"].float().cpu()); ids.extend(batch["ids"])
        labels=torch.cat(labels); reps=torch.cat(reps); normalized=torch.nn.functional.normalize(reps,dim=-1); geometry={"prototype_separation":[],"positive_margin":[],"negative_margin":[],"positive_intra_cosine":[],"negative_intra_cosine":[],"positive_negative_cosine":[]}
        for label in range(cfg["num_labels"]):
            pos=labels[:,label]>0.5; neg=~pos; p=normalized[pos,label]; n=normalized[neg,label]; pp=p@proto.positive_prototypes[label]; pn=p@proto.negative_prototypes[label]; nn=n@proto.negative_prototypes[label]; np_=n@proto.positive_prototypes[label]
            geometry["prototype_separation"].append(float(proto.positive_prototypes[label]@proto.negative_prototypes[label])); geometry["positive_margin"].append(float((pp-pn).mean())); geometry["negative_margin"].append(float((nn-np_).mean())); geometry["positive_intra_cosine"].append(float((p@p.T).mean())); geometry["negative_intra_cosine"].append(float((n@n.T).mean())); geometry["positive_negative_cosine"].append(float((p@n.T).mean()))
        for index in range(min(10,len(labels))):
            if int(labels[index].sum())>=2:
                row={"id":ids[index],"labels":[cfg["label_names"][i] for i in range(cfg["num_labels"]) if labels[index,i]>0.5],"prototype_similarity":{}}
                for label in range(cfg["num_labels"]): row["prototype_similarity"][cfg["label_names"][label]]={"positive":float(normalized[index,label]@proto.positive_prototypes[label]),"negative":float(normalized[index,label]@proto.negative_prototypes[label])}
                examples.append(row)
                if len(examples)>=3: break
    verdict="PENDING"; recommendation="Run C1 and C2 full experiments."
    if "C2 + Label-Decoupled Prototype" in by_name:
        c2_macro=by_name["C2 + Label-Decoupled Prototype"]["metrics"]["tuned"]["macro_f1"]; c1_macro=by_name.get("C1 + Pairwise SupCon",{"metrics":{"tuned":{"macro_f1":-1}}})["metrics"]["tuned"]["macro_f1"]
        if c2_macro>c0_macro and c2_macro>c1_macro: verdict="SUPPORTED"; recommendation="A. Keep prototype mechanism as a second contribution"
        elif c2_macro>c0_macro: verdict="PARTIALLY SUPPORTED"; recommendation="B. Keep as an auxiliary experiment, not a core contribution"
        else: verdict="NOT SUPPORTED"; recommendation="C. Remove prototype mechanism"
    report={"dataset":"DIVE_main6_opcode_process01","validation_only":True,"test_checked":False,"records":records,"geometry":geometry,"multi_label_examples":examples,"hypothesis":"Label-decoupled prototype contrast provides useful representation-level supervision beyond BCE.","hypothesis_verdict":verdict,"recommendation":recommendation,"prototype_inference_overhead":"zero; prototype branch is training-only"}
    (root/"prototype_diagnosis.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines=["# Label-Decoupled Prototype Contrast","","Dataset: `DIVE_main6_opcode_process01`.","Validation only; seed 42; test remains locked.","","## Main Results","","| Variant | Tuned Macro-F1 | Micro-F1 | Detection-F1 | Trainable params |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {x['variant']} | {x['metrics']['tuned']['macro_f1']:.6f} | {x['metrics']['tuned']['micro_f1']:.6f} | {x['metrics']['detection_f1']:.6f} | {x['trainable_params']} |" for x in records]
    lines += ["", "## Interpretation", "",f"Hypothesis verdict: **{verdict}**.",f"Recommendation: **{recommendation}**.","", "Prototype vectors are EMA buffers and are used only during training. They are not vulnerability ground truth and add no inference parameters."]
    (report_root/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(root/"prototype_diagnosis.json"),"verdict":verdict,"test_checked":False},indent=2))


if __name__=="__main__": main()
