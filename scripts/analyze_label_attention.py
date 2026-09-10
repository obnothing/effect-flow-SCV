"""Validation-only attention diagnostics for full B2 LabelGuidedOpcodeNet."""

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def metric(labels, logits, thresholds):
    item = compute_multilabel_metrics_from_probs(labels.numpy().astype(int), torch.sigmoid(logits).numpy(), thresholds)
    return {"macro_f1": float(item["recognition_macro_f1"]), "micro_f1": float(item["recognition_micro_f1"]), "per_label_f1": [float(x) for x in item["per_label_f1"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/light_label/b2_label_attention.yaml")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    runtime = resolve("results/light_label/resolved_runtime.json")
    if runtime.exists(): config.update(json.loads(runtime.read_text(encoding="utf-8")))
    if config.get("allow_test"): raise ValueError("test is locked")
    root = resolve("results/light_label")
    comparison = json.loads((root / "full_comparison.json").read_text(encoding="utf-8"))
    records = {x["variant"]: x for x in comparison["records"]}
    if records["b2_label_attention"]["tuned"]["macro_f1"] <= records["b1_shared_attention"]["tuned"]["macro_f1"]:
        report = {"status": "stopped", "reason": "B2 did not exceed B1", "hypothesis_verdict": "NOT SUPPORTED", "test_checked": False}
        (root / "attention_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2)); return
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    mask_token_id = tokenizer.vocab[tokenizer.mask_token]
    valid = LightLabelDataset(resolve(config["cache_dir"]) / "valid_max8192.pt", runtime_max_len=config["max_len"])
    loader = DataLoader(valid, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, collate_fn=partial(collate_light_label, pad_id=tokenizer.pad_token_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(resolve(config["checkpoint_dir"]) / "full" / "best.pt", map_location="cpu")
    model = LabelGuidedOpcodeNet("b2_label_attention", len(tokenizer), tokenizer.pad_token_id, config["embedding_dim"], config["gru_hidden_size"], config["num_labels"], True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval(); thresholds = checkpoint["metrics"]["thresholds"]
    labels=[]; normal=[]; shuffled=[]; top_removed=[]; random_removed=[]; lengths=[]
    l=int(config["num_labels"]); cos_sum=torch.zeros(l,l); js_sum=torch.zeros(l,l); query_norm=torch.nn.functional.normalize(model.label_queries.detach().cpu(),dim=-1); rng=random.Random(config["seed"]); contracts=0
    with torch.no_grad():
        for raw in loader:
            ids=raw["input_ids"].to(device); mask=raw["mask"].to(device); seq_lengths=raw["lengths"]
            out=model(ids,seq_lengths,mask); shuffled_out=model(ids,seq_lengths,mask,query_permutation=[1,2,3,4,5,0])
            attention=out["attention"].float().cpu(); norm=torch.nn.functional.normalize(attention,dim=-1); cos_sum += torch.einsum("blt,bmt->lm",norm,norm)
            for left in range(l):
                for right in range(l):
                    middle=0.5*(attention[:,left]+attention[:,right]); left_part=attention[:,left]; right_part=attention[:,right]
                    js_sum[left,right] += 0.5*((left_part*(left_part.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum(1)+(right_part*(right_part.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum()).sum()
            changed_top=[]; changed_random=[]
            for label in range(l):
                top_ids=ids.clone(); random_ids=ids.clone()
                for row in range(ids.shape[0]):
                    positions=torch.nonzero(mask[row],as_tuple=False).flatten().tolist(); selected=attention[row,label].masked_fill(~mask[row].cpu(),-1).topk(min(args.top_k,len(positions))).indices.tolist(); random_positions=rng.sample(positions,len(selected))
                    top_ids[row,selected]=mask_token_id; random_ids[row,random_positions]=mask_token_id
                changed_top.append(model(top_ids,seq_lengths,mask)["logits"][:,label]); changed_random.append(model(random_ids,seq_lengths,mask)["logits"][:,label])
            labels.append(raw["labels"].cpu()); normal.append(out["logits"].float().cpu()); shuffled.append(shuffled_out["logits"].float().cpu()); top_removed.append(torch.stack(changed_top,1).float().cpu()); random_removed.append(torch.stack(changed_random,1).float().cpu()); lengths.extend(raw["original_lengths"].tolist()); contracts += ids.shape[0]
    labels=torch.cat(labels); normal=torch.cat(normal); shuffled=torch.cat(shuffled); top_removed=torch.cat(top_removed); random_removed=torch.cat(random_removed); length_values=np.asarray(lengths)
    normal_metric=metric(labels,normal,thresholds); shuffle_metric=metric(labels,shuffled,thresholds); top_metric=metric(labels,top_removed,thresholds); random_metric=metric(labels,random_removed,thresholds)
    length_rows=[]
    for name,select in (("0-2048",length_values<=2048),("2049-4096",(length_values>2048)&(length_values<=4096)),("4097-8192",(length_values>4096)&(length_values<=8192)),(">8192",length_values>8192)):
        if select.any():
            value=metric(labels[select],normal[select],thresholds); length_rows.append({"bucket":name,"contracts":int(select.sum()),"macro_f1":value["macro_f1"],"micro_f1":value["micro_f1"],"per_label_f1":value["per_label_f1"]})
    b2_b1=records["b2_label_attention"]["tuned"]["macro_f1"]-records["b1_shared_attention"]["tuned"]["macro_f1"]; shuffle_drop=normal_metric["macro_f1"]-shuffle_metric["macro_f1"]; top_drop=normal_metric["macro_f1"]-top_metric["macro_f1"]; random_drop=normal_metric["macro_f1"]-random_metric["macro_f1"]
    offdiag=~torch.eye(l,dtype=torch.bool); attention_cos=(cos_sum/max(contracts,1)); attention_js=(js_sum/max(contracts,1)); diverse=float(attention_cos[offdiag].mean())<0.99 or float(attention_js[offdiag].mean())>0.001
    if b2_b1>0 and shuffle_drop>0 and top_drop>random_drop and diverse: verdict="SUPPORTED"; recommendation="A. Continue this lightweight label-guided direction"
    elif b2_b1>0: verdict="PARTIALLY SUPPORTED"; recommendation="B. Modify one key mechanism: use a longer-sequence strategy before changing the attention mechanism"
    else: verdict="NOT SUPPORTED"; recommendation="C. Abandon the hypothesis"
    report={"dataset":"DIVE_main6_opcode_process01","validation_only":True,"test_checked":False,"attention_cosine":attention_cos.tolist(),"attention_js":attention_js.tolist(),"query_cosine":(query_norm@query_norm.T).tolist(),"normal":normal_metric,"query_shuffle":shuffle_metric,"topk_masking":top_metric,"random_masking":random_metric,"mean_logit_drop_topk":(normal-top_removed).mean(0).tolist(),"mean_logit_drop_random":(normal-random_removed).mean(0).tolist(),"length_buckets":length_rows,"truncated_valid_count":int((length_values>config["max_len"]).sum()),"truncated_valid_ratio":float((length_values>config["max_len"]).mean()),"hypothesis_verdict":verdict,"recommendation":recommendation}
    (root / "attention_diagnostics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    report_dir=resolve(config["report_dir"]); report_dir.mkdir(parents=True,exist_ok=True); lines=["# Label Attention Diagnostics","",f"B2-B1 tuned Macro-F1: {b2_b1:.6f}",f"Query-shuffle Macro-F1 drop: {shuffle_drop:.6f}",f"Top-{args.top_k} masking Macro-F1 drop: {top_drop:.6f}",f"Random-{args.top_k} masking Macro-F1 drop: {random_drop:.6f}",f"Valid truncation ratio: {report['truncated_valid_ratio']:.4f}","",f"Hypothesis verdict: **{verdict}**.",f"Recommendation: **{recommendation}**."]
    (report_dir / "attention_diagnostics.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps({"verdict":verdict,"test_checked":False},indent=2))
    final_report=report_dir / "final_report.md"
    if final_report.exists():
        text=final_report.read_text(encoding="utf-8")
        text=text.replace("Hypothesis verdict: `[Pending attention diagnostics]`.",f"Hypothesis verdict: **{verdict}**.")
        text=text.replace("Recommendation: `[Pending attention diagnostics]`.",f"Recommendation: **{recommendation}**.")
        final_report.write_text(text,encoding="utf-8")


if __name__=="__main__": main()
