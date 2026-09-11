"""Compare untouched B2 with symmetric and directional local context variants."""

import argparse
import json
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]


def resolve(value):
    path=Path(value); return path if path.is_absolute() else ROOT/path


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default="reports/light_label_model/local_pattern/final_report.md"); args=parser.parse_args()
    paths={"L0 B2":"results/light_label/b2_label_attention/full/metrics.json","L1 Symmetric Local":"results/light_label/local_pattern/l1_symmetric_local/full/metrics.json","L2 Directional Local":"results/light_label/local_pattern/l2_directional_local/full/metrics.json"}
    records=[]
    for name,path in paths.items():
        target=resolve(path)
        if target.exists():
            value=json.loads(target.read_text(encoding="utf-8")); records.append({"name":name,"metrics":value["metrics"],"params":value["total_params"],"memory":value["peak_memory_mb"],"epoch_seconds":value["mean_epoch_seconds"]})
            continue
        checkpoint_path=resolve(path.replace("results/light_label/", "checkpoints/light_label/")).with_name("best.pt")
        if checkpoint_path.exists():
            payload=torch.load(checkpoint_path,map_location="cpu")
            records.append({"name":name,"metrics":payload["metrics"],"params":sum(value.numel() for value in payload["model_state_dict"].values()),"memory":None,"epoch_seconds":None,"source_checkpoint":str(checkpoint_path)})
    if not records: raise FileNotFoundError("No local-pattern metrics found")
    base=records[0]["metrics"]["tuned"]["macro_f1"]
    for row in records: row["delta_vs_l0"]=row["metrics"]["tuned"]["macro_f1"]-base
    l1=next((x for x in records if x["name"].startswith("L1")),None); l2=next((x for x in records if x["name"].startswith("L2")),None)
    if l2 and l2["metrics"]["tuned"]["macro_f1"]>base and (not l1 or l2["metrics"]["tuned"]["macro_f1"]>l1["metrics"]["tuned"]["macro_f1"]): verdict="SUPPORTED"; recommendation="Keep directional local context as a second mechanism"
    elif l1 and l1["metrics"]["tuned"]["macro_f1"]>base: verdict="PARTIALLY SUPPORTED"; recommendation="Keep local context only; directional refinement is not established"
    else: verdict="NOT SUPPORTED"; recommendation="Remove the local-pattern module and retain B2"
    report={"dataset":"DIVE_main6_opcode_process01","validation_only":True,"test_checked":False,"records":records,"topology_verdict":"LOCAL","hypothesis_verdict":verdict,"recommendation":recommendation}
    output=resolve(args.output); output.parent.mkdir(parents=True,exist_ok=True); (output.parent/"local_pattern_results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines=["# Local Pattern Refinement","","Dataset: `DIVE_main6_opcode_process01`; validation only; seed 42; test locked.","","| Variant | Tuned Macro-F1 | Micro-F1 | Detection-F1 | Params | Δ L0 |","|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {x['name']} | {x['metrics']['tuned']['macro_f1']:.6f} | {x['metrics']['tuned']['micro_f1']:.6f} | {x['metrics']['detection_f1']:.6f} | {x['params']} | {x['delta_vs_l0']:.6f} |" for x in records]
    lines += ["", "Topology verdict: **LOCAL**.", f"Local-pattern verdict: **{verdict}**.", f"Recommendation: **{recommendation}**."]
    output.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"verdict":verdict,"test_checked":False},indent=2))


if __name__=="__main__": main()
