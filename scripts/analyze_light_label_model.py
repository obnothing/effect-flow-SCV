"""Aggregate local B0/B1/B2 results and gate attention diagnostics."""

import argparse
import csv
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
from light_label_data import LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs  # noqa: E402


def resolve(value):
    path=Path(value); return path if path.is_absolute() else ROOT/path


def config(path): return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def metric(labels, probs, thresholds):
    value=compute_multilabel_metrics_from_probs(labels,probs,thresholds)
    return {"macro_f1":float(value["recognition_macro_f1"]),"micro_f1":float(value["recognition_micro_f1"]),"per_label_f1":[float(x) for x in value["per_label_f1"]]}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run",default="full",choices=["smoke","full"]); args=parser.parse_args()
    configs=[config(f"configs/light_label/{name}.yaml") for name in ("b0_mean","b1_shared_attention","b2_label_attention")]
    records=[]; predictions={}
    for item in configs:
        path=resolve(item["result_dir"])/args.run/"metrics.json"; pred=resolve(item["result_dir"])/args.run/"valid_predictions.pt"
        if not path.exists() or not pred.exists(): continue
        report=json.loads(path.read_text(encoding="utf-8")); payload=torch.load(pred,map_location="cpu")
        record={"variant":item["variant"],"fixed":report["metrics"]["fixed"],"tuned":report["metrics"]["tuned"],"detection_f1":report["metrics"]["detection_f1"],"params":report["total_params"],"memory":report["peak_memory_mb"],"epoch_seconds":report["mean_epoch_seconds"],"thresholds":report["metrics"]["thresholds"]}
        records.append(record); predictions[item["variant"]]=payload
    base={x["variant"]:x for x in records}
    for item in records:
        item["delta_b0"]=item["tuned"]["macro_f1"]-base["b0_mean"]["tuned"]["macro_f1"] if "b0_mean" in base else None
        item["delta_b1"]=item["tuned"]["macro_f1"]-base["b1_shared_attention"]["tuned"]["macro_f1"] if "b1_shared_attention" in base else None
    gate=bool("b2_label_attention" in base and "b1_shared_attention" in base and base["b2_label_attention"]["tuned"]["macro_f1"]>base["b1_shared_attention"]["tuned"]["macro_f1"])
    root=resolve("results/light_label"); root.mkdir(parents=True,exist_ok=True)
    output={"dataset":"DIVE_main6_opcode_process01","run":args.run,"records":records,"attention_diagnostics_allowed":gate,"test_checked":False}
    (root/f"{args.run}_comparison.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    rows=[{"variant":x["variant"],"fixed_macro":x["fixed"]["macro_f1"],"tuned_macro":x["tuned"]["macro_f1"],"tuned_micro":x["tuned"]["micro_f1"],"detection_f1":x["detection_f1"],"params":x["params"],"peak_memory_mb":x["memory"],"epoch_seconds":x["epoch_seconds"],"delta_b0":x["delta_b0"],"delta_b1":x["delta_b1"]} for x in records]
    if rows:
        with (root/f"{args.run}_comparison.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if args.run=="smoke":
        report_dir=resolve(configs[0]["report_dir"]); report_dir.mkdir(parents=True,exist_ok=True)
        lines=["# Local Label Model Smoke Test","",f"Dataset: `DIVE_main6_opcode_process01`.","", "| Variant | Tuned Macro-F1 | Peak GPU MB | Epoch seconds |", "|---|---:|---:|---:|"]
        lines += [f"| {x['variant']} | {x['tuned']['macro_f1']:.6f} | {x['memory']:.1f} | {x['epoch_seconds']:.2f} |" for x in records]
        lines += ["",f"B2 attention diagnostics gate: `{gate}`."]
        (report_dir/"smoke_test.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if args.run=="full":
        report_dir=resolve(configs[0]["report_dir"]); report_dir.mkdir(parents=True,exist_ok=True)
        lines=["# LabelGuidedOpcodeNet Hypothesis Experiment","","Dataset: `DIVE_main6_opcode_process01`.","Validation only; seed 42; test remains locked.","","## Main Results","","| Variant | Fixed Macro-F1 | Tuned Macro-F1 | Micro-F1 | Detection-F1 | Params | Δ B0 | Δ B1 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {x['variant']} | {x['fixed']['macro_f1']:.6f} | {x['tuned']['macro_f1']:.6f} | {x['tuned']['micro_f1']:.6f} | {x['detection_f1']:.6f} | {x['params']} | {x['delta_b0']:.6f} | {x['delta_b1']:.6f} |" for x in records]
        lines += ["", "## Current Interpretation", "", f"B2 attention gate: **{gate}**.", "Mechanism diagnostics are generated separately after B2 exceeds B1.", "", "Hypothesis verdict: `[Pending attention diagnostics]`.", "Recommendation: `[Pending attention diagnostics]`."]
        (report_dir/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"run":args.run,"variants":len(records),"attention_diagnostics_allowed":gate,"test_checked":False},indent=2))


if __name__=="__main__": main()
