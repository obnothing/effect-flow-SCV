"""Read-only architecture and data audit for VulProbe-V1."""

import argparse
import json
from pathlib import Path

import yaml
from transformers import BertConfig

ROOT=Path(__file__).resolve().parents[1]


def resolve(value):
    path=Path(value); return path if path.is_absolute() else ROOT/path


def scan(path,num_labels):
    count=0; ids=set(); positives=[0]*num_labels; malformed=[]
    with path.open(encoding="utf-8") as handle:
        for line_no,line in enumerate(handle,1):
            if not line.strip(): continue
            item=json.loads(line); count+=1
            if len(item.get("multi_labels",[]))!=num_labels: malformed.append(line_no)
            sample_id=str(item.get("id",line_no)); ids.add(sample_id)
            if len(item.get("multi_labels",[]))==num_labels:
                positives=[a+int(b) for a,b in zip(positives,item["multi_labels"])]
    return {"records":count,"unique_ids":len(ids),"duplicate_ids":count-len(ids),"positive_counts":positives,"malformed_label_rows":malformed[:20]}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/vulprobe/v1.yaml"); args=parser.parse_args()
    config=yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if config.get("allow_test"): raise ValueError("VulProbe audit requires test lock")
    data=resolve(config["data_dir"]); backbone=resolve(config["backbone_path"]); vocab=resolve(config["vocab_path"])
    report={"route":config["route_name"],"dataset":"DIVE_main6_opcode_process01","seed":config["seed"],"test_checked":False,
            "paths":{"train":str(data/"train.jsonl"),"valid":str(data/"valid.jsonl"),"test_exists":(data/"test.jsonl").exists(),"backbone":str(backbone),"vocab":str(vocab)},
            "splits":{"train":scan(data/"train.jsonl",config["num_labels"]),"valid":scan(data/"valid.jsonl",config["num_labels"])},
            "test_policy":"Existence checked only; test file content was not opened.",
            "chunking":{"max_len":config["max_len"],"content_tokens":config["max_len"]-2,"stride":config["chunk_stride"],"max_chunks":config["max_chunks"]}}
    if not (backbone/"config.json").exists(): raise FileNotFoundError(backbone/"config.json")
    if not vocab.exists(): raise FileNotFoundError(vocab)
    bert=BertConfig.from_pretrained(backbone,local_files_only=True)
    report["backbone_config"]={"hidden_size":bert.hidden_size,"layers":bert.num_hidden_layers,"heads":bert.num_attention_heads,"vocab_size":bert.vocab_size}
    if bert.hidden_size!=768: raise ValueError("VulProbe-V1 expects hidden_size=768")
    output=resolve(config["report_root"])/"vulprobe_v1_architecture_audit.json"; output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"output":str(output),"train":report["splits"]["train"]["records"],"valid":report["splits"]["valid"]["records"],"test_checked":False},indent=2))


if __name__=="__main__": main()
