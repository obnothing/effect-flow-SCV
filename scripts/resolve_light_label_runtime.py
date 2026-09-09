"""Resolve one common 8 GiB runtime configuration using longest sequences."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet  # noqa: E402


def resolve(value):
    path=Path(value); return path if path.is_absolute() else ROOT/path


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/light_label/b2_label_attention.yaml"); args=parser.parse_args()
    config=yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")); tokenizer=EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    cache=resolve(config["cache_dir"])/"train_max8192.pt"; payload=torch.load(cache,map_location="cpu")
    order=torch.argsort(payload["original_lengths"],descending=True).tolist()
    candidates=[(64,8192,128),(32,8192,128),(16,8192,128),(8,8192,128),(4,8192,128),(2,8192,128),(2,4096,128),(2,4096,96)]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("CUDA PyTorch environment required; use the learnDL310 environment")
    attempts=[]; selected=None
    for batch_size,max_len,hidden in candidates:
        try:
            data=LightLabelDataset(cache,runtime_max_len=max_len,indices=order[:batch_size])
            loader=DataLoader(data,batch_size=batch_size,collate_fn=partial(collate_light_label,pad_id=tokenizer.pad_token_id))
            batch=next(iter(loader)); model=LabelGuidedOpcodeNet("b2_label_attention",len(tokenizer),tokenizer.pad_token_id,config["embedding_dim"],hidden,config["num_labels"],True).to(device)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
                output=model(batch["input_ids"].to(device),batch["lengths"],batch["mask"].to(device)); loss=F.binary_cross_entropy_with_logits(output["logits"],batch["labels"].to(device))
            loss.backward(); peak=torch.cuda.max_memory_allocated(device)/2**20
            if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
            selected={"batch_size":batch_size,"max_len":max_len,"gru_hidden_size":hidden,"memory_preflight_peak_mb":peak}
            attempts.append({**selected,"status":"ok"}); break
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch_size":batch_size,"max_len":max_len,"gru_hidden_size":hidden,"status":"oom"}); torch.cuda.empty_cache()
        finally:
            for name in ("model","output","loss","batch"):
                if name in locals(): del locals()[name]
            torch.cuda.empty_cache()
    if selected is None: raise RuntimeError(f"all runtime candidates failed: {attempts}")
    selected["gradient_accumulation_steps"]=int(config["gradient_accumulation_steps"])
    output=resolve("results/light_label/resolved_runtime.json"); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({**selected,"attempts":attempts,"gpu":torch.cuda.get_device_name(0),"vram_gib":torch.cuda.get_device_properties(0).total_memory/2**30,"test_checked":False},indent=2)+"\n",encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__=="__main__": main()
