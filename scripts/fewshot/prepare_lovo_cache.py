"""Build only original train/valid token caches for the few-shot pilot."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from light_label_data import build_sequence_cache
import yaml


def main():
    config = yaml.safe_load((ROOT / "configs/fewshot_query_pilot/pilot0.yaml").read_text(encoding="utf-8"))
    out = ROOT / config["cache_dir"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / config["dataset"] / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("split_sizes", {}).get("train") != 17864 or manifest.get("split_sizes", {}).get("valid") != 2233:
        raise ValueError("Unexpected DIVE-8 train/valid split sizes")
    for split in ("train", "valid"):
        path = out / f"{split}_max{config['max_len']}.pt"
        if path.exists():
            cache = torch.load(path, map_location="cpu")
            valid_shape = len(cache.get("ids", [])) == (17864 if split == "train" else 2233)
            valid_labels = ("labels" in cache and cache["labels"].shape[1] == 8) if split == "train" else "labels" not in cache
            if cache.get("max_len") == config["max_len"] and valid_shape and valid_labels:
                print(f"[{split}] reuse {path} rows={len(cache['ids'])}", flush=True)
                continue
        if split == "train":
            build_sequence_cache(ROOT / config["dataset"] / "train.jsonl", ROOT / config["vocab_path"], path,
                                 config["max_len"], config["num_labels"])
        else:
            from evm_tokenizer import EVMOpcodeTokenizer
            tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
            token_rows, offsets, ids, lengths = [], [0], [], []
            with (ROOT / config["dataset"] / "valid.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tokens = tokenizer.encode(row["opcode"], add_special_tokens=False)[:config["max_len"]]
                    token_rows.append(torch.tensor(tokens or [tokenizer.unk_token_id], dtype=torch.int32))
                    offsets.append(offsets[-1] + len(token_rows[-1])); ids.append(str(row["id"])); lengths.append(len(tokens))
            torch.save({"token_ids": torch.cat(token_rows), "offsets": torch.tensor(offsets, dtype=torch.int64),
                "ids": ids, "original_lengths": torch.tensor(lengths, dtype=torch.int32), "max_len": config["max_len"],
                "vocab_path": str(ROOT / config["vocab_path"]), "source_split": "valid",
                "labels_deferred_until_final_eval": True, "test_checked": False}, path)
        print(f"[{split}] built {path}", flush=True)
    print(json.dumps({"cache_dir": str(out), "splits": ["train", "valid"], "test_checked": False}))


if __name__ == "__main__":
    main()
