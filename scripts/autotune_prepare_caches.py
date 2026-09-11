"""Prepare train/valid caches for every length in the AutoTune search space."""

import argparse
from pathlib import Path

import yaml

from prepare_light_label_data import load_config, resolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/light_label/b2_label_attention.yaml")
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192, 12288, 16384])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    base = load_config(args.base_config)
    for limit in args.lengths:
        config = dict(base); config["max_len"] = limit
        output = resolve(config["cache_dir"])
        output.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid"):
            path = output / f"{split}_max{limit}.pt"
            if path.exists() and not args.overwrite:
                print(f"[{split}] max_len={limit} exists", flush=True)
                continue
            from light_label_data import build_sequence_cache
            build_sequence_cache(resolve(config["data_dir"]) / f"{split}.jsonl", resolve(config["vocab_path"]), path, limit, config["num_labels"])
            print(f"[{split}] max_len={limit} written={path}", flush=True)


if __name__ == "__main__": main()

