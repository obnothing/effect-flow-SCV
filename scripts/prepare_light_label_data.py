"""Create train/valid full-sequence token caches for the local experiment."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from light_label_data import build_sequence_cache  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/light_label/b0_mean.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if config.get("allow_test"):
        raise ValueError("test cache is forbidden")
    output_dir = resolve(config["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": "DIVE_main6_opcode_process01", "splits": {}, "test_checked": False}
    for split in ("train", "valid"):
        output = output_dir / f"{split}_max{config['max_len']}.pt"
        if output.exists() and not args.overwrite:
            payload = __import__("torch").load(output, map_location="cpu")
        else:
            payload = build_sequence_cache(
                resolve(config["data_dir"]) / f"{split}.jsonl", resolve(config["vocab_path"]),
                output, config["max_len"], config["num_labels"],
            )
        summary["splits"][split] = {
            "samples": len(payload["ids"]), "stored_tokens": int(payload["token_ids"].numel()),
            "truncated": int((payload["original_lengths"] > config["max_len"]).sum()), "path": str(output),
        }
        print(f"[{split}] samples={len(payload['ids'])} tokens={payload['token_ids'].numel()} truncated={summary['splits'][split]['truncated']}", flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

