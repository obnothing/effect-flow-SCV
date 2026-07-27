"""Create GraphCodeBERT source caches; test cache requires an explicit opt-in."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidity_graph_dataset import build_cache  # noqa: E402


def load_config(path, variant=None):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = dict(payload["common"])
    if variant:
        config.update(payload["variants"][variant])
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--allow-test-cache", action="store_true")
    parser.add_argument("--tokenizer-path", default=None)
    args = parser.parse_args()
    if "test" in args.splits and not args.allow_test_cache:
        raise SystemExit("Test source cache is locked; pass --allow-test-cache only during final evaluation")
    config = load_config(args.config)
    if args.tokenizer_path:
        manifest = Path(args.tokenizer_path) / "asset_manifest.json"
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")).get("revision") != config["source_model_revision"]:
            raise RuntimeError("Pinned GraphCodeBERT asset manifest is missing or has the wrong revision")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or config["source_model_path"], use_fast=True, local_files_only=True
    )
    data_dir, cache_dir = ROOT / config["data_dir"], ROOT / config["graph_cache_dir"]
    for split in args.splits:
        count = build_cache(data_dir / f"{split}.jsonl", cache_dir / f"{split}.pt", tokenizer, config, ROOT)
        print(f"[OK] cached {split}: {count}")


if __name__ == "__main__":
    main()
