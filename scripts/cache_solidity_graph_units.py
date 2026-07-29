"""Create GraphCodeBERT source caches; test cache requires an explicit opt-in."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidity_graph_dataset import build_cache as build_graph_cache  # noqa: E402
from solidity_source_v2_dataset import build_cache as build_source_v2_cache, coverage_report  # noqa: E402


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
        output = cache_dir / f"{split}.pt"
        if config.get("cache_schema") == "solidity_source_windows_v2":
            count = build_source_v2_cache(data_dir / f"{split}.jsonl", output, tokenizer, config, ROOT)
            report = coverage_report(output)
            report_path = ROOT / config["report_dir"] / f"{split}_window_coverage.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        else:
            count = build_graph_cache(data_dir / f"{split}.jsonl", output, tokenizer, config, ROOT)
        print(f"[OK] cached {split}: {count}")


if __name__ == "__main__":
    main()
