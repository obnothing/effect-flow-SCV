"""Audit one light-label config before an experiment starts.

This is deliberately a fail-fast check: a configuration that does not map to
the constructed model is an invalid experiment, not a warning to ignore.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet, validate_model_config  # noqa: E402
from train_light_label_model import load_config, resolve  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("allow_test"):
        raise ValueError("test remains locked")
    if str(config["variant"]) not in {
        "b0_mean", "b1_shared_attention", "b2_label_attention",
        "l1_symmetric_local", "l2_directional_local",
    }:
        raise ValueError(f"unsupported audit variant: {config['variant']}")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    model = LabelGuidedOpcodeNet(
        config["variant"], len(tokenizer), tokenizer.pad_token_id,
        config["embedding_dim"], config["gru_hidden_size"], config["num_labels"],
        config["bidirectional"], config.get("local_radius", 8), config.get("gru_layers", 1),
    )
    validate_model_config(model, config)
    cache = resolve(config["cache_dir"]) / f"train_max{config['max_len']}.pt"
    if not cache.exists():
        raise FileNotFoundError(f"train cache missing for max_len={config['max_len']}: {cache}")
    valid_cache = resolve(config["cache_dir"]) / f"valid_max{config['max_len']}.pt"
    if not valid_cache.exists():
        raise FileNotFoundError(f"valid cache missing for max_len={config['max_len']}: {valid_cache}")
    params = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "config": str(resolve(args.config)),
        "variant": config["variant"],
        "embedding_dim": int(model.embedding.embedding_dim),
        "gru_hidden_size": int(model.encoder.hidden_size),
        "gru_layers": int(model.encoder.num_layers),
        "bidirectional": bool(model.encoder.bidirectional),
        "num_labels": int(model.num_labels),
        "params": int(params),
        "cache": str(cache),
        "valid_cache": str(valid_cache),
        "test_checked": False,
    }
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
