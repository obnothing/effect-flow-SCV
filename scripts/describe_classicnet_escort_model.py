import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_chunk_mil import build_model, load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    args = parse_args()
    config = load_config(args.config, args.variant)
    model = build_model(config)
    branch_rows = []
    if model.classic_label_branches is not None:
        for label_name, branch in zip(model.label_names, model.classic_label_branches):
            branch_rows.append(
                {
                    "label_name": label_name,
                    "parameters": count_parameters(branch),
                    "bottleneck_dim": branch.down_projection.out_features,
                }
            )
    report = {
        "variant": args.variant,
        "recognition_head_type": model.recognition_head_type,
        "chunk_context_encoder_type": config.get(
            "chunk_context_encoder_type", "transformer"
        ),
        "input_shape": ["B", config["max_chunks"], config["feature_dim"]],
        "hidden_dim": config["hidden_dim"],
        "attention_dim": config["attn_dim"],
        "transformer_layers": config.get("chunk_context_num_layers"),
        "transformer_heads": config.get("chunk_context_num_heads"),
        "label_adapter_dim": config.get("classic_label_adapter_dim"),
        "total_parameters": count_parameters(model),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "shared_label_adapter_parameters": (
            count_parameters(model.classic_label_adapter)
            if model.classic_label_adapter is not None
            else 0
        ),
        "branches": branch_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
