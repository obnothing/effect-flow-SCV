import argparse
import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_SPECS = {
    "global_residual": {
        "major_global_residual_enabled": True,
        "major_multipool_enabled": False,
        "description": "Add gated global contract residual logits for major labels.",
    },
    "multipool": {
        "major_global_residual_enabled": False,
        "major_multipool_enabled": True,
        "description": "Add label-specific multi-pool logits for major labels.",
    },
    "global_residual_multipool": {
        "major_global_residual_enabled": True,
        "major_multipool_enabled": True,
        "description": "Combine global contract residual and label-specific multi-pool logits.",
    },
}
MAJOR_LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def dump_yaml(path, payload):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def make_config(base, args, variant):
    spec = VARIANT_SPECS[variant]
    config = copy.deepcopy(base)
    experiment_name = f"dive_major_label_v2_{variant}"
    config["experiment_name"] = experiment_name
    config["checkpoint_dir"] = str(Path(args.checkpoint_root) / experiment_name).replace("\\", "/")
    config["result_dir"] = str(Path(args.result_root) / experiment_name).replace("\\", "/")
    config["ablation_dataset"] = "DIVE"
    config["ablation_variant"] = f"major_label_v2_{variant}"
    config["ablation_description"] = spec["description"]
    config["major_enhancement_labels"] = MAJOR_LABELS
    config["major_enhancement_scale_init"] = float(args.scale_init)
    config["major_enhancement_dropout"] = float(args.dropout)
    config["major_global_residual_hidden_dim"] = int(args.global_hidden_dim)
    config["major_multipool_top_k"] = int(args.multipool_top_k)
    config["major_global_residual_enabled"] = bool(spec["major_global_residual_enabled"])
    config["major_multipool_enabled"] = bool(spec["major_multipool_enabled"])
    config["front_running_special_enabled"] = False
    config["front_hard_negative_loss_enabled"] = False
    config["front_contrastive_loss_enabled"] = False
    config["rare_negative_subsampling_enabled"] = False
    config["graph_evidence_enabled"] = False
    config["generated_from"] = "scripts/generate_dive_major_label_v2_configs.py"
    config["generated_base_config"] = args.base_config
    control = dict(config.get("ablation_control", {}))
    control.update(
        {
            "same_feature_cache": True,
            "same_semantic_cache": True,
            "same_random_split": True,
            "same_seed": True,
            "dataset": "DIVE",
            "variant": config["ablation_variant"],
        }
    )
    config["ablation_control"] = control
    if args.debug_num_train_samples is not None:
        config["debug_num_train_samples"] = int(args.debug_num_train_samples)
    if args.debug_num_valid_samples is not None:
        config["debug_num_valid_samples"] = int(args.debug_num_valid_samples)
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
    if args.early_stopping_patience is not None:
        config["early_stopping_patience"] = int(args.early_stopping_patience)
    if args.num_workers is not None:
        config["num_workers"] = int(args.num_workers)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DIVE major-label v2 configs.")
    parser.add_argument(
        "--base_config",
        default="configs/generated/dive_side_scale_search/train_dive_side_scale_150_ep50.yaml",
    )
    parser.add_argument(
        "--output_config_dir",
        default="configs/generated/dive_major_label_v2",
    )
    parser.add_argument("--checkpoint_root", default="checkpoints/dive_major_label_v2")
    parser.add_argument("--result_root", default="results/dive_major_label_v2/members")
    parser.add_argument("--variants", nargs="*", default=list(VARIANT_SPECS))
    parser.add_argument("--scale_init", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--global_hidden_dim", type=int, default=256)
    parser.add_argument("--multipool_top_k", type=int, default=4)
    parser.add_argument("--debug_num_train_samples", type=int, default=None)
    parser.add_argument("--debug_num_valid_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = sorted(set(args.variants) - set(VARIANT_SPECS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    base = load_yaml(args.base_config)
    manifest = {
        "base_config": args.base_config,
        "output_config_dir": args.output_config_dir,
        "checkpoint_root": args.checkpoint_root,
        "result_root": args.result_root,
        "major_labels": MAJOR_LABELS,
        "variants": [],
    }
    output_dir = resolve(args.output_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        config = make_config(base, args, variant)
        path = output_dir / f"{config['experiment_name']}.yaml"
        dump_yaml(path, config)
        manifest["variants"].append(
            {
                "variant": variant,
                "config": display_path(path),
                "experiment_name": config["experiment_name"],
                "checkpoint_dir": config["checkpoint_dir"],
                "result_dir": config["result_dir"],
            }
        )
        print(f"[OK] wrote {display_path(path)}")
    manifest_path = output_dir / "manifest.yaml"
    dump_yaml(manifest_path, manifest)
    print(f"[OK] wrote {display_path(manifest_path)}")


if __name__ == "__main__":
    main()
