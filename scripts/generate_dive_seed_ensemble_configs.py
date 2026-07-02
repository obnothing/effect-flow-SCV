import argparse
import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 123, 777, 2024, 3407]


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path):
    with resolve(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path, payload):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def make_config(base, args, seed):
    config = copy.deepcopy(base)
    experiment_name = f"{args.experiment_prefix}_seed{seed}"
    variant = f"{args.variant_prefix}_seed{seed}"
    config["seed"] = int(seed)
    config["experiment_name"] = experiment_name
    config["checkpoint_dir"] = str(
        Path(args.checkpoint_root) / experiment_name
    ).replace("\\", "/")
    config["result_dir"] = str(Path(args.result_root) / experiment_name).replace(
        "\\", "/"
    )
    config["ablation_dataset"] = "DIVE"
    config["ablation_variant"] = variant
    config["ablation_description"] = (
        "DIVE side_scale_150_ep50 seed member for probability ensemble."
    )
    control = dict(config.get("ablation_control", {}))
    control.update(
        {
            "same_feature_cache": True,
            "same_random_split": True,
            "same_batch_size": args.batch_size is None,
            "same_seed": False,
            "dataset": "DIVE",
            "variant": variant,
        }
    )
    config["ablation_control"] = control
    config["ensemble_group"] = args.ensemble_group
    config["ensemble_seed"] = int(seed)
    config["generated_from"] = "scripts/generate_dive_seed_ensemble_configs.py"
    config["generated_base_config"] = args.base_config
    if args.debug_num_train_samples is not None:
        config["debug_num_train_samples"] = int(args.debug_num_train_samples)
    if args.debug_num_valid_samples is not None:
        config["debug_num_valid_samples"] = int(args.debug_num_valid_samples)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
    if args.early_stopping_patience is not None:
        config["early_stopping_patience"] = int(args.early_stopping_patience)
    if args.num_workers is not None:
        config["num_workers"] = int(args.num_workers)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DIVE side_scale_150_ep50 seed ensemble configs."
    )
    parser.add_argument(
        "--base_config",
        default="configs/generated/dive_side_scale_search/train_dive_side_scale_150_ep50.yaml",
    )
    parser.add_argument(
        "--output_config_dir",
        default="configs/generated/dive_seed_ensemble",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints/dive_seed_ensemble",
    )
    parser.add_argument(
        "--result_root",
        default="results/dive_seed_ensemble/members",
    )
    parser.add_argument("--experiment_prefix", default="train_dive_side_scale_150")
    parser.add_argument("--variant_prefix", default="side_scale_150_ep50")
    parser.add_argument("--ensemble_group", default="dive_seed_ensemble")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--debug_num_train_samples", type=int, default=None)
    parser.add_argument("--debug_num_valid_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds or DEFAULT_SEEDS
    base = load_yaml(args.base_config)
    output_dir = resolve(args.output_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_config": args.base_config,
        "output_config_dir": args.output_config_dir,
        "checkpoint_root": args.checkpoint_root,
        "result_root": args.result_root,
        "ensemble_group": args.ensemble_group,
        "seeds": [int(seed) for seed in seeds],
        "configs": [],
    }
    for seed in seeds:
        config = make_config(base, args, seed)
        config_path = output_dir / f"{config['experiment_name']}.yaml"
        dump_yaml(config_path, config)
        manifest["configs"].append(
            {
                "dataset": "DIVE",
                "variant": config["ablation_variant"],
                "seed": int(seed),
                "config": str(config_path.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
                "experiment_name": config["experiment_name"],
                "checkpoint_dir": config["checkpoint_dir"],
                "result_dir": config["result_dir"],
            }
        )
        print(f"[OK] wrote {config_path.relative_to(PROJECT_ROOT)}")
    manifest_path = output_dir / "manifest.yaml"
    dump_yaml(manifest_path, manifest)
    print(f"[OK] wrote {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
