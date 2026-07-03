import argparse
import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAMBDAS = [0.005, 0.01, 0.02]


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


def lambda_tag(value):
    return f"lam{int(round(float(value) * 1000)):04d}"


def tau_tag(value):
    return f"tau{int(round(float(value) * 100)):03d}"


def make_config(base, args, lambda_value):
    config = copy.deepcopy(base)
    tag = f"front_ctr_mined_{lambda_tag(lambda_value)}_{tau_tag(args.temperature)}"
    config["seed"] = int(args.seed)
    config["experiment_name"] = f"train_dive_{tag}"
    config["checkpoint_dir"] = str(Path(args.checkpoint_root) / tag).replace("\\", "/")
    config["result_dir"] = str(Path(args.result_root) / tag / "train").replace("\\", "/")
    config["ablation_dataset"] = "DIVE"
    config["ablation_variant"] = tag
    config["ablation_description"] = (
        "DIVE side_scale_150_ep50 with Front Running mined conditional contrastive loss."
    )
    config["front_running_special_enabled"] = False
    config["front_hard_negative_loss_enabled"] = False
    config["front_hard_negative_lambda"] = 0.0
    config["front_contrastive_loss_enabled"] = True
    config["front_contrastive_lambda"] = float(lambda_value)
    config["front_contrastive_temperature"] = float(args.temperature)
    config["front_contrastive_dim"] = int(args.contrastive_dim)
    config["front_contrastive_enable_epoch"] = int(args.enable_epoch)
    config["front_contrastive_confounder_labels"] = list(args.confounder_labels)
    config["front_contrastive_negative_mining"] = args.negative_mining
    config["front_contrastive_hard_negative_ratio"] = float(args.hard_negative_ratio)
    config["front_contrastive_hard_negative_min_k"] = int(args.hard_negative_min_k)
    config["front_contrastive_hard_negative_max_k"] = int(args.hard_negative_max_k)
    config["front_running_label_name"] = "Front Running"
    config["batch_size"] = int(args.batch_size)
    config["epochs"] = int(args.epochs)
    config["num_workers"] = int(args.num_workers)
    control = dict(config.get("ablation_control", {}))
    control.update(
        {
            "same_feature_cache": True,
            "same_random_split": True,
            "same_batch_size": True,
            "same_seed": True,
            "dataset": "DIVE",
            "variant": tag,
            "opcode_only": True,
            "front_contrastive_negative_mining": args.negative_mining,
            "no_front_special_features": True,
            "no_hard_negative_loss": True,
        }
    )
    config["ablation_control"] = control
    config["generated_from"] = "scripts/generate_dive_front_contrastive_mined_configs.py"
    config["generated_base_config"] = args.base_config
    if args.debug_num_train_samples is not None:
        config["debug_num_train_samples"] = int(args.debug_num_train_samples)
    if args.debug_num_valid_samples is not None:
        config["debug_num_valid_samples"] = int(args.debug_num_valid_samples)
    if args.early_stopping_patience is not None:
        config["early_stopping_patience"] = int(args.early_stopping_patience)
    return tag, config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DIVE Front Running mined contrastive configs."
    )
    parser.add_argument(
        "--base_config",
        default="configs/generated/dive_side_scale_search/train_dive_side_scale_150_ep50.yaml",
    )
    parser.add_argument(
        "--output_config_dir",
        default="configs/generated/dive_front_contrastive_mined_search",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints/dive_front_contrastive_mined_search",
    )
    parser.add_argument(
        "--result_root",
        default="results/dive_front_contrastive_mined_search",
    )
    parser.add_argument("--lambdas", nargs="*", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--contrastive_dim", type=int, default=128)
    parser.add_argument("--enable_epoch", type=int, default=4)
    parser.add_argument(
        "--confounder_labels",
        nargs="*",
        default=["Access Control", "Reentrancy", "Time manipulation"],
    )
    parser.add_argument(
        "--negative_mining",
        default="front_prob_topk",
        choices=["front_prob_topk"],
    )
    parser.add_argument("--hard_negative_ratio", type=float, default=2.0)
    parser.add_argument("--hard_negative_min_k", type=int, default=16)
    parser.add_argument("--hard_negative_max_k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--debug_num_train_samples", type=int, default=None)
    parser.add_argument("--debug_num_valid_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    base = load_yaml(args.base_config)
    output_dir = resolve(args.output_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_config": args.base_config,
        "output_config_dir": args.output_config_dir,
        "checkpoint_root": args.checkpoint_root,
        "result_root": args.result_root,
        "seed": int(args.seed),
        "negative_mining": args.negative_mining,
        "hard_negative_ratio": float(args.hard_negative_ratio),
        "hard_negative_min_k": int(args.hard_negative_min_k),
        "hard_negative_max_k": int(args.hard_negative_max_k),
        "configs": [],
    }
    for lambda_value in args.lambdas or DEFAULT_LAMBDAS:
        variant, config = make_config(base, args, lambda_value)
        config_path = output_dir / f"{config['experiment_name']}.yaml"
        dump_yaml(config_path, config)
        manifest["configs"].append(
            {
                "dataset": "DIVE",
                "variant": variant,
                "lambda": float(lambda_value),
                "temperature": float(args.temperature),
                "negative_mining": args.negative_mining,
                "config": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "experiment_name": config["experiment_name"],
                "checkpoint_dir": config["checkpoint_dir"],
                "train_result_dir": config["result_dir"],
                "eval_result_root": str(Path(args.result_root) / variant).replace("\\", "/"),
            }
        )
        print(f"[OK] wrote {config_path.relative_to(PROJECT_ROOT)}")
    manifest_path = output_dir / "manifest.yaml"
    dump_yaml(manifest_path, manifest)
    print(f"[OK] wrote {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
