import argparse
import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_SPECS = {
    "branch_mlp_recognition_only": {
        "description": (
            "Recognition-only training with shared chunk backbone and per-label "
            "branch MLP heads, keeping side evidence fusion."
        ),
        "label_branch_hidden_dim": 256,
        "label_branch_dropout": 0.1,
        "label_branch_use_layernorm": True,
    },
}


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
    config["ablation_variant"] = f"major_label_v2_branch_single_task_{variant}"
    config["ablation_description"] = spec["description"]

    config["task_mode"] = "recognition_only"
    config["detection_head_enabled"] = False
    config["detection_loss_weight"] = 0.0
    config["recognition_loss_weight"] = 1.0
    config["derived_detection_from_multilabel"] = True

    config["recognition_head_type"] = "label_branch_mlp"
    config["label_branch_hidden_dim"] = int(spec["label_branch_hidden_dim"])
    config["label_branch_dropout"] = float(spec["label_branch_dropout"])
    config["label_branch_use_layernorm"] = bool(spec["label_branch_use_layernorm"])

    config["side_evidence_enabled"] = True
    config["coefficient_scale"] = 1.5
    config["major_global_residual_enabled"] = False
    config["major_multipool_enabled"] = False
    config["front_running_special_enabled"] = False
    config["front_hard_negative_loss_enabled"] = False
    config["front_contrastive_loss_enabled"] = False
    config["rare_negative_subsampling_enabled"] = False
    config["graph_evidence_enabled"] = False
    config["generated_from"] = "scripts/generate_dive_major_label_v2_branch_single_task_configs.py"
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
            "single_task_recognition_only": True,
            "label_branch_mlp": True,
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
    parser = argparse.ArgumentParser(
        description="Generate DIVE major-label v2 branch single-task configs."
    )
    parser.add_argument(
        "--base_config",
        default="configs/generated/dive_side_scale_search/train_dive_side_scale_150_ep50.yaml",
    )
    parser.add_argument(
        "--output_config_dir",
        default="configs/generated/dive_major_label_v2_branch_single_task",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints/dive_major_label_v2_branch_single_task",
    )
    parser.add_argument(
        "--result_root",
        default="results/dive_major_label_v2_branch_single_task/members",
    )
    parser.add_argument("--variants", nargs="*", default=list(VARIANT_SPECS))
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
