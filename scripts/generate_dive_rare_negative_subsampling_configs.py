import argparse
import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATIOS = ["frac050", "frac025", "frac0125", "ratio005", "ratio003"]
RATIO_SPECS = {
    "frac050": {
        "policy": "random_fraction",
        "negative_fraction": 0.5,
        "neg_per_pos": None,
        "description": "rare-label negative BCE loss keeps one half of negatives",
    },
    "frac025": {
        "policy": "random_fraction",
        "negative_fraction": 0.25,
        "neg_per_pos": None,
        "description": "rare-label negative BCE loss keeps one quarter of negatives",
    },
    "frac0125": {
        "policy": "random_fraction",
        "negative_fraction": 0.125,
        "neg_per_pos": None,
        "description": "rare-label negative BCE loss keeps one eighth of negatives",
    },
    "ratio005": {
        "policy": "random_pos_ratio",
        "negative_fraction": None,
        "neg_per_pos": 5.0,
        "description": "rare-label negative BCE loss keeps at most five negatives per positive",
    },
    "ratio003": {
        "policy": "random_pos_ratio",
        "negative_fraction": None,
        "neg_per_pos": 3.0,
        "description": "rare-label negative BCE loss keeps at most three negatives per positive",
    },
}
KNOWN_LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation",
]


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


def normalize_target_labels(values):
    if not values:
        return ["Front Running", "Bad Randomness"]
    text = " ".join(values).replace(",", " ")
    labels = []
    cursor = text.strip()
    while cursor:
        match = None
        for label in sorted(KNOWN_LABELS, key=len, reverse=True):
            if cursor == label or cursor.startswith(label + " "):
                match = label
                break
        if match is None:
            raise ValueError(
                "Could not parse rare negative subsampling target labels from: "
                f"{values}. Use known DIVE label names."
            )
        labels.append(match)
        cursor = cursor[len(match) :].strip()
    return labels


def make_config(base, args, ratio_name, target_labels):
    if ratio_name not in RATIO_SPECS:
        raise ValueError(
            f"Unknown ratio {ratio_name}. Expected one of: {sorted(RATIO_SPECS)}"
        )
    spec = RATIO_SPECS[ratio_name]
    config = copy.deepcopy(base)
    tag = f"rare_neg_{ratio_name}"
    config["seed"] = int(args.seed)
    config["experiment_name"] = f"train_dive_{tag}"
    config["checkpoint_dir"] = str(Path(args.checkpoint_root) / tag).replace("\\", "/")
    config["result_dir"] = str(Path(args.result_root) / tag / "train").replace("\\", "/")
    config["ablation_dataset"] = "DIVE"
    config["ablation_variant"] = tag
    config["ablation_description"] = (
        "DIVE side_scale_150_ep50 with label-wise random negative "
        f"subsampling BCE for rare labels; {spec['description']}."
    )
    config["front_running_special_enabled"] = False
    config["front_hard_negative_loss_enabled"] = False
    config["front_hard_negative_lambda"] = 0.0
    config["front_contrastive_loss_enabled"] = False
    config["front_contrastive_lambda"] = 0.0
    config["graph_evidence_enabled"] = False
    config["rare_negative_subsampling_enabled"] = True
    config["rare_negative_subsampling_labels"] = list(target_labels)
    config["rare_negative_subsampling_policy"] = spec["policy"]
    config["rare_negative_subsampling_negative_fraction"] = (
        1.0 if spec["negative_fraction"] is None else float(spec["negative_fraction"])
    )
    config["rare_negative_subsampling_neg_per_pos"] = (
        5.0 if spec["neg_per_pos"] is None else float(spec["neg_per_pos"])
    )
    config["rare_negative_subsampling_skip_when_no_positive"] = True
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
            "rare_negative_subsampling": True,
            "rare_negative_subsampling_ratio": ratio_name,
            "rare_negative_subsampling_random_only": True,
            "no_front_special_features": True,
            "no_hard_negative_loss": True,
            "no_front_contrastive_loss": True,
            "no_graph_evidence": True,
        }
    )
    config["ablation_control"] = control
    config["generated_from"] = "scripts/generate_dive_rare_negative_subsampling_configs.py"
    config["generated_base_config"] = args.base_config
    if args.debug_num_train_samples is not None:
        config["debug_num_train_samples"] = int(args.debug_num_train_samples)
    if args.debug_num_valid_samples is not None:
        config["debug_num_valid_samples"] = int(args.debug_num_valid_samples)
    if args.early_stopping_patience is not None:
        config["early_stopping_patience"] = int(args.early_stopping_patience)
    return tag, config, spec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DIVE rare-label random negative subsampling configs."
    )
    parser.add_argument(
        "--base_config",
        default="configs/generated/dive_side_scale_search/train_dive_side_scale_150_ep50.yaml",
    )
    parser.add_argument(
        "--output_config_dir",
        default="configs/generated/dive_rare_negative_subsampling_search",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints/dive_rare_negative_subsampling_search",
    )
    parser.add_argument(
        "--result_root",
        default="results/dive_rare_negative_subsampling_search",
    )
    parser.add_argument("--ratios", nargs="*", default=DEFAULT_RATIOS)
    parser.add_argument(
        "--target_labels",
        nargs="*",
        default=["Front Running", "Bad Randomness"],
    )
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
    target_labels = normalize_target_labels(args.target_labels)
    output_dir = resolve(args.output_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_config": args.base_config,
        "output_config_dir": args.output_config_dir,
        "checkpoint_root": args.checkpoint_root,
        "result_root": args.result_root,
        "seed": int(args.seed),
        "target_labels": target_labels,
        "configs": [],
    }
    for ratio_name in args.ratios or DEFAULT_RATIOS:
        variant, config, spec = make_config(base, args, ratio_name, target_labels)
        config_path = output_dir / f"{config['experiment_name']}.yaml"
        dump_yaml(config_path, config)
        manifest["configs"].append(
            {
                "dataset": "DIVE",
                "variant": variant,
                "ratio": ratio_name,
                "policy": spec["policy"],
                "negative_fraction": spec["negative_fraction"],
                "neg_per_pos": spec["neg_per_pos"],
                "target_labels": list(target_labels),
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
