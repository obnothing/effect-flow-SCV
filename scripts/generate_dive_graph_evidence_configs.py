import argparse
import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALES = [0.5, 1.0, 1.5]
DEFAULT_GRAPH_DIM = 12


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def dump_yaml(path, payload):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def scale_tag(value):
    return f"graph_scale_{int(round(float(value) * 100)):03d}"


def make_config(base, args, scale):
    config = copy.deepcopy(base)
    tag = scale_tag(scale)
    config["seed"] = int(args.seed)
    config["experiment_name"] = f"train_dive_{tag}_ep{args.epochs}"
    config["checkpoint_dir"] = str(Path(args.checkpoint_root) / tag).replace("\\", "/")
    config["result_dir"] = str(Path(args.result_root) / tag / "train").replace("\\", "/")
    config["ablation_dataset"] = "DIVE"
    config["ablation_variant"] = tag
    config["ablation_description"] = (
        "DIVE side_scale_150_ep50 with contract-level graph-derived evidence."
    )
    config["graph_evidence_enabled"] = True
    config["graph_evidence_dir"] = args.graph_evidence_dir
    config["graph_evidence_dim"] = int(args.graph_evidence_dim)
    config["graph_evidence_scale"] = float(scale)
    config["graph_evidence_attention_scale"] = float(scale)
    config["graph_evidence_logit_scale"] = float(scale)
    config["graph_evidence_hidden_dim"] = int(args.graph_hidden_dim)
    config["graph_evidence_dropout"] = float(args.graph_dropout)
    config["graph_evidence_enable_epoch"] = int(args.enable_epoch)
    config["beta_graph_init"] = float(args.beta_graph_init)
    config["gamma_graph_init"] = float(args.gamma_graph_init)
    config["front_running_special_enabled"] = False
    config["front_hard_negative_loss_enabled"] = False
    config["front_hard_negative_lambda"] = 0.0
    config["front_contrastive_loss_enabled"] = False
    config["front_contrastive_lambda"] = 0.0
    config["batch_size"] = int(args.batch_size)
    config["epochs"] = int(args.epochs)
    config["num_workers"] = int(args.num_workers)
    control = dict(config.get("ablation_control", {}))
    control.update(
        {
            "same_feature_cache": True,
            "same_semantic_cache": True,
            "same_random_split": True,
            "same_batch_size": True,
            "same_seed": True,
            "dataset": "DIVE",
            "variant": tag,
            "opcode_only": True,
            "contract_level_graph": True,
            "no_gnn": True,
            "no_front_special_features": True,
            "no_hard_negative_loss": True,
            "no_front_contrastive_loss": True,
        }
    )
    config["ablation_control"] = control
    config["generated_from"] = "scripts/generate_dive_graph_evidence_configs.py"
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
        description="Generate DIVE graph evidence MIL configs."
    )
    parser.add_argument(
        "--base_config",
        default="configs/generated/dive_side_scale_search/train_dive_side_scale_150_ep50.yaml",
    )
    parser.add_argument(
        "--output_config_dir",
        default="configs/generated/dive_graph_evidence_search",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints/dive_graph_evidence_search",
    )
    parser.add_argument(
        "--result_root",
        default="results/dive_graph_evidence_search",
    )
    parser.add_argument(
        "--graph_evidence_dir",
        default="data/features/graph_evidence/dive_random_stride256_max64",
    )
    parser.add_argument("--scales", nargs="*", type=float, default=DEFAULT_SCALES)
    parser.add_argument("--graph_evidence_dim", type=int, default=DEFAULT_GRAPH_DIM)
    parser.add_argument("--graph_hidden_dim", type=int, default=32)
    parser.add_argument("--graph_dropout", type=float, default=0.1)
    parser.add_argument("--enable_epoch", type=int, default=4)
    parser.add_argument("--beta_graph_init", type=float, default=0.1)
    parser.add_argument("--gamma_graph_init", type=float, default=0.1)
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
        "graph_evidence_dir": args.graph_evidence_dir,
        "graph_evidence_dim": int(args.graph_evidence_dim),
        "seed": int(args.seed),
        "configs": [],
    }
    for scale in args.scales or DEFAULT_SCALES:
        variant, config = make_config(base, args, scale)
        config_path = output_dir / f"{config['experiment_name']}.yaml"
        dump_yaml(config_path, config)
        manifest["configs"].append(
            {
                "dataset": "DIVE",
                "variant": variant,
                "graph_scale": float(scale),
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
