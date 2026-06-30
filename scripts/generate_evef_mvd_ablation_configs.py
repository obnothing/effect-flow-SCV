import argparse
import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path):
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path, payload):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def variant_config(dataset_name, dataset_spec, variant, matrix):
    base = load_yaml(dataset_spec["base_config"])
    config = copy.deepcopy(base)
    dataset_key = dataset_name.lower()
    variant_name = variant["name"]
    experiment_name = f"{dataset_spec['experiment_prefix']}_{variant_name}"
    config["experiment_name"] = experiment_name
    config["checkpoint_dir"] = str(
        Path(matrix["checkpoint_root"]) / experiment_name
    ).replace("\\", "/")
    config["result_dir"] = str(
        Path(matrix["output_root"]) / experiment_name
    ).replace("\\", "/")
    config["ablation_dataset"] = dataset_name
    config["ablation_variant"] = variant_name
    config["ablation_description"] = variant.get("description", "")
    config["side_evidence_enabled"] = bool(variant.get("side_evidence", False))
    config["coefficient_scale"] = variant.get("coefficient_scale")
    config["ablation_control"] = {
        "same_feature_cache": True,
        "same_random_split": True,
        "same_batch_size": True,
        "same_seed": True,
        "dataset": dataset_name,
        "variant": variant_name,
    }
    for key, value in variant.get("overrides", {}).items():
        config[key] = value
    if not variant.get("side_evidence", False):
        config["model_type"] = "evm_chunk_mil"
        config["semantic_feature_dir"] = None
        config["use_weak_semantic_features"] = False
    config["generated_from"] = "configs/evef_mvd_v2_side_evidence_ablation.yaml"
    config["generated_dataset_key"] = dataset_key
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate EVEF-MVD v2 side-evidence ablation configs."
    )
    parser.add_argument(
        "--matrix",
        default="configs/evef_mvd_v2_side_evidence_ablation.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    matrix = load_yaml(args.matrix)
    output_dir = resolve_path(matrix["output_config_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "matrix": args.matrix,
        "output_config_dir": matrix["output_config_dir"],
        "configs": [],
    }
    for dataset_name, dataset_spec in matrix["datasets"].items():
        for variant in matrix["variants"]:
            config = variant_config(dataset_name, dataset_spec, variant, matrix)
            path = output_dir / (
                f"{dataset_spec['experiment_prefix']}_{variant['name']}.yaml"
            )
            dump_yaml(path, config)
            manifest["configs"].append(
                {
                    "dataset": dataset_name,
                    "variant": variant["name"],
                    "side_evidence": bool(variant.get("side_evidence", False)),
                    "config": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "experiment_name": config["experiment_name"],
                    "checkpoint_dir": config["checkpoint_dir"],
                    "result_dir": config["result_dir"],
                }
            )
            print(f"[OK] wrote {path.relative_to(PROJECT_ROOT)}")
    manifest_path = output_dir / "manifest.yaml"
    dump_yaml(manifest_path, manifest)
    print(f"[OK] wrote {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
