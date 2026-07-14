import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chunk_feature_dataset import build_chunk_feature_datasets  # noqa: E402
from train_chunk_mil import (  # noqa: E402
    apply_config_overrides,
    build_model,
    collate_batch,
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--target_config", required=True)
    parser.add_argument("--target_variant", required=True)
    parser.add_argument("--target_override", action="append", default=[])
    parser.add_argument("--target_checkpoint", required=True)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def model_inputs(batch):
    keys = [
        "chunk_features",
        "chunk_mask",
        "efpp_probs",
        "etp_distribution",
        "relation_distribution",
    ]
    return {key: batch[key] for key in keys if key in batch}


def main():
    args = parse_args()
    source_checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    target_checkpoint = torch.load(args.target_checkpoint, map_location="cpu")
    source_config = source_checkpoint["config"]
    target_config = apply_config_overrides(
        load_config(args.target_config, args.target_variant),
        args.target_override,
    )
    source_labels = list(source_config["label_names"])
    target_labels = list(target_config["label_names"])

    source_model = build_model(source_config)
    source_model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
    target_model = build_model(target_config)
    target_model.load_state_dict(target_checkpoint["model_state_dict"], strict=True)
    source_model.eval()
    target_model.eval()
    # Compare the deployed state with all reliable side-evidence gates enabled.
    source_model.set_training_epoch(10**6)
    target_model.set_training_epoch(10**6)

    dataset_config = dict(target_config)
    dataset_config["debug_num_valid_samples"] = int(args.max_samples)
    dataset = build_chunk_feature_datasets(dataset_config)["valid"]
    loader = DataLoader(
        dataset,
        batch_size=min(int(args.max_samples), len(dataset)),
        shuffle=False,
        collate_fn=collate_batch,
    )
    batch = next(iter(loader))
    inputs = model_inputs(batch)
    with torch.no_grad():
        source_logits = source_model(**inputs)["recognition_logits"]
        target_logits = target_model(**inputs)["recognition_logits"]
    target_old_indices = [target_labels.index(name) for name in source_labels]
    target_old_logits = target_logits[:, target_old_indices]
    max_logit_difference = float(
        (source_logits - target_old_logits).abs().max().item()
    )

    source_state = source_checkpoint["model_state_dict"]
    target_state = target_checkpoint["model_state_dict"]
    parameter_mismatches = []
    for source_idx, label_name in enumerate(source_labels):
        target_idx = target_labels.index(label_name)
        source_prefix = f"classic_label_branches.{source_idx}."
        target_prefix = f"classic_label_branches.{target_idx}."
        for source_key, source_value in source_state.items():
            if not source_key.startswith(source_prefix):
                continue
            target_key = target_prefix + source_key[len(source_prefix) :]
            if target_key not in target_state or not torch.equal(
                source_value, target_state[target_key]
            ):
                parameter_mismatches.append(
                    {"label_name": label_name, "parameter": source_key}
                )
    label_specific_prefixes = (
        "classic_label_branches.",
        "label_attn",
        "label_out",
        "label_bias",
        "chunk_classifier.",
        "beta_reliable_raw",
        "gamma_reliable_raw",
        "evidence_logit_weight",
        "evidence_logit_bias",
        "template_",
        "active_global_indices",
        "major_enhancement_label_mask",
    )
    shared_parameter_mismatches = []
    for source_key, source_value in source_state.items():
        if source_key.startswith(label_specific_prefixes):
            continue
        if source_key not in target_state:
            continue
        if source_value.shape != target_state[source_key].shape:
            continue
        if not torch.equal(source_value, target_state[source_key]):
            shared_parameter_mismatches.append(source_key)

    report = {
        "source_checkpoint": args.source_checkpoint,
        "target_checkpoint": args.target_checkpoint,
        "source_labels": source_labels,
        "target_labels": target_labels,
        "samples_checked": int(source_logits.shape[0]),
        "max_old_label_logit_difference": max_logit_difference,
        "logit_tolerance": float(args.tolerance),
        "logits_within_tolerance": max_logit_difference <= float(args.tolerance),
        "old_branch_parameter_mismatch_count": len(parameter_mismatches),
        "old_branch_parameters_exact": not parameter_mismatches,
        "parameter_mismatches": parameter_mismatches,
        "shared_parameter_mismatch_count": len(shared_parameter_mismatches),
        "shared_parameters_exact": not shared_parameter_mismatches,
        "shared_parameter_mismatches": shared_parameter_mismatches,
        "status": (
            "ok"
            if max_logit_difference <= float(args.tolerance)
            and not parameter_mismatches
            and not shared_parameter_mismatches
            else "error"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
