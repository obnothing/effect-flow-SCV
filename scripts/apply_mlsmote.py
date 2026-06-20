import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlsmote import (  # noqa: E402
    compute_multilabel_stats,
    get_label_matrix,
    load_jsonl,
    oversample_text_multilabel,
    save_jsonl,
)


WARNING_TEXT = (
    "This stage uses text-compatible oversampling by duplicating real opcode "
    "samples. It does not synthesize opcode sequences by linear interpolation."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply MLSMOTE-compatible text oversampling to BJUT SC01 train set."
    )
    parser.add_argument(
        "--config",
        default="configs/mlsmote.yaml",
        help="Path to MLSMOTE YAML config.",
    )
    return parser.parse_args()


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_label_mapping(path):
    path = resolve_project_path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_label_names(config_label_names, mapping):
    mapping_label_names = mapping.get("label_names", [])
    if mapping_label_names and mapping_label_names != config_label_names:
        raise ValueError(
            "label_names in MLSMOTE config do not match label_mapping.json."
        )


def build_target_strategy(config):
    strategy = {
        "name": config["target_strategy"],
        "max_duplicate_per_sample": config.get("max_duplicate_per_sample", 5),
    }
    for key in [
        "paper_before_counts",
        "paper_after_counts",
        "minimum_positive_ratio",
    ]:
        if key in config:
            strategy[key] = config[key]
    return strategy


def compute_per_label_diff(label_names, targets, achieved):
    return {
        label: int(achieved[label] - targets[label])
        for label in label_names
    }


def write_text_report(path, report):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{report['dataset_name']} MLSMOTE-compatible oversampling report",
        "",
        f"status: {report['status']}",
        f"input_train_path: {report['input_train_path']}",
        f"output_train_path: {report['output_train_path']}",
        f"original_train_samples: {report['original_train_samples']}",
        f"oversampled_train_samples: {report['oversampled_train_samples']}",
        f"added_samples: {report['added_samples']}",
        f"max_duplicate_per_sample: {report['max_duplicate_per_sample']}",
        f"random_seed: {report['random_seed']}",
        f"warning: {report['warning']}",
        "",
        "Label Counts:",
        "Label | before | target_after | achieved_after | diff | IRLbl_before | IRLbl_after",
    ]
    for label in report["label_names"]:
        lines.append(
            f"{label} | "
            f"{report['before_label_counts'][label]} | "
            f"{report['target_train_after_counts'][label]} | "
            f"{report['achieved_after_counts'][label]} | "
            f"{report['per_label_diff'][label]} | "
            f"{report['irlbl_before'][label]} | "
            f"{report['irlbl_after'][label]}"
        )

    lines.extend(
        [
            "",
            f"MeanIR before: {report['mean_ir_before']}",
            f"MeanIR after: {report['mean_ir_after']}",
            "",
            "Minority labels before:",
        ]
    )
    lines.extend([f"- {label}" for label in report["minority_labels"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_report(path, report):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    label_names = config["label_names"]
    mapping = load_label_mapping(config["label_mapping_path"])
    validate_label_names(label_names, mapping)

    input_train_path = resolve_project_path(config["input_train_path"])
    output_train_path = resolve_project_path(config["output_train_path"])

    print(f"[LOADING] {input_train_path}")
    train_samples = load_jsonl(input_train_path)
    before_stats = compute_multilabel_stats(get_label_matrix(train_samples), label_names)
    target_strategy = build_target_strategy(config)

    print("[OVERSAMPLING] duplicating real opcode samples for minority labels")
    oversampled_samples, oversample_meta = oversample_text_multilabel(
        train_samples,
        label_names,
        target_strategy,
        seed=int(config.get("seed", 42)),
    )
    after_stats = compute_multilabel_stats(
        get_label_matrix(oversampled_samples), label_names
    )

    print(f"[WRITING] {output_train_path}")
    save_jsonl(oversampled_samples, output_train_path)

    targets = oversample_meta["target_train_after_counts"]
    achieved = oversample_meta["achieved_after_counts"]
    status = "ok"
    if any(achieved[label] < targets[label] for label in label_names):
        status = "warning"

    report = {
        "status": status,
        "dataset_name": config.get("dataset_name", "BJUT SC01"),
        "target_strategy": target_strategy,
        "input_train_path": input_train_path.relative_to(PROJECT_ROOT).as_posix(),
        "output_train_path": output_train_path.relative_to(PROJECT_ROOT).as_posix(),
        "original_train_samples": len(train_samples),
        "oversampled_train_samples": len(oversampled_samples),
        "added_samples": oversample_meta["added_samples"],
        "label_names": label_names,
        "before_label_counts": before_stats["label_counts"],
        "after_label_counts": after_stats["label_counts"],
        "irlbl_before": before_stats["irlbl"],
        "irlbl_after": after_stats["irlbl"],
        "mean_ir_before": before_stats["mean_ir"],
        "mean_ir_after": after_stats["mean_ir"],
        "minority_labels": before_stats["minority_labels"],
        "target_train_after_counts": targets,
        "achieved_after_counts": achieved,
        "per_label_diff": compute_per_label_diff(label_names, targets, achieved),
        "max_duplicate_per_sample": oversample_meta["max_duplicate_per_sample"],
        "random_seed": int(config.get("seed", 42)),
        "warning": WARNING_TEXT,
    }

    write_text_report(config["report_path"], report)
    write_json_report(config["report_json_path"], report)
    print(f"[OK] wrote {config['output_train_path']}")
    print(f"[OK] wrote {config['report_path']}")
    print(f"[OK] wrote {config['report_json_path']}")


if __name__ == "__main__":
    main()
