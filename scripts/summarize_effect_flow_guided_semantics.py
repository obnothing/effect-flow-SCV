import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from effect_flow_schema import RELATION_TYPES  # noqa: E402
from effect_flow_utils import GLOBAL_VULNERABILITY_LABELS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize effect-flow semantic patterns for guided MIL predictions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--predictions",
        default=None,
        help="Defaults to result_dir/test_predictions_per_label.jsonl.",
    )
    parser.add_argument(
        "--patterns",
        default="configs/effect_flow_efpp_conservative_22.json",
    )
    parser.add_argument("--output_prefix", default=None)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_yaml(path):
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_predictions(path):
    rows = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_patterns_for_indices(contract_pattern_means, indices, pattern_names, top_k):
    if not indices:
        return []
    values = contract_pattern_means[indices].mean(axis=0)
    order = np.argsort(-values)[:top_k]
    return [
        {
            "pattern": pattern_names[int(idx)],
            "mean_probability": float(values[int(idx)]),
        }
        for idx in order
    ]


def mean_top_values_for_indices(values, indices, names, top_k, value_name="mean_probability"):
    if values is None or not indices:
        return []
    mean_values = values[indices].mean(axis=0)
    order = np.argsort(-mean_values)[:top_k]
    return [
        {
            "name": names[int(idx)],
            value_name: float(mean_values[int(idx)]),
        }
        for idx in order
    ]


def mean_scalar_for_indices(values, indices, label_index):
    if values is None or not indices or label_index is None:
        return None
    return float(values[indices, label_index].mean())


def format_optional_float(value):
    return "n/a" if value is None else f"{value:.6f}"


def dataset_label_to_global_indices(label_names):
    global_index = {name: idx for idx, name in enumerate(GLOBAL_VULNERABILITY_LABELS)}
    return [global_index.get(name) for name in label_names]


def main():
    args = parse_args()
    config = load_yaml(args.config)
    result_dir = resolve_path(config["result_dir"])
    prediction_path = (
        resolve_path(args.predictions)
        if args.predictions
        else result_dir / "test_predictions_per_label.jsonl"
    )
    semantic = torch.load(
        resolve_path(config["semantic_feature_dir"]) / "test.pt",
        map_location="cpu",
    )
    predictions = load_predictions(prediction_path)
    if [str(row["id"]) for row in predictions] != [str(v) for v in semantic["ids"]]:
        raise ValueError("prediction ids do not match semantic cache ids")
    pattern_config = json.loads(resolve_path(args.patterns).read_text(encoding="utf-8"))
    pattern_names = pattern_config["included_pattern_names"]
    efpp = semantic["efpp_probs"].float()
    chunk_mask = semantic["chunk_mask"].bool()
    weights = chunk_mask.unsqueeze(-1).float()
    contract_pattern_means = (
        (efpp * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    ).numpy()
    relation = semantic.get("relation_distribution")
    evidence = semantic.get("vulnerability_evidence_probs")
    template = semantic.get("template_match_scores")
    if relation is not None:
        relation = relation.float()
        contract_relation_means = (
            (relation * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        ).numpy()
    else:
        contract_relation_means = None
    if evidence is not None:
        evidence = evidence.float()
        contract_evidence_means = (
            (evidence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        ).numpy()
    else:
        contract_evidence_means = None
    if template is not None:
        template = template.float()
        contract_template_means = (
            (template * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        ).numpy()
    else:
        contract_template_means = None
    label_names = config.get("label_names", [f"label_{idx}" for idx in range(config["num_labels"])])
    global_label_indices = dataset_label_to_global_indices(label_names)
    top_k = int(config.get("semantic_summary_top_k", 8))
    per_label = []
    for label_id, label_name in enumerate(label_names):
        global_label_id = global_label_indices[label_id]
        true_positive_indices = [
            idx for idx, row in enumerate(predictions) if int(row["multi_true"][label_id]) == 1
        ]
        predicted_positive_indices = [
            idx for idx, row in enumerate(predictions) if int(row["multi_pred"][label_id]) == 1
        ]
        true_and_pred_indices = [
            idx
            for idx, row in enumerate(predictions)
            if int(row["multi_true"][label_id]) == 1 and int(row["multi_pred"][label_id]) == 1
        ]
        per_label.append(
            {
                "label_id": int(label_id),
                "label_name": label_name,
                "true_positive_contract_count": len(true_positive_indices),
                "predicted_positive_contract_count": len(predicted_positive_indices),
                "true_and_predicted_positive_contract_count": len(true_and_pred_indices),
                "global_label_id": None if global_label_id is None else int(global_label_id),
                "mean_vep_for_true_positive_contracts": mean_scalar_for_indices(
                    contract_evidence_means,
                    true_positive_indices,
                    global_label_id,
                ),
                "mean_vep_for_predicted_positive_contracts": mean_scalar_for_indices(
                    contract_evidence_means,
                    predicted_positive_indices,
                    global_label_id,
                ),
                "mean_template_score_for_true_positive_contracts": mean_scalar_for_indices(
                    contract_template_means,
                    true_positive_indices,
                    global_label_id,
                ),
                "mean_template_score_for_predicted_positive_contracts": mean_scalar_for_indices(
                    contract_template_means,
                    predicted_positive_indices,
                    global_label_id,
                ),
                "top_patterns_for_true_positive_contracts": mean_patterns_for_indices(
                    contract_pattern_means,
                    true_positive_indices,
                    pattern_names,
                    top_k,
                ),
                "top_patterns_for_predicted_positive_contracts": mean_patterns_for_indices(
                    contract_pattern_means,
                    predicted_positive_indices,
                    pattern_names,
                    top_k,
                ),
                "top_patterns_for_true_and_predicted_positive_contracts": mean_patterns_for_indices(
                    contract_pattern_means,
                    true_and_pred_indices,
                    pattern_names,
                    top_k,
                ),
                "top_relations_for_true_positive_contracts": mean_top_values_for_indices(
                    contract_relation_means,
                    true_positive_indices,
                    RELATION_TYPES,
                    min(top_k, len(RELATION_TYPES)),
                ),
                "top_relations_for_predicted_positive_contracts": mean_top_values_for_indices(
                    contract_relation_means,
                    predicted_positive_indices,
                    RELATION_TYPES,
                    min(top_k, len(RELATION_TYPES)),
                ),
            }
        )
    report = {
        "experiment_name": config["experiment_name"],
        "semantic_feature_dir": config["semantic_feature_dir"],
        "prediction_path": project_relative(prediction_path),
        "pattern_source": args.patterns,
        "semantic_cache_schema": "v2",
        "uses_relation_distribution": contract_relation_means is not None,
        "uses_vulnerability_evidence_probs": contract_evidence_means is not None,
        "uses_template_match_scores": contract_template_means is not None,
        "top_k": top_k,
        "per_label": per_label,
    }
    prefix = resolve_path(args.output_prefix) if args.output_prefix else result_dir / "effect_flow_semantic_contribution_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    txt_path = prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["Effect-flow semantic contribution summary", ""]
    lines.append(f"experiment_name: {report['experiment_name']}")
    lines.append(f"semantic_feature_dir: {report['semantic_feature_dir']}")
    lines.append(f"prediction_path: {report['prediction_path']}")
    for row in per_label:
        lines.append("")
        lines.append(
            f"{row['label_name']}: true={row['true_positive_contract_count']} "
            f"pred={row['predicted_positive_contract_count']} "
            f"tp_pred={row['true_and_predicted_positive_contract_count']}"
        )
        if row["mean_vep_for_true_positive_contracts"] is not None:
            lines.append(
                "  mean VEP/template for true positives: "
                f"vep={format_optional_float(row['mean_vep_for_true_positive_contracts'])} "
                f"template={format_optional_float(row['mean_template_score_for_true_positive_contracts'])}"
            )
        if row["mean_vep_for_predicted_positive_contracts"] is not None:
            lines.append(
                "  mean VEP/template for predicted positives: "
                f"vep={format_optional_float(row['mean_vep_for_predicted_positive_contracts'])} "
                f"template={format_optional_float(row['mean_template_score_for_predicted_positive_contracts'])}"
            )
        lines.append("  top true-positive patterns:")
        for pattern in row["top_patterns_for_true_positive_contracts"]:
            lines.append(f"  - {pattern['pattern']}: {pattern['mean_probability']:.6f}")
        lines.append("  top predicted-positive patterns:")
        for pattern in row["top_patterns_for_predicted_positive_contracts"]:
            lines.append(f"  - {pattern['pattern']}: {pattern['mean_probability']:.6f}")
        if row["top_relations_for_true_positive_contracts"]:
            lines.append("  top true-positive relations:")
            for relation_row in row["top_relations_for_true_positive_contracts"]:
                lines.append(f"  - {relation_row['name']}: {relation_row['mean_probability']:.6f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {project_relative(txt_path)}")
    print(f"[OK] wrote {project_relative(json_path)}")


if __name__ == "__main__":
    main()
