import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chunk_feature_dataset import ChunkFeatureDataset  # noqa: E402
from evaluate_chunk_mil import (  # noqa: E402
    feature_path,
    json_default,
    load_best_global_macro_threshold,
    load_config,
    load_model,
    load_threshold_selection,
    sigmoid,
)
from metrics import precision_recall_f1  # noqa: E402
from train_chunk_mil import collate_batch  # noqa: E402


GROUPS = ("tp", "fp", "fn", "tn", "predicted_positive", "true_positive", "all")
SCORE_KEYS = (
    "risk_evidence_scores",
    "protective_evidence_scores",
    "missing_check_evidence_scores",
    "final_evidence_scores",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize risk/protective/missing-check contribution by label."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--threshold", default="auto")
    parser.add_argument("--threshold_file", default=None)
    parser.add_argument("--global_threshold_file", default=None)
    parser.add_argument("--output_prefix", default=None)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_loader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def threshold_values(args, checkpoint, config, num_labels):
    if args.threshold_file:
        values = load_threshold_selection(resolve_path(args.threshold_file))
        return np.asarray(values, dtype=float), "validation_selected_per_label"
    if args.global_threshold_file:
        value = load_best_global_macro_threshold(resolve_path(args.global_threshold_file))
        return np.full(num_labels, float(value)), "validation_best_global_macro"
    if args.threshold == "auto":
        value = checkpoint.get("best_threshold_by_macro_f1") or config.get("threshold", 0.5)
        return np.full(num_labels, float(value)), "checkpoint_best_threshold_by_macro_f1"
    value = float(args.threshold)
    return np.full(num_labels, value), "cli"


def init_stats(num_labels):
    return {
        "tp": np.zeros(num_labels, dtype=np.int64),
        "fp": np.zeros(num_labels, dtype=np.int64),
        "fn": np.zeros(num_labels, dtype=np.int64),
        "tn": np.zeros(num_labels, dtype=np.int64),
        "sum": {
            group: {
                score_key: np.zeros(num_labels, dtype=np.float64)
                for score_key in SCORE_KEYS
            }
            for group in GROUPS
        },
        "count": {group: np.zeros(num_labels, dtype=np.int64) for group in GROUPS},
    }


def add_group(stats, group, mask, score_arrays):
    counts = mask.sum(axis=0).astype(np.int64)
    stats["count"][group] += counts
    for score_key, values in score_arrays.items():
        stats["sum"][group][score_key] += (values * mask).sum(axis=0)


def group_means(stats, group, label_id):
    count = int(stats["count"][group][label_id])
    if count == 0:
        return {
            "count": 0,
            "risk_mean": None,
            "protective_mean": None,
            "missing_check_mean": None,
            "final_evidence_mean": None,
        }
    return {
        "count": count,
        "risk_mean": float(
            stats["sum"][group]["risk_evidence_scores"][label_id] / count
        ),
        "protective_mean": float(
            stats["sum"][group]["protective_evidence_scores"][label_id] / count
        ),
        "missing_check_mean": float(
            stats["sum"][group]["missing_check_evidence_scores"][label_id] / count
        ),
        "final_evidence_mean": float(
            stats["sum"][group]["final_evidence_scores"][label_id] / count
        ),
    }


def format_value(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def aggregate_contributions(model, loader, device, thresholds, num_labels):
    stats = init_stats(num_labels)
    evaluated_samples = 0
    for batch in tqdm(loader, desc="behavior", leave=False):
        inputs = {
            "chunk_features": batch["chunk_features"].to(device),
            "chunk_mask": batch["chunk_mask"].to(device),
            "binary_label": batch["binary_label"].to(device),
            "multi_labels": batch["multi_labels"].to(device),
        }
        if "efpp_probs" in batch:
            inputs["efpp_probs"] = batch["efpp_probs"].to(device)
            inputs["etp_distribution"] = batch["etp_distribution"].to(device)
            inputs["relation_distribution"] = batch["relation_distribution"].to(device)
            inputs["vulnerability_evidence_probs"] = batch[
                "vulnerability_evidence_probs"
            ].to(device)
            inputs["template_match_scores"] = batch["template_match_scores"].to(device)
            inputs["chunk_vulnerability_evidence"] = batch[
                "chunk_vulnerability_evidence"
            ].to(device)
            inputs["vulnerability_template_matches"] = batch[
                "vulnerability_template_matches"
            ].to(device)
            inputs["active_vulnerability_label_mask"] = batch[
                "active_vulnerability_label_mask"
            ].to(device)
        if "front_special_features" in batch:
            inputs["front_special_features"] = batch["front_special_features"].to(device)
        outputs = model(**inputs)
        missing = [key for key in SCORE_KEYS if key not in outputs]
        if missing:
            raise ValueError(
                "Model output does not contain behavior contribution tensors: "
                f"{missing}. Use an effect-flow side-evidence model."
            )
        if "attn_weights" not in outputs:
            raise ValueError("Model output does not contain attn_weights.")

        probs = sigmoid(outputs["recognition_logits"].detach().cpu().numpy())
        pred = (probs >= thresholds.reshape(1, -1)).astype(np.int64)
        true = inputs["multi_labels"].detach().cpu().numpy().astype(np.int64)

        attn = outputs["attn_weights"].detach().float()
        score_arrays = {}
        for score_key in SCORE_KEYS:
            scores = outputs[score_key].detach().float()
            score_arrays[score_key] = (
                attn * scores
            ).sum(dim=1).cpu().numpy().astype(np.float64)

        tp_mask = (pred == 1) & (true == 1)
        fp_mask = (pred == 1) & (true == 0)
        fn_mask = (pred == 0) & (true == 1)
        tn_mask = (pred == 0) & (true == 0)
        stats["tp"] += tp_mask.sum(axis=0)
        stats["fp"] += fp_mask.sum(axis=0)
        stats["fn"] += fn_mask.sum(axis=0)
        stats["tn"] += tn_mask.sum(axis=0)
        add_group(stats, "tp", tp_mask, score_arrays)
        add_group(stats, "fp", fp_mask, score_arrays)
        add_group(stats, "fn", fn_mask, score_arrays)
        add_group(stats, "tn", tn_mask, score_arrays)
        add_group(stats, "predicted_positive", pred == 1, score_arrays)
        add_group(stats, "true_positive", true == 1, score_arrays)
        add_group(stats, "all", np.ones_like(true, dtype=bool), score_arrays)
        evaluated_samples += true.shape[0]
    return stats, evaluated_samples


def build_report(config, args, checkpoint, thresholds, threshold_source, stats, samples):
    label_names = config.get(
        "label_names",
        [f"label_{idx}" for idx in range(int(config.get("num_labels", 10)))],
    )
    per_label = []
    for label_id, label_name in enumerate(label_names):
        tp = int(stats["tp"][label_id])
        fp = int(stats["fp"][label_id])
        fn = int(stats["fn"][label_id])
        tn = int(stats["tn"][label_id])
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        row = {
            "label_id": int(label_id),
            "label_name": label_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(tp + fn),
            "predicted_positive_count": int(tp + fp),
            "true_negative_count": tn,
            "threshold": float(thresholds[label_id]),
            "contribution_scope": (
                "attention-weighted chunk evidence; primary means are "
                "over predicted-positive contracts"
            ),
            "predicted_positive_contribution": group_means(
                stats,
                "predicted_positive",
                label_id,
            ),
            "true_positive_contract_contribution": group_means(
                stats,
                "true_positive",
                label_id,
            ),
            "tp_contribution": group_means(stats, "tp", label_id),
            "fp_contribution": group_means(stats, "fp", label_id),
            "fn_contribution": group_means(stats, "fn", label_id),
            "tn_contribution": group_means(stats, "tn", label_id),
            "all_contribution": group_means(stats, "all", label_id),
        }
        per_label.append(row)
    return {
        "experiment_name": config["experiment_name"],
        "model_type": config.get("model_type"),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "evaluated_samples": samples,
        "threshold_source": threshold_source,
        "thresholds": [float(v) for v in thresholds.tolist()],
        "feature_dir": config["feature_dir"],
        "semantic_feature_dir": config.get("semantic_feature_dir"),
        "risk_evidence_weight": config.get("risk_evidence_weight"),
        "missing_check_evidence_weight": config.get("missing_check_evidence_weight"),
        "protective_evidence_weight": config.get("protective_evidence_weight"),
        "effect_type_evidence_weight": config.get("effect_type_evidence_weight"),
        "relation_evidence_weight": config.get("relation_evidence_weight"),
        "behavior_weight_path": config.get("behavior_weight_path"),
        "use_weighted_behavior_scoring": bool(config.get("behavior_weight_path")),
        "beta_reliable_init": config.get("beta_reliable_init"),
        "gamma_reliable_init": config.get("gamma_reliable_init"),
        "per_label": per_label,
    }


def write_report(prefix, report):
    prefix = resolve_path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    txt_path = prefix.with_suffix(".txt")
    json_path.write_text(
        json.dumps(report, indent=2, default=json_default, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "Behavior contribution summary",
        "",
        f"experiment_name: {report['experiment_name']}",
        f"split: {report['split']}",
        f"checkpoint_epoch: {report['checkpoint_epoch']}",
        f"evaluated_samples: {report['evaluated_samples']}",
        f"threshold_source: {report['threshold_source']}",
        "",
        "Primary table: predicted-positive, attention-weighted behavior means",
        (
            "label | accuracy | precision | recall | f1 | support | predicted | "
            "risk_mean | protective_mean | missing_check_mean | "
            "final_evidence_mean | tp_final | fp_final | fn_final"
        ),
        "-" * 180,
    ]
    for row in report["per_label"]:
        pred_pos = row["predicted_positive_contribution"]
        tp = row["tp_contribution"]
        fp = row["fp_contribution"]
        fn = row["fn_contribution"]
        lines.append(
            f"{row['label_name']} | {row['accuracy']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {row['support']} | "
            f"{row['predicted_positive_count']} | "
            f"{format_value(pred_pos['risk_mean'])} | "
            f"{format_value(pred_pos['protective_mean'])} | "
            f"{format_value(pred_pos['missing_check_mean'])} | "
            f"{format_value(pred_pos['final_evidence_mean'])} | "
            f"{format_value(tp['final_evidence_mean'])} | "
            f"{format_value(fp['final_evidence_mean'])} | "
            f"{format_value(fn['final_evidence_mean'])}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


def main():
    args = parse_args()
    config = load_config(resolve_path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_path = None
    if config.get("semantic_feature_dir"):
        semantic_path = resolve_path(config["semantic_feature_dir"]) / f"{args.split}.pt"
    front_special_path = None
    if config.get("front_special_feature_dir"):
        front_special_path = resolve_path(config["front_special_feature_dir"]) / f"{args.split}.pt"
    dataset = ChunkFeatureDataset(
        resolve_path(feature_path(config, args.split)),
        seed=config.get("seed", 42),
        num_labels=config.get("num_labels"),
        semantic_path=semantic_path,
        front_special_path=front_special_path,
    )
    loader = make_loader(dataset, config)
    model, checkpoint = load_model(config, resolve_path(args.checkpoint), device)
    thresholds, threshold_source = threshold_values(
        args,
        checkpoint,
        config,
        int(config.get("num_labels", len(config.get("label_names", [])))),
    )
    stats, evaluated_samples = aggregate_contributions(
        model,
        loader,
        device,
        thresholds,
        int(config["num_labels"]),
    )
    report = build_report(
        config,
        args,
        checkpoint,
        thresholds,
        threshold_source,
        stats,
        evaluated_samples,
    )
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = Path(config["result_dir"]) / "behavior_contribution_summary"
    write_report(output_prefix, report)


if __name__ == "__main__":
    main()
