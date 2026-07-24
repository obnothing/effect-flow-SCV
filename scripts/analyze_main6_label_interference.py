"""Validation-only diagnostics for label-conditioned ETP attention."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chunk_feature_dataset import ChunkFeatureDataset
from evaluate_chunk_mil import load_config, load_model, make_loader
from metrics import compute_multilabel_metrics_from_probs, sigmoid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--faithfulness-per-label", type=int, default=32)
    return parser.parse_args()


def js_divergence(left, right):
    left = left / max(1e-12, left.sum())
    right = right / max(1e-12, right.sum())
    midpoint = 0.5 * (left + right)
    return float(0.5 * ((left * np.log((left + 1e-12) / (midpoint + 1e-12))).sum() + (right * np.log((right + 1e-12) / (midpoint + 1e-12))).sum()))


def main():
    args = parse_args()
    config = load_config(args.config, args.variant)
    if config.get("model_type") != "ld_etp_cross_attention_mil":
        raise ValueError("Interference analysis requires an LD-ETPCA candidate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = Path(config["feature_dir"]) / "valid.pt"
    tokens = Path(config["token_semantic_dir"]) / "valid.pt"
    dataset = ChunkFeatureDataset(feature, num_labels=config["num_labels"], source_label_names=config["source_label_names"], label_names=config["label_names"], token_semantic_path=tokens)
    loader = make_loader(dataset, config)
    model, _ = load_model(config, args.checkpoint, device)
    ids = []
    labels = []
    probabilities = []
    profiles = []
    overlaps_discordant = []
    overlaps_copositive = []
    for batch in loader:
        inputs = {
            "chunk_features": batch["chunk_features"].to(device),
            "chunk_mask": batch["chunk_mask"].to(device),
            "etp_top2_ids": batch["etp_top2_ids"].to(device),
            "etp_top2_confidence": batch["etp_top2_confidence"].to(device),
            "return_attention": True,
        }
        with torch.no_grad():
            outputs = model(**inputs)
        profile = outputs["label_effect_profile"].cpu().numpy()
        profile = profile / np.maximum(profile.sum(axis=-1, keepdims=True), 1e-12)
        y = batch["multi_labels"].numpy().astype(int)
        for item, row, item_profile in zip(batch["id"], y, profile):
            ids.append(str(item))
            labels.append(row)
            profiles.append(item_profile)
            for left in range(row.size):
                for right in range(left + 1, row.size):
                    similarity = float(np.dot(item_profile[left], item_profile[right]) / max(1e-12, np.linalg.norm(item_profile[left]) * np.linalg.norm(item_profile[right])))
                    (overlaps_copositive if row[left] == row[right] == 1 else overlaps_discordant if row[left] != row[right] else []).append(similarity)
        probabilities.append(torch.sigmoid(outputs["recognition_logits"]).cpu().numpy())
    labels = np.asarray(labels)
    probabilities = np.concatenate(probabilities)
    profiles = np.asarray(profiles)
    groups = {}
    for name, mask in {
        "one": labels.sum(axis=1) == 1,
        "two": labels.sum(axis=1) == 2,
        "three": labels.sum(axis=1) == 3,
        "four_plus": labels.sum(axis=1) >= 4,
    }.items():
        groups[name] = compute_multilabel_metrics_from_probs(labels[mask], probabilities[mask], 0.5) if mask.any() else None
    rng = np.random.default_rng(42)
    role_lifts = []
    role_positive = []
    role_negative = []
    for label_id, name in enumerate(config["label_names"]):
        positive = profiles[labels[:, label_id] == 1, label_id]
        negative = profiles[labels[:, label_id] == 0, label_id]
        pos_mean = positive.mean(axis=0)
        neg_mean = negative.mean(axis=0)
        samples = []
        for _ in range(args.bootstrap):
            samples.append((positive[rng.integers(len(positive), size=len(positive))].mean(axis=0) - negative[rng.integers(len(negative), size=len(negative))].mean(axis=0)))
        ci = np.quantile(np.asarray(samples), [0.025, 0.975], axis=0)
        role_positive.append(pos_mean.tolist())
        role_negative.append(neg_mean.tolist())
        role_lifts.append({"label": name, "lift": (pos_mean - neg_mean).tolist(), "ci_low": ci[0].tolist(), "ci_high": ci[1].tolist(), "has_positive_lift": bool(np.max(ci[0]) > 0)})
    pair_js = []
    label_profiles = np.asarray(role_positive)
    for left in range(len(config["label_names"])):
        for right in range(left + 1, len(config["label_names"])):
            pair_js.append({"left": config["label_names"][left], "right": config["label_names"][right], "js_divergence": js_divergence(label_profiles[left], label_profiles[right])})
    predicted = (probabilities >= 0.5).astype(int)
    conditional_fpr = []
    for left, name in enumerate(config["label_names"]):
        for right, other in enumerate(config["label_names"]):
            if left == right:
                continue
            mask = (labels[:, left] == 0) & (labels[:, right] == 1)
            if mask.any():
                conditional_fpr.append({"negative_label": name, "positive_confounder": other, "count": int(mask.sum()), "fpr": float(predicted[mask, left].mean())})
    faithfulness = []
    for label_id, name in enumerate(config["label_names"]):
        positive_indices = np.flatnonzero(labels[:, label_id] == 1)
        chosen = rng.choice(positive_indices, size=min(len(positive_indices), args.faithfulness_per_label), replace=False)
        top_drops = []
        random_drops = []
        for index in chosen:
            item = dataset[int(index)]
            base_inputs = {
                "chunk_features": item["chunk_features"].unsqueeze(0).to(device),
                "chunk_mask": item["chunk_mask"].unsqueeze(0).to(device),
                "etp_top2_ids": item["etp_top2_ids"].unsqueeze(0).to(device),
                "etp_top2_confidence": item["etp_top2_confidence"].unsqueeze(0).to(device),
                "return_attention": True,
            }
            with torch.no_grad():
                base = model(**base_inputs)
            original = float(torch.sigmoid(base["recognition_logits"])[0, label_id])
            token_weights = (
                base["token_attention"][0, :, label_id]
                * base["slot_weights"][0, :, label_id].unsqueeze(-1)
                * base["chunk_attention"][0, :, label_id].view(-1, 1, 1)
            ).sum(dim=1)
            valid = base_inputs["etp_top2_ids"][0].ne(255).any(dim=-1)
            valid_positions = valid.flatten().nonzero(as_tuple=False).flatten()
            count = max(1, int(np.ceil(valid_positions.numel() * 0.1)))
            ranked = token_weights.flatten().masked_fill(~valid.flatten(), -1.0).topk(count).indices
            random_order = torch.randperm(
                valid_positions.numel(),
                generator=torch.Generator().manual_seed(42 + int(index) + label_id),
            )[:count].to(valid_positions.device)
            random_positions = valid_positions[random_order]
            def masked_probability(positions):
                changed_ids = base_inputs["etp_top2_ids"].clone()
                changed_conf = base_inputs["etp_top2_confidence"].clone()
                chunk_ids = torch.div(positions, valid.shape[1], rounding_mode="floor")
                token_ids = positions % valid.shape[1]
                changed_ids[0, chunk_ids, token_ids] = 255
                changed_conf[0, chunk_ids, token_ids] = 0
                with torch.no_grad():
                    value = model(
                        chunk_features=base_inputs["chunk_features"],
                        chunk_mask=base_inputs["chunk_mask"],
                        etp_top2_ids=changed_ids,
                        etp_top2_confidence=changed_conf,
                    )["recognition_logits"]
                return float(torch.sigmoid(value)[0, label_id])
            top_drops.append(original - masked_probability(ranked))
            random_drops.append(original - masked_probability(random_positions))
        faithfulness.append({"label": name, "samples": int(len(chosen)), "top_mask_probability_drop": float(np.mean(top_drops)), "random_mask_probability_drop": float(np.mean(random_drops)), "passes": float(np.mean(top_drops)) > float(np.mean(random_drops))})
    report = {
        "split": "valid",
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "ids": ids,
        "cardinality_metrics": groups,
        "conditional_false_positive_rate": conditional_fpr,
        "positive_effect_profiles": role_positive,
        "negative_effect_profiles": role_negative,
        "attention_lift": role_lifts,
        "pairwise_js_divergence": pair_js,
        "discordant_attention_overlap": float(np.mean(overlaps_discordant)) if overlaps_discordant else None,
        "copositive_attention_overlap": float(np.mean(overlaps_copositive)) if overlaps_copositive else None,
        "attention_lift_pass_count": sum(item["has_positive_lift"] for item in role_lifts),
        "interpretability_pass": sum(item["has_positive_lift"] for item in role_lifts) >= 4,
        "faithfulness": faithfulness,
        "faithfulness_pass": bool(faithfulness) and float(np.mean([item["top_mask_probability_drop"] for item in faithfulness])) > float(np.mean([item["random_mask_probability_drop"] for item in faithfulness])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
