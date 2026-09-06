"""Validation-only diagnostics for three-view standalone CRER.

No training and no test data are used.  Routing and evidence ablations are
diagnostic interventions on a frozen checkpoint, not evidence ground truth.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crer_classifier import CRERClassifier  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs, select_per_label_thresholds  # noqa: E402
from train_crer import load_config, load_split, pos_weight  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    p = int(y.sum())
    n = len(y) - p
    if not p or not n:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def ap(y, score):
    y = np.asarray(y).astype(int)
    p = int(y.sum())
    if not p:
        return None
    ranked = y[np.argsort(-np.asarray(score), kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(y) + 1)
    return float((precision * ranked).sum() / p)


def metrics(labels, logits, thresholds):
    result = compute_multilabel_metrics_from_probs(labels, torch.sigmoid(logits).numpy(), thresholds)
    return {
        "macro_f1": float(result["recognition_macro_f1"]),
        "micro_f1": float(result["recognition_micro_f1"]),
        "per_label_f1": [float(x) for x in result["per_label_f1"]],
        "per_label_precision": [float(x) for x in result["per_label_precision"]],
        "per_label_recall": [float(x) for x in result["per_label_recall"]],
    }


def build_model(config):
    return CRERClassifier(
        feature_dim=int(config["feature_dim"]),
        num_labels=int(config["num_labels"]),
        max_chunks=int(config["max_chunks"]),
        chunk_view_index=int(config.get("chunk_view_index", 1)),
        view_indices=config.get("view_indices", [config.get("chunk_view_index", 1)]),
        hidden_dim=int(config["hidden_dim"]),
        num_heads=int(config["num_heads"]),
        shared_encoder_layers=int(config["shared_encoder_layers"]),
        evidence_blocks=int(config["evidence_blocks"]),
        dropout=0.0,
        temperature=float(config["temperature"]),
        use_label_queries=True,
        sparsity_target=float(config["sparsity_target"]),
    )


def load_model(config, checkpoint):
    model = build_model(config)
    payload = torch.load(resolve(checkpoint), map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def collect(model, data, device, batch_size, routing_override=None):
    loader = DataLoader(TensorDataset(data["features"], data["mask"]), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for features, mask in loader:
            outputs.append(model(features.to(device), mask.to(device), return_diagnostics=True, routing_override=routing_override))
    keys = ("recognition_logits", "evidence_representation", "chunk_representation", "routing_weights")
    return {key: torch.cat([item[key].float().cpu() for item in outputs]) for key in keys}


def evidence_ablation(model, data, labels, device, batch_size, thresholds):
    loader = DataLoader(TensorDataset(data["features"], data["mask"]), batch_size=batch_size, shuffle=False)
    real_logits, removed_logits = [], []
    model.eval()
    with torch.no_grad():
        for features, mask in loader:
            features, mask = features.to(device), mask.to(device)
            real = model(features, mask, return_diagnostics=True)
            override = real["evidence_representation"].clone()
            current = []
            for label_id in range(override.shape[1]):
                altered = override.clone()
                altered[:, label_id] = 0.0
                current.append(model(features, mask, evidence_override=altered)["recognition_logits"].float().cpu())
            real_logits.append(real["recognition_logits"].float().cpu())
            removed_logits.append(torch.stack(current, dim=1))
    real = torch.cat(real_logits)
    removed = torch.cat(removed_logits)
    rows = []
    for label_id, name in enumerate(["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]):
        per_sample = []
        for idx in range(removed.shape[0]):
            per_sample.append(float(torch.sigmoid(real[idx, label_id]) - torch.sigmoid(removed[idx, label_id, label_id])))
        rows.append({"label": name, "mean_probability_drop": float(np.mean(per_sample)), "std_probability_drop": float(np.std(per_sample))})
    return rows


def gradient_directions(model, data, device, batch_size, weights):
    model.eval()
    features = data["features"][: min(batch_size * 2, len(data["labels"]))].to(device)
    mask = data["mask"][: len(features)].to(device)
    labels = data["labels"][: len(features)].to(device)
    groups = {
        "label_embedding": lambda n: n.startswith("label_embedding"),
        "query_projection": lambda n: n.startswith("query_projection") or ".query_projection" in n,
        "key_projection": lambda n: n.startswith("gate_key_projection") or ".key_projection" in n,
        "value_projection": lambda n: n.startswith("gate_value_projection") or ".value_projection" in n,
        "feature_encoder": lambda n: n.startswith("encoder."),
    }
    vectors = {group: [] for group in groups}
    for label_id in range(labels.shape[1]):
        model.zero_grad(set_to_none=True)
        output = model(features, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output["recognition_logits"][:, label_id], labels[:, label_id], pos_weight=weights[label_id])
        loss.backward()
        for group, predicate in groups.items():
            values = [parameter.grad.detach().float().flatten().cpu() for name, parameter in model.named_parameters() if predicate(name) and parameter.grad is not None]
            vectors[group].append(torch.cat(values) if values else torch.zeros(1))
    result = {}
    for group, values in vectors.items():
        matrix = torch.stack([value / value.norm().clamp_min(1e-12) for value in values])
        result[group] = (matrix @ matrix.T).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_crer_three_view.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/crer_main6_three_view/diagnosis.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("allow_test"):
        raise ValueError("test is locked")
    train, valid = load_split(config, "train"), load_split(config, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint).to(device)
    labels = valid["labels"].numpy().astype(int)
    learned = collect(model, valid, device, args.batch_size)
    learned_probs = torch.sigmoid(learned["recognition_logits"]).numpy()
    threshold_report = select_per_label_thresholds(labels, learned_probs, config["thresholds"], config["label_names"], global_threshold=0.2)
    thresholds = threshold_report["thresholds"]
    routing = {}
    for name, override in (("learned", None), ("uniform", "uniform"), ("zero", "zero")):
        output = learned if name == "learned" else collect(model, valid, device, args.batch_size, override)
        routing[name] = metrics(labels, output["recognition_logits"], thresholds)

    h = learned["chunk_representation"]
    mask = valid["mask"]
    global_mean = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    z = learned["evidence_representation"]
    cosine = torch.nn.functional.cosine_similarity(z, global_mean.unsqueeze(1), dim=-1)
    label_cos = torch.nn.functional.cosine_similarity(z.unsqueeze(2), z.unsqueeze(1), dim=-1)
    card = valid["labels"].sum(dim=1).numpy()
    off_diag = (~torch.eye(label_cos.shape[1], dtype=torch.bool)).unsqueeze(0).expand(label_cos.shape[0], -1, -1)
    similarity = {
        "all_mean": float(label_cos[off_diag].mean()),
        "matrix_all": label_cos.mean(dim=0).tolist(),
        "single_label_contract_mean": float(label_cos[card == 1][:, ~torch.eye(6, dtype=torch.bool)].mean()) if (card == 1).any() else None,
        "single_label_matrix": label_cos[card == 1].mean(dim=0).tolist() if (card == 1).any() else None,
        "multi_label_contract_mean": float(label_cos[card >= 2][:, ~torch.eye(6, dtype=torch.bool)].mean()) if (card >= 2).any() else None,
        "multi_label_matrix": label_cos[card >= 2].mean(dim=0).tolist() if (card >= 2).any() else None,
        "evidence_to_global_mean_cosine_by_label": cosine.mean(dim=0).tolist(),
    }
    score_rows = []
    for idx, name in enumerate(config["label_names"]):
        score_rows.append({"label": name, "roc_auc": auc(labels[:, idx], learned_probs[:, idx]), "average_precision": ap(labels[:, idx], learned_probs[:, idx]), "positive_mean_probability": float(learned_probs[labels[:, idx] == 1, idx].mean()), "negative_mean_probability": float(learned_probs[labels[:, idx] == 0, idx].mean())})
    report = {
        "route": "DIVE Main6 CRER three-view diagnosis",
        "dataset": "DIVE Main6 random split",
        "view_indices": config.get("view_indices", [1]),
        "view_fusion": "elementwise_mean_preserving_768d",
        "checkpoint": str(resolve(args.checkpoint)),
        "validation_only": True,
        "test_checked": False,
        "routing_ablation": routing,
        "evidence_removal": evidence_ablation(model, valid, labels, device, args.batch_size, thresholds),
        "global_similarity": similarity,
        "gradient_direction_cosine": gradient_directions(model, train, device, args.batch_size, pos_weight(train["labels"], config).to(device)),
        "dos_score_diagnostics": score_rows[4],
        "score_diagnostics": score_rows,
        "thresholds_from_learned_validation": thresholds,
        "interpretation": {"warning": "All routing, removal, similarity, and gradient values are diagnostics; they are not local evidence ground truth."},
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "view_indices": report["view_indices"], "learned_macro_f1": routing["learned"]["macro_f1"], "uniform_macro_f1": routing["uniform"]["macro_f1"], "zero_macro_f1": routing["zero"]["macro_f1"], "test_checked": False}, indent=2))


if __name__ == "__main__":
    main()
