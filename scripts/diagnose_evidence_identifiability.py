"""Validation-only diagnosis of local evidence identifiability on Main-6.

This route deliberately stops at Phase 1-4.  Contract labels are used only as
weak inherited chunk labels or for the requested influence stratification;
they are never treated as chunk-level ground truth.  Test data is not loaded.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evm_chunk_mil_model import MLM8ViewMultiSlotMIL  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs  # noqa: E402
from train_chunk_mil import load_config as load_mil_config  # noqa: E402
from train_learned_evidence_retrieval import EvidenceRelevanceScorer  # noqa: E402


LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Time manipulation",
]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    raw = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    base = yaml.safe_load(resolve(raw.get("base_config", "configs/retrieval_validation.yaml")).read_text(encoding="utf-8"))
    base.update({key: value for key, value in raw.items() if key != "base_config"})
    return base


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_raw_cache(config, split):
    if split not in ("train", "valid"):
        raise ValueError("evidence identifiability accepts only train and valid")
    path = resolve(config["feature_dir"]) / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].float()
    mask = payload["chunk_mask"].bool()
    labels = payload["multi_labels"].float()
    if features.ndim != 4 or features.shape[2:] != (8, int(config["feature_dim"])):
        raise ValueError(f"{split}: unexpected feature shape {tuple(features.shape)}")
    if mask.shape != features.shape[:2]:
        raise ValueError(f"{split}: chunk mask shape mismatch")
    if labels.shape != (features.shape[0], int(config["num_labels"])):
        raise ValueError(f"{split}: label shape mismatch")
    if (~mask).all(dim=1).any():
        raise ValueError(f"{split}: sample without a valid chunk")
    if not torch.isfinite(features).all():
        raise ValueError(f"{split}: NaN/Inf in features")
    return {
        "ids": [str(value) for value in payload["ids"]],
        "features": features,
        "mask": mask,
        "labels": labels,
    }


def validate_alignment(config, data, split):
    data_dir = resolve(config["data_dir"])
    expected = []
    with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected.append(str(json.loads(line)["id"]))
    if data["ids"] != expected:
        raise ValueError(f"{split}: feature cache IDs do not match JSONL order")


def active_examples(data, max_per_class, seed):
    """Return a deterministic, balanced-enough chunk sample for Phase 1."""
    rng = np.random.default_rng(seed)
    active = data["mask"].reshape(-1).numpy()
    contract_rows = np.repeat(np.arange(len(data["ids"])), data["mask"].shape[1])
    chunk_rows = np.tile(np.arange(data["mask"].shape[1]), len(data["ids"]))
    active_indices = np.flatnonzero(active)
    labels = data["labels"].numpy()[contract_rows[active_indices]]
    selected = []
    for label_id in range(labels.shape[1]):
        positive = active_indices[labels[:, label_id] > 0.5]
        negative = active_indices[labels[:, label_id] <= 0.5]
        take_positive = min(len(positive), int(max_per_class))
        take_negative = min(len(negative), int(max_per_class))
        if take_positive < len(positive):
            positive = rng.choice(positive, take_positive, replace=False)
        if take_negative < len(negative):
            negative = rng.choice(negative, take_negative, replace=False)
        selected.append(np.concatenate([positive, negative]))
    # A union avoids storing six copies of identical features for the MLP.
    indices = np.unique(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
    return indices.astype(np.int64), contract_rows[indices], chunk_rows[indices]


def chunk_matrix(data, flat_indices):
    # View 1 is the historical masked-mean view used by M0's sequence branch.
    return data["features"][:, :, 1, :].reshape(-1, data["features"].shape[-1])[flat_indices]


def safe_auc(y_true, scores):
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else 0.5


def chunk_metrics(y_true, scores):
    predictions = (scores >= 0.5).astype(np.int64)
    return {
        "roc_auc": safe_auc(y_true, scores),
        "average_precision": float(average_precision_score(y_true, scores)) if y_true.sum() else 0.0,
        "f1_at_0.5": float(f1_score(y_true, predictions, zero_division=0)),
        "positive_chunks": int(y_true.sum()),
        "negative_chunks": int((y_true == 0).sum()),
    }


class SmallChunkMLP(nn.Module):
    def __init__(self, feature_dim, hidden_dim, labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, labels),
        )

    def forward(self, x):
        return self.net(x)


def run_chunk_separability(config, train, valid, device):
    train_indices, train_contracts, _ = active_examples(
        train, int(config["chunk_max_train_per_class"]), int(config["seed"]) + 11
    )
    valid_indices, valid_contracts, _ = active_examples(
        valid, int(config["chunk_max_valid_per_class"]), int(config["seed"]) + 12
    )
    x_train = chunk_matrix(train, train_indices)
    x_valid = chunk_matrix(valid, valid_indices)
    y_train = train["labels"][train_contracts]
    y_valid = valid["labels"][valid_contracts]
    rows = []
    logistic_details = {}
    for label_id, label_name in enumerate(config["label_names"]):
        classifier = LogisticRegression(
            max_iter=100,
            class_weight="balanced",
            solver="liblinear",
            random_state=int(config["seed"]),
        )
        classifier.fit(x_train.numpy(), y_train[:, label_id].numpy().astype(int))
        scores = classifier.predict_proba(x_valid.numpy())[:, 1]
        result = chunk_metrics(y_valid[:, label_id].numpy().astype(int), scores)
        result.update({"model": "logistic_regression", "label": label_name})
        rows.append(result)
        logistic_details[label_name] = result

    mlp = SmallChunkMLP(
        int(config["feature_dim"]), int(config["chunk_classifier_hidden_dim"]), int(config["num_labels"])
    ).to(device)
    train_ds = TensorDataset(x_train, y_train)
    loader = DataLoader(train_ds, batch_size=int(config["chunk_classifier_batch_size"]), shuffle=True)
    positive = y_train.sum(0).to(device)
    negative = y_train.shape[0] - positive
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(negative / positive.clamp_min(1)).clamp(1.0, 20.0))
    optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=float(config["chunk_classifier_learning_rate"]),
        weight_decay=float(config["chunk_classifier_weight_decay"]),
    )
    history = []
    best_state = None
    best_loss = float("inf")
    patience = 0
    for epoch in range(1, int(config["chunk_classifier_epochs"]) + 1):
        mlp.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(mlp(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mlp.eval()
        with torch.no_grad():
            valid_logits = []
            valid_loss = []
            for left in range(0, len(x_valid), int(config["chunk_classifier_batch_size"])):
                batch_x = x_valid[left:left + int(config["chunk_classifier_batch_size"])].to(device)
                batch_y = y_valid[left:left + int(config["chunk_classifier_batch_size"])].to(device)
                logits = mlp(batch_x)
                valid_logits.append(logits.cpu())
                valid_loss.append(float(loss_fn(logits, batch_y).item()))
        valid_logits = torch.cat(valid_logits)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "valid_loss": float(np.mean(valid_loss)),
        }
        history.append(record)
        print(
            f"[chunk_mlp] epoch={epoch} train_loss={record['train_loss']:.6f} "
            f"valid_loss={record['valid_loss']:.6f}", flush=True,
        )
        if record["valid_loss"] < best_loss:
            best_loss = record["valid_loss"]
            best_state = {key: value.detach().cpu() for key, value in mlp.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(config["chunk_classifier_patience"]):
            break
    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        mlp_scores = []
        for left in range(0, len(x_valid), int(config["chunk_classifier_batch_size"])):
            mlp_scores.append(torch.sigmoid(mlp(x_valid[left:left + int(config["chunk_classifier_batch_size"])].to(device))).cpu())
    mlp_scores = torch.cat(mlp_scores).numpy()
    for label_id, label_name in enumerate(config["label_names"]):
        result = chunk_metrics(y_valid[:, label_id].numpy().astype(int), mlp_scores[:, label_id])
        result.update({"model": "small_mlp", "label": label_name})
        rows.append(result)
    return {
        "schema": "main6_evidence_identifiability_chunk_separability_v1",
        "weak_label_warning": "Every active chunk inherits its contract label; positive chunks are noisy proxies, not evidence ground truth.",
        "train_chunks_sampled": int(len(x_train)),
        "valid_chunks_evaluated": int(len(x_valid)),
        "models": rows,
        "mlp_history": history,
    }


def load_m0(config, device):
    baseline_config = load_mil_config(resolve(config["baseline_config"]), config.get("baseline_variant", "mlm8_slot3"))
    checkpoint_path = resolve(config["baseline_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = MLM8ViewMultiSlotMIL(baseline_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def infer_m0(model, features, masks, device, batch_size, return_attention=False):
    logits, attentions = [], []
    with torch.no_grad():
        for left in range(0, len(features), int(batch_size)):
            right = min(left + int(batch_size), len(features))
            output = model(
                features[left:right].to(device),
                masks[left:right].to(device),
                return_attention=return_attention,
            )
            logits.append(output["recognition_logits"].float().cpu())
            if return_attention:
                attentions.append(output["chunk_attention"].float().cpu())
    result = {"logits": torch.cat(logits)}
    if return_attention:
        result["attention"] = torch.cat(attentions)
    return result


def build_deletion_batch(data, row, chunk_ids):
    """Create valid M0 inputs for deleting one chunk per batch item."""
    selected = torch.as_tensor(chunk_ids, dtype=torch.long)
    if selected.numel() == 0:
        raise ValueError("chunk_ids must not be empty")
    masks = data["mask"][row:row + 1].expand(len(selected), -1).clone()
    for batch_row, chunk_id in enumerate(selected.tolist()):
        masks[batch_row, chunk_id] = False
    if (~masks).all(dim=1).any():
        raise ValueError("deletion would remove the only valid chunk")
    features = data["features"][row:row + 1].expand(len(selected), -1, -1, -1).contiguous()
    return features, masks


def leave_one_chunk_out(config, data, model, device):
    baseline = infer_m0(
        model, data["features"], data["mask"], device, int(config["influence_batch_size"]), True
    )
    base_probs = torch.sigmoid(baseline["logits"])
    influence = torch.zeros((len(data["ids"]), data["mask"].shape[1], int(config["num_labels"])))
    influence[:] = float("nan")
    single_chunk_contracts = 0
    for row in range(len(data["ids"])):
        chunk_ids = torch.where(data["mask"][row])[0]
        if len(chunk_ids) <= 1:
            # Removing the only real chunk would violate M0's input contract;
            # its influence is undefined rather than zero.
            single_chunk_contracts += 1
            continue
        deleted_probs = []
        for left in range(0, len(chunk_ids), int(config["influence_batch_size"])):
            selected = chunk_ids[left:left + int(config["influence_batch_size"])]
            repeated_features, repeated_masks = build_deletion_batch(data, row, selected)
            output = infer_m0(model, repeated_features, repeated_masks, device, len(selected), False)
            deleted_probs.append(torch.sigmoid(output["logits"]))
        deleted_probs = torch.cat(deleted_probs)
        influence[row, chunk_ids] = base_probs[row].unsqueeze(0) - deleted_probs
        if (row + 1) % int(config["influence_progress_every"]) == 0:
            print(f"[influence] processed_valid_contracts={row + 1}/{len(data['ids'])}", flush=True)
    return {
        "baseline_logits": baseline["logits"],
        "baseline_probs": base_probs,
        "baseline_attention": baseline["attention"],
        "influence": influence,
        "single_chunk_contracts": single_chunk_contracts,
    }


def rank_local(scores, mask, k, largest=True):
    valid = torch.where(mask)[0].tolist()
    if not valid:
        return []
    ordered = sorted(valid, key=lambda idx: float(scores[idx]), reverse=largest)
    return ordered[: min(int(k), len(ordered))]


def spearman(values_a, values_b):
    if len(values_a) < 2:
        return 0.0
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    if np.std(rank_a) == 0 or np.std(rank_b) == 0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def load_retrieval_inputs(config):
    root = resolve("results/retrieval_validation")
    forbidden = [
        root / "test_queries.pt",
        resolve(config["result_dir"]) / "test_queries.pt",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("test retrieval artifacts exist; refusing evidence identifiability diagnosis")
    memory = torch.load(root / "retrieval_memory.pt", map_location="cpu")
    valid_path = root / "valid_queries.pt"
    if not valid_path.exists():
        raise FileNotFoundError(f"run retrieval prepare first: {valid_path}")
    valid = torch.load(valid_path, map_location="cpu")
    if memory.get("train_only") is not True:
        raise RuntimeError("retrieval memory is not train-only")
    if any(key.startswith("test") for key in memory) or any(key.startswith("test") for key in valid):
        raise RuntimeError("test artifacts are not allowed in this diagnosis")
    return memory, valid


def candidate_refs(config, memory, payload, row):
    masks = memory["chunk_mask"].bool()
    candidates = payload["candidate_contracts"][row].tolist()[:int(config["retrieval_candidate_k"])]
    refs = [(int(contract_id), int(chunk_id)) for contract_id in candidates for chunk_id in torch.where(masks[contract_id])[0].tolist()]
    budget = int(config["retrieval_chunk_budget"])
    if int(config["retrieval_balanced_chunks_per_contract"]) > 0:
        m = int(config["retrieval_balanced_chunks_per_contract"])
        balanced = [(int(contract_id), int(chunk_id)) for contract_id in candidates for chunk_id in torch.where(masks[contract_id])[0].tolist()[:m]]
        refs = balanced[:budget]
    return refs


def load_learned_scorer(config, device):
    path = resolve(config.get("learned_scorer_path", "results/retrieval_validation/learned_evidence_v1/evidence_scorer.pt"))
    if not path.exists():
        return None
    scorer = EvidenceRelevanceScorer(
        int(config["feature_dim"]), int(config["num_labels"]),
        int(config.get("scorer_hidden_dim", 128)), float(config.get("scorer_dropout", 0.1)),
    ).to(device)
    scorer.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return scorer.eval()


def retrieval_selections(config, memory, payload, influence_report, device):
    chunks = memory["chunks"].float()
    train_labels = memory["labels"].bool()
    query_contract = payload["query_contract"].float()
    label_prototypes = []
    train_contract = memory["contract"].float()
    for label_id in range(train_labels.shape[1]):
        positive = train_labels[:, label_id]
        proto = train_contract[positive].mean(0) if bool(positive.any()) else train_contract.mean(0)
        label_prototypes.append(torch.nn.functional.normalize(proto, dim=0))
    label_prototypes = torch.stack(label_prototypes)
    learned = load_learned_scorer(config, device)
    rng = random.Random(int(config["seed"]) + 31)
    selections = {"random": {}, "cosine": {}, "learned_scorer": {}, "influence": {}}
    native_scores = {name: {} for name in selections}
    candidate_counts = {}
    for row in range(len(query_contract)):
        refs = candidate_refs(config, memory, payload, row)
        candidate_counts[row] = len(refs)
        if not refs:
            continue
        values = torch.stack([chunks[c, k] for c, k in refs])
        for label_id in range(int(config["num_labels"])):
            key = (row, label_id)
            take_count = min(max(int(k) for k in config["ranking_top_k_values"]), len(refs))
            random_order = list(range(len(refs)))
            rng.shuffle(random_order)
            selections["random"][key] = [refs[index] for index in random_order[:take_count]]
            native_scores["random"][key] = [0.0] * take_count
            cosine_scores = values @ torch.nn.functional.normalize(query_contract[row] + label_prototypes[label_id], dim=0)
            cosine_order = torch.argsort(cosine_scores, descending=True)[:take_count].tolist()
            selections["cosine"][key] = [refs[index] for index in cosine_order]
            native_scores["cosine"][key] = [float(cosine_scores[index]) for index in cosine_order]
            if learned is not None:
                with torch.no_grad():
                    q = query_contract[row].to(device).expand(len(refs), -1)
                    e = values.to(device)
                    label_ids = torch.full((len(refs),), label_id, dtype=torch.long, device=device)
                    learned_scores = learned(q, e, label_ids).cpu()
                learned_order = torch.argsort(learned_scores, descending=True)[:take_count].tolist()
                selections["learned_scorer"][key] = [refs[index] for index in learned_order]
                native_scores["learned_scorer"][key] = [float(learned_scores[index]) for index in learned_order]
            local_influence = influence_report["influence"][row, :, label_id]
            local_mask = torch.isfinite(local_influence)
            local_order = rank_local(local_influence, local_mask, take_count, True)
            selections["influence"][key] = [(row, int(index)) for index in local_order]
            native_scores["influence"][key] = [float(local_influence[index]) for index in local_order]
    return selections, native_scores, candidate_counts, learned is not None


def ranking_comparison(config, memory, payload, data, influence_report, device):
    selections, native_scores, candidate_counts, learned_available = retrieval_selections(config, memory, payload, influence_report, device)
    rows = []
    top_values = [int(value) for value in config["ranking_top_k_values"]]
    labels = payload["labels"].bool()
    influence = influence_report["influence"]
    attention = influence_report["baseline_attention"]
    methods = ["random", "cosine", "influence"] + (["learned_scorer"] if learned_available else [])
    for method in methods:
        for label_id, label_name in enumerate(config["label_names"]):
            for top_k in top_values:
                score_values, purities, margins, correlations = [], [], [], []
                positive_count = 0
                for row in range(len(payload["query_ids"])):
                    if not bool(labels[row, label_id]):
                        continue
                    refs = selections[method].get((row, label_id), [])[:top_k]
                    if not refs:
                        continue
                    positive_count += 1
                    selected_scores = native_scores[method][(row, label_id)][:top_k]
                    score_values.extend(selected_scores)
                    if method == "influence":
                        # This is the query's own positive contract by construction.
                        purities.extend([1.0] * len(refs))
                        local = influence[row, :, label_id]
                        chosen = [chunk for _, chunk in refs]
                        remaining = [float(local[index]) for index in torch.where(torch.isfinite(local))[0].tolist() if index not in chosen]
                        margins.append(float(np.mean([local[index] for index in chosen]) - (np.mean(remaining) if remaining else 0.0)))
                        correlations.append(spearman(
                            [float(local[index]) for index in torch.where(torch.isfinite(local))[0].tolist()],
                            [float(attention[row, index, label_id]) for index in torch.where(torch.isfinite(local))[0].tolist()],
                        ))
                    else:
                        for contract_id, _ in refs:
                            purities.append(float(memory["labels"][contract_id, label_id] > 0.5))
                        margins.append(float(selected_scores[0] - np.mean(selected_scores)))
                rows.append({
                    "method": method,
                    "label": label_name,
                    "top_k": top_k,
                    "positive_query_count": positive_count,
                    "selected_score_mean": float(np.mean(score_values)) if score_values else 0.0,
                    "selected_score_p50": float(np.percentile(score_values, 50)) if score_values else 0.0,
                    "selected_score_p10": float(np.percentile(score_values, 10)) if score_values else 0.0,
                    "selected_score_p90": float(np.percentile(score_values, 90)) if score_values else 0.0,
                    "top1_margin_proxy_mean": float(np.mean(margins)) if margins else 0.0,
                    "ranking_consistency_vs_m0_attention": float(np.mean(correlations)) if correlations else None,
                    "contract_label_purity": float(np.mean(purities)) if purities else 0.0,
                    "purity_note": "Influence purity is trivial because the selected source is the positive query contract; other methods use train-contract labels as a weak proxy, not evidence ground truth.",
                })
    return rows, selections, native_scores


def random_local_indices(data, row, k, rng):
    valid = torch.where(data["mask"][row])[0].tolist()
    rng.shuffle(valid)
    return valid[: min(int(k), len(valid))]


def phase4_sanity(config, data, model, influence_report, device):
    rng = random.Random(int(config["phase4_random_seed"]))
    max_k = max(int(k) for k in config["ranking_top_k_values"])
    random_selection = {row: random_local_indices(data, row, max_k, rng) for row in range(len(data["ids"]))}
    results = []
    baseline_metric = compute_multilabel_metrics_from_probs(
        data["labels"].numpy(),
        influence_report["baseline_probs"].numpy(),
        0.5,
    )
    results.append({
        "method": "full_contract_m0",
        "top_k": "all",
        "threshold": 0.5,
        "macro_f1": float(baseline_metric["recognition_macro_f1"]),
        "micro_f1": float(baseline_metric["recognition_micro_f1"]),
        "per_label_f1": [float(value) for value in baseline_metric["per_label_f1"]],
    })
    for method in ("random", "influence"):
        for top_k in config["ranking_top_k_values"]:
            logits = torch.zeros((len(data["ids"]), int(config["num_labels"])))
            for label_id in range(int(config["num_labels"])):
                # M0 has one shared chunk input but six output labels. To test a
                # label-specific evidence set fairly, each label's logit is
                # evaluated with that label's selected chunk mask and then joined.
                selected_masks = torch.zeros_like(data["mask"])
                for row in range(len(data["ids"])):
                    if method == "influence" and bool(data["labels"][row, label_id]):
                        indices = rank_local(
                            influence_report["influence"][row, :, label_id],
                            torch.isfinite(influence_report["influence"][row, :, label_id]),
                            int(top_k),
                            True,
                        )
                        if not indices:
                            indices = random_selection[row][: min(int(top_k), len(random_selection[row]))]
                    else:
                        indices = random_selection[row][: min(int(top_k), len(random_selection[row]))]
                    if indices:
                        selected_masks[row, indices] = True
                output = infer_m0(
                    model,
                    data["features"],
                    selected_masks,
                    device,
                    int(config["phase4_batch_size"]),
                    False,
                )
                logits[:, label_id] = output["logits"][:, label_id]
            probabilities = torch.sigmoid(logits).numpy()
            metric = compute_multilabel_metrics_from_probs(data["labels"].numpy(), probabilities, 0.5)
            results.append({
                "method": method,
                "top_k": int(top_k),
                "threshold": 0.5,
                "macro_f1": float(metric["recognition_macro_f1"]),
                "micro_f1": float(metric["recognition_micro_f1"]),
                "per_label_f1": [float(value) for value in metric["per_label_f1"]],
            })
            print(f"[phase4] method={method} top_k={top_k} macro_f1={metric['recognition_macro_f1']:.6f}", flush=True)
    return {
        "protocol": "For positive labels, influence selects M0-sensitive chunks; negative labels use the same random fallback. All scores use fixed threshold 0.5 and no validation threshold tuning.",
        "results": results,
        "note": "M0 is evaluated with a separate label-specific mask for each output label, then the six logits are joined. This is a diagnostic evidence-isolation protocol, not a deployable classifier.",
    }


def assess_case(phase1, phase4):
    """Give a conservative post-run case suggestion, never a final claim."""
    by_model = {}
    for row in phase1["models"]:
        by_model.setdefault(row["model"], []).append(float(row["roc_auc"]))
    linear_auc = float(np.mean(by_model.get("logistic_regression", [0.5])))
    mlp_auc = float(np.mean(by_model.get("small_mlp", [0.5])))
    random_rows = {int(row["top_k"]): row for row in phase4["results"] if row["method"] == "random"}
    influence_rows = {int(row["top_k"]): row for row in phase4["results"] if row["method"] == "influence"}
    comparison = {}
    for top_k in sorted(set(random_rows) & set(influence_rows)):
        comparison[str(top_k)] = float(influence_rows[top_k]["macro_f1"] - random_rows[top_k]["macro_f1"])
    max_gain = max(comparison.values()) if comparison else 0.0
    strong_chunk_signal = max(linear_auc, mlp_auc) >= 0.65
    influence_better = max_gain >= 0.01
    if strong_chunk_signal and influence_better:
        case = "A"
        conclusion = "Chunk-level signal and M0 influence evidence are both detectable; weak evidence supervision is the leading bottleneck."
    elif not strong_chunk_signal and not influence_better:
        case = "B"
        conclusion = "Neither weak chunk separability nor influence-isolated evidence is convincing; a single local chunk may be too small as the evidence unit."
    elif strong_chunk_signal and not influence_better:
        case = "C"
        conclusion = "Chunk-level signal exists but influence selection does not improve the controlled check; retrieval/query/ranking or context interaction remains the leading bottleneck."
    else:
        case = "undetermined"
        conclusion = "The two diagnostics disagree; inspect per-label results before designing a new retriever."
    return {
        "case": case,
        "conclusion": conclusion,
        "mean_logistic_regression_auc": linear_auc,
        "mean_small_mlp_auc": mlp_auc,
        "influence_minus_random_macro_f1": comparison,
        "strong_chunk_signal_threshold_auc": 0.65,
        "influence_gain_threshold_macro_f1": 0.01,
        "note": "Case labels are diagnostic heuristics, not evidence ground-truth claims.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evidence_identifiability_diagnosis.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if bool(config.get("allow_test", False)) or bool(config.get("allow_test_cache", False)):
        raise RuntimeError("Evidence identifiability is validation-only; test access is forbidden")
    set_seed(int(config["seed"]))
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    train = load_raw_cache(config, "train")
    valid = load_raw_cache(config, "valid")
    validate_alignment(config, train, "train")
    validate_alignment(config, valid, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} train={len(train['ids'])} valid={len(valid['ids'])}", flush=True)

    phase1 = run_chunk_separability(config, train, valid, device)
    write_json(root / "chunk_separability.json", phase1)
    phase1_rows = phase1["models"]
    write_csv(root / "chunk_separability_per_label.csv", phase1_rows, list(phase1_rows[0]))

    model = load_m0(config, device)
    influence_report = leave_one_chunk_out(config, valid, model, device)
    influence_summary = {
        "schema": "main6_evidence_identifiability_influence_v1",
        "weak_supervision_only": True,
        "definition": "delta_i_l = p_l(C) - p_l(C - e_i), using the frozen historical M0 checkpoint",
        "top_k_values": config["ranking_top_k_values"],
        "single_chunk_contracts_influence_undefined": int(influence_report["single_chunk_contracts"]),
        "per_label": {},
        "score_distribution": {},
    }
    influence_jsonl = root / "influence_chunks.jsonl"
    with influence_jsonl.open("w", encoding="utf-8") as handle:
        for row in range(len(valid["ids"])):
            entry = {"id": valid["ids"][row], "row": row, "labels": {}}
            for label_id, label_name in enumerate(config["label_names"]):
                if not bool(valid["labels"][row, label_id]):
                    continue
                local = influence_report["influence"][row, :, label_id]
                valid_ids = torch.where(torch.isfinite(local))[0].tolist()
                ordered = sorted(valid_ids, key=lambda index: float(local[index]), reverse=True)
                entry["labels"][label_name] = {
                    "top_1": [{"chunk": int(index), "influence": float(local[index])} for index in ordered[:1]],
                    "top_3": [{"chunk": int(index), "influence": float(local[index])} for index in ordered[:3]],
                    "top_5": [{"chunk": int(index), "influence": float(local[index])} for index in ordered[:5]],
                    "top_10": [{"chunk": int(index), "influence": float(local[index])} for index in ordered[:10]],
                    "bottom_10": [{"chunk": int(index), "influence": float(local[index])} for index in sorted(valid_ids, key=lambda index: float(local[index]))[:10]],
                }
            handle.write(json.dumps(entry) + "\n")
    for label_id, label_name in enumerate(config["label_names"]):
        values = influence_report["influence"][:, :, label_id][torch.isfinite(influence_report["influence"][:, :, label_id])].numpy()
        positive_rows = valid["labels"][:, label_id].bool()
        positive_top = []
        top1_values = []
        for row in torch.where(positive_rows)[0].tolist():
            local = influence_report["influence"][row, :, label_id]
            finite = torch.isfinite(local)
            positive_top.extend(local[finite].numpy().tolist())
            if bool(finite.any()):
                top1_values.append(float(local[finite].max()))
        per_contract_top = {}
        for top_k in (1, 3, 5):
            values_top = []
            for row in torch.where(positive_rows)[0].tolist():
                local = influence_report["influence"][row, :, label_id]
                selected = rank_local(local, torch.isfinite(local), top_k, True)
                if selected:
                    values_top.append(float(local[selected].mean()))
            per_contract_top[f"top_{top_k}_mean_influence"] = float(np.mean(values_top)) if values_top else 0.0
        influence_summary["per_label"][label_name] = {
            "positive_contracts": int(positive_rows.sum()),
            "mean_all_valid_chunk_influence": float(np.mean(values)) if len(values) else 0.0,
            "mean_positive_contract_chunk_influence": float(np.mean(positive_top)) if positive_top else 0.0,
            "top1_mean": float(np.mean(top1_values)) if top1_values else 0.0,
            **per_contract_top,
        }
        influence_summary["score_distribution"][label_name] = {
            "p10": float(np.percentile(values, 10)) if len(values) else 0.0,
            "p50": float(np.percentile(values, 50)) if len(values) else 0.0,
            "p90": float(np.percentile(values, 90)) if len(values) else 0.0,
        }
    write_json(root / "influence_summary.json", influence_summary)
    torch.save({"schema": "main6_evidence_identifiability_influence_cache_v1", "ids": valid["ids"], "baseline_probs": influence_report["baseline_probs"], "baseline_attention": influence_report["baseline_attention"], "influence": influence_report["influence"], "test_checked": False}, root / "influence_cache.pt")

    memory, valid_payload = load_retrieval_inputs(config)
    ranking_rows, selections, native_scores = ranking_comparison(config, memory, valid_payload, valid, influence_report, device)
    write_csv(root / "influence_ranking_comparison.csv", ranking_rows, list(ranking_rows[0]))
    write_json(root / "influence_ranking_comparison.json", {"rows": ranking_rows, "test_checked": False, "candidate_contract_k": int(config["retrieval_candidate_k"]), "candidate_count_note": "Retrieval methods use train-only candidates; influence uses the validation query contract itself."})
    phase4 = phase4_sanity(config, valid, model, influence_report, device)
    write_json(root / "influence_sanity_check.json", phase4)

    report = {
        "route": config["route_name"],
        "dataset": "DIVE Main6 random split",
        "seed": int(config["seed"]),
        "train_only_memory": True,
        "test_checked": False,
        "phases_completed": [1, 2, 3, 4],
        "phase5_started": False,
        "case_assessment": assess_case(phase1, phase4),
        "oracle_upper_bound_reference": 0.963081,
        "m0_reference": 0.797254,
        "artifacts": {
            "chunk_separability": "chunk_separability.json",
            "influence": "influence_summary.json",
            "influence_chunks": "influence_chunks.jsonl",
            "ranking": "influence_ranking_comparison.csv",
            "sanity_check": "influence_sanity_check.json",
        },
        "interpretation_rules": {
            "A": "chunk classifier strong and influence evidence beats random: evidence exists, supervision is the bottleneck",
            "B": "chunk classifier weak and influence does not beat random: single local chunk may be too simple; consider multi-chunk spans",
            "C": "chunk signal and influence signal exist but retriever fails: query/ranking objective is the bottleneck",
        },
        "warnings": [
            "Positive chunk labels are inherited contract labels and are not evidence ground truth.",
            "Oracle Evidence remains an upper bound only and is not included as a final model.",
            "No CFG, DFG, Stack, LLM, LoRA, new backbone, or test artifact is used by this route.",
        ],
    }
    write_json(root / "evidence_identifiability_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
