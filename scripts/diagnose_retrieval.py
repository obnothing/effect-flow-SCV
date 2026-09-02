"""Systematic train/valid diagnosis for the Main-6 evidence-retrieval MVP."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_retrieval_validation import (  # noqa: E402
    M2,
    build_retrieval_indices,
    gather_features,
    load_cache,
    load_prepared,
    metrics,
    select_thresholds,
)
from train_learned_evidence_retrieval import (  # noqa: E402
    EvidenceRelevanceScorer,
    build_pairs,
    rerank,
    train_scorer,
)


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


def load_route_data(config):
    if bool(config.get("allow_test", False)) or bool(config.get("allow_test_cache", False)):
        raise RuntimeError("Evidence diagnosis is validation-only; test access is forbidden")
    base_root = resolve("results/retrieval_validation")
    forbidden = [base_root / "test_queries.pt", base_root / "diagnosis" / "test_queries.pt"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("test retrieval artifacts exist; refusing validation diagnosis")
    memory = torch.load(base_root / "retrieval_memory.pt", map_location="cpu")
    if memory.get("train_only") is not True:
        raise RuntimeError("retrieval memory is not marked train_only")
    train = load_prepared({**config, "result_dir": "results/retrieval_validation"}, "train")
    valid = load_prepared({**config, "result_dir": "results/retrieval_validation"}, "valid")
    if "candidate_contracts" not in train or "candidate_contracts" not in valid:
        raise RuntimeError("retrieval query cache lacks candidate_contracts; rerun run_retrieval_validation.sh prepare")
    if any(key.startswith("test") for key in memory) or any(key.startswith("test") for key in train) or any(key.startswith("test") for key in valid):
        raise RuntimeError("test artifacts are not allowed in evidence diagnosis inputs")
    return memory, train, valid


def contract_candidates(memory, query_payload, candidate_k):
    train_contract = memory["contract"].float()
    query_contract = query_payload["query_contract"].float()
    scores = query_contract @ train_contract.T
    train_ids = memory["ids"]
    own = {sample_id: index for index, sample_id in enumerate(train_ids)}
    for row, sample_id in enumerate(query_payload["query_ids"]):
        if sample_id in own:
            scores[row, own[sample_id]] = -2.0
    count = min(int(candidate_k), train_contract.shape[0])
    return torch.topk(scores, k=count, dim=1).indices


def build_scope_indices(config, memory, query_payload, candidate_k, mode="prototype", top_k=5, budget=512, balanced_m=8):
    """Build deterministic candidate scopes without reading holdout labels."""
    chunks = memory["chunks"].float()
    masks = memory["chunk_mask"].bool()
    train_labels = memory["labels"].float()
    queries = query_payload["query_contract"].float()
    candidates = contract_candidates(memory, query_payload, candidate_k)
    label_prototypes = []
    for label_id in range(train_labels.shape[1]):
        positive = train_labels[:, label_id] > 0.5
        value = memory["contract"].float()[positive].mean(0) if bool(positive.any()) else memory["contract"].float().mean(0)
        label_prototypes.append(torch.nn.functional.normalize(value, dim=0))
    label_prototypes = torch.stack(label_prototypes)
    output = torch.full((len(queries), train_labels.shape[1], top_k, 2), -1, dtype=torch.long)
    output_scores = torch.zeros((len(queries), train_labels.shape[1], top_k))
    rng = random.Random(int(config.get("seed", 42)))
    for row in range(len(queries)):
        candidate_ids = candidates[row].tolist()
        refs = [(contract_id, chunk_id) for contract_id in candidate_ids for chunk_id in torch.where(masks[contract_id])[0].tolist()]
        if mode in ("flatten_512", "prototype", "query_cosine"):
            refs = refs[:budget]
        elif mode == "balanced_512":
            refs = [(contract_id, chunk_id) for contract_id in candidate_ids for chunk_id in torch.where(masks[contract_id])[0].tolist()[:balanced_m]]
            refs = refs[:budget]
        elif mode == "all":
            pass
        elif mode in ("random", "random_topk"):
            rng.shuffle(refs)
            refs = refs[:budget]
        if not refs:
            continue
        values = torch.stack([chunks[contract_id, chunk_id] for contract_id, chunk_id in refs])
        for label_id in range(train_labels.shape[1]):
            if mode == "query_cosine":
                score_query = queries[row]
            else:
                score_query = torch.nn.functional.normalize(queries[row] + label_prototypes[label_id], dim=0)
            scores = values @ score_query
            take = min(top_k, len(refs))
            selected_values, positions = torch.topk(scores, k=take)
            if mode == "random_topk":
                positions = torch.tensor(rng.sample(range(len(refs)), take), dtype=torch.long)
                selected_values = scores[positions]
            for slot, (position, value) in enumerate(zip(positions.tolist(), selected_values.tolist())):
                output[row, label_id, slot] = torch.tensor(refs[position])
                output_scores[row, label_id, slot] = value
    return output, output_scores, candidates


def oracle_indices(config, memory, payload, top_k=5):
    """Upper bound: use the query's true label to expose positive evidence only.

    Negative labels intentionally receive no oracle evidence. This is a diagnostic
    upper bound, never a legal retrieval procedure for a final model.
    """
    chunks = memory["chunks"].float()
    masks = memory["chunk_mask"].bool()
    source_labels = memory["labels"].bool()
    query_contract = payload["query_contract"].float()
    query_labels = payload["labels"].bool()
    output = torch.full((len(query_contract), source_labels.shape[1], top_k, 2), -1, dtype=torch.long)
    for row in range(len(query_contract)):
        contract_scores = query_contract[row] @ memory["contract"].float().T
        own_id = payload["query_ids"][row]
        own_index = memory["ids"].index(own_id) if own_id in memory["ids"] else None
        for label_id in range(source_labels.shape[1]):
            if not bool(query_labels[row, label_id]):
                continue
            wanted = source_labels[:, label_id].clone()
            if own_index is not None:
                wanted[own_index] = False
            candidate_scores = contract_scores.masked_fill(~wanted, -2.0)
            count = min(int(config.get("oracle_candidate_contracts", 256)), int(wanted.sum().item()))
            if count == 0:
                continue
            candidate_ids = torch.topk(candidate_scores, k=count).indices
            refs = [(contract_id, chunk_id) for contract_id in candidate_ids.tolist() for chunk_id in torch.where(masks[contract_id])[0].tolist()]
            if not refs:
                continue
            values = torch.stack([chunks[contract_id, chunk_id] for contract_id, chunk_id in refs])
            scores = values @ query_contract[row]
            take = min(top_k, len(refs))
            selected, positions = torch.topk(scores, k=take)
            for slot, (position, value) in enumerate(zip(positions.tolist(), selected.tolist())):
                output[row, label_id, slot] = torch.tensor(refs[position])
    return output


def train_residual(config, name, memory, train_payload, valid_payload, train_indices, valid_indices, device):
    set_seed(int(config.get("seed", 42)))
    train_payload = dict(train_payload)
    valid_payload = dict(valid_payload)
    train_payload["evidence_indices"] = train_indices
    valid_payload["evidence_indices"] = valid_indices
    train_q, train_contract, train_evidence = gather_features(memory, train_payload, device)
    valid_q, valid_contract, valid_evidence = gather_features(memory, valid_payload, device)
    train_base = train_payload["base_logits"].float().to(device)
    valid_base = valid_payload["base_logits"].float().to(device)
    train_labels = train_payload["labels"].float().to(device)
    valid_labels = valid_payload["labels"].float().to(device)
    model = M2(int(config["feature_dim"]), int(config.get("diagnosis_hidden_dim", 256)), int(config["num_labels"]), float(config.get("diagnosis_dropout", 0.1))).to(device)
    positive = train_labels.sum(0)
    weight = torch.sqrt((len(train_labels) - positive) / positive.clamp_min(1)).clamp(1.0, 5.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("diagnosis_learning_rate", 1e-4)), weight_decay=0.01)
    loader = DataLoader(TensorDataset(train_q, train_contract, train_evidence, train_base, train_labels), batch_size=int(config.get("diagnosis_batch_size", 128)), shuffle=True)
    best = None
    patience = 0
    history = []
    for epoch in range(1, int(config.get("diagnosis_epochs", 5)) + 1):
        model.train()
        losses = []
        for query, contract_ev, evidence, base, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(query, contract_evidence=contract_ev, evidence=evidence, base_logits=base)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_q, contract_evidence=valid_contract, evidence=valid_evidence, base_logits=valid_base)
        valid_metrics, valid_probs = metrics(valid_labels, valid_logits)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": float(loss_fn(valid_logits, valid_labels).item()), "fixed_macro_f1": valid_metrics["recognition_macro_f1"], "fixed_micro_f1": valid_metrics["recognition_micro_f1"]}
        history.append(row)
        print(f"[{name}] epoch={epoch} train_loss={row['train_loss']:.6f} valid_loss={row['valid_loss']:.6f} fixed_macro={row['fixed_macro_f1']:.6f}", flush=True)
        if best is None or row["fixed_macro_f1"] > best["score"]:
            best = {"score": row["fixed_macro_f1"], "epoch": epoch, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "probs": valid_probs}
            patience = 0
        else:
            patience += 1
        if patience >= int(config.get("diagnosis_patience", 2)):
            break
    thresholds = select_thresholds(valid_labels.cpu().numpy().astype(int), best["probs"], config["thresholds"])
    tuned = metrics(valid_labels, torch.as_tensor(np.log(best["probs"] / np.clip(1 - best["probs"], 1e-7, 1.0))), thresholds)[0]
    return {"variant": name, "best_epoch": best["epoch"], "fixed": metrics(valid_labels, torch.as_tensor(np.log(best["probs"] / np.clip(1 - best["probs"], 1e-7, 1.0))), None)[0], "tuned": tuned, "thresholds": thresholds, "history": history, "valid_probs": best["probs"].tolist()}


def purity_rows(config, memory, payload, index_sets, retrieval_names):
    source_labels = memory["labels"].numpy().astype(int)
    query_labels = payload["labels"].numpy().astype(int)
    rows = []
    for name, indices in zip(retrieval_names, index_sets):
        for k in config["ranking_top_k_values"]:
            take = indices[:, :, : min(int(k), indices.shape[2]), 0]
            for label_id, label_name in enumerate(config["label_names"]):
                values = []
                positive_values = []
                for row in range(len(query_labels)):
                    refs = take[row, label_id]
                    refs = refs[refs >= 0].numpy()
                    value = float(source_labels[refs, label_id].mean()) if len(refs) else 0.0
                    values.append(value)
                    if query_labels[row, label_id] == 1:
                        positive_values.append(value)
                rows.append({"retrieval": name, "top_k": int(k), "label": label_name, "label_purity": float(np.mean(values)), "positive_query_label_purity": float(np.mean(positive_values)) if positive_values else 0.0, "proxy_note": "source contract label purity; not evidence ground truth"})
    return rows


def scope_refs(memory, payload, row, candidate_k, mode, budget, balanced_m, rng, candidates=None):
    masks = memory["chunk_mask"].bool()
    if candidates is None:
        cached = payload.get("candidate_contracts")
        if cached is not None and cached.shape[1] >= candidate_k:
            candidates = cached[:, :candidate_k]
        else:
            candidates = contract_candidates(memory, payload, candidate_k)
    candidate_ids = candidates[row].tolist()
    refs = [(contract_id, chunk_id) for contract_id in candidate_ids for chunk_id in torch.where(masks[contract_id])[0].tolist()]
    if mode in ("flatten_512", "prototype", "query_cosine"):
        return refs[:budget]
    if mode == "balanced_512":
        refs = [(contract_id, chunk_id) for contract_id in candidate_ids for chunk_id in torch.where(masks[contract_id])[0].tolist()[:balanced_m]]
        return refs[:budget]
    if mode in ("random", "random_topk"):
        rng.shuffle(refs)
        return refs[:budget]
    if mode == "all":
        return refs
    raise ValueError(f"unknown retrieval scope mode: {mode}")


def candidate_statistics(config, memory, payload):
    source_labels = memory["labels"].numpy().astype(int)
    query_labels = payload["labels"].numpy().astype(int)
    rows = []
    for candidate_k in config["candidate_k_values"]:
        candidates = contract_candidates(memory, payload, int(candidate_k))
        candidate_array = candidates.numpy()
        for label_id, label_name in enumerate(config["label_names"]):
            source_positive = source_labels[candidate_array, label_id]
            query_positive = query_labels[:, label_id] == 1
            rows.append({
                "candidate_contracts": int(candidate_k),
                "label": label_name,
                "query_positive_count": int(query_positive.sum()),
                "candidate_positive_recall": float((source_positive[query_positive].sum(1) > 0).mean()) if query_positive.any() else 0.0,
                "mean_positive_candidates": float(source_positive[query_positive].sum(1).mean()) if query_positive.any() else 0.0,
                "mean_candidate_label_purity": float(source_positive.mean()),
                "proxy_note": "contract-level label coverage; not local evidence ground truth",
            })
    return rows


def ranked_references(config, memory, payload, method, scorer, device, candidate_k=64, budget=512):
    """Return complete ranked references within a fixed candidate scope.

    The ranking is label-conditioned for the prototype and learned methods;
    labels are used only later for validation statistics, never for ranking.
    """
    chunks = memory["chunks"].float()
    source_labels = memory["labels"].float()
    queries = payload["query_contract"].float()
    masks = memory["chunk_mask"].bool()
    candidates = contract_candidates(memory, payload, candidate_k)
    prototypes = []
    for label_id in range(source_labels.shape[1]):
        positive = source_labels[:, label_id] > 0.5
        proto = memory["contract"].float()[positive].mean(0) if bool(positive.any()) else memory["contract"].float().mean(0)
        prototypes.append(torch.nn.functional.normalize(proto, dim=0))
    prototypes = torch.stack(prototypes)
    stable_method_seed = sum((index + 1) * ord(char) for index, char in enumerate(method))
    rng = random.Random(int(config.get("seed", 42)) + stable_method_seed)
    ranked = []
    for row in range(len(queries)):
        refs = scope_refs(memory, payload, row, candidate_k, "flatten_512", budget, 8, rng, candidates)
        if method == "random":
            rng.shuffle(refs)
            ranked.append([list(refs) for _ in range(source_labels.shape[1])])
            continue
        values = torch.stack([chunks[c, k] for c, k in refs]) if refs else torch.empty((0, chunks.shape[-1]))
        row_ranked = []
        for label_id in range(source_labels.shape[1]):
            if method == "cosine":
                rank_query = queries[row]
                scores = values @ rank_query
            elif method == "fixed_prototype":
                rank_query = torch.nn.functional.normalize(queries[row] + prototypes[label_id], dim=0)
                scores = values @ rank_query
            elif method == "learned_scorer":
                if not refs:
                    row_ranked.append([])
                    continue
                with torch.no_grad():
                    q = queries[row].to(device).expand(len(refs), -1)
                    e = values.to(device)
                    labels = torch.full((len(refs),), label_id, dtype=torch.long, device=device)
                    scores = scorer(q, e, labels).float().cpu()
            else:
                raise ValueError(f"unknown ranking method: {method}")
            order = torch.argsort(scores, descending=True).tolist()
            row_ranked.append([refs[index] for index in order])
        ranked.append(row_ranked)
    return ranked


def ranking_metric_rows(config, memory, payload, ranked_by_method):
    source_labels = memory["labels"].numpy().astype(int)
    query_labels = payload["labels"].numpy().astype(int)
    rows = []
    top_values = [int(value) for value in config["ranking_top_k_values"]]
    for method, ranked in ranked_by_method.items():
        for label_id, label_name in enumerate(config["label_names"]):
            auc_values, reciprocal_values = [], []
            recalls = {k: [] for k in top_values}
            purities = {k: [] for k in top_values}
            positive_count = int(query_labels[:, label_id].sum())
            for row, row_labels in enumerate(query_labels):
                if row_labels[label_id] != 1:
                    continue
                refs = ranked[row][label_id]
                relevance = np.asarray([source_labels[c, label_id] for c, _ in refs], dtype=int)
                if len(relevance) == 0:
                    auc_values.append(0.5)
                    reciprocal_values.append(0.0)
                    for k in top_values:
                        recalls[k].append(0.0)
                        purities[k].append(0.0)
                    continue
                if relevance.max() != relevance.min():
                    auc_values.append(float(roc_auc_score(relevance, -np.arange(len(relevance), dtype=float))))
                else:
                    auc_values.append(0.5)
                hits = np.where(relevance == 1)[0]
                reciprocal_values.append(float(1.0 / (hits[0] + 1)) if len(hits) else 0.0)
                for k in top_values:
                    head = relevance[:k]
                    recalls[k].append(float(head.sum() > 0))
                    purities[k].append(float(head.mean()) if len(head) else 0.0)
            row = {
                "retrieval": method,
                "label": label_name,
                "positive_query_count": positive_count,
                "auc": float(np.mean(auc_values)) if auc_values else 0.0,
                "mrr": float(np.mean(reciprocal_values)) if reciprocal_values else 0.0,
                "score_definition": "ranking order within train-only candidate chunk scope",
                "proxy_note": "contract label relevance proxy; not local evidence ground truth",
            }
            for k in top_values:
                row[f"recall_at_{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
                row[f"top_{k}_purity"] = float(np.mean(purities[k])) if purities[k] else 0.0
            rows.append(row)
    return rows


def pair_diagnostics(config, memory, train_payload, device):
    pairs = build_pairs(memory, train_payload, config)
    scorer = EvidenceRelevanceScorer(int(config["feature_dim"]), int(config["num_labels"]), int(config.get("scorer_hidden_dim", 128)), float(config.get("scorer_dropout", 0.1))).to(device)
    path = resolve("results/retrieval_validation/learned_evidence_v1/evidence_scorer.pt")
    if not path.exists():
        return {"status": "missing_scorer", "pairs": len(pairs["query"])}
    scorer.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    scorer.eval()
    with torch.no_grad():
        positive = scorer(pairs["query"].to(device), pairs["positive"].to(device), pairs["labels"].to(device)).cpu().numpy()
        negative = scorer(pairs["query"].to(device), pairs["negative"].to(device), pairs["labels"].to(device)).cpu().numpy()
    target = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    scores = np.concatenate([positive, negative])
    pair_order = positive > negative
    hist_edges = np.linspace(float(min(scores.min(), -1.0)), float(max(scores.max(), 1.0)), 21)
    positive_hist, _ = np.histogram(positive, bins=hist_edges)
    negative_hist, _ = np.histogram(negative, bins=hist_edges)
    return {
        "pairs": len(positive),
        "positive_score_mean": float(positive.mean()),
        "negative_score_mean": float(negative.mean()),
        "score_margin": float((positive - negative).mean()),
        "score_margin_median": float(np.median(positive - negative)),
        "pairwise_order_accuracy": float(pair_order.mean()),
        "auc": float(roc_auc_score(target, scores)),
        "positive_score_p50": float(np.percentile(positive, 50)),
        "negative_score_p50": float(np.percentile(negative, 50)),
        "score_histogram": {
            "bin_edges": hist_edges.tolist(),
            "positive_counts": positive_hist.tolist(),
            "negative_counts": negative_hist.tolist(),
        },
        "note": "Pairwise diagnostics; MRR and Recall@K are reported in retrieval_ranking_metrics.csv.",
    }


def categorized_pairs(memory, payload, config):
    chunks = memory["chunks"].float()
    masks = memory["chunk_mask"].bool()
    source_labels = memory["labels"].bool()
    queries = payload["query_contract"].float()
    query_labels = payload["labels"].bool()
    candidates = payload["candidate_contracts"]
    rng = random.Random(int(config.get("seed", 42)))
    rows = {"positive": [], "negative_a_safe": [], "negative_b_other_vulnerability": [], "labels": [], "query_rows": []}
    for row in range(len(queries)):
        for label_id in range(source_labels.shape[1]):
            if not bool(query_labels[row, label_id]):
                continue
            candidate_ids = candidates[row].tolist()
            positive = [index for index in candidate_ids if bool(source_labels[index, label_id])]
            safe = [index for index in candidate_ids if not bool(source_labels[index].any())]
            other = [index for index in candidate_ids if not bool(source_labels[index, label_id]) and bool(source_labels[index].any())]
            if not positive:
                continue
            if not safe and not other:
                continue
            positive_contract = rng.choice(positive)
            positive_chunks = torch.where(masks[positive_contract])[0].tolist()
            if not positive_chunks:
                continue
            rows["positive"].append(chunks[positive_contract, rng.choice(positive_chunks)])
            rows["labels"].append(label_id)
            rows["query_rows"].append(row)
            for key, pool in (("negative_a_safe", safe), ("negative_b_other_vulnerability", other)):
                if pool:
                    contract = rng.choice(pool)
                    chunk_ids = torch.where(masks[contract])[0].tolist()
                    rows[key].append(chunks[contract, rng.choice(chunk_ids)])
                else:
                    rows[key].append(None)
    if not rows["positive"]:
        empty = torch.empty((0, chunks.shape[-1]))
        return {
            "positive": empty,
            "labels": torch.empty((0,), dtype=torch.long),
            "query_rows": torch.empty((0,), dtype=torch.long),
            "negative_a_safe": (empty, torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), empty),
            "negative_b_other_vulnerability": (empty, torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), empty),
        }
    output = {"positive": torch.stack(rows["positive"]), "labels": torch.tensor(rows["labels"], dtype=torch.long), "query_rows": torch.tensor(rows["query_rows"], dtype=torch.long)}
    for key in ("negative_a_safe", "negative_b_other_vulnerability"):
        valid = [index for index, value in enumerate(rows[key]) if value is not None]
        output[key] = (
            torch.stack([rows[key][index] for index in valid]),
            output["labels"][valid],
            output["query_rows"][valid],
            output["positive"][valid],
        )
    return output


def hard_negative_report(config, memory, train_payload, device):
    path = resolve("results/retrieval_validation/learned_evidence_v1/evidence_scorer.pt")
    if not path.exists():
        return {"status": "missing_scorer"}
    scorer = EvidenceRelevanceScorer(int(config["feature_dim"]), int(config["num_labels"]), int(config.get("scorer_hidden_dim", 128)), float(config.get("scorer_dropout", 0.1))).to(device)
    scorer.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    scorer.eval()
    pairs = categorized_pairs(memory, train_payload, config)
    if len(pairs["positive"]) == 0:
        return {"status": "no_categorized_pairs"}
    report = {}
    with torch.no_grad():
        positive = scorer(train_payload["query_contract"][pairs["query_rows"]].to(device), pairs["positive"].to(device), pairs["labels"].to(device)).cpu().numpy()
        for key in ("negative_a_safe", "negative_b_other_vulnerability"):
            negative, labels, query_rows, positive_chunks = pairs[key]
            if len(negative) == 0:
                report[key] = {"pairs": 0, "status": "no_candidates"}
                continue
            query = train_payload["query_contract"][query_rows]
            positive_score = scorer(query.to(device), positive_chunks.to(device), labels.to(device)).cpu().numpy()
            score = scorer(query.to(device), negative.to(device), labels.to(device)).cpu().numpy()
            report[key] = {"pairs": len(score), "positive_mean": float(positive_score.mean()), "negative_mean": float(score.mean()), "margin": float((positive_score - score).mean()), "auc": float(roc_auc_score(np.r_[np.ones(len(score)), np.zeros(len(score))], np.r_[positive_score, score])) if len(score) and len(np.unique(np.r_[positive_score, score])) > 1 else 0.5}
    return report


def shuffle_scorer_report(config, memory, train_payload, valid_payload, device):
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)) + 1009)
    shuffled_memory = dict(memory)
    shuffled_payload = dict(train_payload)
    shuffled_memory["labels"] = memory["labels"][torch.randperm(len(memory["labels"]), generator=generator)]
    shuffled_payload["labels"] = train_payload["labels"][torch.randperm(len(train_payload["labels"]), generator=generator)]
    scorer, history = train_scorer(config, shuffled_memory, shuffled_payload, device)
    pairs = build_pairs(shuffled_memory, shuffled_payload, config)
    with torch.no_grad():
        pos = scorer(pairs["query"].to(device), pairs["positive"].to(device), pairs["labels"].to(device)).cpu().numpy()
        neg = scorer(pairs["query"].to(device), pairs["negative"].to(device), pairs["labels"].to(device)).cpu().numpy()
    rerank_config = dict(config)
    rerank_config["evidence_top_k"] = 10
    shuffled_indices, _ = rerank(rerank_config, memory, valid_payload, scorer, device)
    purity = purity_rows(config, memory, valid_payload, [shuffled_indices], ["shuffled_scorer"])
    return {"history": history, "train_pair_auc": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])), "train_pair_margin": float((pos - neg).mean()), "valid_purity": purity}


class EvidenceMIL(nn.Module):
    def __init__(self, dim, labels, hidden, scorer_hidden, dropout):
        super().__init__()
        self.scorer = EvidenceRelevanceScorer(dim, labels, scorer_hidden, dropout)
        self.residual = nn.Sequential(nn.Linear(dim * 3 + scorer_hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, query, evidence, base_logits, evidence_mask=None):
        batch, labels, count, dim = evidence.shape
        q = query.unsqueeze(1).unsqueeze(2).expand(-1, labels, count, -1)
        label_ids = torch.arange(labels, device=evidence.device).view(1, labels, 1).expand(batch, -1, count)
        scores = self.scorer(q.reshape(-1, dim), evidence.reshape(-1, dim), label_ids.reshape(-1)).reshape(batch, labels, count)
        if evidence_mask is not None:
            scores = scores.masked_fill(~evidence_mask.bool(), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        pooled = (attention.unsqueeze(-1) * evidence).sum(-2)
        label = self.scorer.label_embedding.unsqueeze(0).expand(batch, -1, -1)
        q_label = query.unsqueeze(1).expand(-1, labels, -1)
        residual = self.residual(torch.cat([q_label, pooled, q_label * pooled, label], dim=-1)).squeeze(-1)
        return base_logits + residual, attention


def gather_candidate_tensor(memory, indices, device):
    chunks = memory["chunks"].float()
    result = torch.zeros((*indices.shape[:-1], chunks.shape[-1]))
    for row in range(len(indices)):
        for label_id in range(indices.shape[1]):
            for slot in range(indices.shape[2]):
                contract_id, chunk_id = indices[row, label_id, slot].tolist()
                if contract_id >= 0:
                    result[row, label_id, slot] = chunks[contract_id, chunk_id]
    return result.to(device)


def candidate_mask(indices):
    return (indices[..., 0] >= 0)


def train_mil_supervision(config, memory, train_payload, valid_payload, train_indices, valid_indices, device):
    train_q = train_payload["query_contract"].float().to(device)
    valid_q = valid_payload["query_contract"].float().to(device)
    train_e = gather_candidate_tensor(memory, train_indices, device)
    valid_e = gather_candidate_tensor(memory, valid_indices, device)
    train_mask = candidate_mask(train_indices).to(device)
    valid_mask = candidate_mask(valid_indices).to(device)
    train_base = train_payload["base_logits"].float().to(device)
    valid_base = valid_payload["base_logits"].float().to(device)
    train_y = train_payload["labels"].float().to(device)
    valid_y = valid_payload["labels"].float().to(device)
    model = EvidenceMIL(int(config["feature_dim"]), int(config["num_labels"]), int(config.get("diagnosis_hidden_dim", 256)), int(config.get("scorer_hidden_dim", 128)), float(config.get("diagnosis_dropout", 0.1))).to(device)
    pos = train_y.sum(0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.sqrt((len(train_y) - pos) / pos.clamp_min(1)).clamp(1, 5))
    loader = DataLoader(TensorDataset(train_q, train_e, train_base, train_y, train_mask), batch_size=int(config.get("diagnosis_batch_size", 128)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("diagnosis_learning_rate", 1e-4)), weight_decay=0.01)
    best = None
    patience = 0
    for epoch in range(1, int(config.get("diagnosis_epochs", 5)) + 1):
        model.train()
        for query, evidence, base, labels, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(query, evidence, base, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, attention = model(valid_q, valid_e, valid_base, valid_mask)
        valid_metrics, probs = metrics(valid_y, logits)
        print(f"[mil_supervision] epoch={epoch} fixed_macro={valid_metrics['recognition_macro_f1']:.6f}", flush=True)
        if best is None or valid_metrics["recognition_macro_f1"] > best["score"]:
            best = {"score": valid_metrics["recognition_macro_f1"], "epoch": epoch, "probs": probs, "attention": attention.detach().cpu()}
            patience = 0
        else:
            patience += 1
        if patience >= int(config.get("diagnosis_patience", 2)):
            break
    thresholds = select_thresholds(valid_y.cpu().numpy().astype(int), best["probs"], config["thresholds"])
    best_logits = torch.as_tensor(np.log(best["probs"] / np.clip(1 - best["probs"], 1e-7, 1.0)))
    fixed = metrics(valid_y, best_logits, None)[0]
    tuned = metrics(valid_y, best_logits, thresholds)[0]
    return {"best_epoch": best["epoch"], "fixed": fixed, "tuned": tuned, "thresholds": thresholds, "attention_mean": best["attention"].mean((-1, -2)).tolist(), "valid_probs": best["probs"].tolist()}


def load_existing_results():
    root = resolve("results/retrieval_validation")
    output = {}
    for key, path in (("M0", root / "baseline.json"), ("M1", root / "contract_retrieval.json"), ("M2_MVP", root / "evidence_retrieval.json"), ("M2_learned", root / "learned_evidence_v1" / "learned_evidence_summary.json")):
        if not path.exists():
            output[key] = {"status": "missing", "path": str(path)}
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if key == "M2_learned":
            value = value.get("classifier", value)
        if "fixed_0.5" not in value and "recomputed_from_checkpoint" in value:
            value = value["recomputed_from_checkpoint"]
        fixed = value.get("fixed_0.5", {})
        tuned = value.get("tuned_valid", {})
        output[key] = {"fixed_macro_f1": fixed.get("recognition_macro_f1"), "fixed_micro_f1": fixed.get("recognition_micro_f1"), "tuned_macro_f1": tuned.get("recognition_macro_f1"), "tuned_micro_f1": tuned.get("recognition_micro_f1"), "fixed_per_label_f1": fixed.get("per_label_f1"), "tuned_per_label_f1": tuned.get("per_label_f1")}
    return output


def result_row(method, result):
    if "fixed_macro_f1" in result:
        return {
            "method": method,
            "fixed_micro_f1": result.get("fixed_micro_f1"),
            "fixed_macro_f1": result.get("fixed_macro_f1"),
            "tuned_micro_f1": result.get("tuned_micro_f1"),
            "tuned_macro_f1": result.get("tuned_macro_f1"),
            "fixed_per_label_f1": json.dumps(result.get("fixed_per_label_f1", [])),
            "tuned_per_label_f1": json.dumps(result.get("tuned_per_label_f1", [])),
            "per_label_f1": json.dumps(result.get("tuned_per_label_f1", result.get("fixed_per_label_f1", []))),
        }
    fixed = result.get("fixed", result.get("fixed_0.5", {}))
    tuned = result.get("tuned", result.get("tuned_valid", {}))
    return {
        "method": method,
        "fixed_micro_f1": fixed.get("recognition_micro_f1"),
        "fixed_macro_f1": fixed.get("recognition_macro_f1"),
        "tuned_micro_f1": tuned.get("recognition_micro_f1"),
        "tuned_macro_f1": tuned.get("recognition_macro_f1"),
        "fixed_per_label_f1": json.dumps(fixed.get("per_label_f1", [])),
        "tuned_per_label_f1": json.dumps(tuned.get("per_label_f1", [])),
        "per_label_f1": json.dumps(tuned.get("per_label_f1", fixed.get("per_label_f1", []))),
    }


def diagnose_bottlenecks(diagnosis):
    """Produce conservative, evidence-backed hypotheses instead of a score-only verdict."""
    existing = diagnosis["existing_results"]
    m0 = existing.get("M0", {}).get("tuned_macro_f1")
    causes = []
    evidence = []
    oracle = diagnosis.get("oracle", {}).get("tuned", {}).get("recognition_macro_f1")
    if m0 is not None and oracle is not None:
        delta = float(oracle - m0)
        evidence.append({"test": "oracle_minus_m0", "value": delta})
        if delta <= 0.005:
            causes.append("A")
    scorer = diagnosis.get("scorer", {})
    if scorer.get("auc") is not None:
        evidence.append({"test": "learned_pair_auc", "value": scorer["auc"]})
        evidence.append({"test": "learned_pair_margin", "value": scorer.get("score_margin")})
        if scorer["auc"] < 0.6 or abs(float(scorer.get("score_margin", 0.0))) < 0.05:
            causes.append("E")
    k_rows = diagnosis.get("candidate_k_ablation", [])
    if k_rows:
        first = k_rows[0].get("tuned_macro_f1")
        last = k_rows[-1].get("tuned_macro_f1")
        evidence.append({"test": "K_256_minus_K_8", "value": (last - first) if first is not None and last is not None else None})
        if first is not None and last is not None and last - first > 0.005:
            causes.append("B")
    budget_rows = [row for row in diagnosis.get("chunk_budget_ablation", []) if row.get("status") != "disabled"]
    if len(budget_rows) >= 2:
        flat = next((row for row in budget_rows if row["scope"] == "flatten_512"), None)
        balanced = next((row for row in budget_rows if row["scope"] == "balanced_512"), None)
        if flat and balanced:
            delta = balanced["tuned_macro_f1"] - flat["tuned_macro_f1"]
            evidence.append({"test": "balanced_minus_flatten", "value": delta})
            if delta > 0.005:
                causes.append("C")
    supervision = diagnosis.get("evidence_supervision_comparison", [])
    random_row = next((row for row in supervision if row["method"] == "random_chunk_aggregation"), None)
    mil = diagnosis.get("mil_supervision", {})
    mil_score = mil.get("tuned", {}).get("recognition_macro_f1")
    if random_row and mil_score is not None:
        delta = mil_score - random_row["tuned_macro_f1"]
        evidence.append({"test": "MIL_minus_random_aggregation", "value": delta})
        if delta > 0.005:
            causes.append("D")
    purity = diagnosis.get("ranking_purity", [])
    if purity and scorer.get("auc") is not None:
        learned = [row["positive_query_label_purity"] for row in purity if row["retrieval"] == "learned_scorer" and row["top_k"] == 10]
        cosine = [row["positive_query_label_purity"] for row in purity if row["retrieval"] == "cosine" and row["top_k"] == 10]
        if learned and cosine:
            evidence.append({"test": "learned_minus_cosine_top10_purity", "value": float(np.mean(learned) - np.mean(cosine))})
            if np.mean(learned) <= np.mean(cosine):
                causes.append("F")
    if not causes:
        causes.append("G")
    return {"likely_bottlenecks": sorted(set(causes)), "evidence": evidence, "interpretation": "Codes: A hypothesis headroom, B candidate recall, C chunk budget, D supervision noise, E scorer ranking, F query/prototype, G aggregation/fusion, H effective retrieval diluted by fusion."}


def positive_supervision_analysis(config, memory, train_payload):
    """Document what the weak positive-chunk label actually guarantees."""
    labels = memory["labels"].bool()
    masks = memory["chunk_mask"].bool()
    rows = []
    for label_id, label_name in enumerate(config["label_names"]):
        positive = labels[:, label_id]
        counts = masks[positive].sum(1).numpy() if bool(positive.any()) else np.asarray([])
        rows.append({
            "label": label_name,
            "positive_contracts": int(positive.sum()),
            "mean_chunks_per_positive_contract": float(counts.mean()) if len(counts) else 0.0,
            "min_chunks": int(counts.min()) if len(counts) else 0,
            "max_chunks": int(counts.max()) if len(counts) else 0,
            "random_chunk_local_relevance_observed": False,
            "noise_statement": "A random chunk inherits only a contract-level positive label; its local vulnerability relevance is unobserved.",
        })
    pairs = build_pairs(memory, train_payload, config)
    return {
        "pair_count": len(pairs["query"]),
        "per_label": rows,
        "conclusion": "The current random-positive supervision is weak and cannot distinguish relevant from irrelevant chunks within a positive contract.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval_diagnosis.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    memory, train_payload, valid_payload = load_route_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    existing = load_existing_results()
    diagnosis = {"route": config["route_name"], "seed": int(config.get("seed", 42)), "test_checked": False, "config": config, "existing_results": existing, "warnings": ["Oracle uses query labels by design and is an upper bound only.", "Purity is a source-contract-label proxy, not evidence ground truth."]}

    oracle_train = oracle_indices(config, memory, train_payload)
    oracle_valid = oracle_indices(config, memory, valid_payload)
    diagnosis["oracle"] = train_residual(config, "oracle_evidence", memory, train_payload, valid_payload, oracle_train, oracle_valid, device)
    oracle_rows = [result_row("M0", existing.get("M0", {})), result_row("M1", existing.get("M1", {})), result_row("M2_MVP", existing.get("M2_MVP", {})), result_row("M2_learned", existing.get("M2_learned", {})), result_row("Oracle Evidence", diagnosis["oracle"])]
    write_json(root / "oracle_evidence.json", {"upper_bound_only": True, "test_checked": False, "models": oracle_rows})
    diagnosis["oracle_comparison"] = oracle_rows

    candidate_stat_rows = candidate_statistics(config, memory, valid_payload)
    k_rows = []
    for candidate_k in config["candidate_k_values"]:
        train_idx, _, _ = build_scope_indices(config, memory, train_payload, candidate_k, mode="prototype", top_k=int(config["evidence_top_k"]), budget=int(config["chunk_budget"]), balanced_m=int(config["balanced_chunks_per_contract"]))
        valid_idx, _, _ = build_scope_indices(config, memory, valid_payload, candidate_k, mode="prototype", top_k=int(config["evidence_top_k"]), budget=int(config["chunk_budget"]), balanced_m=int(config["balanced_chunks_per_contract"]))
        result = train_residual(config, f"candidate_k_{candidate_k}", memory, train_payload, valid_payload, train_idx, valid_idx, device)
        stat = [row for row in candidate_stat_rows if row["candidate_contracts"] == candidate_k]
        k_rows.append({"candidate_contracts": candidate_k, "fixed_macro_f1": result["fixed"]["recognition_macro_f1"], "tuned_macro_f1": result["tuned"]["recognition_macro_f1"], "tuned_micro_f1": result["tuned"]["recognition_micro_f1"], "best_epoch": result["best_epoch"], "mean_candidate_positive_recall": float(np.mean([row["candidate_positive_recall"] for row in stat])), "per_label_candidate_positive_recall": json.dumps({row["label"]: row["candidate_positive_recall"] for row in stat})})
    write_csv(root / "candidate_k_ablation.csv", k_rows, list(k_rows[0]) if k_rows else ["candidate_contracts"])
    diagnosis["candidate_k_ablation"] = k_rows
    write_csv(root / "candidate_statistics.csv", candidate_stat_rows, list(candidate_stat_rows[0]))
    diagnosis["candidate_statistics"] = candidate_stat_rows

    budget_rows = []
    modes = ["flatten_512", "balanced_512"]
    if bool(config.get("candidate_all", False)):
        modes.append("all")
    for mode in modes:
        train_idx, _, _ = build_scope_indices(config, memory, train_payload, 64, mode=mode, top_k=int(config["evidence_top_k"]), budget=int(config["chunk_budget"]), balanced_m=int(config["balanced_chunks_per_contract"]))
        valid_idx, _, _ = build_scope_indices(config, memory, valid_payload, 64, mode=mode, top_k=int(config["evidence_top_k"]), budget=int(config["chunk_budget"]), balanced_m=int(config["balanced_chunks_per_contract"]))
        result = train_residual(config, f"budget_{mode}", memory, train_payload, valid_payload, train_idx, valid_idx, device)
        budget_rows.append({"scope": mode, "status": "completed", "fixed_macro_f1": result["fixed"]["recognition_macro_f1"], "tuned_macro_f1": result["tuned"]["recognition_macro_f1"], "tuned_micro_f1": result["tuned"]["recognition_micro_f1"], "best_epoch": result["best_epoch"]})
    if not bool(config.get("candidate_all", False)):
        budget_rows.append({"scope": "all", "status": "disabled_by_config", "fixed_macro_f1": "", "tuned_macro_f1": "", "tuned_micro_f1": "", "best_epoch": ""})
    write_csv(root / "chunk_budget_ablation.csv", budget_rows, list(budget_rows[0]) if budget_rows else ["scope"])
    diagnosis["chunk_budget_ablation"] = budget_rows

    cosine_idx, _, _ = build_scope_indices(config, memory, valid_payload, 64, mode="query_cosine", top_k=10, budget=int(config["rerank_chunk_budget"]), balanced_m=int(config["balanced_chunks_per_contract"]))
    prototype_train, _, _ = build_scope_indices(config, memory, train_payload, 64, mode="prototype", top_k=5, budget=int(config["rerank_chunk_budget"]), balanced_m=int(config["balanced_chunks_per_contract"]))
    prototype_idx, _, _ = build_scope_indices(config, memory, valid_payload, 64, mode="prototype", top_k=10, budget=int(config["rerank_chunk_budget"]), balanced_m=int(config["balanced_chunks_per_contract"]))
    random_idx, _, _ = build_scope_indices(config, memory, valid_payload, 64, mode="random_topk", top_k=10, budget=int(config["rerank_chunk_budget"]), balanced_m=int(config["balanced_chunks_per_contract"]))
    learned_idx = None
    learned_scorer_path = resolve("results/retrieval_validation/learned_evidence_v1/evidence_scorer.pt")
    if learned_scorer_path.exists():
        scorer = EvidenceRelevanceScorer(int(config["feature_dim"]), int(config["num_labels"]), int(config.get("scorer_hidden_dim", 128)), float(config.get("scorer_dropout", 0.1))).to(device)
        scorer.load_state_dict(torch.load(learned_scorer_path, map_location="cpu")["state_dict"])
        learned_payload_config = dict(config)
        learned_payload_config["evidence_top_k"] = 10
        learned_idx, _ = rerank(learned_payload_config, memory, valid_payload, scorer, device)
    index_sets = [cosine_idx, prototype_idx, random_idx] + ([learned_idx] if learned_idx is not None else [])
    names = ["cosine", "fixed_prototype", "random"] + (["learned_scorer"] if learned_idx is not None else [])
    ranking_rows = purity_rows(config, memory, valid_payload, index_sets, names)
    write_csv(root / "retrieval_ranking_purity.csv", ranking_rows, ["retrieval", "top_k", "label", "label_purity", "positive_query_label_purity", "proxy_note"])
    diagnosis["ranking_purity"] = ranking_rows
    rank_scorers = {"cosine": None, "random": None}
    if learned_idx is not None:
        rank_scorers["learned_scorer"] = scorer
    ranked = {name: ranked_references(config, memory, valid_payload, name, rank_scorers.get(name), device, candidate_k=64, budget=int(config["rerank_chunk_budget"])) for name in rank_scorers}
    ranking_rows_metrics = ranking_metric_rows(config, memory, valid_payload, ranked)
    write_csv(root / "retrieval_ranking_metrics.csv", ranking_rows_metrics, list(ranking_rows_metrics[0]))
    diagnosis["ranking_metrics"] = ranking_rows_metrics
    diagnosis["scorer"] = pair_diagnostics(config, memory, train_payload, device)
    diagnosis["hard_negative"] = hard_negative_report(config, memory, train_payload, device)
    diagnosis["label_shuffle"] = shuffle_scorer_report(config, memory, train_payload, valid_payload, device)
    write_json(root / "scorer_diagnostics.json", diagnosis["scorer"])
    write_json(root / "hard_negative_analysis.json", diagnosis["hard_negative"])
    write_json(root / "label_shuffle_analysis.json", diagnosis["label_shuffle"])
    diagnosis["positive_supervision"] = positive_supervision_analysis(config, memory, train_payload)
    write_json(root / "positive_evidence_supervision.json", diagnosis["positive_supervision"])

    random_train, _, _ = build_scope_indices(config, memory, train_payload, 64, mode="random_topk", top_k=5, budget=512, balanced_m=8)
    random_valid, _, _ = build_scope_indices(config, memory, valid_payload, 64, mode="random_topk", top_k=5, budget=512, balanced_m=8)
    cosine_train, _, _ = build_scope_indices(config, memory, train_payload, 64, mode="query_cosine", top_k=5, budget=512, balanced_m=8)
    cosine_valid, _, _ = build_scope_indices(config, memory, valid_payload, 64, mode="query_cosine", top_k=5, budget=512, balanced_m=8)
    supervision_rows = []
    for name, train_idx, valid_idx in (("random_chunk_aggregation", random_train, random_valid), ("cosine_topk", cosine_train, cosine_valid), ("fixed_prototype_topk", prototype_train, prototype_idx[:, :, :5])):
        result = train_residual(config, name, memory, train_payload, valid_payload, train_idx, valid_idx, device)
        supervision_rows.append({"method": name, "fixed_macro_f1": result["fixed"]["recognition_macro_f1"], "tuned_macro_f1": result["tuned"]["recognition_macro_f1"], "tuned_micro_f1": result["tuned"]["recognition_micro_f1"]})
    learned_summary = load_existing_results().get("M2_learned", {})
    if learned_summary.get("tuned_macro_f1") is not None:
        supervision_rows.append({"method": "learned_scorer_random_positive_supervision", "fixed_macro_f1": learned_summary.get("fixed_macro_f1"), "tuned_macro_f1": learned_summary.get("tuned_macro_f1"), "tuned_micro_f1": learned_summary.get("tuned_micro_f1")})
    write_csv(root / "evidence_supervision_comparison.csv", supervision_rows, list(supervision_rows[0]) if supervision_rows else ["method"])
    diagnosis["evidence_supervision_comparison"] = supervision_rows

    mil_train, _, _ = build_scope_indices(config, memory, train_payload, 64, mode="balanced_512", top_k=8, budget=512, balanced_m=int(config["balanced_chunks_per_contract"]))
    mil_valid, _, _ = build_scope_indices(config, memory, valid_payload, 64, mode="balanced_512", top_k=8, budget=512, balanced_m=int(config["balanced_chunks_per_contract"]))
    diagnosis["mil_supervision"] = train_mil_supervision(config, memory, train_payload, valid_payload, mil_train, mil_valid, device)

    diagnosis["bottleneck_diagnosis"] = diagnose_bottlenecks(diagnosis)
    write_json(root / "diagnosis_report.json", diagnosis)
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
