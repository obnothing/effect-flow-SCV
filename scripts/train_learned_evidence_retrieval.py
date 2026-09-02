"""Train a train-only weakly supervised evidence relevance scorer."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_retrieval_validation import (  # noqa: E402
    gather_features,
    load_cache,
    load_prepared,
    metrics,
    select_thresholds,
    train_one,
    M2,
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


class EvidenceRelevanceScorer(nn.Module):
    def __init__(self, dim, labels, hidden, dropout):
        super().__init__()
        self.query_projection = nn.Linear(dim, hidden)
        self.evidence_projection = nn.Linear(dim, hidden)
        self.label_embedding = nn.Parameter(torch.randn(labels, hidden) * 0.02)
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, query, evidence, label_ids):
        query = self.query_projection(query)
        evidence = self.evidence_projection(evidence)
        label = self.label_embedding[label_ids]
        return self.scorer(torch.cat([query, evidence, query * evidence, label], dim=-1)).squeeze(-1)


def build_pairs(memory, train_payload, config):
    chunks = memory["chunks"].float()
    masks = memory["chunk_mask"].bool()
    labels = memory["labels"].bool()
    queries = train_payload["query_contract"].float()
    candidates = train_payload["candidate_contracts"]
    query_labels = train_payload["labels"].bool()
    query_rows, positive_rows, negative_rows, label_rows = [], [], [], []
    rng = random.Random(int(config.get("seed", 42)))
    for query_id in range(len(queries)):
        candidate_ids = candidates[query_id].tolist()
        for label_id in range(labels.shape[1]):
            if not bool(query_labels[query_id, label_id]):
                continue
            positive_contracts = [index for index in candidate_ids if bool(labels[index, label_id])]
            negative_contracts = [index for index in candidate_ids if not bool(labels[index, label_id])]
            if not positive_contracts or not negative_contracts:
                continue
            for _ in range(int(config.get("pairs_per_query_label", 1))):
                positive_contract = rng.choice(positive_contracts)
                negative_contract = rng.choice(negative_contracts)
                positive_chunks = torch.where(masks[positive_contract])[0].tolist()
                negative_chunks = torch.where(masks[negative_contract])[0].tolist()
                if not positive_chunks or not negative_chunks:
                    continue
                query_rows.append(query_id)
                positive_rows.append((positive_contract, rng.choice(positive_chunks)))
                negative_rows.append((negative_contract, rng.choice(negative_chunks)))
                label_rows.append(label_id)
    if not query_rows:
        raise RuntimeError("no weakly supervised evidence pairs were constructed")
    return {
        "query": queries[query_rows],
        "positive": torch.stack([chunks[c, k] for c, k in positive_rows]),
        "negative": torch.stack([chunks[c, k] for c, k in negative_rows]),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def train_scorer(config, memory, train_payload, device):
    pairs = build_pairs(memory, train_payload, config)
    dataset = TensorDataset(pairs["query"], pairs["positive"], pairs["negative"], pairs["labels"])
    loader = DataLoader(dataset, batch_size=int(config.get("scorer_batch_size", 512)), shuffle=True)
    scorer = EvidenceRelevanceScorer(
        int(config["feature_dim"]), int(config["num_labels"]),
        int(config.get("scorer_hidden_dim", 128)), float(config.get("scorer_dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=float(config.get("scorer_learning_rate", 1e-4)), weight_decay=0.01)
    margin = float(config.get("pair_margin", 0.2))
    history, best_state, best_loss, patience = [], None, float("inf"), 0
    for epoch in range(1, int(config.get("scorer_epochs", 5)) + 1):
        scorer.train()
        losses = []
        for query, positive, negative, label_ids in loader:
            query, positive, negative, label_ids = [value.to(device) for value in (query, positive, negative, label_ids)]
            optimizer.zero_grad(set_to_none=True)
            positive_score = scorer(query, positive, label_ids)
            negative_score = scorer(query, negative, label_ids)
            loss = torch.nn.functional.softplus(margin - positive_score + negative_score).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        value = float(np.mean(losses))
        history.append({"epoch": epoch, "pairwise_loss": value, "pairs": len(dataset)})
        print(f"[learned_scorer] epoch={epoch} pairwise_loss={value:.6f} pairs={len(dataset)}", flush=True)
        if value < best_loss:
            best_loss, patience = value, 0
            best_state = {key: value.detach().cpu() for key, value in scorer.state_dict().items()}
        else:
            patience += 1
        if patience >= int(config.get("scorer_patience", 2)):
            break
    scorer.load_state_dict(best_state)
    return scorer, history


def rerank(config, memory, payload, scorer, device):
    chunks = memory["chunks"].float()
    masks = memory["chunk_mask"].bool()
    candidates = payload["candidate_contracts"]
    queries = payload["query_contract"].float()
    labels = int(config["num_labels"])
    top_k = int(config["evidence_top_k"])
    budget = int(config.get("rerank_chunk_budget", 512))
    output = torch.full((len(queries), labels, top_k, 2), -1, dtype=torch.long)
    scores = torch.zeros((len(queries), labels, top_k), dtype=torch.float32)
    scorer.eval()
    with torch.no_grad():
        for row in range(len(queries)):
            contract_ids = candidates[row].tolist()
            refs = [(contract_id, chunk_id) for contract_id in contract_ids for chunk_id in torch.where(masks[contract_id])[0].tolist()]
            refs = refs[:budget]
            if not refs:
                continue
            evidence = torch.stack([chunks[contract_id, chunk_id] for contract_id, chunk_id in refs]).to(device)
            query = queries[row].to(device).expand(len(refs), -1)
            for label_id in range(labels):
                label_ids = torch.full((len(refs),), label_id, dtype=torch.long, device=device)
                relevance = scorer(query, evidence, label_ids)
                values, positions = torch.topk(relevance, k=min(top_k, len(refs)))
                for position, value in zip(positions.cpu().tolist(), values.cpu().tolist()):
                    slot = positions.cpu().tolist().index(position)
                    output[row, label_id, slot] = torch.tensor(refs[position])
                    scores[row, label_id, slot] = value
    return output, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval_learned_evidence.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    base_root = resolve("results/retrieval_validation")
    memory = torch.load(base_root / "retrieval_memory.pt", map_location="cpu")
    train_payload = load_prepared({**config, "result_dir": str(base_root)}, "train")
    valid_payload = load_prepared({**config, "result_dir": str(base_root)}, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, history = train_scorer(config, memory, train_payload, device)
    output_root = resolve(config["result_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "main6_learned_evidence_scorer_v1", "state_dict": scorer.state_dict(), "history": history, "train_only": True}, output_root / "evidence_scorer.pt")
    for split, payload in (("train", train_payload), ("valid", valid_payload)):
        evidence_indices, evidence_scores = rerank(config, memory, payload, scorer, device)
        updated = dict(payload)
        updated["evidence_indices"] = evidence_indices
        updated["evidence_scores"] = evidence_scores
        updated["retrieval_scorer_train_only"] = True
        torch.save(updated, output_root / f"{split}_queries.pt")
    train_config = dict(config)
    train_config["result_dir"] = str(output_root)
    train_config["train_only_memory"] = True
    train_config["epochs"] = int(config.get("epochs", 30))
    trained = train_one(train_config, "m2_evidence_retrieval_learned", M2, memory, torch.load(output_root / "train_queries.pt", map_location="cpu"), torch.load(output_root / "valid_queries.pt", map_location="cpu"), device)
    (output_root / "learned_evidence_summary.json").write_text(json.dumps({"route": config["route_name"], "test_checked": False, "scorer_history": history, "classifier": trained}, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output_root / 'learned_evidence_summary.json'}")


if __name__ == "__main__":
    main()
