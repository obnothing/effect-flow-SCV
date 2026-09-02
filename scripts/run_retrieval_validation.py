"""Validate train-only contract and vulnerability-evidence retrieval on Main-6."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import compute_multilabel_metrics_from_probs  # noqa: E402


LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path):
    with resolve(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def opcode_hash(value):
    return hashlib.sha256(" ".join(str(value).split()).encode()).hexdigest()


def audit(config):
    data_dir = resolve(config["data_dir"])
    rows = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "valid", "test")}
    ids = {split: [str(row["id"]) for row in values] for split, values in rows.items()}
    hashes = {
        split: {opcode_hash(row.get("opcode", "")) for row in values}
        for split, values in rows.items()
    }
    label_names = config.get("label_names", LABELS)
    report = {
        "route": config["route_name"],
        "status": "ok",
        "seed": int(config.get("seed", 42)),
        "retrieval_memory_splits": ["train"],
        "test_checked": False,
        "data_dir": str(data_dir.relative_to(ROOT)),
        "splits": {},
        "id_overlap": {},
        "opcode_hash_overlap": {},
        "risks": [
            "The official split is a non-grouped random split.",
            "Opcode hash overlap is reported and is not used to alter the split.",
            "The retrieval memory contains training contracts only.",
        ],
    }
    for split, values in rows.items():
        labels = [row.get("multi_labels", []) for row in values]
        lengths = [len(str(row.get("opcode", "")).split()) for row in values]
        report["splits"][split] = {
            "samples": len(values),
            "unique_ids": len(set(ids[split])),
            "unique_opcode_hashes": len(hashes[split]),
            "label_widths": sorted({len(value) for value in labels}),
            "label_counts": [sum(int(value[i]) for value in labels) for i in range(len(label_names))],
            "opcode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "opcode_length_max": max(lengths or [0]),
        }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        report["id_overlap"][f"{left}_{right}"] = len(set(ids[left]) & set(ids[right]))
        report["opcode_hash_overlap"][f"{left}_{right}"] = len(hashes[left] & hashes[right])
    cache_dir = resolve(config["feature_dir"])
    report["feature_cache"] = {}
    for split in ("train", "valid"):
        path = cache_dir / f"{split}.pt"
        row = {"path": str(path.relative_to(ROOT)), "exists": path.exists()}
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            features = payload["features"]
            mask = payload["chunk_mask"].bool()
            row.update({
                "sha256": digest(path),
                "ids": len(payload["ids"]),
                "feature_shape": list(features.shape),
                "mask_shape": list(mask.shape),
                "label_width": int(payload["multi_labels"].shape[1]),
                "nan": bool(torch.isnan(features.float()).any()),
                "inf": bool(torch.isinf(features.float()).any()),
                "min_real_chunks": int(mask.sum(1).min()),
                "max_real_chunks": int(mask.sum(1).max()),
                "cache_ids_match_jsonl": [str(x) for x in payload["ids"]] == ids[split],
            })
        report["feature_cache"][split] = row
    output = resolve(config["result_dir"]) / "project_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(value for value in report["id_overlap"].values()):
        raise RuntimeError("duplicate contract IDs cross splits")
    return report


def load_cache(config, split):
    path = resolve(config["feature_dir"]) / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].float()
    if features.ndim != 4 or features.shape[2] != 8 or features.shape[3] != int(config["feature_dim"]):
        raise ValueError(f"unexpected feature shape in {path}: {tuple(features.shape)}")
    mask = payload["chunk_mask"].bool()
    base = features[:, :, 1, :]
    base = base.masked_fill(~mask.unsqueeze(-1), 0.0)
    contract = base.sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
    contract = torch.nn.functional.normalize(contract, dim=-1)
    return {
        "ids": [str(value) for value in payload["ids"]],
        "labels": payload["multi_labels"].float(),
        "mask": mask,
        "chunks": base,
        "contract": contract,
        "cache_report": payload.get("report", {}),
    }


def validate_cache_alignment(config, data, split):
    expected = [str(row["id"]) for row in read_jsonl(resolve(config["data_dir"]) / f"{split}.jsonl")]
    if data["ids"] != expected:
        first = next((index for index, pair in enumerate(zip(data["ids"], expected)) if pair[0] != pair[1]), 0)
        raise ValueError(f"{split} feature-cache IDs are not aligned at index {first}")
    if data["labels"].shape != (len(expected), int(config["num_labels"])):
        raise ValueError(f"{split} cache labels have unexpected shape: {tuple(data['labels'].shape)}")


def build_retrieval_indices(config, train, query, query_split):
    k_contract = int(config["contract_top_k"])
    k_evidence = int(config["evidence_top_k"])
    candidate_count = int(config["evidence_candidate_contracts"])
    train_contract = train["contract"]
    query_contract = query["contract"]
    scores = query_contract @ train_contract.T
    if query_split == "train":
        own = {sample_id: index for index, sample_id in enumerate(train["ids"])}
        for row, sample_id in enumerate(query["ids"]):
            if sample_id in own:
                scores[row, own[sample_id]] = -2.0
    neighbor_scores, neighbors = torch.topk(scores, k=min(k_contract, train_contract.shape[0]), dim=1)
    candidate_scores, candidates = torch.topk(scores, k=min(candidate_count, train_contract.shape[0]), dim=1)
    label_proto = []
    for label_id in range(train["labels"].shape[1]):
        positive = train["labels"][:, label_id] > 0.5
        proto = train_contract[positive].mean(0) if bool(positive.any()) else train_contract.mean(0)
        label_proto.append(torch.nn.functional.normalize(proto, dim=0))
    label_proto = torch.stack(label_proto)
    evidence_indices = torch.full((len(query["ids"]), train["labels"].shape[1], k_evidence, 2), -1, dtype=torch.long)
    evidence_scores = torch.zeros(evidence_indices.shape[:-1])
    for row in range(len(query["ids"])):
        candidate_ids = candidates[row]
        candidate_chunks = train["chunks"][candidate_ids]
        candidate_mask = train["mask"][candidate_ids]
        flat_chunks = candidate_chunks.reshape(-1, candidate_chunks.shape[-1])
        flat_mask = candidate_mask.reshape(-1)
        flat_contract_ids = candidate_ids.view(-1, 1).expand(-1, train["chunks"].shape[1]).reshape(-1)
        flat_chunk_ids = torch.arange(train["chunks"].shape[1]).view(1, -1).expand(len(candidate_ids), -1).reshape(-1)
        for label_id in range(train["labels"].shape[1]):
            label_query = torch.nn.functional.normalize(query_contract[row] + label_proto[label_id], dim=0)
            local_scores = flat_chunks @ label_query
            local_scores = local_scores.masked_fill(~flat_mask, -2.0)
            values, positions = torch.topk(local_scores, k=min(k_evidence, int(flat_mask.sum().item())))
            evidence_indices[row, label_id, : len(positions), 0] = flat_contract_ids[positions]
            evidence_indices[row, label_id, : len(positions), 1] = flat_chunk_ids[positions]
            evidence_scores[row, label_id, : len(values)] = values
    return {
        "contract_neighbors": neighbors,
        "contract_neighbor_scores": neighbor_scores,
        "evidence_indices": evidence_indices,
        "evidence_scores": evidence_scores,
        "label_prototypes": label_proto,
    }


def prepare(config):
    set_seed(int(config.get("seed", 42)))
    train = load_cache(config, "train")
    valid = load_cache(config, "valid")
    validate_cache_alignment(config, train, "train")
    validate_cache_alignment(config, valid, "valid")
    memory = {
        "schema": "main6_retrieval_memory_v1",
        "route": config["route_name"],
        "train_only": True,
        "ids": train["ids"],
        "contract": train["contract"].half(),
        "chunks": train["chunks"].half(),
        "chunk_mask": train["mask"],
        "labels": train["labels"],
    }
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    torch.save(memory, root / "retrieval_memory.pt")
    for split, data in (("train", train), ("valid", valid)):
        indices = build_retrieval_indices(config, train, data, split)
        torch.save({
            "schema": "main6_retrieval_queries_v1",
            "route": config["route_name"],
            "split": split,
            "train_memory_ids": train["ids"],
            "query_ids": data["ids"],
            "query_contract": data["contract"].half(),
            "query_chunk_counts": data["mask"].sum(1),
            "labels": data["labels"],
            **indices,
        }, root / f"{split}_queries.pt")
        print(f"[OK] wrote {root / (split + '_queries.pt')}")
    print(f"[OK] wrote {root / 'retrieval_memory.pt'}")


class M0(nn.Module):
    def __init__(self, dim, hidden, labels, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, labels))

    def forward(self, query, **_):
        return self.net(query)


class M1(nn.Module):
    def __init__(self, dim, hidden, labels, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, labels))

    def forward(self, query, contract_evidence=None, **_):
        return self.net(torch.cat([query, contract_evidence], dim=-1))


class M2(nn.Module):
    def __init__(self, dim, hidden, labels, dropout):
        super().__init__()
        self.label_embedding = nn.Parameter(torch.randn(labels, dim) * 0.02)
        self.net = nn.Sequential(nn.Linear(dim * 4, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, query, evidence, **_):
        q = query.unsqueeze(1).expand(-1, evidence.shape[1], -1)
        label = self.label_embedding.unsqueeze(0).expand(query.shape[0], -1, -1)
        x = torch.cat([q, evidence, q * evidence, label], dim=-1)
        return self.net(x).squeeze(-1)


def load_prepared(config, split):
    path = resolve(config["result_dir"]) / f"{split}_queries.pt"
    if not path.exists():
        raise FileNotFoundError(f"run prepare first: {path}")
    return torch.load(path, map_location="cpu")


def gather_features(memory, query_payload, device):
    chunks = memory["chunks"].float()
    contract = memory["contract"].float()
    neighbors = query_payload["contract_neighbors"]
    evidence_indices = query_payload["evidence_indices"]
    contract_evidence = contract[neighbors].mean(1)
    evidence = torch.zeros((len(evidence_indices), evidence_indices.shape[1], chunks.shape[-1]))
    for row in range(len(evidence_indices)):
        for label_id in range(evidence_indices.shape[1]):
            refs = evidence_indices[row, label_id]
            valid = refs[:, 0] >= 0
            if bool(valid.any()):
                evidence[row, label_id] = chunks[refs[valid, 0], refs[valid, 1]].mean(0)
    return query_payload["query_contract"].float().to(device), contract_evidence.to(device), evidence.to(device)


def metrics(labels, logits, thresholds=None):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype(int)
    threshold = 0.5 if thresholds is None else np.asarray(thresholds)
    result = compute_multilabel_metrics_from_probs(labels, probs, threshold)
    result["pr_auc_per_label"] = [float(average_precision_score(labels[:, i], probs[:, i])) for i in range(labels.shape[1])]
    result["pr_auc_macro"] = float(np.mean(result["pr_auc_per_label"]))
    return result, probs


def select_thresholds(labels, probs, candidates):
    values = []
    for i in range(labels.shape[1]):
        best = (0.5, -1.0)
        for threshold in candidates:
            pred = (probs[:, i] >= threshold).astype(int)
            tp = int(((pred == 1) & (labels[:, i] == 1)).sum())
            fp = int(((pred == 1) & (labels[:, i] == 0)).sum())
            fn = int(((pred == 0) & (labels[:, i] == 1)).sum())
            score = 2 * tp / max(1, 2 * tp + fp + fn)
            if score > best[1]:
                best = (float(threshold), float(score))
        values.append(best[0])
    return values


def train_one(config, name, model_cls, memory, train_payload, valid_payload, device):
    train_labels = train_payload["labels"].float()
    valid_labels = valid_payload["labels"].float()
    train_q, train_contract, train_evidence = gather_features(memory, train_payload, device)
    valid_q, valid_contract, valid_evidence = gather_features(memory, valid_payload, device)
    model = model_cls(int(config["feature_dim"]), int(config["hidden_dim"]), int(config["num_labels"]), float(config["dropout"])).to(device)
    pos = train_labels.sum(0).to(device)
    neg = train_labels.shape[0] - pos
    weight = torch.sqrt(neg / pos.clamp_min(1)).clamp(1.0, 5.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    train_ds = TensorDataset(train_q, train_contract, train_evidence, train_labels.to(device))
    loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    best = None
    patience = 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for query, contract_ev, evidence, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(query, contract_evidence=contract_ev, evidence=evidence)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_q, contract_evidence=valid_contract, evidence=valid_evidence)
        valid_metrics, valid_probs = metrics(valid_labels, valid_logits)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": float(loss_fn(valid_logits, valid_labels.to(device)).item()), "valid_fixed_macro_f1": valid_metrics["recognition_macro_f1"], "valid_fixed_micro_f1": valid_metrics["recognition_micro_f1"]}
        history.append(record)
        score = record["valid_fixed_macro_f1"]
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "probs": valid_probs}
            patience = 0
        else:
            patience += 1
        if patience >= int(config["early_stopping_patience"]):
            break
    model.load_state_dict(best["state"])
    thresholds = select_thresholds(valid_labels.numpy().astype(int), best["probs"], config["thresholds"])
    fixed, _ = metrics(valid_labels, torch.as_tensor(np.log(best["probs"] / np.clip(1 - best["probs"], 1e-7, 1.0))), None)
    tuned = compute_multilabel_metrics_from_probs(valid_labels.numpy(), best["probs"], thresholds)
    tuned["pr_auc_per_label"] = fixed["pr_auc_per_label"]
    tuned["pr_auc_macro"] = fixed["pr_auc_macro"]
    result = {"variant": name, "best_epoch": best["epoch"], "fixed_0.5": fixed, "tuned_valid": tuned, "thresholds": thresholds, "epoch_history": history, "valid_probs": best["probs"].tolist(), "valid_labels": valid_labels.tolist()}
    out = resolve(config["result_dir"]) / f"{name}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"schema": "main6_retrieval_model_v1", "variant": name, "state_dict": best["state"], "thresholds": thresholds}, resolve(config["result_dir"]) / f"{name}.pt")
    print(f"[{name}] epoch={best['epoch']} fixed_macro={fixed['recognition_macro_f1']:.6f} tuned_macro={tuned['recognition_macro_f1']:.6f}")
    return result


def analyze(config, results, valid_payload, train):
    labels = valid_payload["labels"].numpy().astype(int)
    length = valid_payload.get("query_chunk_counts")
    if length is None:
        length = train["valid_chunk_counts"] if "valid_chunk_counts" in train else None
    counts = np.asarray(length if length is not None else np.ones(len(labels)))
    order = np.argsort(counts)
    groups = {"short": order[: len(order) // 3], "medium": order[len(order) // 3 : 2 * len(order) // 3], "long": order[2 * len(order) // 3 :]}
    length_rows = []
    for group, indices in groups.items():
        row = {"group": group, "samples": int(len(indices)), "mean_chunks": float(counts[indices].mean())}
        for name, result in results.items():
            probs = np.asarray(result["valid_probs"])[indices]
            row[name] = compute_multilabel_metrics_from_probs(labels[indices], probs, result["thresholds"])["recognition_macro_f1"]
        length_rows.append(row)
    (resolve(config["result_dir"]) / "length_analysis.csv").write_text(
        "group,samples,mean_chunks," + ",".join(results) + "\n" + "\n".join(
            ",".join(str(row[key]) for key in ("group", "samples", "mean_chunks", *results)) for row in length_rows
        ) + "\n", encoding="utf-8"
    )
    per_label = []
    for label_id, label_name in enumerate(config["label_names"]):
        row = {"label": label_name, "support": int(labels[:, label_id].sum())}
        for name, result in results.items():
            row[name] = float(result["tuned_valid"]["per_label_f1"][label_id])
        per_label.append(row)
    (resolve(config["result_dir"]) / "per_label.csv").write_text(
        "label,support," + ",".join(results) + "\n" + "\n".join(
            ",".join(str(row[key]) for key in ("label", "support", *results)) for row in per_label
        ) + "\n", encoding="utf-8"
    )


def retrieval_quality(config, memory, train_payload, valid_payload):
    train_labels = memory["labels"].numpy().astype(int)
    valid_labels = valid_payload["labels"].numpy().astype(int)
    rows = []
    for name, payload, query_labels in (
        ("contract", valid_payload, valid_labels),
        ("evidence", valid_payload, valid_labels),
    ):
        if name == "contract":
            refs = payload["contract_neighbors"]
            for k in (1, 3, 5):
                selected = refs[:, : min(k, refs.shape[1])]
                purity = train_labels[selected.numpy()].mean(1)
                for label_id, label_name in enumerate(config["label_names"]):
                    rows.append({
                        "retrieval": name,
                        "top_k": k,
                        "label": label_name,
                        "query_positive_count": int(query_labels[:, label_id].sum()),
                        "label_purity": float(purity[:, label_id].mean()),
                        "positive_query_label_purity": float(purity[query_labels[:, label_id] == 1, label_id].mean()) if bool(query_labels[:, label_id].any()) else 0.0,
                    })
        else:
            refs = payload["evidence_indices"]
            for k in (1, 3, 5):
                selected = refs[:, :, : min(k, refs.shape[2]), 0]
                valid_refs = selected >= 0
                label_purity = np.zeros((len(query_labels), len(config["label_names"])), dtype=float)
                for row in range(len(query_labels)):
                    for label_id in range(len(config["label_names"])):
                        source = selected[row, label_id][valid_refs[row, label_id]].numpy()
                        label_purity[row, label_id] = float(train_labels[source, label_id].mean()) if len(source) else 0.0
                for label_id, label_name in enumerate(config["label_names"]):
                    rows.append({
                        "retrieval": name,
                        "top_k": k,
                        "label": label_name,
                        "query_positive_count": int(query_labels[:, label_id].sum()),
                        "label_purity": float(label_purity[:, label_id].mean()),
                        "positive_query_label_purity": float(label_purity[query_labels[:, label_id] == 1, label_id].mean()) if bool(query_labels[:, label_id].any()) else 0.0,
                    })
    columns = ["retrieval", "top_k", "label", "query_positive_count", "label_purity", "positive_query_label_purity"]
    output = resolve(config["result_dir"]) / "retrieval_quality.csv"
    output.write_text(",".join(columns) + "\n" + "\n".join(
        ",".join(str(row[column]) for column in columns) for row in rows
    ) + "\n", encoding="utf-8")


def train(config):
    root = resolve(config["result_dir"])
    memory = torch.load(root / "retrieval_memory.pt", map_location="cpu")
    train_payload = load_prepared(config, "train")
    valid_payload = load_prepared(config, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name, cls in (("m0_control", M0), ("m1_contract_retrieval", M1), ("m2_evidence_retrieval", M2)):
        results[name] = train_one(config, name, cls, memory, train_payload, valid_payload, device)
    baseline_path = resolve(config["baseline_summary"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {"missing": str(baseline_path)}
    (root / "baseline.json").write_text(json.dumps({"variant": "M0 historical opcode-only mlm8_slot3", "source": str(baseline_path), "summary": baseline, "control_model": results["m0_control"]}, indent=2), encoding="utf-8")
    (root / "contract_retrieval.json").write_text(json.dumps(results["m1_contract_retrieval"], indent=2), encoding="utf-8")
    (root / "evidence_retrieval.json").write_text(json.dumps(results["m2_evidence_retrieval"], indent=2), encoding="utf-8")
    analyze(config, results, valid_payload, {"valid_chunk_counts": torch.load(resolve(config["feature_dir"]) / "valid.pt", map_location="cpu")["chunk_mask"].sum(1).numpy()})
    retrieval_quality(config, memory, train_payload, valid_payload)
    (root / "retrieval_validation_summary.json").write_text(json.dumps({"route": config["route_name"], "test_checked": False, "models": {name: {"fixed_macro_f1": value["fixed_0.5"]["recognition_macro_f1"], "tuned_macro_f1": value["tuned_valid"]["recognition_macro_f1"], "tuned_micro_f1": value["tuned_valid"]["recognition_micro_f1"]} for name, value in results.items()}}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "prepare", "train"])
    parser.add_argument("--config", default="configs/retrieval_validation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "audit":
        audit(config)
    elif args.mode == "prepare":
        if not (resolve(config["result_dir"]) / "project_audit.json").exists():
            audit(config)
        prepare(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
