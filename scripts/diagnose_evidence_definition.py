"""Validation-only diagnosis of the definition of local vulnerability evidence.

The route intentionally stops before training a new evidence retriever.  It
uses train/valid feature caches and frozen or train-only M0 copies only.  Test
data and test artifacts are rejected.
"""

import argparse
import csv
import itertools
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

from evm_chunk_mil_model import MLM8ViewMultiSlotMIL  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs  # noqa: E402
from train_chunk_mil import load_config as load_mil_config  # noqa: E402


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_cache(config, split):
    if split not in ("train", "valid"):
        raise ValueError("this route accepts only train and valid")
    path = resolve(config["feature_dir"]) / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].float()
    mask = payload["chunk_mask"].bool()
    labels = payload["multi_labels"].float()
    if features.ndim != 4 or tuple(features.shape[2:]) != (8, int(config["feature_dim"])):
        raise ValueError(f"{split}: unexpected feature shape {tuple(features.shape)}")
    if mask.shape != features.shape[:2] or labels.shape != (features.shape[0], int(config["num_labels"])):
        raise ValueError(f"{split}: cache shape mismatch")
    if (~mask).all(dim=1).any() or not torch.isfinite(features).all():
        raise ValueError(f"{split}: invalid mask or NaN/Inf feature")
    return {"ids": [str(value) for value in payload["ids"]], "features": features, "mask": mask, "labels": labels}


def validate_alignment(config, data, split):
    expected = []
    with (resolve(config["data_dir"]) / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected.append(str(json.loads(line)["id"]))
    if expected != data["ids"]:
        raise ValueError(f"{split}: cache IDs are not aligned with JSONL")


def active_examples(data, max_per_class, seed):
    rng = np.random.default_rng(seed)
    width = data["mask"].shape[1]
    flat_active = data["mask"].reshape(-1).numpy()
    contract_rows = np.repeat(np.arange(len(data["ids"])), width)
    active_indices = np.flatnonzero(flat_active)
    inherited = data["labels"].numpy()[contract_rows[active_indices]]
    selected = []
    for label_id in range(inherited.shape[1]):
        for positive in (True, False):
            indices = active_indices[(inherited[:, label_id] > 0.5) == positive]
            take = min(len(indices), int(max_per_class))
            if take < len(indices):
                indices = rng.choice(indices, take, replace=False)
            selected.append(indices)
    indices = np.unique(np.concatenate(selected))
    return indices, contract_rows[indices]


def all_active_examples(data):
    width = data["mask"].shape[1]
    flat_active = data["mask"].reshape(-1).numpy()
    indices = np.flatnonzero(flat_active).astype(np.int64)
    contract_rows = np.repeat(np.arange(len(data["ids"])), width)
    return indices, contract_rows[indices]


class ChunkScoreMLP(nn.Module):
    def __init__(self, dim, hidden, labels):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, labels))

    def forward(self, x):
        return self.net(x)


def train_chunk_score(config, train, valid, device):
    train_idx, train_contract = active_examples(train, int(config["chunk_score_max_train_per_class"]), int(config["seed"]) + 11)
    # The diagnostic must produce a score for every real validation chunk so
    # correlation and Top-K overlap are not affected by evaluation sampling.
    valid_idx, valid_contract = all_active_examples(valid)
    train_x = train["features"][:, :, 1, :].reshape(-1, train["features"].shape[-1])[train_idx]
    valid_x = valid["features"][:, :, 1, :].reshape(-1, valid["features"].shape[-1])[valid_idx]
    train_y = train["labels"][train_contract]
    valid_y = valid["labels"][valid_contract]
    model = ChunkScoreMLP(int(config["feature_dim"]), int(config["chunk_score_hidden_dim"]), int(config["num_labels"])).to(device)
    pos = train_y.sum(0).to(device)
    neg = train_y.shape[0] - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(neg / pos.clamp_min(1)).clamp(1.0, 20.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["chunk_score_learning_rate"]), weight_decay=float(config["chunk_score_weight_decay"]))
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=int(config["chunk_score_batch_size"]), shuffle=True)
    history = []
    best_state, best_loss = None, float("inf")
    for epoch in range(1, int(config["chunk_score_epochs"]) + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_logits = []
            for left in range(0, len(valid_x), int(config["chunk_score_batch_size"])):
                val_logits.append(model(valid_x[left:left + int(config["chunk_score_batch_size"])].to(device)).cpu())
        val_logits = torch.cat(val_logits)
        val_loss = float(loss_fn(val_logits.to(device), valid_y.to(device)).item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": val_loss})
        print(f"[chunk_score] epoch={epoch} train_loss={history[-1]['train_loss']:.6f} valid_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = []
        for left in range(0, len(valid_x), int(config["chunk_score_batch_size"])):
            scores.append(torch.sigmoid(model(valid_x[left:left + int(config["chunk_score_batch_size"])].to(device)).cpu()))
    return {"scores": torch.cat(scores), "valid_indices": valid_idx, "valid_contracts": valid_contract, "history": history}


def build_m0(config, device):
    baseline_config = load_mil_config(resolve(config["baseline_config"]), config.get("baseline_variant", "mlm8_slot3"))
    return MLM8ViewMultiSlotMIL(baseline_config).to(device)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(resolve(path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError(f"checkpoint has no model state: {path}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def infer(model, features, masks, device, batch_size, attention=False):
    logits, weights = [], []
    with torch.no_grad():
        for left in range(0, len(features), int(batch_size)):
            right = min(left + int(batch_size), len(features))
            result = model(features[left:right].to(device), masks[left:right].to(device), return_attention=attention)
            logits.append(result["recognition_logits"].float().cpu())
            if attention:
                weights.append(result["chunk_attention"].float().cpu())
    output = {"logits": torch.cat(logits)}
    if attention:
        output["attention"] = torch.cat(weights)
    return output


def build_delete_batch(data, row, chunk_ids):
    chunk_ids = torch.as_tensor(chunk_ids, dtype=torch.long)
    masks = data["mask"][row:row + 1].expand(len(chunk_ids), -1).clone()
    for item, chunk_id in enumerate(chunk_ids.tolist()):
        masks[item, chunk_id] = False
    if (~masks).all(dim=1).any():
        raise ValueError("cannot delete the only valid chunk")
    features = data["features"][row:row + 1].expand(len(chunk_ids), -1, -1, -1).contiguous()
    return features, masks


def compute_influence(config, data, model, device):
    batch_size = int(config["influence_batch_size"])
    baseline = infer(model, data["features"], data["mask"], device, batch_size, True)
    base_probs = torch.sigmoid(baseline["logits"])
    influence = torch.full((len(data["ids"]), data["mask"].shape[1], int(config["num_labels"])), float("nan"))
    skipped = 0
    for row in range(len(data["ids"])):
        chunks = torch.where(data["mask"][row])[0]
        if len(chunks) <= 1:
            skipped += 1
            continue
        deleted = []
        for left in range(0, len(chunks), batch_size):
            current = chunks[left:left + batch_size]
            features, masks = build_delete_batch(data, row, current)
            deleted.append(torch.sigmoid(infer(model, features, masks, device, len(current))["logits"]))
        deleted = torch.cat(deleted)
        influence[row, chunks] = base_probs[row].unsqueeze(0) - deleted
        if (row + 1) % int(config["influence_progress_every"]) == 0:
            print(f"[influence] model_processed={row + 1}/{len(data['ids'])}", flush=True)
    return {"baseline_probs": base_probs, "baseline_attention": baseline["attention"], "influence": influence, "skipped_single_chunk": skipped}


def rank_indices(values, finite_mask, k, largest=True):
    valid = torch.where(finite_mask)[0].tolist()
    return sorted(valid, key=lambda i: float(values[i]), reverse=largest)[:min(int(k), len(valid))]


def jaccard(left, right):
    a, b = set(left), set(right)
    return float(len(a & b) / len(a | b)) if a or b else 1.0


def pairwise_stability(influences, data, config):
    rows = []
    for left, right in itertools.combinations(range(len(influences)), 2):
        for label_id, label_name in enumerate(config["label_names"]):
            for k in config["ranking_top_k_values"]:
                values = []
                for row in range(len(data["ids"])):
                    finite_left = torch.isfinite(influences[left]["influence"][row, :, label_id])
                    finite_right = torch.isfinite(influences[right]["influence"][row, :, label_id])
                    if not bool(finite_left.any() and finite_right.any()):
                        continue
                    a = rank_indices(influences[left]["influence"][row, :, label_id], finite_left, k)
                    b = rank_indices(influences[right]["influence"][row, :, label_id], finite_right, k)
                    values.append(jaccard(a, b))
                rows.append({"model_left": left, "model_right": right, "label": label_name, "top_k": int(k), "jaccard": float(np.mean(values)) if values else 0.0, "contracts": len(values)})
    return rows


def stability_summary(rows):
    overall = {}
    per_label = {}
    for top_k in sorted({int(row["top_k"]) for row in rows}):
        values = [float(row["jaccard"]) for row in rows if int(row["top_k"]) == top_k]
        overall[str(top_k)] = {"mean_jaccard": float(np.mean(values)) if values else 0.0, "comparisons": len(values)}
    for label in sorted({row["label"] for row in rows}):
        per_label[label] = {}
        for top_k in sorted({int(row["top_k"]) for row in rows}):
            values = [float(row["jaccard"]) for row in rows if row["label"] == label and int(row["top_k"]) == top_k]
            per_label[label][str(top_k)] = {"mean_jaccard": float(np.mean(values)) if values else 0.0, "comparisons": len(values)}
    return {"overall": overall, "per_label": per_label}


def pearson_spearman(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0, 0.0
    pearson = float(np.corrcoef(left, right)[0, 1])
    left_rank = np.argsort(np.argsort(left)).astype(float)
    right_rank = np.argsort(np.argsort(right)).astype(float)
    spearman = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return pearson, spearman


def signal_alignment(config, data, chunk_scores, influence):
    valid_indices = chunk_scores["valid_indices"]
    contract_rows = chunk_scores["valid_contracts"]
    score_lookup = {(int(contract), int(flat % data["mask"].shape[1])): chunk_scores["scores"][pos] for pos, (flat, contract) in enumerate(zip(valid_indices, contract_rows))}
    rows = []
    for label_id, label_name in enumerate(config["label_names"]):
        for k in config["ranking_top_k_values"]:
            pearsons, spearmans, overlaps = [], [], []
            for row in range(len(data["ids"])):
                finite = torch.isfinite(influence["influence"][row, :, label_id])
                chunks = torch.where(finite)[0].tolist()
                if len(chunks) < 2:
                    continue
                inf_values = [float(influence["influence"][row, c, label_id]) for c in chunks]
                label_values = [float(score_lookup.get((row, c), torch.tensor(0.0))[label_id]) for c in chunks]
                pearson, spearman = pearson_spearman(inf_values, label_values)
                influence_top = rank_indices(influence["influence"][row, :, label_id], finite, k)
                score_tensor = torch.tensor(label_values)
                signal_top = [chunks[i] for i in torch.argsort(score_tensor, descending=True)[:min(k, len(chunks))].tolist()]
                pearsons.append(pearson)
                spearmans.append(spearman)
                overlaps.append(jaccard(influence_top, signal_top))
            rows.append({"label": label_name, "top_k": int(k), "pearson": float(np.mean(pearsons)) if pearsons else 0.0, "spearman": float(np.mean(spearmans)) if spearmans else 0.0, "top_k_jaccard": float(np.mean(overlaps)) if overlaps else 0.0, "contracts": len(pearsons)})
    return rows


def contiguous_spans(values, finite_mask, span_count, seed_k):
    """Merge adjacent high-influence chunks into ranked local evidence spans."""
    valid = torch.where(finite_mask)[0].tolist()
    if not valid:
        return []
    seeds = rank_indices(values, finite_mask, max(int(seed_k), int(span_count)))
    runs = []
    for seed in sorted(seeds):
        if not runs or seed != runs[-1][-1] + 1:
            runs.append([seed])
        else:
            runs[-1].append(seed)
    candidates = sorted(
        ((float(values[run].sum()), -run[0], run) for run in runs),
        reverse=True,
    )
    chosen = []
    occupied = set()
    for _, _, window in candidates:
        if occupied.intersection(window):
            continue
        chosen.append(window)
        occupied.update(window)
        if len(chosen) >= int(span_count):
            break
    if not chosen:
        return [[index] for index in seeds[:int(span_count)]]
    return chosen


def spatial_rows(config, data, influence):
    rows = []
    for label_id, label_name in enumerate(config["label_names"]):
        positive_rows = torch.where(data["labels"][:, label_id] > 0.5)[0].tolist()
        for k in config["ranking_top_k_values"]:
            distances, runs, run_lengths, densities = [], [], [], []
            for row in positive_rows:
                local = influence["influence"][row, :, label_id]
                finite = torch.isfinite(local)
                selected = sorted(rank_indices(local, finite, k))
                if not selected:
                    continue
                distances.extend([selected[i + 1] - selected[i] for i in range(len(selected) - 1)])
                current_runs = 1
                lengths = []
                for index in range(1, len(selected)):
                    if selected[index] == selected[index - 1] + 1:
                        current_runs += 1
                    else:
                        lengths.append(current_runs)
                        current_runs = 1
                lengths.append(current_runs)
                runs.append(len(lengths))
                run_lengths.append(float(np.mean(lengths)))
                densities.append(float(len(selected) / max(1, int(finite.sum()))))
            rows.append({"label": label_name, "top_k": int(k), "mean_adjacent_distance": float(np.mean(distances)) if distances else 0.0, "mean_contiguous_runs": float(np.mean(runs)) if runs else 0.0, "mean_run_length": float(np.mean(run_lengths)) if run_lengths else 0.0, "mean_evidence_density": float(np.mean(densities)) if densities else 0.0, "positive_contracts": len(runs)})
    return rows


def specificity_rows(config, data, influence):
    rows = []
    for left, right in itertools.combinations(range(int(config["num_labels"])), 2):
        both = torch.where((data["labels"][:, left] > 0.5) & (data["labels"][:, right] > 0.5))[0].tolist()
        for k in (5, 10):
            values = []
            for row in both:
                left_values = influence["influence"][row, :, left]
                right_values = influence["influence"][row, :, right]
                left_top = rank_indices(left_values, torch.isfinite(left_values), k)
                right_top = rank_indices(right_values, torch.isfinite(right_values), k)
                values.append(jaccard(left_top, right_top))
            rows.append({"label_left": config["label_names"][left], "label_right": config["label_names"][right], "top_k": k, "jaccard": float(np.mean(values)) if values else 0.0, "contracts_with_both_labels": len(values)})
    return rows


def isolated_classification(config, data, influence, attention, model, device):
    rng = random.Random(int(config["phase4_random_seed"]))
    results = []
    baseline = infer(model, data["features"], data["mask"], device, int(config["phase4_batch_size"]), False)
    base_metric = compute_multilabel_metrics_from_probs(data["labels"].numpy(), torch.sigmoid(baseline["logits"]).numpy(), 0.5)
    results.append({"method": "full_contract_m0", "top_k": "all", "macro_f1": float(base_metric["recognition_macro_f1"]), "micro_f1": float(base_metric["recognition_micro_f1"]), "per_label_f1": [float(x) for x in base_metric["per_label_f1"]]})
    random_choices = {}
    for row in range(len(data["ids"])):
        values = torch.where(data["mask"][row])[0].tolist()
        rng.shuffle(values)
        random_choices[row] = values
    for method in ("random", "attention", "influence"):
        for k in config["ranking_top_k_values"]:
            joined = torch.zeros((len(data["ids"]), int(config["num_labels"])))
            for label_id in range(int(config["num_labels"])):
                masks = torch.zeros_like(data["mask"])
                for row in range(len(data["ids"])):
                    if method == "random":
                        selected = random_choices[row][:min(int(k), len(random_choices[row]))]
                    else:
                        values = attention[row, :, label_id] if method == "attention" else influence["influence"][row, :, label_id]
                        selected = rank_indices(values, data["mask"][row] & torch.isfinite(values), k)
                    if selected:
                        masks[row, selected] = True
                output = infer(model, data["features"], masks, device, int(config["phase4_batch_size"]), False)
                joined[:, label_id] = output["logits"][:, label_id]
            metric = compute_multilabel_metrics_from_probs(data["labels"].numpy(), torch.sigmoid(joined).numpy(), 0.5)
            results.append({"method": method, "top_k": int(k), "macro_f1": float(metric["recognition_macro_f1"]), "micro_f1": float(metric["recognition_micro_f1"]), "per_label_f1": [float(x) for x in metric["per_label_f1"]]})
            print(f"[isolation] method={method} top_k={k} macro_f1={metric['recognition_macro_f1']:.6f}", flush=True)
    return results


def span_classification(config, data, influence, model, device):
    results = []
    selections = []
    seed_k = int(config["contiguous_span_seed_k"])
    for span_count in config["contiguous_span_count_values"]:
        joined = torch.zeros((len(data["ids"]), int(config["num_labels"])))
        for label_id in range(int(config["num_labels"])):
            masks = torch.zeros_like(data["mask"])
            for row in range(len(data["ids"])):
                local = influence["influence"][row, :, label_id]
                spans = contiguous_spans(local, data["mask"][row] & torch.isfinite(local), int(span_count), seed_k)
                selected = [chunk for span in spans for chunk in span]
                selections.append({
                    "row": row,
                    "label": config["label_names"][label_id],
                    "span_count": int(span_count),
                    "spans": spans,
                    "span_lengths": [len(span) for span in spans],
                })
                if selected:
                    masks[row, selected] = True
            output = infer(model, data["features"], masks, device, int(config["phase4_batch_size"]), False)
            joined[:, label_id] = output["logits"][:, label_id]
        metric = compute_multilabel_metrics_from_probs(data["labels"].numpy(), torch.sigmoid(joined).numpy(), 0.5)
        results.append({"method": "contiguous_span", "span_seed_k": seed_k, "span_count": int(span_count), "macro_f1": float(metric["recognition_macro_f1"]), "micro_f1": float(metric["recognition_micro_f1"]), "per_label_f1": [float(x) for x in metric["per_label_f1"]]})
        print(f"[span] seed_k={seed_k} count={span_count} macro_f1={metric['recognition_macro_f1']:.6f}", flush=True)
    return {"results": results, "selections": selections, "span_seed_k": seed_k}


def train_m0_copy(config, train, device, seed, output_path):
    set_seed(seed)
    model = build_m0(config, device)
    model.train()
    labels = train["labels"].to(device)
    pos = labels.sum(0)
    neg = labels.shape[0] - pos
    model.set_recognition_pos_weight((neg / pos.clamp_min(1)).clamp(1.0, 5.0))
    loader = DataLoader(TensorDataset(train["features"], train["mask"], train["labels"]), batch_size=int(config["stability_train_batch_size"]), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["stability_train_learning_rate"]), weight_decay=float(config["stability_train_weight_decay"]))
    for epoch in range(1, int(config["stability_train_epochs"]) + 1):
        losses = []
        for features, masks, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(features.to(device), masks.to(device), multi_labels=targets.to(device))
            loss = output.get("loss")
            # Some historical M0 implementations serialize the aggregate loss
            # as a Python scalar. Rebuild it from the component losses so the
            # automatically trained stability copy still has a grad graph.
            if not torch.is_tensor(loss) or not loss.requires_grad:
                recognition_loss = output.get("recognition_loss")
                detection_loss = output.get("detection_loss")
                if recognition_loss is None:
                    raise RuntimeError(
                        f"M0 copy returned no differentiable recognition loss at seed={seed}, epoch={epoch}"
                    )
                loss = model.compute_weighted_task_loss(
                    detection_loss,
                    recognition_loss,
                )
            loss = loss.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite M0 copy loss at seed={seed}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["stability_train_max_grad_norm"]))
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"[m0_copy] seed={seed} epoch={epoch} train_loss={np.mean(losses):.6f}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "seed": seed, "train_only": True}, output_path)
    return output_path


def resolve_stability_checkpoints(config, train, device, root):
    paths = []
    for value in config.get("stability_checkpoints", []):
        path = resolve(value)
        if path.exists() and path not in paths:
            paths.append(path)
    if not paths:
        raise FileNotFoundError("no same-architecture M0 checkpoint found")
    if len(paths) < 3 and bool(config.get("stability_auto_train", True)):
        for seed in config.get("stability_train_seeds", []):
            target = root / "m0_copies" / f"seed_{int(seed)}.pt"
            if not target.exists():
                train_m0_copy(config, train, device, int(seed), target)
            paths.append(target)
            if len(paths) >= 3:
                break
    if len(paths) < 3:
        raise RuntimeError("need at least three same-architecture M0 checkpoints")
    return paths[:max(3, len(paths))]


def report_case(config, stability_rows, alignment_rows, isolation_rows, specificity):
    top5_stability = [row["jaccard"] for row in stability_rows if row["top_k"] == 5]
    influence = {row["top_k"]: row["macro_f1"] for row in isolation_rows if row["method"] == "influence"}
    attention = {row["top_k"]: row["macro_f1"] for row in isolation_rows if row["method"] == "attention"}
    random_rows = {row["top_k"]: row["macro_f1"] for row in isolation_rows if row["method"] == "random"}
    margins = {str(k): float(influence[k] - attention[k]) for k in influence if k in attention}
    pair_top5 = [row["jaccard"] for row in specificity if row["top_k"] == 5 and row["contracts_with_both_labels"] > 0]
    alignment_spearman = float(np.mean([row["spearman"] for row in alignment_rows])) if alignment_rows else 0.0
    stable = bool(top5_stability) and float(np.mean(top5_stability)) >= float(config["go_min_top5_stability_jaccard"])
    influence_beats_attention = bool(margins) and max(margins.values()) >= float(config["go_min_influence_attention_macro_margin"])
    label_specific = not pair_top5 or float(np.mean(pair_top5)) <= float(config["go_max_label_pair_top5_jaccard"])
    go = stable and influence_beats_attention and label_specific
    return {
        "decision": "GO" if go else "NO-GO",
        "conclusion": "Evidence definition is sufficiently stable and label-specific for a minimal weakly-supervised retriever." if go else "Do not enter the weakly-supervised retriever yet; revise the evidence definition or collect stronger evidence supervision.",
        "criteria": {
            "mean_top5_stability_jaccard": float(np.mean(top5_stability)) if top5_stability else 0.0,
            "influence_minus_attention_macro_f1": margins,
            "mean_influence_minus_attention_macro_f1": float(np.mean(list(margins.values()))) if margins else 0.0,
            "mean_label_pair_top5_jaccard": float(np.mean(pair_top5)) if pair_top5 else None,
            "mean_influence_chunk_signal_spearman": alignment_spearman,
            "stability_pass": stable,
            "influence_attention_pass": influence_beats_attention,
            "label_specificity_pass": label_specific,
        },
        "thresholds": {
            "min_top5_stability_jaccard": float(config["go_min_top5_stability_jaccard"]),
            "min_influence_attention_macro_margin": float(config["go_min_influence_attention_macro_margin"]),
            "max_label_pair_top5_jaccard": float(config["go_max_label_pair_top5_jaccard"]),
        },
        "protocol_warning": "Influence isolation and label-conditioned selection use validation labels by design; they are model-decision sensitivity diagnostics, not ground-truth localization or final model performance.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evidence_definition_diagnosis.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if bool(config.get("allow_test", False)) or bool(config.get("allow_test_cache", False)):
        raise RuntimeError("test access is forbidden")
    set_seed(int(config["seed"]))
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    train, valid = load_cache(config, "train"), load_cache(config, "valid")
    validate_alignment(config, train, "train")
    validate_alignment(config, valid, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} train={len(train['ids'])} valid={len(valid['ids'])}", flush=True)

    chunk_score = train_chunk_score(config, train, valid, device)
    model_paths = resolve_stability_checkpoints(config, train, device, root)
    model_reports, influences = [], []
    for index, path in enumerate(model_paths):
        model = load_checkpoint(build_m0(config, device), path, device)
        report = compute_influence(config, valid, model, device)
        model_reports.append({"model_index": index, "checkpoint": str(path), "baseline_macro_f1": float(compute_multilabel_metrics_from_probs(valid["labels"].numpy(), report["baseline_probs"].numpy(), 0.5)["recognition_macro_f1"]), "skipped_single_chunk": int(report["skipped_single_chunk"])})
        influences.append(report)
        torch.save({"checkpoint": str(path), "baseline_probs": report["baseline_probs"], "baseline_attention": report["baseline_attention"], "influence": report["influence"], "test_checked": False}, root / f"influence_model_{index}.pt")
    stability = pairwise_stability(influences, valid, config)
    write_csv(root / "influence_stability.csv", stability, list(stability[0]))
    write_json(root / "influence_stability.json", {"models": model_reports, "rows": stability, "summary": stability_summary(stability), "test_checked": False})

    alignment = signal_alignment(config, valid, chunk_score, influences[0])
    write_csv(root / "influence_chunk_signal_alignment.csv", alignment, list(alignment[0]))
    spatial = spatial_rows(config, valid, influences[0])
    write_csv(root / "influence_spatial_distribution.csv", spatial, list(spatial[0]))
    specificity = specificity_rows(config, valid, influences[0])
    write_csv(root / "influence_label_specificity.csv", specificity, list(specificity[0]))

    model = load_checkpoint(build_m0(config, device), model_paths[0], device)
    isolation = isolated_classification(config, valid, influences[0], influences[0]["baseline_attention"], model, device)
    spans = span_classification(config, valid, influences[0], model, device)
    write_json(root / "influence_attention_random_isolation.json", {"results": isolation, "test_checked": False})
    write_json(root / "individual_vs_contiguous_span_isolation.json", {"individual_and_baseline": isolation, "contiguous_spans": spans, "test_checked": False})
    isolation_rows = []
    for result in isolation:
        if not isinstance(result.get("top_k"), int):
            continue
        for label_id, label_name in enumerate(config["label_names"]):
            isolation_rows.append({"method": result["method"], "top_k": result["top_k"], "label": label_name, "f1": result["per_label_f1"][label_id], "macro_f1": result["macro_f1"], "micro_f1": result["micro_f1"]})
    write_csv(root / "influence_attention_random_per_label.csv", isolation_rows, list(isolation_rows[0]))
    span_rows = []
    for result in spans["results"]:
        for label_id, label_name in enumerate(config["label_names"]):
            span_rows.append({"span_count": result["span_count"], "span_seed_k": result["span_seed_k"], "label": label_name, "f1": result["per_label_f1"][label_id], "macro_f1": result["macro_f1"], "micro_f1": result["micro_f1"]})
    write_csv(root / "contiguous_span_per_label.csv", span_rows, list(span_rows[0]))

    final = report_case(config, stability, alignment, isolation, specificity)
    report = {
        "route": config["route_name"],
        "dataset": "DIVE Main6 random split",
        "seed": int(config["seed"]),
        "train_only_memory": True,
        "test_checked": False,
        "phases_completed": ["influence_stability", "chunk_signal_alignment", "single_chunk_vs_span", "spatial_distribution", "label_specificity", "influence_attention_random"],
        "phase5_started": False,
        "m0_models": model_reports,
        "decision": final,
        "interpretation": {
            "stability": "Jaccard compares Top-K influence chunk sets across same-architecture M0 models.",
            "alignment": "Chunk labels are inherited contract labels and remain noisy proxies, not evidence ground truth.",
            "spans": "Top influence seed chunks are merged when adjacent; the highest-summed non-overlapping runs are selected as Top-1/3/5 spans. Span lengths are recorded in the span selection artifact. Chunk order is preserved; retrospective full-contract influence selection is used only for this offline diagnosis, not online future prediction.",
            "specificity": "Label-pair overlap is computed only on validation contracts carrying both labels.",
            "isolation": "Per-label validation-label-conditioned masks are joined into a diagnostic multi-label prediction; this is not deployable retrieval evaluation.",
        },
        "artifacts": {
            "stability": "influence_stability.csv",
            "alignment": "influence_chunk_signal_alignment.csv",
            "spatial": "influence_spatial_distribution.csv",
            "specificity": "influence_label_specificity.csv",
            "isolation": "influence_attention_random_isolation.json",
            "isolation_per_label": "influence_attention_random_per_label.csv",
            "span": "individual_vs_contiguous_span_isolation.json",
            "span_per_label": "contiguous_span_per_label.csv",
        },
        "warnings": [
            "No test data, test cache, or test prediction was read or created.",
            "No evidence scorer was modified or trained.",
            "Influence is model-decision sensitivity, not causal ground-truth localization.",
            "Phase 4/6 use validation labels for the requested controlled diagnostic and must not be reported as final validation performance.",
        ],
    }
    write_json(root / "evidence_definition_diagnosis.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
