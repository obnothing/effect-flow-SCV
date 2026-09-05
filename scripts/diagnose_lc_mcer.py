"""Validation-only diagnosis of the LC-MCER residual/evidence path.

This route deliberately does not use influence pseudo-labels, test data, or a
new evidence supervision objective. Oracle evidence is an upper-bound selector
only: it uses the query label to choose local chunks and is never a final model.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evm_chunk_mil_model import MLM8ViewMultiSlotMIL  # noqa: E402
from lc_mcer import CRER, LCMCER  # noqa: E402
from metrics import (  # noqa: E402
    compute_multilabel_metrics_from_probs,
    select_per_label_thresholds,
)
from train_chunk_mil import load_config as load_mil_config  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def json_safe(value):
    """Convert diagnostic values to JSON without serializing model tensors."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_config(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(config, split):
    if split not in ("train", "valid"):
        raise ValueError("LC-MCER diagnosis is train/valid only")
    payload = torch.load(resolve(config["feature_dir"]) / f"{split}.pt", map_location="cpu")
    features = payload["features"].float()
    mask = payload["chunk_mask"].bool()
    labels = payload["multi_labels"].float()
    if features.ndim != 4 or tuple(features.shape[2:]) != (8, int(config["feature_dim"])):
        raise ValueError(f"{split}: unexpected feature shape {tuple(features.shape)}")
    if mask.shape != features.shape[:2] or labels.shape != (features.shape[0], int(config["num_labels"])):
        raise ValueError(f"{split}: feature/mask/label alignment error")
    if (~mask).all(dim=1).any() or not torch.isfinite(features).all():
        raise ValueError(f"{split}: empty sample or NaN/Inf")
    return {"features": features, "mask": mask, "labels": labels, "ids": payload.get("ids", [])}


def load_m0(config, device):
    model_config = load_mil_config(resolve(config["baseline_config"]), config["baseline_variant"])
    model = MLM8ViewMultiSlotMIL(model_config).to(device)
    checkpoint = torch.load(resolve(config["baseline_checkpoint"]), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("baseline checkpoint has no model state")
    model.load_state_dict(state, strict=True)
    return model.eval()


def pos_weight(labels, config):
    positive = labels.sum(dim=0)
    negative = labels.shape[0] - positive
    if config.get("pos_weight_mode", "sqrt_ratio") == "ratio":
        value = negative / positive.clamp_min(1.0)
    else:
        value = torch.sqrt(negative / positive.clamp_min(1.0))
    return value.clamp(1.0, float(config.get("max_pos_weight", 5.0)))


def metric_pack(config, labels, logits):
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    fixed = compute_multilabel_metrics_from_probs(labels_np, probs, 0.5)
    selection = select_per_label_thresholds(
        labels_np, probs, config["thresholds"], config["label_names"], global_threshold=0.2
    )
    tuned = compute_multilabel_metrics_from_probs(labels_np, probs, selection["thresholds"])
    return {
        "fixed_macro_f1": float(fixed["recognition_macro_f1"]),
        "fixed_micro_f1": float(fixed["recognition_micro_f1"]),
        "tuned_macro_f1": float(tuned["recognition_macro_f1"]),
        "tuned_micro_f1": float(tuned["recognition_micro_f1"]),
        "fixed_per_label_f1": [float(x) for x in fixed["per_label_f1"]],
        "per_label_f1": [float(x) for x in tuned["per_label_f1"]],
        "per_label_precision": [float(x) for x in tuned["per_label_precision"]],
        "per_label_recall": [float(x) for x in tuned["per_label_recall"]],
        "thresholds": [float(x) for x in selection["thresholds"]],
    }


def build_model(
    config,
    device,
    *,
    scale=None,
    top_k=None,
    use_competition=None,
    diversity_lambda=None,
    simple=False,
    explicit_logit=False,
):
    base = load_m0(config, device)
    if simple:
        use_competition = False
        diversity_lambda = 0.0
    if use_competition is None:
        use_competition = True
    if diversity_lambda is None:
        diversity_lambda = float(config["diversity_lambda"])
    return LCMCER(
        base,
        feature_dim=int(config["feature_dim"]),
        num_labels=int(config["num_labels"]),
        retrieval_dim=int(config["retrieval_dim"]),
        top_k=int(config["top_k"] if top_k is None else top_k),
        diversity_lambda=float(diversity_lambda),
        use_competition=bool(use_competition),
        residual_scale=float(config["residual_scale"] if scale is None else scale),
        residual_hidden_dim=int(config["residual_hidden_dim"]),
        dropout=float(config["dropout"]),
        include_base_logit=explicit_logit,
    ).to(device)


def random_evidence(data, config, seed):
    """Fixed random evidence with the same [N, label, feature] interface."""
    generator = torch.Generator().manual_seed(int(seed))
    chunks = data["features"][:, :, 1, :]
    result = torch.zeros(
        len(chunks), int(config["num_labels"]), chunks.shape[-1], dtype=chunks.dtype
    )
    top_k = int(config["top_k"])
    for row in range(len(chunks)):
        active = torch.where(data["mask"][row])[0].tolist()
        take = min(top_k, len(active))
        if not take:
            continue
        for label_id in range(int(config["num_labels"])):
            order = torch.randperm(len(active), generator=generator)[:take].tolist()
            indices = torch.tensor([active[index] for index in order], dtype=torch.long)
            result[row, label_id] = chunks[row, indices].mean(dim=0)
    return result


def parameter_group_norms(model):
    groups = {
        "query_projection": [],
        "chunk_projection": [],
        "label_embedding": [],
        "residual_classifier": [],
    }
    for name, parameter in model.named_parameters():
        for group in groups:
            if name.startswith(group):
                groups[group].append(parameter)
    result = {}
    for group, parameters in groups.items():
        values = [parameter.detach().float().norm() ** 2 for parameter in parameters]
        result[group] = float(torch.stack(values).sum().sqrt()) if values else 0.0
    return result


def parameter_group_update_norms(model, initial_parameters):
    groups = {
        "query_projection": [],
        "chunk_projection": [],
        "label_embedding": [],
        "residual_classifier": [],
    }
    for name, parameter in model.named_parameters():
        if name not in initial_parameters:
            continue
        for group in groups:
            if name.startswith(group):
                groups[group].append((parameter.detach().float() - initial_parameters[name]).norm() ** 2)
    return {
        group: float(torch.stack(values).sum().sqrt()) if values else 0.0
        for group, values in groups.items()
    }


def parameter_group_gradient_norms(model):
    groups = {
        "query_projection": [],
        "chunk_projection": [],
        "label_embedding": [],
        "residual_classifier": [],
    }
    for name, parameter in model.named_parameters():
        for group in groups:
            if name.startswith(group) and parameter.grad is not None:
                groups[group].append(parameter.grad.detach().float().norm() ** 2)
    return {
        group: float(torch.stack(values).sum().sqrt()) if values else 0.0
        for group, values in groups.items()
    }


def oracle_evidence(train, payload, config):
    """Create a label-gated upper-bound evidence representation.

    The local chunk score is based on train-only positive-chunk centroids. The
    query label gates which label receives evidence, so this is intentionally an
    oracle diagnostic and not valid retrieval supervision.
    """
    train_chunks = train["features"][:, :, 1, :]
    train_mask = train["mask"]
    train_labels = train["labels"].bool()
    centroids = []
    all_active = train_chunks[train_mask]
    fallback = all_active.mean(0)
    for label_id in range(int(config["num_labels"])):
        active = train_chunks[train_mask & train_labels[:, None, label_id]]
        centroids.append(active.mean(0) if len(active) else fallback)
    centroids = F.normalize(torch.stack(centroids), dim=-1)
    chunks = payload["features"][:, :, 1, :]
    norm_chunks = F.normalize(chunks, dim=-1)
    scores = torch.einsum("ncd,ld->ncl", norm_chunks, centroids)
    scores = scores.masked_fill(~payload["mask"].unsqueeze(-1), -1e9)
    k = min(int(config.get("oracle_top_k", 5)), chunks.shape[1])
    result = torch.zeros((len(chunks), int(config["num_labels"]), chunks.shape[-1]))
    for row in range(len(chunks)):
        for label_id in range(int(config["num_labels"])):
            if not bool(payload["labels"][row, label_id]):
                continue
            values, indices = torch.topk(scores[row, :, label_id], k=k)
            valid = values > -1e8
            values, indices = values[valid], indices[valid]
            if len(indices):
                weights = torch.softmax(values, dim=0)
                result[row, label_id] = (chunks[row, indices] * weights[:, None]).sum(0)
    return result


def forward_batches(model, data, device, batch_size, override=None, return_diag=False):
    model.eval()
    loader = DataLoader(
        TensorDataset(data["features"], data["mask"]), batch_size=int(batch_size), shuffle=False
    )
    logits, residuals, bases, evidences, ranked, ranked_scores, competitions = [], [], [], [], [], [], []
    with torch.no_grad():
        for start, (features, masks) in enumerate(loader):
            left = start * int(batch_size)
            right = left + len(features)
            kwargs = {"return_diagnostics": return_diag}
            if override is not None:
                kwargs["evidence_override"] = override[left:right].to(device)
            output = model(features.to(device), masks.to(device), **kwargs)
            logits.append(output["recognition_logits"].float().cpu())
            residuals.append(output["residual_logits"].float().cpu())
            bases.append(output["base_logits"].float().cpu())
            evidences.append(output["evidence_representation"].float().cpu())
            if return_diag:
                ranked.append(output["ranked_indices"].cpu())
                ranked_scores.append(output["ranked_scores"].float().cpu())
                competitions.append(output["competition_scores"].float().cpu())
    result = {
        "logits": torch.cat(logits),
        "residual": torch.cat(residuals),
        "base": torch.cat(bases),
        "evidence": torch.cat(evidences),
    }
    if return_diag:
        result["ranked_indices"] = torch.cat(ranked)
        result["ranked_scores"] = torch.cat(ranked_scores)
        result["competition_scores"] = torch.cat(competitions)
    return result


def baseline_batches(model, data, device, batch_size):
    loader = DataLoader(
        TensorDataset(data["features"], data["mask"]),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    logits = []
    with torch.no_grad():
        for features, masks in loader:
            output = model(features.to(device), masks.to(device))
            logits.append(output["recognition_logits"].float().cpu())
    return torch.cat(logits)


def train_model(
    config,
    train,
    valid,
    device,
    name,
    *,
    scale=None,
    top_k=None,
    use_competition=None,
    diversity_lambda=None,
    simple=False,
    explicit_logit=False,
    train_override=None,
    valid_override=None,
):
    set_seed(int(config["seed"]))
    model = build_model(
        config,
        device,
        scale=scale,
        top_k=top_k,
        use_competition=use_competition,
        diversity_lambda=diversity_lambda,
        simple=simple,
        explicit_logit=explicit_logit,
    )
    weight = pos_weight(train["labels"], config).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(config.get("fp16", True)))
    if train_override is None:
        values = TensorDataset(train["features"], train["mask"], train["labels"])
    else:
        values = TensorDataset(train["features"], train["mask"], train["labels"], train_override)
    loader = DataLoader(values, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = []
    initial_norms = parameter_group_norms(model)
    initial_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if any(name.startswith(group) for group in ("query_projection", "chunk_projection", "label_embedding", "residual_classifier"))
    }
    gradient_history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            features, masks, labels = batch[:3]
            override = batch[3] if train_override is not None else None
            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and bool(config.get("fp16", True))):
                kwargs = {} if override is None else {"evidence_override": override.to(device)}
                output = model(features.to(device), masks.to(device), **kwargs)
                loss = F.binary_cross_entropy_with_logits(output["recognition_logits"], labels.to(device), pos_weight=weight)
            scaled = loss / int(config["gradient_accumulation_steps"])
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
                if step % int(config["gradient_accumulation_steps"]) == 0:
                    scaler.unscale_(optimizer)
                    gradient_history.append(parameter_group_gradient_norms(model))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            else:
                scaled.backward()
                if step % int(config["gradient_accumulation_steps"]) == 0:
                    gradient_history.append(parameter_group_gradient_norms(model))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if len(loader) % int(config["gradient_accumulation_steps"]):
            if scaler.is_enabled():
                scaler.unscale_(optimizer); gradient_history.append(parameter_group_gradient_norms(model)); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])); optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        valid_out = forward_batches(model, valid, device, config["eval_batch_size"], valid_override)
        metric = metric_pack(config, valid["labels"], valid_out["logits"])
        valid_loss = F.binary_cross_entropy_with_logits(
            valid_out["logits"].to(device), valid["labels"].to(device), pos_weight=weight
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "valid_loss": float(valid_loss.detach().cpu()),
            **metric,
        }
        history.append(row)
        print(
            f"[{name}] epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"valid_loss={row['valid_loss']:.6f} "
            f"tuned_macro={row['tuned_macro_f1']:.6f}",
            flush=True,
        )
        if row["tuned_macro_f1"] > best_score:
            best_score, best_epoch, stale = row["tuned_macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    model.load_state_dict(best_state, strict=True)
    valid_out = forward_batches(model, valid, device, config["eval_batch_size"], valid_override, return_diag=True)
    final_norms = parameter_group_norms(model)
    update_norms = parameter_group_update_norms(model, {
        name: value.to(device)
        for name, value in initial_parameters.items()
    })
    gradient_summary = {
        "initial_parameter_norm": initial_norms,
        "final_parameter_norm": final_norms,
        "parameter_norm_change_abs": update_norms,
        "mean_gradient_norm": {
            group: float(np.mean([row[group] for row in gradient_history]))
            if gradient_history else 0.0
            for group in initial_norms
        },
        "final_gradient_norm": gradient_history[-1] if gradient_history else {group: 0.0 for group in initial_norms},
        "nonzero_gradient_step_fraction": {
            group: float(np.mean([row[group] > 1e-12 for row in gradient_history]))
            if gradient_history else 0.0
            for group in initial_norms
        },
    }
    result = {"name": name, "seed": int(config["seed"]), "best_epoch": int(best_epoch), "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "frozen_parameter_count": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)), **metric_pack(config, valid["labels"], valid_out["logits"]), "history": history, "valid_output": valid_out, "model": model, "gradient_diagnostics": gradient_summary}
    return result


def residual_stats(config, valid, output, scale):
    residual = output["residual"].numpy()
    base = output["base"].numpy()
    labels = valid["labels"].numpy().astype(int)
    error = ((base >= 0).astype(int) != labels).astype(float)
    rows = []
    for label_id, name in enumerate(config["label_names"]):
        r, b, e = residual[:, label_id], base[:, label_id], error[:, label_id]
        def corr(x, y):
            return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0
        rows.append({"label": name, "mean_residual": float(r.mean()), "residual_std": float(r.std()), "mean_abs_residual": float(np.abs(r).mean()), "mean_abs_scaled_residual": float(np.abs(scale * r).mean()), "positive_abs_residual": float(np.abs(r[labels[:, label_id] == 1]).mean()), "negative_abs_residual": float(np.abs(r[labels[:, label_id] == 0]).mean()), "corr_m0_logit_residual": corr(b, r), "corr_abs_residual_classification_error": corr(np.abs(r), e)})
    return {"per_label": rows, "mean_abs_residual": float(np.abs(residual).mean()), "residual_std": float(residual.std()), "mean_abs_scaled_residual": float(np.abs(scale * residual).mean())}


def eval_override(config, model, valid, device, override, name):
    out = forward_batches(model, valid, device, config["eval_batch_size"], override)
    return {"name": name, **metric_pack(config, valid["labels"], out["logits"]), "residual_stats": residual_stats(config, valid, out, model.residual_scale)}


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def finalize_conclusions(report):
    m0 = float(report["m0"]["tuned_macro_f1"])
    experiments = {item["name"]: item for item in report["experiments"]}
    scales = [experiments[f"scale_{value}"]["tuned_macro_f1"] for value in (0.1, 0.25, 0.5, 1.0)]
    oracle = experiments["oracle_same_head_scale_0.1"]["tuned_macro_f1"]
    real = next(item for item in report["evidence_ablation"] if item["name"] == "learned_real")["tuned_macro_f1"]
    shuffled = next(item for item in report["evidence_ablation"] if item["name"] == "learned_shuffled_label_condition")["tuned_macro_f1"]
    zero = next(item for item in report["evidence_ablation"] if item["name"] == "learned_zero_evidence")["tuned_macro_f1"]
    simple = experiments["simple_lc_mcer"]["tuned_macro_f1"]
    conclusion = report["diagnostic_conclusions"]
    conclusion["residual_scale_major_bottleneck"] = bool(max(scales) - scales[0] > 0.005)
    conclusion["oracle_residual_head_capacity"] = bool(oracle - m0 > 0.005)
    conclusion["learned_evidence_used"] = bool(real - zero > 0.005 and real - shuffled > 0.005)
    conclusion["competition_diversity_effect"] = {
        "full_minus_simple_macro_f1": float(scales[0] - simple),
        "full_better": bool(scales[0] > simple + 0.005),
    }
    if conclusion["oracle_residual_head_capacity"] and not conclusion["learned_evidence_used"]:
        conclusion["recommended_next_architecture"] = "Keep the residual head, simplify the learned selector; the bottleneck is learned evidence selection, not fusion."
    elif not conclusion["oracle_residual_head_capacity"]:
        conclusion["recommended_next_architecture"] = "Do not expand the selector; the current residual/fusion head cannot exploit even Oracle evidence."
    else:
        conclusion["recommended_next_architecture"] = "Retain the simplest selector that matches the full variant; confirm on three seeds before any expansion."


def legacy_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lc_mcer_diagnosis.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("allow_test"):
        raise ValueError("LC-MCER diagnosis keeps test locked")
    set_seed(int(config["seed"]))
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    train, valid = load_split(config, "train"), load_split(config, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} train={len(train['labels'])} valid={len(valid['labels'])}", flush=True)

    m0 = load_m0(config, device)
    m0_valid_logits = baseline_batches(m0, valid, device, config["eval_batch_size"])
    m0_metrics = metric_pack(config, valid["labels"], m0_valid_logits)
    del m0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    oracle_train = oracle_evidence(train, train, config)
    oracle_valid = oracle_evidence(train, valid, config)
    random_train = random_evidence(train, config, int(config["seed"]))
    random_valid = random_evidence(valid, config, int(config["seed"]) + 1)
    results, start = [], time.perf_counter()

    def run(name, **kwargs):
        item = train_model(config, train, valid, device, name, **kwargs)
        item["residual_stats"] = residual_stats(
            config, valid, item["valid_output"], item["model"].residual_scale
        )
        return item

    def store(item, keep=False):
        if not keep:
            item.pop("model", None)
            item.pop("valid_output", None)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(item)
        return item

    learned = store(run("scale_0.1", scale=0.1), keep=True)
    for scale in (0.25, 0.5, 1.0):
        store(run(f"scale_{scale}", scale=scale))

    store(run("random_evidence", scale=0.1, train_override=random_train, valid_override=random_valid))
    store(run("oracle_same_head_scale_0.1", scale=0.1, train_override=oracle_train, valid_override=oracle_valid))
    for scale in (0.25, 0.5, 1.0):
        store(run(f"oracle_same_head_scale_{scale}", scale=scale, train_override=oracle_train, valid_override=oracle_valid))

    store(run("no_competition", scale=0.1, use_competition=False, diversity_lambda=float(config["diversity_lambda"])))
    store(run("no_diversity", scale=0.1, use_competition=True, diversity_lambda=0.0))
    store(run("simple_lc_mcer", scale=0.1, simple=True))
    store(run("explicit_m0_logit_scale_0.5", scale=0.5, explicit_logit=True))
    store(run("explicit_m0_logit_scale_1.0", scale=1.0, explicit_logit=True))

    topk_items = []
    for top_k in (1, 3, 5, 10):
        if top_k == 5:
            item = learned
        else:
            item = run(f"top_k_{top_k}", scale=0.1, top_k=top_k)
            store(item)
        topk_items.append(item)

    learned_model = learned["model"]
    learned_output = learned["valid_output"]
    evidence = learned_output["evidence"]
    learned_eval = eval_override(config, learned_model, valid, device, evidence, "learned_real")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    shuffled = torch.stack(
        [evidence[row, torch.randperm(int(config["num_labels"]), generator=generator)] for row in range(len(evidence))]
    )
    ablations = [
        learned_eval,
        eval_override(config, learned_model, valid, device, shuffled, "learned_shuffled_label_condition"),
        eval_override(config, learned_model, valid, device, torch.zeros_like(evidence), "learned_zero_evidence"),
    ]
    learned_model = None
    learned.pop("model", None)
    learned.pop("valid_output", None)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    scale_summary = []
    for item in results:
        if not item["name"].startswith("scale_"):
            continue
        scale_summary.append({
            "experiment": item["name"],
            "macro_f1": item["tuned_macro_f1"],
            "micro_f1": item["tuned_micro_f1"],
            "fixed_macro_f1": item["fixed_macro_f1"],
            "fixed_micro_f1": item["fixed_micro_f1"],
            "delta_vs_m0": item["tuned_macro_f1"] - m0_metrics["tuned_macro_f1"],
            "valid_loss": item["history"][-1]["valid_loss"],
            "mean_abs_residual": item["residual_stats"]["mean_abs_residual"],
            "mean_abs_scaled_residual": item["residual_stats"]["mean_abs_scaled_residual"],
            "residual_std": item["residual_stats"]["residual_std"],
            "per_label_f1": json.dumps(item["per_label_f1"]),
        })

    innovation_names = ["random_evidence", "scale_0.1"]
    innovation_rows = [{
        "experiment": "M0",
        "macro_f1": m0_metrics["tuned_macro_f1"],
        "micro_f1": m0_metrics["tuned_micro_f1"],
        "delta_vs_m0": 0.0,
        "per_label_f1": json.dumps(m0_metrics["per_label_f1"]),
    }]
    for name in innovation_names:
        item = next(item for item in results if item["name"] == name)
        innovation_rows.append({
            "experiment": "M0 + Random Evidence" if name == "random_evidence" else "M0 + Current Retrieved Evidence",
            "macro_f1": item["tuned_macro_f1"],
            "micro_f1": item["tuned_micro_f1"],
            "delta_vs_m0": item["tuned_macro_f1"] - m0_metrics["tuned_macro_f1"],
            "per_label_f1": json.dumps(item["per_label_f1"]),
        })
    oracle_item = next(item for item in results if item["name"] == "oracle_same_head_scale_0.1")
    innovation_rows.append({
        "experiment": "M0 + Oracle Evidence (same residual head)",
        "macro_f1": oracle_item["tuned_macro_f1"],
        "micro_f1": oracle_item["tuned_micro_f1"],
        "delta_vs_m0": oracle_item["tuned_macro_f1"] - m0_metrics["tuned_macro_f1"],
        "per_label_f1": json.dumps(oracle_item["per_label_f1"]),
    })

    def named_rows(names):
        return [
            {"experiment": item["name"], "macro_f1": item["tuned_macro_f1"], "micro_f1": item["tuned_micro_f1"], "delta_vs_m0": item["tuned_macro_f1"] - m0_metrics["tuned_macro_f1"], "per_label_f1": json.dumps(item["per_label_f1"])}
            for item in results if item["name"] in names
        ]

    topk_rows = [{
        "top_k": int(top_k),
        "macro_f1": item["tuned_macro_f1"],
        "micro_f1": item["tuned_micro_f1"],
        "delta_vs_m0": item["tuned_macro_f1"] - m0_metrics["tuned_macro_f1"],
        "mean_abs_residual": item["residual_stats"]["mean_abs_residual"],
        "residual_std": item["residual_stats"]["residual_std"],
        "per_label_f1": json.dumps(item["per_label_f1"]),
    } for top_k, item in zip((1, 3, 5, 10), topk_items)]

    ranking_rows = []
    ranked_scores = learned_output["ranked_scores"]
    ranked_indices = learned_output["ranked_indices"]
    competition_scores = learned_output["competition_scores"]
    for label_id, label_name in enumerate(config["label_names"]):
        values = ranked_scores[:, label_id]
        permutation = torch.stack(
            [torch.randperm(values.shape[1], generator=generator) for _ in range(len(values))]
        )
        random_values = values.gather(1, permutation)
        margins, accuracies = [], []
        for row, is_positive in enumerate(valid["labels"][:, label_id].bool().tolist()):
            if not is_positive:
                continue
            indices = ranked_indices[row, label_id, :min(5, ranked_indices.shape[2])]
            indices = torch.as_tensor(
                [int(index) for index in indices.tolist() if bool(valid["mask"][row, int(index)])],
                dtype=torch.long,
            )
            if len(indices) == 0:
                continue
            target = competition_scores[row, indices, label_id].mean().item()
            other_ids = [index for index in range(int(config["num_labels"])) if index != label_id]
            other = competition_scores[row, indices][:, other_ids].mean().item()
            margins.append(target - other)
            accuracies.append(float(target > other))
        for method, method_values, note in (
            ("learned_lc_mcer", values, "current label-conditioned retrieval score"),
            ("random_ranking", random_values, "same score distribution with within-contract rank permutation"),
        ):
            ranking_rows.append({
                "method": method,
                "label": label_name,
                "score_mean": float(method_values.mean()),
                "score_std": float(method_values.std()),
                "top1_score_mean": float(method_values[:, 0].mean()),
                "top5_score_mean": float(method_values[:, :min(5, method_values.shape[1])].mean()),
                "evidence_norm_mean": float(evidence[:, label_id].norm(dim=-1).mean()),
                "weak_other_label_margin": float(np.mean(margins)) if margins else 0.0,
                "weak_other_label_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
                "ranking_proxy_note": note,
            })
    ranking_rows.append({
        "method": "real_vs_shuffled_evidence_ablation",
        "label": "all",
        "score_mean": float(learned_eval["tuned_macro_f1"]),
        "score_std": float(next(item for item in ablations if item["name"] == "learned_shuffled_label_condition")["tuned_macro_f1"]),
        "top1_score_mean": float(learned_eval["tuned_macro_f1"]),
        "top5_score_mean": float(next(item for item in ablations if item["name"] == "learned_zero_evidence")["tuned_macro_f1"]),
        "evidence_norm_mean": float(evidence.norm(dim=-1).mean()),
        "weak_other_label_margin": 0.0,
        "weak_other_label_accuracy": 0.0,
        "ranking_proxy_note": "values are validation F1 ablation references, not local evidence ground truth",
    })

    gradient_rows = []
    for item in results:
        for group, value in item["gradient_diagnostics"]["mean_gradient_norm"].items():
            gradient_rows.append({
                "experiment": item["name"],
                "parameter_group": group,
                "trainable_parameter_count": item["trainable_parameter_count"],
                "frozen_parameter_count": item["frozen_parameter_count"],
                "initial_parameter_norm": item["gradient_diagnostics"]["initial_parameter_norm"][group],
                "final_parameter_norm": item["gradient_diagnostics"]["final_parameter_norm"][group],
                "parameter_norm_change_abs": item["gradient_diagnostics"]["parameter_norm_change_abs"][group],
                "mean_gradient_norm": value,
                "final_gradient_norm": item["gradient_diagnostics"]["final_gradient_norm"][group],
                "nonzero_gradient_step_fraction": item["gradient_diagnostics"]["nonzero_gradient_step_fraction"][group],
            })

    oracle_rows = [
        {"experiment": item["name"], "scale": float(item["name"].rsplit("_", 1)[-1]), "macro_f1": item["tuned_macro_f1"], "micro_f1": item["tuned_micro_f1"], "delta_vs_m0": item["tuned_macro_f1"] - m0_metrics["tuned_macro_f1"], "per_label_f1": json.dumps(item["per_label_f1"])}
        for item in results if item["name"].startswith("oracle_same_head_scale_")
    ]
    report = {
        "route": config["route_name"],
        "dataset": "DIVE Main6 random split",
        "seed": int(config["seed"]),
        "train_only": True,
        "test_checked": False,
        "phase5_started": False,
        "m0": {**m0_metrics, "per_label_names": config["label_names"]},
        "experiments": results,
        "evidence_ablation": ablations,
        "scale_summary": scale_summary,
        "diagnostic_conclusions": {
            "residual_scale_major_bottleneck": "pending_review",
            "oracle_residual_head_capacity": "pending_review",
            "learned_evidence_used": "pending_review",
            "competition_diversity_effect": "pending_review",
            "recommended_next_architecture": "pending_review",
        },
        "restrictions": ["No test data/cache/predictions", "No influence pseudo-labels", "No random-positive evidence supervision", "Oracle evidence is validation upper-bound analysis only"],
        "wall_time_seconds": float(time.perf_counter() - start),
    }
    finalize_conclusions(report)
    safe_report = json_safe(report)
    (root / "lc_mcer_diagnosis_report.json").write_text(json.dumps(safe_report, indent=2), encoding="utf-8")
    save_csv(root / "innovation_ablation.csv", innovation_rows)
    save_csv(root / "residual_scale.csv", scale_summary)
    save_csv(root / "residual_scale_summary.csv", scale_summary)
    save_csv(root / "topk_ablation.csv", topk_rows)
    save_csv(root / "competition_diversity.csv", named_rows(["scale_0.1", "no_competition", "no_diversity", "simple_lc_mcer"]))
    save_csv(root / "base_logit_ablation.csv", named_rows(["scale_0.1", "explicit_m0_logit_scale_0.5", "explicit_m0_logit_scale_1.0"]))
    save_csv(root / "evidence_ranking_diagnostics.csv", ranking_rows)
    save_csv(root / "gradient_diagnostics.csv", gradient_rows)
    save_csv(root / "oracle_head_diagnostics.csv", oracle_rows)
    (root / "final_diagnosis.json").write_text(json.dumps(safe_report, indent=2), encoding="utf-8")
    lines = [
        "# FINAL DIAGNOSIS: LC-MCER",
        "",
        "DIVE Main6 random split; seed 42; validation-only; M0 checkpoint fixed and frozen; test remains locked.",
        "",
        "| Experiment | Tuned Macro-F1 | Delta vs M0 | Tuned Micro-F1 |",
        "|---|---:|---:|---:|",
        f"| M0 | {m0_metrics['tuned_macro_f1']:.6f} | +0.000000 | {m0_metrics['tuned_micro_f1']:.6f} |",
    ]
    for row in innovation_rows[1:]:
        lines.append(f"| {row['experiment']} | {row['macro_f1']:.6f} | {row['delta_vs_m0']:+.6f} | {row['micro_f1']:.6f} |")
    lines += [
        "",
        "## Scope",
        "All evidence results are diagnostic representations, not evidence ground truth. Oracle uses labels only for upper-bound selection and is not a deployable model. No influence pseudo-labels or test artifacts are used.",
        "",
        "## Automatic checks",
        f"- Residual scale major bottleneck: `{report['diagnostic_conclusions']['residual_scale_major_bottleneck']}`",
        f"- Same residual head can use Oracle evidence: `{report['diagnostic_conclusions']['oracle_residual_head_capacity']}`",
        f"- Learned evidence is used label-specifically: `{report['diagnostic_conclusions']['learned_evidence_used']}`",
        f"- Competition/diversity comparison: `{report['diagnostic_conclusions']['competition_diversity_effect']}`",
        f"- Recommended next architecture: {report['diagnostic_conclusions']['recommended_next_architecture']}",
        "",
        "Per-label F1, thresholds, residual correlations, gradient norms, and update diagnostics are in `final_diagnosis.json` and the CSV artifacts.",
    ]
    (root / "FINAL_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(root / "final_diagnosis.json"), "m0_tuned_macro_f1": m0_metrics["tuned_macro_f1"], "completed": True}, indent=2), flush=True)


def crer_safe_auc(positive, negative):
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    values = np.concatenate([positive, negative])
    if not np.isfinite(values).all():
        values = np.nan_to_num(values)
    targets = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    return float(roc_auc_score(targets, values))


def crer_forward_batches(model, data, device, batch_size):
    loader = DataLoader(
        TensorDataset(data["features"], data["mask"]),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    keys = (
        "recognition_logits", "base_logits", "residual_logits", "residual_empty",
        "counterfactual_delta", "evidence_representation", "gate", "gate_entropy",
        "active_chunk_count", "routing_scores", "routing_weights", "chunk_mask",
    )
    collected = {key: [] for key in keys}
    model.eval()
    with torch.no_grad():
        for features, masks in loader:
            output = model(features.to(device), masks.to(device))
            for key in keys:
                collected[key].append(output[key].detach().float().cpu())
    return {key: torch.cat(values) for key, values in collected.items()}


def crer_gate_diagnostics(config, data, output):
    labels = data["labels"].bool()
    mask = output["chunk_mask"].bool()
    scores = output["routing_scores"].float()
    gates = output["gate"].float()
    valid = mask.unsqueeze(-1).expand_as(gates)
    valid_gates = gates[valid]
    current_vs_other_positive = []
    other_vs_current_positive = []
    positive_vs_negative = []
    for label_id in range(int(config["num_labels"])):
        positive_values, negative_values = [], []
        for row in range(len(labels)):
            active = mask[row]
            current = scores[row, active, label_id]
            if bool(labels[row, label_id]):
                positive_values.extend(current.tolist())
                other_ids = [x for x in range(scores.shape[-1]) if x != label_id]
                if other_ids:
                    other = scores[row, active][:, other_ids].mean(dim=-1)
                    current_vs_other_positive.extend(current.tolist())
                    other_vs_current_positive.extend(other.tolist())
            else:
                negative_values.extend(current.tolist())
        positive_vs_negative.append(crer_safe_auc(positive_values, negative_values))
    # This is a weak score proxy: the current label is known at contract level,
    # but no chunk-level evidence target is available.
    other_label_auc = crer_safe_auc(
        current_vs_other_positive, other_vs_current_positive
    )
    return {
        "mean_gate": float(valid_gates.mean()) if valid_gates.numel() else 0.0,
        "gate_std": float(valid_gates.std()) if valid_gates.numel() > 1 else 0.0,
        "mean_gate_entropy": float(output["gate_entropy"].mean()),
        "mean_active_chunk_count": float(output["active_chunk_count"].mean()),
        "current_label_ranking_auc": float(np.mean(positive_vs_negative)),
        "other_label_ranking_auc": float(other_label_auc),
        "ranking_proxy_note": "Contract-level label proxy; not chunk-level evidence ground truth.",
        "mean_counterfactual_delta": float(output["counterfactual_delta"].mean()),
    }


def crer_error_reports(config, data, output):
    labels = data["labels"].numpy().astype(bool)
    base = output["base_logits"].numpy() >= 0.0
    final = output["recognition_logits"].numpy() >= 0.0
    delta = output["counterfactual_delta"].numpy()
    correction_rows, delta_rows = [], []
    all_wrong, all_correct, all_fixed, all_damaged = 0, 0, 0, 0
    for label_id, label_name in enumerate(config["label_names"]):
        wrong = base[:, label_id] != labels[:, label_id]
        correct = ~wrong
        fixed = wrong & (final[:, label_id] == labels[:, label_id])
        damaged = correct & (final[:, label_id] != labels[:, label_id])
        all_wrong += int(wrong.sum())
        all_correct += int(correct.sum())
        all_fixed += int(fixed.sum())
        all_damaged += int(damaged.sum())
        correction_rows.append({
            "label": label_name,
            "m0_wrong_count": int(wrong.sum()),
            "m0_correct_count": int(correct.sum()),
            "error_correction_rate": float(fixed.sum() / max(1, wrong.sum())),
            "error_damage_rate": float(damaged.sum() / max(1, correct.sum())),
        })
        fn = labels[:, label_id] & ~base[:, label_id]
        fp = ~labels[:, label_id] & base[:, label_id]
        m0_correct = correct
        delta_rows.append({
            "label": label_name,
            "mean_delta": float(delta[:, label_id].mean()),
            "positive_error_delta_fn": float(delta[fn, label_id].mean()) if fn.any() else 0.0,
            "negative_error_delta_fp": float(delta[fp, label_id].mean()) if fp.any() else 0.0,
            "m0_correct_delta": float(delta[m0_correct, label_id].mean()) if m0_correct.any() else 0.0,
            "m0_correct_abs_delta": float(np.abs(delta[m0_correct, label_id]).mean()) if m0_correct.any() else 0.0,
        })
    correction_rows.append({
        "label": "ALL",
        "m0_wrong_count": all_wrong,
        "m0_correct_count": all_correct,
        "error_correction_rate": float(all_fixed / max(1, all_wrong)),
        "error_damage_rate": float(all_damaged / max(1, all_correct)),
    })
    return {"error_correction": correction_rows, "counterfactual": delta_rows}


def crer_build_model(config, device):
    from lc_mcer import CRER
    return CRER(
        load_m0(config, device),
        feature_dim=int(config["feature_dim"]),
        num_labels=int(config["num_labels"]),
        retrieval_dim=int(config["retrieval_dim"]),
        gate_temperature=float(config["crer_gate_temperature"]),
        residual_scale=float(config["residual_scale"]),
        residual_hidden_dim=int(config["residual_hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)


def crer_train_one(config, train, valid, device, name, *, cf_weight=0.0,
                   error_weighting=False, preserve_weight=0.0, sparse_weight=0.0):
    set_seed(int(config["seed"]))
    model = crer_build_model(config, device)
    weight = pos_weight(train["labels"], config).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and bool(config.get("fp16", True))
    )
    loader = DataLoader(
        TensorDataset(train["features"], train["mask"], train["labels"]),
        batch_size=int(config["batch_size"]), shuffle=True, num_workers=0,
    )
    root = resolve(config["result_dir"])
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals, classes, cfs, preserves, sparsities = [], [], [], [], []
        for step, (features, masks, labels) in enumerate(loader, 1):
            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda" and bool(config.get("fp16", True))
            ):
                output = model(features.to(device), masks.to(device))
                parts = model.auxiliary_loss(
                    output, labels.to(device), weight,
                    counterfactual_weight=float(cf_weight),
                    error_weighting=bool(error_weighting),
                    preservation_weight=float(preserve_weight),
                    sparsity_weight=float(sparse_weight),
                    sparsity_target=float(config["crer_sparse_target"]),
                    margin=float(config["crer_margin"]),
                )
                loss = parts["loss"]
            scaled = loss / int(config["gradient_accumulation_steps"])
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
                if step % int(config["gradient_accumulation_steps"]) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            else:
                scaled.backward()
                if step % int(config["gradient_accumulation_steps"]) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            totals.append(float(loss.detach().cpu()))
            classes.append(float(parts["classification_loss"].detach().cpu()))
            cfs.append(float(parts["counterfactual_loss"].detach().cpu()))
            preserves.append(float(parts["preservation_loss"].detach().cpu()))
            sparsities.append(float(parts["sparsity_loss"].detach().cpu()))
        if len(loader) % int(config["gradient_accumulation_steps"]):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        valid_output = crer_forward_batches(model, valid, device, config["eval_batch_size"])
        valid_loss = F.binary_cross_entropy_with_logits(
            valid_output["recognition_logits"].to(device), valid["labels"].to(device), pos_weight=weight
        )
        metric = metric_pack(config, valid["labels"], valid_output["recognition_logits"])
        gates = crer_gate_diagnostics(config, valid, valid_output)
        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(totals)),
            "train_classification_loss": float(np.mean(classes)),
            "train_counterfactual_loss": float(np.mean(cfs)),
            "train_preservation_loss": float(np.mean(preserves)),
            "train_sparsity_loss": float(np.mean(sparsities)),
            "valid_loss": float(valid_loss.detach().cpu()),
            **metric, **gates,
        }
        history.append(row)
        (out_dir / "training_history.json").write_text(
            json.dumps(json_safe(history), indent=2), encoding="utf-8"
        )
        (root / f"training_history_{name}.json").write_text(
            json.dumps(json_safe(history), indent=2), encoding="utf-8"
        )
        print(
            f"[{name}] epoch={epoch} train_total={row['train_total_loss']:.6f} "
            f"train_cls={row['train_classification_loss']:.6f} "
            f"train_cf={row['train_counterfactual_loss']:.6f} "
            f"valid_loss={row['valid_loss']:.6f} tuned_macro={row['tuned_macro_f1']:.6f}",
            flush=True,
        )
        if row["tuned_macro_f1"] > best_score:
            best_score, best_epoch, stale = row["tuned_macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {"model_state_dict": best_state, "name": name, "epoch": epoch,
                 "train_only": True, "test_checked": False}, out_dir / "best.pt"
            )
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    model.load_state_dict(best_state, strict=True)
    valid_output = crer_forward_batches(model, valid, device, config["eval_batch_size"])
    metrics = metric_pack(config, valid["labels"], valid_output["recognition_logits"])
    result = {
        "name": name, "seed": int(config["seed"]), "best_epoch": int(best_epoch),
        "train_only": True, "test_checked": False,
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "frozen_parameter_count": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)),
        "routing_mode_train": "soft", "routing_mode_validation": "soft",
        "counterfactual_weight": float(cf_weight), "error_weighting": bool(error_weighting),
        "preservation_weight": float(preserve_weight), "sparsity_weight": float(sparse_weight),
        **metrics,
        "gate_diagnostics": crer_gate_diagnostics(config, valid, valid_output),
        **crer_error_reports(config, valid, valid_output),
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    return result


def crer_write_report(config, root, m0, results):
    rows = [{
        "experiment": "M0", "macro_f1": m0["tuned_macro_f1"],
        "micro_f1": m0["tuned_micro_f1"], "fixed_macro_f1": m0["fixed_macro_f1"],
        "delta_vs_m0": 0.0, **{f"f1_{i}": value for i, value in enumerate(m0["per_label_f1"])},
    }]
    for item in results:
        rows.append({
            "experiment": item["name"], "macro_f1": item["tuned_macro_f1"],
            "micro_f1": item["tuned_micro_f1"], "fixed_macro_f1": item["fixed_macro_f1"],
            "delta_vs_m0": item["tuned_macro_f1"] - m0["tuned_macro_f1"],
            **{f"f1_{i}": value for i, value in enumerate(item["per_label_f1"])},
            "mean_gate": item.get("gate_diagnostics", {}).get("mean_gate", 0.0),
            "mean_delta": item.get("gate_diagnostics", {}).get("mean_counterfactual_delta", 0.0),
            "error_correction_rate": item.get("error_correction", [{}])[-1].get("error_correction_rate", 0.0),
            "error_damage_rate": item.get("error_correction", [{}])[-1].get("error_damage_rate", 0.0),
        })
    save_csv(root / "ablation.csv", rows)
    correction_rows, delta_rows = [], []
    for item in results:
        for row in item.get("error_correction", []):
            correction_rows.append({"experiment": item["name"], **row})
        for row in item.get("counterfactual", []):
            delta_rows.append({"experiment": item["name"], **row})
    save_csv(root / "error_correction.csv", correction_rows)
    save_csv(root / "counterfactual_diagnostics.csv", delta_rows)
    report = {
        "route": config["route_name"], "dataset": "DIVE Main6 random split",
        "seed": int(config["seed"]), "train_only": True, "test_checked": False,
        "phase5_started": False, "m0": m0, "experiments": results,
        "restrictions": [
            "No test data/cache/predictions", "M0 frozen", "No prototype/EMA/contrastive loss",
            "No competition or diversity", "Training and validation use the same soft gate",
        ],
    }
    (root / "crer_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    full = next((x for x in results if x["name"] == "crer_full"), None)
    current = next((x for x in results if x["name"] == "current_lc_mcer"), None)
    full_correction = full["error_correction"][-1] if full else {}
    current_correction = current["error_correction"][-1] if current else {}
    go = bool(
        full
        and full["tuned_macro_f1"] > m0["tuned_macro_f1"] + 0.005
        and full["fixed_macro_f1"] > m0["fixed_macro_f1"] + 0.005
        and full_correction.get("error_correction_rate", 0.0) > current_correction.get("error_correction_rate", 0.0) + 0.01
        and full_correction.get("error_damage_rate", 1.0) <= current_correction.get("error_damage_rate", 0.0) + 0.01
    )
    report["decision"] = "GO" if go else "NO-GO"
    report["decision_rule"] = "Full CRER must improve tuned and fixed Macro-F1 by >0.005, improve correction by >0.01, and not increase damage by >0.01 versus current LC-MCER."
    (root / "final_diagnosis.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    lines = [
        "# CRER Diagnosis", "", "DIVE Main6 random split; seed 42; M0 frozen; validation-only; test locked.", "",
        "| Experiment | Tuned Macro-F1 | Delta vs M0 | Fixed Macro-F1 | Tuned Micro-F1 |", "|---|---:|---:|---:|---:|",
        f"| M0 | {m0['tuned_macro_f1']:.6f} | +0.000000 | {m0['fixed_macro_f1']:.6f} | {m0['tuned_micro_f1']:.6f} |",
    ]
    for row in rows[1:]:
        lines.append(f"| {row['experiment']} | {row['macro_f1']:.6f} | {row['delta_vs_m0']:+.6f} | {row['fixed_macro_f1']:.6f} | {row['micro_f1']:.6f} |")
    lines += [
        "", f"## Decision: {'GO' if go else 'NO-GO'}", "",
        "CRER evidence is a learned diagnostic representation, not local evidence ground truth.",
        "Counterfactual delta is measured as the residual difference between full and zero evidence.",
        "All A0-A5 results remain validation-only; no test data or predictions were read or written.",
    ]
    (root / "FINAL_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def crer_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lc_mcer_diagnosis.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("allow_test"):
        raise ValueError("CRER keeps test locked")
    set_seed(int(config["seed"]))
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    train, valid = load_split(config, "train"), load_split(config, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} train={len(train['labels'])} valid={len(valid['labels'])}", flush=True)
    baseline = load_m0(config, device)
    m0_logits = baseline_batches(baseline, valid, device, config["eval_batch_size"])
    m0 = metric_pack(config, valid["labels"], m0_logits)
    (root / "config_snapshot.yaml").write_text(resolve(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    (root / "m0_reference.json").write_text(json.dumps(json_safe({"name": "M0", **m0, "train_only": True, "test_checked": False}), indent=2), encoding="utf-8")
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []
    # A1 is the current hard Top-K LC-MCER reference; CRER itself is soft-only.
    current = train_model(
        config, train, valid, device, "current_lc_mcer",
        scale=float(config["residual_scale"]), use_competition=True,
        diversity_lambda=float(config["diversity_lambda"]),
    )
    current_report = {
        "name": "current_lc_mcer", "seed": int(config["seed"]), "train_only": True,
        "test_checked": False, "tuned_macro_f1": current["tuned_macro_f1"],
        "tuned_micro_f1": current["tuned_micro_f1"], "fixed_macro_f1": current["fixed_macro_f1"],
        "fixed_micro_f1": current["fixed_micro_f1"], "per_label_f1": current["per_label_f1"],
        "per_label_precision": current["per_label_precision"], "per_label_recall": current["per_label_recall"],
        "thresholds": current["thresholds"],
        "error_correction": crer_error_reports(
            config, valid, {"base_logits": current["valid_output"]["base"], "recognition_logits": current["valid_output"]["logits"], "counterfactual_delta": torch.zeros_like(current["valid_output"]["logits"])}
        )["error_correction"],
        "counterfactual": [], "gate_diagnostics": {},
        "trainable_parameter_count": current["trainable_parameter_count"],
    }
    results.append(current_report)
    (root / "current_lc_mcer.json").write_text(json.dumps(json_safe(current_report), indent=2), encoding="utf-8")
    (root / "training_history_current_lc_mcer.json").write_text(json.dumps(json_safe(current["history"]), indent=2), encoding="utf-8")
    current.pop("model", None); current.pop("valid_output", None)
    del current
    if device.type == "cuda":
        torch.cuda.empty_cache()

    definitions = {
        "A2": ("crer_classification_only", 0.0, False, 0.0, 0.0),
        "A3": ("crer_counterfactual", float(config["crer_lambda_cf"]), False, 0.0, 0.0),
        "A4": ("crer_error_weighted", float(config["crer_lambda_cf"]), True, 0.0, 0.0),
        "A5": ("crer_full", float(config["crer_lambda_cf"]), True, float(config["crer_lambda_preserve"]), float(config["crer_lambda_sparse"])),
    }
    for experiment in config.get("experiments", ["A0", "A1", "A2", "A3", "A4", "A5"]):
        if experiment in ("A0", "A1"):
            continue
        if experiment not in definitions:
            raise ValueError(f"unknown CRER experiment {experiment}")
        name, cf, weighted, preserve, sparse = definitions[experiment]
        item = crer_train_one(
            config, train, valid, device, name, cf_weight=cf,
            error_weighting=weighted, preserve_weight=preserve, sparse_weight=sparse,
        )
        results.append(item)
        crer_write_report(config, root, m0, results)
        (root / f"{name}.json").write_text(json.dumps(json_safe(item), indent=2), encoding="utf-8")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = crer_write_report(config, root, m0, results)
    print(json.dumps({"result_dir": str(root), "decision": report["decision"], "completed_experiments": [x["name"] for x in results], "test_checked": False}, indent=2), flush=True)


if __name__ == "__main__":
    crer_main()
