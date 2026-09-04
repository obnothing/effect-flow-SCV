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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evm_chunk_mil_model import MLM8ViewMultiSlotMIL  # noqa: E402
from lc_mcer import LCMCER  # noqa: E402
from metrics import (  # noqa: E402
    compute_multilabel_metrics_from_probs,
    select_per_label_thresholds,
)
from train_chunk_mil import load_config as load_mil_config  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def build_model(config, device, *, scale=None, simple=False, explicit_logit=False):
    base = load_m0(config, device)
    return LCMCER(
        base,
        feature_dim=int(config["feature_dim"]),
        num_labels=int(config["num_labels"]),
        retrieval_dim=int(config["retrieval_dim"]),
        top_k=int(config["top_k"]),
        diversity_lambda=0.0 if simple else float(config["diversity_lambda"]),
        use_competition=not simple,
        residual_scale=float(config["residual_scale"] if scale is None else scale),
        residual_hidden_dim=int(config["residual_hidden_dim"]),
        dropout=float(config["dropout"]),
        include_base_logit=explicit_logit,
    ).to(device)


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
    logits, residuals, bases, evidences, ranked = [], [], [], [], []
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
    result = {
        "logits": torch.cat(logits),
        "residual": torch.cat(residuals),
        "base": torch.cat(bases),
        "evidence": torch.cat(evidences),
    }
    if return_diag:
        result["ranked_indices"] = torch.cat(ranked)
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


def train_model(config, train, valid, device, name, *, scale=None, simple=False, explicit_logit=False, oracle_train=None, oracle_valid=None):
    set_seed(int(config["seed"]))
    model = build_model(config, device, scale=scale, simple=simple, explicit_logit=explicit_logit)
    weight = pos_weight(train["labels"], config).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(config.get("fp16", True)))
    if oracle_train is None:
        values = TensorDataset(train["features"], train["mask"], train["labels"])
    else:
        values = TensorDataset(train["features"], train["mask"], train["labels"], oracle_train)
    loader = DataLoader(values, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            features, masks, labels = batch[:3]
            override = batch[3] if oracle_train is not None else None
            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and bool(config.get("fp16", True))):
                kwargs = {} if override is None else {"evidence_override": override.to(device)}
                output = model(features.to(device), masks.to(device), **kwargs)
                loss = F.binary_cross_entropy_with_logits(output["recognition_logits"], labels.to(device), pos_weight=weight)
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
            losses.append(float(loss.detach().cpu()))
        if len(loader) % int(config["gradient_accumulation_steps"]):
            if scaler.is_enabled():
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])); optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        valid_out = forward_batches(model, valid, device, config["eval_batch_size"], oracle_valid)
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
    valid_out = forward_batches(model, valid, device, config["eval_batch_size"], oracle_valid, return_diag=True)
    result = {"name": name, "seed": int(config["seed"]), "best_epoch": int(best_epoch), **metric_pack(config, valid["labels"], valid_out["logits"]), "history": history, "valid_output": valid_out, "model": model}
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def finalize_conclusions(report):
    m0 = float(report["m0"]["tuned_macro_f1"])
    experiments = {item["name"]: item for item in report["experiments"]}
    scales = [experiments[f"scale_{value}"]["tuned_macro_f1"] for value in (0.1, 0.3, 0.5, 1.0)]
    oracle = experiments["oracle_same_head"]["tuned_macro_f1"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lc_mcer_diagnosis.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("allow_test"):
        raise ValueError("LC-MCER diagnosis keeps test locked")
    set_seed(int(config["seed"]))
    root = resolve(config["result_dir"]); root.mkdir(parents=True, exist_ok=True)
    train, valid = load_split(config, "train"), load_split(config, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} train={len(train['labels'])} valid={len(valid['labels'])}", flush=True)

    m0 = load_m0(config, device)
    m0_valid_logits = baseline_batches(m0, valid, device, config["eval_batch_size"])
    m0_metrics = metric_pack(config, valid["labels"], m0_valid_logits)
    del m0
    if device.type == "cuda":
        torch.cuda.empty_cache()
    oracle_train, oracle_valid = oracle_evidence(train, train, config), oracle_evidence(train, valid, config)

    results, start = [], time.perf_counter()
    for scale in (0.1, 0.3, 0.5, 1.0):
        item = train_model(config, train, valid, device, f"scale_{scale}", scale=scale)
        item["residual_stats"] = residual_stats(config, valid, item["valid_output"], item["model"].residual_scale)
        if scale != 0.1:
            item.pop("model", None); item.pop("valid_output", None)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(item)
    learned = results[0]
    for name, kwargs in (
        ("oracle_same_head", {"scale": 0.1, "oracle_train": oracle_train, "oracle_valid": oracle_valid}),
        ("simple_lc_mcer", {"scale": 0.1, "simple": True}),
        ("explicit_m0_logit_scale_0.5", {"scale": 0.5, "explicit_logit": True}),
        ("explicit_m0_logit_scale_1.0", {"scale": 1.0, "explicit_logit": True}),
    ):
        item = train_model(config, train, valid, device, name, **kwargs)
        item["residual_stats"] = residual_stats(config, valid, item["valid_output"], item["model"].residual_scale)
        item.pop("model", None); item.pop("valid_output", None)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results.append(item)

    learned_model = learned["model"]
    learned_eval = eval_override(config, learned_model, valid, device, learned["valid_output"]["evidence"], "learned_real")
    evidence = learned["valid_output"]["evidence"]
    generator = torch.Generator().manual_seed(int(config["seed"]))
    shuffled = torch.stack([evidence[row, torch.randperm(int(config["num_labels"]), generator=generator)] for row in range(len(evidence))])
    ablations = [learned_eval, eval_override(config, learned_model, valid, device, shuffled, "learned_shuffled_label_condition"), eval_override(config, learned_model, valid, device, torch.zeros_like(evidence), "learned_zero_evidence")]
    scale_stats = [{"experiment": item["name"], "macro_f1": item["tuned_macro_f1"], "micro_f1": item["tuned_micro_f1"], "delta_vs_m0": item["tuned_macro_f1"] - m0_metrics["tuned_macro_f1"], "valid_loss": item["history"][-1]["valid_loss"], "mean_abs_residual": item["residual_stats"]["mean_abs_residual"], "mean_abs_scaled_residual": item["residual_stats"]["mean_abs_scaled_residual"], "residual_std": item["residual_stats"]["residual_std"], "per_label_f1": item["per_label_f1"]} for item in results]
    report = {
        "route": config["route_name"], "dataset": "DIVE Main6 random split", "seed": int(config["seed"]), "train_only": True, "test_checked": False, "phase5_started": False,
        "m0": {**m0_metrics, "residual_stats": {"mean_abs_residual": 0.0}},
        "experiments": results, "evidence_ablation": ablations, "scale_summary": scale_stats,
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
    (root / "lc_mcer_diagnosis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_csv(root / "residual_scale_summary.csv", scale_stats)
    lines = ["# LC-MCER Residual Diagnosis", "", "DIVE Main6 random split; validation-only diagnostic, seed 42; M0 is frozen.", "", "| Experiment | Tuned Macro-F1 | Delta vs M0 |", "|---|---:|---:|"]
    lines.append(f"| M0 | {m0_metrics['tuned_macro_f1']:.6f} | 0.000000 |")
    for row in scale_stats:
        lines.append(f"| {row['experiment']} | {row['macro_f1']:.6f} | {row['delta_vs_m0']:+.6f} |")
    for row in ablations:
        lines.append(f"| {row['name']} | {row['tuned_macro_f1']:.6f} | {row['tuned_macro_f1'] - m0_metrics['tuned_macro_f1']:+.6f} |")
    lines += ["", "All evidence outputs are diagnostic representations, not evidence ground truth. Oracle uses labels only to select an upper-bound evidence representation; it is not a deployable model.", "", "Per-label F1 and residual correlations are in `lc_mcer_diagnosis_report.json`."]
    (root / "lc_mcer_diagnosis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(root / "lc_mcer_diagnosis_report.json"), "m0_tuned_macro_f1": m0_metrics["tuned_macro_f1"], "completed": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
