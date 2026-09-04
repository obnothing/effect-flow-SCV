"""Train/evaluate the minimal LC-MCER-v2 ablation matrix.

Only the Main-6 train/valid caches are read.  Prototype updates are train-only
EMA updates; evidence outputs are diagnostic representations, not ground truth.
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

from diagnose_lc_mcer import (  # noqa: E402
    baseline_batches,
    json_safe,
    load_m0,
    load_split,
    metric_pack,
    pos_weight,
)
from lc_mcer import LCMCER, LCMCERV2  # noqa: E402


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


def build_model(config, device, *, routing_mode, evidence_loss, use_competition, diversity_lambda):
    cls = LCMCERV2 if evidence_loss else LCMCER
    kwargs = {
        "feature_dim": int(config["feature_dim"]),
        "num_labels": int(config["num_labels"]),
        "retrieval_dim": int(config["retrieval_dim"]),
        "top_k": int(config["top_k"]),
        "diversity_lambda": float(diversity_lambda),
        "use_competition": bool(use_competition),
        "residual_scale": float(config["residual_scale"]),
        "residual_hidden_dim": int(config["residual_hidden_dim"]),
        "dropout": float(config["dropout"]),
        "routing_mode": routing_mode,
        "routing_temperature": float(config["evidence_temperature"]),
    }
    if evidence_loss:
        kwargs["prototype_momentum"] = float(config["prototype_momentum"])
    return cls(load_m0(config, device), **kwargs).to(device)


def safe_auc(positive, negative):
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    values = np.concatenate([np.asarray(positive), np.asarray(negative)])
    targets = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    return float(roc_auc_score(targets, values)) if len(np.unique(targets)) > 1 else 0.5


def selector_diagnostics(output, data):
    scores = output["routing_scores"].float().cpu()
    weights = output["routing_weights"].float().cpu().clamp_min(1e-12)
    mask = data["mask"]
    entropy = -(weights * weights.log()).sum(dim=1)
    current, other, contract_positive, contract_negative = [], [], [], []
    for row in range(len(scores)):
        active = mask[row]
        for label_id in torch.where(data["labels"][row] > 0.5)[0].tolist():
            current_values = scores[row, active, label_id].numpy()
            other_ids = [index for index in range(scores.shape[-1]) if index != label_id]
            other_values = scores[row, active][:, other_ids].mean(dim=-1).numpy()
            current.extend(current_values.tolist())
            other.extend(other_values.tolist())
            contract_positive.append(float(scores[row, active, label_id].mean()))
            contract_negative.append(float(scores[row, active][:, other_ids].mean()))
    return {
        "selector_current_label_auc": safe_auc(current, other),
        "selector_other_label_auc": safe_auc(contract_positive, contract_negative),
        "selector_current_label_margin": float(np.mean(np.asarray(contract_positive) - np.asarray(contract_negative))) if current else 0.0,
        "mean_evidence_entropy": float(entropy.mean()) if entropy.numel() else 0.0,
        "evidence_weight_max_mean": float(weights.max(dim=1).values.mean()),
    }


def train_one(config, train, valid, device, name, *, routing_mode, evidence_loss, use_competition, diversity_lambda):
    set_seed(int(config["seed"]))
    model = build_model(
        config,
        device,
        routing_mode=routing_mode,
        evidence_loss=evidence_loss,
        use_competition=use_competition,
        diversity_lambda=diversity_lambda,
    )
    weight = pos_weight(train["labels"], config).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(config.get("fp16", True)))
    loader = DataLoader(
        TensorDataset(train["features"], train["mask"], train["labels"]),
        batch_size=int(config["batch_size"]), shuffle=True, num_workers=0,
    )
    out_dir = resolve(config["result_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history, grad_history = [], []
    start = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        model_loss, cls_losses, evidence_losses = [], [], []
        optimizer.zero_grad(set_to_none=True)
        for step, (features, masks, labels) in enumerate(loader, 1):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and bool(config.get("fp16", True))):
                output = model(features.to(device), masks.to(device))
                cls_loss = F.binary_cross_entropy_with_logits(
                    output["recognition_logits"], labels.to(device), pos_weight=weight
                )
                if evidence_loss:
                    ev_loss = model.evidence_discrimination_loss(
                        output["evidence_representation"], labels.to(device),
                        temperature=float(config["evidence_contrastive_temperature"]),
                    )
                else:
                    ev_loss = cls_loss * 0.0
                loss = cls_loss + float(config["evidence_loss_weight"]) * ev_loss
            scaled = loss / int(config["gradient_accumulation_steps"])
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
                if step % int(config["gradient_accumulation_steps"]) == 0:
                    scaler.unscale_(optimizer)
                    grad_history.append({
                        name: float(value) for name, value in zip(
                            ("query_projection", "chunk_projection", "label_embedding"),
                            [parameter.grad.detach().float().norm().item() if parameter.grad is not None else 0.0 for parameter in (model.query_projection.weight, model.chunk_projection.weight, model.label_embedding)],
                        )
                    })
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            else:
                scaled.backward()
                if step % int(config["gradient_accumulation_steps"]) == 0:
                    grad_history.append({
                        "query_projection": float(model.query_projection.weight.grad.detach().norm()) if model.query_projection.weight.grad is not None else 0.0,
                        "chunk_projection": float(model.chunk_projection.weight.grad.detach().norm()) if model.chunk_projection.weight.grad is not None else 0.0,
                        "label_embedding": float(model.label_embedding.grad.detach().norm()) if model.label_embedding.grad is not None else 0.0,
                    })
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if evidence_loss:
                model.update_prototypes(output["evidence_representation"], labels.to(device))
            model_loss.append(float(loss.detach().cpu())); cls_losses.append(float(cls_loss.detach().cpu())); evidence_losses.append(float(ev_loss.detach().cpu()))
        if len(loader) % int(config["gradient_accumulation_steps"]):
            if scaler.is_enabled():
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])); scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])); optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        model.eval()
        valid_outputs = []
        with torch.no_grad():
            valid_loader = DataLoader(TensorDataset(valid["features"], valid["mask"]), batch_size=int(config["eval_batch_size"]), shuffle=False)
            for features, masks in valid_loader:
                valid_outputs.append(model(features.to(device), masks.to(device)))
        valid_logits = torch.cat([item["recognition_logits"].float().cpu() for item in valid_outputs])
        valid_loss = F.binary_cross_entropy_with_logits(valid_logits.to(device), valid["labels"].to(device), pos_weight=weight)
        metrics = metric_pack(config, valid["labels"], valid_logits)
        selector = selector_diagnostics({key: torch.cat([item[key].float().cpu() for item in valid_outputs]) for key in ("routing_scores", "routing_weights")}, valid)
        record = {
            "epoch": epoch, "train_cls_loss": float(np.mean(cls_losses)), "train_evidence_loss": float(np.mean(evidence_losses)), "train_total_loss": float(np.mean(model_loss)), "valid_loss": float(valid_loss.cpu()), **metrics, **selector,
            "mean_residual": float(torch.cat([item["residual_logits"].float().cpu() for item in valid_outputs]).mean()),
            "mean_abs_residual": float(torch.cat([item["residual_logits"].float().cpu() for item in valid_outputs]).abs().mean()),
            "prototype_norm": float(model.label_prototypes.norm().detach().cpu()) if evidence_loss else 0.0,
            "prototype_counts": model.prototype_counts.detach().cpu().tolist() if evidence_loss else [],
        }
        history.append(record)
        (out_dir / "training_history.json").write_text(json.dumps(json_safe(history), indent=2), encoding="utf-8")
        print(f"[{name}] epoch={epoch} train_cls={record['train_cls_loss']:.6f} train_ev={record['train_evidence_loss']:.6f} valid_loss={record['valid_loss']:.6f} tuned_macro={record['tuned_macro_f1']:.6f}", flush=True)
        if record["tuned_macro_f1"] > best_score:
            best_score, best_epoch, stale = record["tuned_macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "name": name, "epoch": epoch, "train_only": True, "test_checked": False}, out_dir / "best.pt")
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        valid_outputs = []
        valid_loader = DataLoader(TensorDataset(valid["features"], valid["mask"]), batch_size=int(config["eval_batch_size"]), shuffle=False)
        for features, masks in valid_loader:
            valid_outputs.append(model(features.to(device), masks.to(device)))
    valid_logits = torch.cat([item["recognition_logits"].float().cpu() for item in valid_outputs])
    summary = {"name": name, "best_epoch": int(best_epoch), "train_only": True, "test_checked": False, "routing_mode_train": routing_mode, "routing_mode_inference": "hard", "evidence_loss_enabled": evidence_loss, "evidence_loss_weight": float(config["evidence_loss_weight"] if evidence_loss else 0.0), "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)), "frozen_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)), "elapsed_seconds": float(time.perf_counter() - start), **metric_pack(config, valid["labels"], valid_logits), "selector_diagnostics": selector_diagnostics({key: torch.cat([item[key].float().cpu() for item in valid_outputs]) for key in ("routing_scores", "routing_weights")}, valid), "gradient_diagnostics": {"mean": {key: float(np.mean([row[key] for row in grad_history])) if grad_history else 0.0 for key in ("query_projection", "chunk_projection", "label_embedding")}, "final": grad_history[-1] if grad_history else {}, "steps": len(grad_history)}, "prototype_norm": float(model.label_prototypes.norm().detach().cpu()) if evidence_loss else 0.0, "history": history}
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    (out_dir / "selector_diagnostics.json").write_text(
        json.dumps(json_safe({"selector": summary["selector_diagnostics"], "gradient": summary["gradient_diagnostics"], "prototype_norm": summary["prototype_norm"], "test_checked": False}), indent=2),
        encoding="utf-8",
    )
    return summary


def write_report(config, root, m0, summaries):
    report = {"route": config["route_name"], "dataset": "DIVE Main6 random split", "seed": int(config["seed"]), "train_only": True, "test_checked": False, "phase5_started": False, "m0": m0, "experiments": summaries, "restrictions": ["No test data/cache/predictions", "No influence pseudo-labels", "Prototype/EMA updates use train only", "Evidence is diagnostic and not ground truth"]}
    (root / "final_diagnosis.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    rows = [{"experiment": "M0", "macro_f1": m0["tuned_macro_f1"], "micro_f1": m0["tuned_micro_f1"], "delta_vs_m0": 0.0}]
    for item in summaries:
        rows.append({"experiment": item["name"], "macro_f1": item["tuned_macro_f1"], "micro_f1": item["tuned_micro_f1"], "delta_vs_m0": item["tuned_macro_f1"] - m0["tuned_macro_f1"]})
    with (root / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    best = max(summaries, key=lambda item: item["tuned_macro_f1"]) if summaries else None
    lines = ["# LC-MCER-v2 Diagnosis", "", "DIVE Main6 random split; seed 42; M0 fixed/frozen; validation-only; test locked.", "", "| Experiment | Tuned Macro-F1 | Delta vs M0 | Tuned Micro-F1 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['macro_f1']:.6f} | {row['delta_vs_m0']:+.6f} | {row['micro_f1']:.6f} |")
    lines += ["", "## Interpretation", "", f"Best completed experiment: `{best['name'] if best else 'none'}`.", "A0 hard routing/classification-only is the v1 reference. A1 tests soft routing alone. A2 tests evidence discrimination with hard inference. A3 is the full v2 setting: soft training routing, train-only EMA prototype loss, and hard Top-K inference.", "", "All evidence and selector metrics are weak/diagnostic proxies, not local evidence ground truth."]
    (root / "FINAL_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lc_mcer_v2.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("allow_test"):
        raise ValueError("LC-MCER-v2 keeps test locked")
    set_seed(int(config["seed"]))
    root = resolve(config["result_dir"]); root.mkdir(parents=True, exist_ok=True)
    train, valid = load_split(config, "train"), load_split(config, "valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = load_m0(config, device)
    m0_logits = baseline_batches(baseline, valid, device, config["eval_batch_size"])
    m0 = metric_pack(config, valid["labels"], m0_logits)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config_snapshot.yaml").write_text(
        resolve(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_report(config, root, m0, [])
    (root / "m0_reference.json").write_text(
        json.dumps(json_safe({"name": "M0", **m0, "train_only": True, "test_checked": False}), indent=2),
        encoding="utf-8",
    )
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    definitions = {
        "A0": ("baseline", "hard", False, True, float(config.get("baseline_diversity_lambda", 0.1))),
        "A1": ("soft_routing", "soft", False, False, 0.0),
        "A2": ("evidence_loss", "hard", True, False, 0.0),
        "A3": ("full_v2", "soft", True, False, 0.0),
    }
    summaries = []
    for experiment in config.get("experiments", ["A0", "A1", "A2", "A3"]):
        if experiment not in definitions:
            raise ValueError(f"unknown experiment {experiment}")
        name, mode, evidence_loss, use_competition, diversity_lambda = definitions[experiment]
        summary = train_one(config, train, valid, device, name, routing_mode=mode, evidence_loss=evidence_loss, use_competition=use_competition, diversity_lambda=diversity_lambda)
        summaries.append(summary)
        write_report(config, root, m0, summaries)
        mapping = {"baseline": "baseline.json", "soft_routing": "soft_routing.json", "evidence_loss": "evidence_loss.json", "full_v2": "full_v2.json"}
        (root / mapping[name]).write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_report(config, root, m0, summaries)
    print(json.dumps({"result_dir": str(root), "m0_tuned_macro_f1": m0["tuned_macro_f1"], "completed_experiments": [item["name"] for item in summaries], "test_checked": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
