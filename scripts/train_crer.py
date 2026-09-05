"""Train and diagnose the standalone CRER Main-6 detector.

CRER never loads or calls M0. M0 is loaded separately only for the reference
row in the validation report.
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

from crer_classifier import CRERClassifier, parameter_group_diagnostics  # noqa: E402
from evm_chunk_mil_model import MLM8ViewMultiSlotMIL  # noqa: E402
from metrics import (  # noqa: E402
    compute_multilabel_metrics_from_probs,
    derived_detection_metrics_from_multilabel_probs,
    select_per_label_thresholds,
)
from train_chunk_mil import load_config as load_mil_config  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def json_safe(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def load_config(path):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_split(config, split):
    if split not in ("train", "valid"):
        raise ValueError("CRER keeps test locked")
    payload = torch.load(resolve(config["feature_dir"]) / f"{split}.pt", map_location="cpu")
    features = payload["features"].float()
    mask = payload["chunk_mask"].bool()
    labels = payload["multi_labels"].float()
    expected = (int(config["num_views"]), int(config["feature_dim"]))
    if features.ndim != 4 or tuple(features.shape[2:]) != expected:
        raise ValueError(f"{split}: unexpected feature shape {tuple(features.shape)}")
    if mask.shape != features.shape[:2] or labels.shape != (features.shape[0], int(config["num_labels"])):
        raise ValueError(f"{split}: feature/mask/label alignment error")
    if (~mask).all(dim=1).any() or not torch.isfinite(features).all():
        raise ValueError(f"{split}: empty sample or NaN/Inf")
    return {
        "features": features,
        "mask": mask,
        "labels": labels,
        "ids": [str(x) for x in payload.get("ids", range(len(labels)))],
    }


def pos_weight(labels, config):
    positive = labels.sum(dim=0)
    negative = labels.shape[0] - positive
    if config.get("pos_weight_mode", "sqrt_ratio") == "ratio":
        result = negative / positive.clamp_min(1.0)
    else:
        result = torch.sqrt(negative / positive.clamp_min(1.0))
    return result.clamp(1.0, float(config.get("max_pos_weight", 5.0)))


def metric_pack(config, labels, logits):
    probabilities = torch.sigmoid(logits).cpu().numpy()
    targets = labels.cpu().numpy()
    fixed = compute_multilabel_metrics_from_probs(targets, probabilities, 0.5)
    selected = select_per_label_thresholds(
        targets, probabilities, config["thresholds"], config["label_names"],
        global_threshold=float(config.get("global_threshold", 0.2)),
    )
    tuned = compute_multilabel_metrics_from_probs(targets, probabilities, selected["thresholds"])
    detection = derived_detection_metrics_from_multilabel_probs(
        targets, probabilities, selected["thresholds"]
    )
    return {
        "fixed_macro_f1": float(fixed["macro_f1"]),
        "fixed_micro_f1": float(fixed["micro_f1"]),
        "fixed_per_label_f1": [float(x) for x in fixed["per_label_f1"]],
        "tuned_macro_f1": float(tuned["macro_f1"]),
        "tuned_micro_f1": float(tuned["micro_f1"]),
        "tuned_per_label_f1": [float(x) for x in tuned["per_label_f1"]],
        "tuned_per_label_precision": [float(x) for x in tuned["per_label_precision"]],
        "tuned_per_label_recall": [float(x) for x in tuned["per_label_recall"]],
        "thresholds": [float(x) for x in selected["thresholds"]],
        "detection_f1": float(detection["detection_f1"]),
        "detection_precision": float(detection["detection_precision"]),
        "detection_recall": float(detection["detection_recall"]),
    }


def build_model(config, variant):
    return CRERClassifier(
        feature_dim=int(config["feature_dim"]),
        num_labels=int(config["num_labels"]),
        max_chunks=int(config["max_chunks"]),
        chunk_view_index=int(config.get("chunk_view_index", 1)),
        hidden_dim=int(config["hidden_dim"]),
        num_heads=int(config["num_heads"]),
        shared_encoder_layers=int(config["shared_encoder_layers"]),
        evidence_blocks=1 if variant == "A4_single_block" else int(config["evidence_blocks"]),
        dropout=float(config["dropout"]),
        temperature=float(config["temperature"]),
        use_label_queries=variant != "A5_shared_query",
        sparsity_target=float(config["sparsity_target"]),
    )


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_outputs(outputs):
    if not outputs:
        return {}
    result = {}
    for key in ("gate_fraction", "evidence_entropy", "effective_evidence_count"):
        value = torch.cat([item[key].float().cpu() for item in outputs], dim=0)
        result[key + "_mean"] = float(value.mean())
        result[key + "_std"] = float(value.std(unbiased=False))
    evidence = torch.cat([item["evidence_representation"].float().cpu() for item in outputs], dim=0)
    result["evidence_representation_norm_mean"] = float(evidence.norm(dim=-1).mean())
    normalized = F.normalize(evidence, dim=-1)
    similarity = torch.einsum("blh,bmh->blm", normalized, normalized)
    off_diag = (~torch.eye(evidence.shape[1], dtype=torch.bool)).unsqueeze(0).expand(evidence.shape[0], -1, -1)
    result["label_evidence_cosine_offdiag_mean"] = float(similarity[off_diag].mean())
    weights = torch.cat([item["routing_weights"].float().cpu() for item in outputs], dim=0)
    result["routing_top1_weight_mean"] = float(weights.max(dim=1).values.mean())
    result["routing_top5_weight_mean"] = float(weights.topk(min(5, weights.shape[1]), dim=1).values.sum(dim=1).mean())
    return result


def evaluate(model, data, config, device, weight, batch_size, collect=False):
    loader = DataLoader(TensorDataset(data["features"], data["mask"], data["labels"]), batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    logits, losses, outputs = [], [], []
    with torch.no_grad():
        for features, mask, labels in loader:
            output = model(features.to(device), mask.to(device), return_diagnostics=collect)
            loss = model.compute_loss(output, labels.to(device), weight, lambda_cf=0.0, lambda_sparse=0.0, lambda_entropy=0.0)["classification_loss"]
            logits.append(output["recognition_logits"].float().cpu())
            losses.append(float(loss.cpu()))
            if collect:
                outputs.append(output)
    return torch.cat(logits), float(np.mean(losses)), summarize_outputs(outputs)


def train_one(config, variant, seed, train, valid, device, root):
    set_seed(seed)
    model = build_model(config, variant).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(config.get("fp16", True)))
    weight = pos_weight(train["labels"], config).to(device)
    loader = DataLoader(TensorDataset(train["features"], train["mask"], train["labels"]), batch_size=int(config["batch_size"]), shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    run_dir = root / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    gradient_rows = []
    history = []
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    start_time = time.perf_counter()
    accumulation = max(1, int(config["gradient_accumulation_steps"]))
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        for step, (features, mask, labels) in enumerate(loader, 1):
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(features.to(device), mask.to(device))
                values = model.compute_loss(
                    output, labels.to(device), weight,
                    lambda_cf=0.0 if variant == "A2_no_counterfactual" else float(config["lambda_cf"]),
                    lambda_sparse=0.0 if variant == "A3_no_sparsity" else float(config["lambda_sparse"]),
                    lambda_entropy=float(config.get("lambda_entropy", 0.0)),
                    margin=float(config["counterfactual_margin"]),
                )
                loss = values["loss"] / accumulation
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            epoch_losses.append(float(values["loss"].detach().cpu()))
            if step % accumulation != 0 and step != len(loader):
                continue
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            gradient_rows.extend({"epoch": epoch, "step": step, "phase": "gradient", **row} for row in parameter_group_diagnostics(model))
            before = {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()}
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gradient_rows.extend({"epoch": epoch, "step": step, "phase": "update", **row} for row in parameter_group_diagnostics(model, before))
        valid_logits, valid_loss, valid_diag = evaluate(model, valid, config, device, weight, config["eval_batch_size"], collect=True)
        score = metric_pack(config, valid["labels"], valid_logits)
        record = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "valid_loss": valid_loss, **score, "evidence_diagnostics": valid_diag, "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0}
        history.append(record)
        print(f"[{variant}] seed={seed} epoch={epoch} train_loss={record['train_loss']:.6f} valid_loss={valid_loss:.6f} fixed_macro={score['fixed_macro_f1']:.6f} tuned_macro={score['tuned_macro_f1']:.6f}", flush=True)
        if score["tuned_macro_f1"] > best_score:
            best_score, best_epoch, best_state, stale = score["tuned_macro_f1"], epoch, {name: p.detach().cpu() for name, p in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("CRER did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()
    valid_logits, valid_loss, valid_diag = evaluate(model, valid, config, device, weight, config["eval_batch_size"], collect=True)
    final_metrics = metric_pack(config, valid["labels"], valid_logits)
    summary = {"variant": variant, "seed": int(seed), "best_epoch": int(best_epoch), "train_only": True, "test_checked": False, **final_metrics, "valid_loss": valid_loss, "evidence_diagnostics": valid_diag, "total_params": int(sum(p.numel() for p in model.parameters())), "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "peak_memory_mb": max([x["peak_memory_mb"] for x in history] or [0.0]), "valid_inference_seconds": float(time.perf_counter() - inference_start), "wall_time_seconds": float(time.perf_counter() - start_time), "history": history}
    torch.save({"model_state_dict": model.state_dict(), "config": config, "variant": variant, "seed": seed, "test_checked": False}, run_dir / "best.pt")
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(json_safe(history), indent=2), encoding="utf-8")
    write_csv(run_dir / "gradient_diagnostics.csv", gradient_rows, ["epoch", "step", "phase", "group", "grad_norm", "param_norm", "update_norm"])
    return summary


def load_m0_reference(config, valid, device):
    model_config = load_mil_config(resolve(config["baseline_config"]), config["baseline_variant"])
    model = MLM8ViewMultiSlotMIL(model_config).to(device)
    checkpoint = torch.load(resolve(config["baseline_checkpoint"]), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("M0 checkpoint has no model state")
    model.load_state_dict(state, strict=True)
    model.eval()
    logits = []
    loader = DataLoader(TensorDataset(valid["features"], valid["mask"]), batch_size=int(config["eval_batch_size"]), shuffle=False)
    with torch.no_grad():
        for features, mask in loader:
            logits.append(model(features.to(device), mask.to(device))["recognition_logits"].float().cpu())
    result = metric_pack(config, valid["labels"], torch.cat(logits))
    result.update({"name": "M0", "total_params": int(sum(p.numel() for p in model.parameters())), "trainable_params": 0, "test_checked": False})
    return result


def aggregate(config, reference, summaries, root):
    keys = ("tuned_macro_f1", "tuned_micro_f1", "fixed_macro_f1", "fixed_micro_f1", "detection_f1", "total_params", "trainable_params", "best_epoch", "valid_inference_seconds", "peak_memory_mb")
    rows = [{"model": "M0", "seed": "reference", **{key: reference.get(key) for key in keys}}]
    rows.extend({"model": item["variant"], "seed": item["seed"], **{key: item.get(key) for key in keys}} for item in summaries)
    write_csv(root / "model_comparison.csv", rows, sorted({key for row in rows for key in row}))
    grouped = {}
    for variant in config["variants"]:
        items = [x for x in summaries if x["variant"] == variant]
        values = [x["tuned_macro_f1"] for x in items]
        grouped[variant] = {"runs": items, "mean_tuned_macro_f1": float(np.mean(values)) if values else 0.0, "std_tuned_macro_f1": float(np.std(values)) if values else 0.0, "delta_vs_m0": float(np.mean(values) - reference["tuned_macro_f1"]) if values else 0.0}
    report = {"route": config["route_name"], "dataset": "DIVE Main6 random split", "seed_protocol": [int(x) for x in config["seeds"]], "train_only": True, "test_checked": False, "crer_is_standalone": True, "m0_reference": reference, "runs": summaries, "variants": grouped, "artifacts": {"model_comparison": "model_comparison.csv"}, "evidence_ground_truth": False, "restrictions": ["No test data", "No M0 input to CRER", "No local evidence labels or pseudo-labels"]}
    (root / "model_comparison.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    (root / "crer_main6_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    lines = ["# Standalone CRER Main-6", "", "Validation-only; DIVE Main6 random split; seed 42 protocol; test locked.", "", "CRER is an independent detector: cached opcode chunk features -> shared encoder -> label-conditioned evidence blocks -> label-wise heads.", "", "| Variant | Mean tuned Macro-F1 | Std | Delta vs M0 |", "|---|---:|---:|---:|"]
    lines.extend(f"| {name} | {item['mean_tuned_macro_f1']:.6f} | {item['std_tuned_macro_f1']:.6f} | {item['delta_vs_m0']:.6f} |" for name, item in grouped.items())
    lines.extend(["", f"M0 tuned Macro-F1: **{reference['tuned_macro_f1']:.6f}**", "", "Evidence routing diagnostics are representation/sensitivity diagnostics, not chunk-level ground truth."])
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "report", "all"), nargs="?", default="all")
    parser.add_argument("--config", default="configs/train_crer.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if bool(config.get("allow_test", False)):
        raise ValueError("Standalone CRER keeps test locked")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, valid = load_split(config, "train"), load_split(config, "valid")
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    if args.stage in ("train", "all"):
        for variant in config["variants"]:
            for seed in config["seeds"]:
                summaries.append(train_one(config, variant, int(seed), train, valid, device, root))
        (root / "run_summaries.json").write_text(json.dumps(json_safe(summaries), indent=2), encoding="utf-8")
    if args.stage in ("report", "all"):
        if not summaries:
            summaries = json.loads((root / "run_summaries.json").read_text(encoding="utf-8"))
        report = aggregate(config, load_m0_reference(config, valid, device), summaries, root)
        print(json.dumps({"m0_tuned_macro_f1": report["m0_reference"]["tuned_macro_f1"], "variants": {key: value["mean_tuned_macro_f1"] for key, value in report["variants"].items()}, "test_checked": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
