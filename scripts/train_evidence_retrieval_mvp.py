"""Validation-only training for the LC-MCER MVP.

Only contract-level weighted BCE is used. Test data and evidence pseudo-labels
are deliberately unavailable in this route.
"""

import argparse
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
        raise ValueError("LC-MCER MVP is train/valid only; test is locked")
    payload = torch.load(resolve(config["feature_dir"]) / f"{split}.pt", map_location="cpu")
    features = payload["features"].float()
    mask = payload["chunk_mask"].bool()
    labels = payload["multi_labels"].float()
    if features.ndim != 4 or tuple(features.shape[2:]) != (8, int(config["feature_dim"])):
        raise ValueError(f"{split}: unexpected cache shape {tuple(features.shape)}")
    if mask.shape != features.shape[:2] or labels.shape != (features.shape[0], int(config["num_labels"])):
        raise ValueError(f"{split}: cache alignment/shape error")
    if (~mask).all(dim=1).any() or not torch.isfinite(features).all():
        raise ValueError(f"{split}: empty sample or NaN/Inf")
    return {
        "ids": [str(x) for x in payload["ids"]],
        "features": features,
        "mask": mask,
        "labels": labels,
    }


def load_m0(config, device):
    model_config = load_mil_config(
        resolve(config["baseline_config"]), config["baseline_variant"]
    )
    model = MLM8ViewMultiSlotMIL(model_config).to(device)
    checkpoint = torch.load(resolve(config["baseline_checkpoint"]), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("baseline checkpoint has no model state")
    model.load_state_dict(state, strict=True)
    return model.eval()


def build_model(config, device, variant):
    base = load_m0(config, device)
    return LCMCER(
        base,
        feature_dim=int(config["feature_dim"]),
        num_labels=int(config["num_labels"]),
        retrieval_dim=int(config["retrieval_dim"]),
        top_k=int(config["top_k"]),
        diversity_lambda=(
            float(config["diversity_lambda"]) if variant != "no_diversity" else 0.0
        ),
        use_competition=variant != "no_competition",
        residual_scale=float(config["residual_scale"]),
        residual_hidden_dim=int(config["residual_hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)


def make_pos_weight(labels, config):
    positive = labels.sum(dim=0)
    negative = labels.shape[0] - positive
    if config.get("pos_weight_mode", "sqrt_ratio") == "ratio":
        value = negative / positive.clamp_min(1.0)
    else:
        value = torch.sqrt(negative / positive.clamp_min(1.0))
    return value.clamp(1.0, float(config.get("max_pos_weight", 5.0)))


def threshold_metrics(config, labels, logits):
    probs = torch.sigmoid(logits).numpy()
    fixed = compute_multilabel_metrics_from_probs(labels.numpy(), probs, 0.5)
    selected = select_per_label_thresholds(
        labels.numpy(),
        probs,
        config["thresholds"],
        config["label_names"],
        global_threshold=0.2,
    )
    tuned = compute_multilabel_metrics_from_probs(
        labels.numpy(), probs, selected["thresholds"]
    )
    return fixed, tuned, selected


def length_metrics(config, labels, probs, mask, thresholds):
    lengths = mask.sum(dim=1).numpy()
    q1, q2 = np.quantile(lengths, [1 / 3, 2 / 3])
    groups = {
        "short": lengths <= q1,
        "medium": (lengths > q1) & (lengths <= q2),
        "long": lengths > q2,
    }
    result = {
        "boundaries": {"short_max": float(q1), "medium_max": float(q2)},
        "groups": {},
    }
    for name, selected in groups.items():
        metric = compute_multilabel_metrics_from_probs(
            labels.numpy()[selected], probs[selected], thresholds
        )
        result["groups"][name] = {
            "contracts": int(selected.sum()),
            "macro_f1": float(metric["recognition_macro_f1"]),
            "micro_f1": float(metric["recognition_micro_f1"]),
        }
    rare_ids = [
        config["label_names"].index(name) for name in config["rare_label_names"]
    ]
    overall = compute_multilabel_metrics_from_probs(
        labels.numpy(), probs, thresholds
    )
    result["rare_label_macro_f1"] = float(
        np.mean([overall["per_label_f1"][idx] for idx in rare_ids])
    )
    return result


def run_epoch(
    model,
    data,
    device,
    batch_size,
    pos_weight,
    config,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None
    model.train(training)
    loader = DataLoader(
        TensorDataset(data["features"], data["mask"], data["labels"]),
        batch_size=int(batch_size),
        shuffle=training,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    logits = []
    losses = []
    accumulation = int(config["gradient_accumulation_steps"])
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (features, masks, labels) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda" and bool(config.get("fp16", True))
        ):
            output = model(features, masks)
            loss = F.binary_cross_entropy_with_logits(
                output["recognition_logits"], labels, pos_weight=pos_weight.to(device)
            )
        if training:
            scaled = loss / accumulation
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled).backward()
                if step % accumulation == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["gradient_clip_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                scaled.backward()
                if step % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["gradient_clip_norm"])
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        logits.append(output["recognition_logits"].detach().float().cpu())
        losses.append(float(loss.detach().cpu().item()))
    if training and len(loader) % accumulation != 0:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return torch.cat(logits), float(np.mean(losses))


def save_evidence_analysis(config, model, valid, device, output_path):
    model.eval()
    loader = DataLoader(
        TensorDataset(valid["features"], valid["mask"]),
        batch_size=int(config["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    indices, scores, weights = [], [], []
    selected_indices, selected_scores = [], []
    with torch.no_grad():
        for features, masks in loader:
            output = model(
                features.to(device),
                masks.to(device),
                return_diagnostics=True,
            )
            indices.append(output["ranked_indices"].cpu())
            scores.append(output["ranked_scores"].cpu())
            weights.append(output["evidence_weights"].cpu())
            selected_indices.append(output["evidence_indices"].cpu())
            selected_scores.append(output["evidence_scores"].cpu())
    ranked_indices = torch.cat(indices)
    ranked_scores = torch.cat(scores)
    evidence_weights = torch.cat(weights)
    evidence_indices = torch.cat(selected_indices)
    evidence_scores = torch.cat(selected_scores)
    active_count = valid["mask"].sum(dim=1).float()
    per_label = {}
    examples = []
    analysis_k = int(config["analysis_top_k"])
    adjacency = int(config["region_adjacency"])
    for label_id, label_name in enumerate(config["label_names"]):
        selected = ranked_indices[:, label_id, :analysis_k]
        region_counts, region_lengths, densities = [], [], []
        for row in range(len(valid["ids"])):
            chunks = sorted(
                {
                    int(index)
                    for index, score in zip(
                        selected[row].tolist(),
                        ranked_scores[row, label_id, :analysis_k].tolist(),
                    )
                    if bool(valid["mask"][row, int(index)]) and float(score) > -1e20
                }
            )
            runs = []
            for chunk in chunks:
                if not runs or chunk - runs[-1][-1] > adjacency:
                    runs.append([chunk])
                else:
                    runs[-1].append(chunk)
            region_counts.append(len(runs))
            region_lengths.append(
                float(np.mean([len(item) for item in runs])) if runs else 0.0
            )
            densities.append(len(chunks) / max(1.0, float(active_count[row])))
            if row < int(config["example_count"]):
                examples.append(
                    {
                        "id": valid["ids"][row],
                        "label": label_name,
                        "top_10_indices": [
                            int(x)
                            for x in ranked_indices[row, label_id, :analysis_k].tolist()
                        ],
                        "top_10_scores": [
                            float(x)
                            for x in ranked_scores[row, label_id, :analysis_k].tolist()
                        ],
                        "top_5_weights": [
                            float(x)
                            for x in evidence_weights[row, label_id].tolist()
                        ],
                    }
                )
        valid_scores = []
        for row in range(len(valid["ids"])):
            for index, score in zip(
                ranked_indices[row, label_id, :analysis_k].tolist(),
                ranked_scores[row, label_id, :analysis_k].tolist(),
            ):
                if bool(valid["mask"][row, int(index)]) and float(score) > -1e20:
                    valid_scores.append(float(score))
        per_label[label_name] = {
            "mean_region_count": float(np.mean(region_counts)),
            "mean_region_length": float(np.mean(region_lengths)),
            "mean_evidence_density": float(np.mean(densities)),
            "mean_top_10_score": float(np.mean(valid_scores)) if valid_scores else 0.0,
        }
    overlaps = []
    overlap_k = int(config["evidence_overlap_k"])
    for left in range(int(config["num_labels"])):
        for right in range(left + 1, int(config["num_labels"])):
            values = []
            for row in range(len(valid["ids"])):
                left_set = {
                    int(index)
                    for index in ranked_indices[row, left, :overlap_k].tolist()
                    if bool(valid["mask"][row, int(index)])
                }
                right_set = {
                    int(index)
                    for index in ranked_indices[row, right, :overlap_k].tolist()
                    if bool(valid["mask"][row, int(index)])
                }
                values.append(len(left_set & right_set) / max(1, len(left_set | right_set)))
            overlaps.append(
                {
                    "label_left": config["label_names"][left],
                    "label_right": config["label_names"][right],
                    "top_k": overlap_k,
                    "jaccard": float(np.mean(values)),
                }
            )
    report = {
        "diagnostic_only": True,
        "ground_truth_claim": False,
        "region_rule": "chunk index difference <= 1",
        "per_label": per_label,
        "label_overlap": overlaps,
        "examples": examples,
        "test_checked": False,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {
            "ids": valid["ids"],
            "ranked_indices": ranked_indices,
            "ranked_scores": ranked_scores,
            "evidence_indices": evidence_indices,
            "evidence_scores": evidence_scores,
            "evidence_weights": evidence_weights,
            "test_checked": False,
        },
        output_path.with_suffix(".pt"),
    )


def train_one(config, variant, seed, train, valid, device, root):
    set_seed(seed)
    model = build_model(config, device, variant)
    pos_weight = make_pos_weight(train["labels"], config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and bool(config.get("fp16", True))
    )
    run_dir = root / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    best_state = None
    history = []
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _, train_loss = run_epoch(
            model,
            train,
            device,
            config["batch_size"],
            pos_weight,
            config,
            optimizer,
            scaler,
        )
        with torch.no_grad():
            valid_logits, valid_loss = run_epoch(
                model,
                valid,
                device,
                config["eval_batch_size"],
                pos_weight,
                config,
            )
        fixed, tuned, threshold_report = threshold_metrics(
            config, valid["labels"], valid_logits
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "fixed_macro_f1": float(fixed["recognition_macro_f1"]),
            "fixed_micro_f1": float(fixed["recognition_micro_f1"]),
            "tuned_macro_f1": float(tuned["recognition_macro_f1"]),
            "tuned_micro_f1": float(tuned["recognition_micro_f1"]),
            "per_label_f1": [float(x) for x in tuned["per_label_f1"]],
            "thresholds": threshold_report["thresholds"],
            "max_memory_allocated_mb": (
                float(torch.cuda.max_memory_allocated(device) / 2**20)
                if device.type == "cuda"
                else 0.0
            ),
        }
        history.append(record)
        print(
            f"[{variant}] seed={seed} epoch={epoch} "
            f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"fixed_macro={record['fixed_macro_f1']:.6f} "
            f"tuned_macro={record['tuned_macro_f1']:.6f}",
            flush=True,
        )
        if record["tuned_macro_f1"] > best_score:
            best_score = record["tuned_macro_f1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    model.load_state_dict(best_state, strict=True)
    inference_start = time.perf_counter()
    with torch.no_grad():
        valid_logits, _ = run_epoch(
            model,
            valid,
            device,
            config["eval_batch_size"],
            pos_weight,
            config,
        )
    inference_seconds = float(time.perf_counter() - inference_start)
    fixed, tuned, threshold_report = threshold_metrics(
        config, valid["labels"], valid_logits
    )
    probabilities = torch.sigmoid(valid_logits).numpy()
    summary = {
        "variant": variant,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "train_only": True,
        "test_checked": False,
        "fixed_macro_f1": float(fixed["recognition_macro_f1"]),
        "fixed_micro_f1": float(fixed["recognition_micro_f1"]),
        "tuned_macro_f1": float(tuned["recognition_macro_f1"]),
        "tuned_micro_f1": float(tuned["recognition_micro_f1"]),
        "per_label_f1": [float(x) for x in tuned["per_label_f1"]],
        "per_label_precision": [float(x) for x in tuned["per_label_precision"]],
        "per_label_recall": [float(x) for x in tuned["per_label_recall"]],
        "thresholds": threshold_report["thresholds"],
        "length_metrics": length_metrics(
            config,
            valid["labels"],
            probabilities,
            valid["mask"],
            threshold_report["thresholds"],
        ),
        "parameter_count_total": int(sum(p.numel() for p in model.parameters())),
        "parameter_count_trainable": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "max_memory_allocated_mb": max(
            [x["max_memory_allocated_mb"] for x in history] or [0.0]
        ),
        "wall_time_seconds": float(time.perf_counter() - start),
        "valid_inference_seconds": inference_seconds,
        "history": history,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "variant": variant,
            "seed": seed,
            "test_checked": False,
        },
        run_dir / "best.pt",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_evidence_analysis(
        config, model, valid, device, run_dir / "evidence_analysis.json"
    )
    return summary


def load_reference(config, valid, device):
    model = load_m0(config, device)
    with torch.no_grad():
        logits, _ = run_epoch(
            model,
            valid,
            device,
            config["eval_batch_size"],
            torch.ones(int(config["num_labels"])),
            config,
        )
    fixed, tuned, selected = threshold_metrics(config, valid["labels"], logits)
    return {
        "name": "M0",
        "checkpoint": str(config["baseline_checkpoint"]),
        "fixed_macro_f1": float(fixed["recognition_macro_f1"]),
        "fixed_micro_f1": float(fixed["recognition_micro_f1"]),
        "tuned_macro_f1": float(tuned["recognition_macro_f1"]),
        "tuned_micro_f1": float(tuned["recognition_micro_f1"]),
        "per_label_f1": [float(x) for x in tuned["per_label_f1"]],
        "thresholds": selected["thresholds"],
        "test_checked": False,
    }


def aggregate(config, summaries, m0, root):
    variants = {}
    for variant in config["variants"]:
        rows = [item for item in summaries if item["variant"] == variant]
        values = [item["tuned_macro_f1"] for item in rows]
        deltas = [value - m0["tuned_macro_f1"] for value in values]
        variants[variant] = {
            "runs": rows,
            "mean_tuned_macro_f1": float(np.mean(values)),
            "std_tuned_macro_f1": float(np.std(values)),
            "mean_delta_vs_m0": float(np.mean(deltas)),
            "improved_seed_count": int(sum(delta > 0 for delta in deltas)),
            "seed_count": len(rows),
            "per_label_mean_f1": [
                float(np.mean([item["per_label_f1"][index] for item in rows]))
                for index in range(int(config["num_labels"]))
            ],
        }
    full = variants["full"]
    go = full["mean_delta_vs_m0"] > 0 and full["improved_seed_count"] >= 2
    report = {
        "route": config["route_name"],
        "dataset": "DIVE Main6 random split",
        "seed_protocol": [int(x) for x in config["seeds"]],
        "test_checked": False,
        "phase5_started": False,
        "m0_reference": m0,
        "vanilla_mil_reference": {
            **m0,
            "name": "vanilla_mil",
            "note": "The historical M0 is the existing vanilla label-conditioned MIL reference.",
        },
        "variants": variants,
        "decision": "GO" if go else "NO-GO",
        "decision_rule": {
            "mean_delta_vs_m0_positive": True,
            "minimum_improved_seeds": 2,
        },
        "next_step": (
            "Proceed only with validation review if GO; otherwise stop and perform failure analysis."
        ),
        "restrictions": [
            "No influence pseudo-label",
            "No random-positive evidence supervision",
            "No prototype memory, contract retrieval, ranking loss, or contrastive loss",
            "No test data, label, cache, or prediction",
            "Evidence outputs are diagnostic and not ground-truth localization",
        ],
    }
    (root / "evidence_retrieval_mvp_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Evidence Retrieval MVP",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "DIVE Main6 random split; train/valid only. Evidence indices and weights are diagnostic outputs, not ground-truth evidence.",
        "",
        "| Variant | Mean tuned Macro-F1 | Std | Mean delta vs M0 | Improved seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in variants.items():
        lines.append(
            f"| {name} | {item['mean_tuned_macro_f1']:.6f} | "
            f"{item['std_tuned_macro_f1']:.6f} | "
            f"{item['mean_delta_vs_m0']:.6f} | "
            f"{item['improved_seed_count']}/{item['seed_count']} |"
        )
    lines.extend(
        [
            "",
            f"M0 tuned Macro-F1: **{m0['tuned_macro_f1']:.6f}**",
            "",
            "The MVP uses residual fusion, cross-label score competition, Top-5 MMR-style diversity selection, and classification BCE only.",
        ]
    )
    (root / "evidence_retrieval_mvp_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "report", "all"), nargs="?", default="all")
    parser.add_argument("--config", default="configs/evidence_retrieval_mvp.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if bool(config.get("allow_test", False)):
        raise ValueError("Evidence Retrieval MVP keeps test locked")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(config, "train")
    valid = load_split(config, "valid")
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("train", "all"):
        summaries = []
        for variant in config["variants"]:
            for seed in config["seeds"]:
                summaries.append(
                    train_one(config, variant, int(seed), train, valid, device, root)
                )
        (root / "run_summaries.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
    if args.stage in ("report", "all"):
        summaries = json.loads((root / "run_summaries.json").read_text(encoding="utf-8"))
        report = aggregate(config, summaries, load_reference(config, valid, device), root)
        full = report["variants"]["full"]
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "m0": report["m0_reference"]["tuned_macro_f1"],
                    "full_mean": full["mean_tuned_macro_f1"],
                    "full_std": full["std_tuned_macro_f1"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
