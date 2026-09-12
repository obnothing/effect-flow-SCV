"""Validation-only TDVP dictionary, context-use, and shortcut diagnostics."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from functools import partial

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e5_tdvp.model import E5TDVPModel  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet  # noqa: E402
from light_label_runtime import merge_runtime_config  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs, derived_detection_metrics_from_multilabel_probs  # noqa: E402


LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
VARIANTS = ["d1_param_control", "d2_global_mean", "d3_shuffled_dictionary", "d4_tdvp"]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    config = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if config.get("base_config"):
        base = yaml.safe_load(resolve(config["base_config"]).read_text(encoding="utf-8"))
        base.update(config)
        config = base
    runtime = resolve(config.get("runtime_path", "results/light_label/resolved_runtime.json"))
    return merge_runtime_config(config, runtime)


def make_model(config, tokenizer):
    return E5TDVPModel(config["variant"], len(tokenizer), tokenizer.pad_token_id, config["embedding_dim"],
                       config["gru_hidden_size"], config["num_labels"], config["bidirectional"],
                       config.get("local_radius", 8), config.get("gru_layers", 1),
                       dictionary_size=config.get("dictionary_size", 16), joint_dim=config.get("joint_dim", 128)).to(
                           torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def metric(config, labels, logits, thresholds):
    probs = torch.sigmoid(logits).numpy()
    targets = labels.numpy().astype(int)
    value = compute_multilabel_metrics_from_probs(targets, probs, thresholds)
    detection = derived_detection_metrics_from_multilabel_probs(targets, probs, thresholds)
    return {"macro_f1": float(value["recognition_macro_f1"]), "micro_f1": float(value["recognition_micro_f1"]),
            "detection_f1": float(detection["detection_f1"]), "per_label_f1": [float(x) for x in value["per_label_f1"]],
            "thresholds": [float(x) for x in thresholds]}


def entropy(values):
    values = np.asarray(values, dtype=np.float64)
    if values.sum() <= 0:
        return None
    values = values / values.sum()
    return float(-(values * np.log(np.maximum(values, 1e-12))).sum())


def cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


@torch.no_grad()
def evaluate_variant(config, variant, seed, batch_size, device):
    config = dict(config); config["variant"] = variant
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    data = LightLabelDataset(resolve(config["cache_dir"]) / f"valid_max{config['max_len']}.pt", runtime_max_len=config["max_len"])
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate_light_label, pad_id=tokenizer.pad_token_id))
    model = make_model(config, tokenizer)
    path = resolve(config["checkpoint_root"]) / variant / f"seed_{seed}" / "best.pt"
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    thresholds = payload["metrics"]["thresholds"]
    outputs = {name: {"logits": [], "labels": []} for name in ("normal", "context_zero", "context_permutation", "centroid_permutation")}
    context_entropy, context_top1, context_top3, pair_cosine, pair_js = [], [], [], [], []
    amp = device.type == "cuda" and bool(config.get("amp", True))
    for batch in loader:
        inputs = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"].cpu()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            normal = model(inputs, batch["lengths"], mask)
            zero = model(inputs, batch["lengths"], mask, context_zero=True)
            permutation = torch.roll(torch.arange(inputs.shape[0], device=device), shifts=1)
            shuffled = model(inputs, batch["lengths"], mask, context_batch_permutation=permutation)
            centroid = model(inputs, batch["lengths"], mask, centroid_permutation=torch.arange(model.dictionary_size - 1, -1, -1, device=device))
        for name, output in (("normal", normal), ("context_zero", zero), ("context_permutation", shuffled), ("centroid_permutation", centroid)):
            outputs[name]["logits"].append(output["logits"].float().cpu())
            outputs[name]["labels"].append(labels)
        weights = normal["context_attention"].float().cpu().numpy()
        for row in weights:
            for label_id in range(len(LABELS)):
                distribution = row[label_id]
                if distribution.sum() > 0:
                    context_entropy.append((label_id, entropy(distribution)))
                    context_top1.append((label_id, float(np.max(distribution))))
                    context_top3.append((label_id, float(np.sort(distribution)[-min(3, len(distribution)):].sum())))
        row_weights = normal["context_attention"].float().cpu().numpy()
        for row in row_weights:
            norm = row / np.maximum(np.linalg.norm(row, axis=1, keepdims=True), 1e-12)
            matrix = norm @ norm.T
            tri = matrix[np.triu_indices(len(LABELS), 1)]
            pair_cosine.append(float(tri.mean()))
            js_values = []
            for left in range(len(LABELS)):
                for right in range(left + 1, len(LABELS)):
                    p = row[left] / max(row[left].sum(), 1e-12); q = row[right] / max(row[right].sum(), 1e-12)
                    midpoint = 0.5 * (p + q)
                    js_values.append(0.5 * float((p * np.log(np.maximum(p, 1e-12) / np.maximum(midpoint, 1e-12))).sum()) + 0.5 * float((q * np.log(np.maximum(q, 1e-12) / np.maximum(midpoint, 1e-12))).sum()))
            pair_js.append(float(np.mean(js_values)))
    metrics = {}
    for name, values in outputs.items():
        logits = torch.cat(values["logits"]); labels = torch.cat(values["labels"])
        metrics[name] = metric(config, labels, logits, thresholds)
    per_label_entropy = {label: float(np.mean([value for index, value in context_entropy if index == i])) if any(index == i for index, _ in context_entropy) else None for i, label in enumerate(LABELS)}
    return {"variant": variant, "seed": seed, "metrics": metrics,
            "context_attention": {"entropy_mean": per_label_entropy,
                                   "top1_mean": {label: float(np.mean([value for index, value in context_top1 if index == i])) if any(index == i for index, _ in context_top1) else None for i, label in enumerate(LABELS)},
                                   "top3_mass_mean": {label: float(np.mean([value for index, value in context_top3 if index == i])) if any(index == i for index, _ in context_top3) else None for i, label in enumerate(LABELS)},
                                   "label_pair_cosine_mean": float(np.mean(pair_cosine)) if pair_cosine else None,
                                   "label_pair_js_mean": float(np.mean(pair_js)) if pair_js else None},
            "test_checked": False}


def cluster_diagnostics(variant, seed):
    path = ROOT / f"results/e5_tdvp/dictionaries/random/{variant}_seed{seed}.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    assignments = payload["assignments"].numpy()
    labels = payload["labels"].numpy()
    lengths = payload["original_lengths"].numpy()
    sizes = np.bincount(assignments, minlength=payload["metadata"]["clusters"])
    label_prevalence = [[float(labels[assignments == cluster, index].mean()) if np.any(assignments == cluster) else None for index in range(6)] for cluster in range(len(sizes))]
    length_stats = [{"mean": float(lengths[assignments == cluster].mean()), "median": float(np.median(lengths[assignments == cluster]))} if np.any(assignments == cluster) else {"mean": None, "median": None} for cluster in range(len(sizes))]
    train_path = ROOT / "data/processed/DIVE_main6_opcode_process01/train.jsonl"
    skeleton_by_id = {}
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                normalized = " ".join(str(item.get("opcode", "")).split())
                skeleton_by_id[str(item.get("id"))] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    skeleton_stats = []
    for cluster in range(len(sizes)):
        cluster_ids = [str(value) for index, value in enumerate(payload["ids"]) if assignments[index] == cluster]
        values = [skeleton_by_id[value] for value in cluster_ids if value in skeleton_by_id]
        counts = np.asarray(list(__import__("collections").Counter(values).values()), dtype=np.float64)
        probabilities = counts / max(counts.sum(), 1.0)
        skeleton_stats.append({"unique_skeletons": int(len(counts)),
                               "dominant_skeleton_ratio": float(counts.max() / counts.sum()) if len(counts) else None,
                               "skeleton_entropy": float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()) if len(counts) else None})
    return {"metadata": payload["metadata"], "cluster_sizes": sizes.tolist(), "cluster_size_min": int(sizes.min()),
            "cluster_size_max": int(sizes.max()), "cluster_size_mean": float(sizes.mean()),
            "cluster_size_cv": float(sizes.std() / max(sizes.mean(), 1e-12)),
            "cluster_label_prevalence": label_prevalence, "cluster_length": length_stats,
            "cluster_skeleton": skeleton_stats,
            "test_checked": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    config_path = "configs/e5_tdvp/d4_tdvp.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports = []
    for variant in VARIANTS:
        path = resolve(config["checkpoint_root"]) / variant / f"seed_{args.seed}" / "best.pt"
        if not path.exists():
            continue
        item = evaluate_variant(config, variant, args.seed, args.batch_size, device)
        item["dictionary"] = cluster_diagnostics(variant, args.seed)
        reports.append(item)
        output = ROOT / f"results/e5_tdvp/diagnostics/{variant}_seed{args.seed}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    aggregate = {"dataset": "DIVE_main6_opcode_process01", "protocol": "random", "seed": args.seed,
                 "reports": reports, "test_checked": False}
    root = ROOT / "results/e5_tdvp/diagnostics"; root.mkdir(parents=True, exist_ok=True)
    (root / f"diagnostics_seed{args.seed}.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(reports), "seed": args.seed, "device": str(device), "test_checked": False}, indent=2))


if __name__ == "__main__": main()
