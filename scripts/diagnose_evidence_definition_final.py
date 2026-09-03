"""Final validation-only diagnosis of the evidence unit.

This script does not train a model.  It reuses three existing M0 influence
caches (or computes leave-one-chunk-out influence from three checkpoints) and
compares individual chunks with budget-matched contiguous regions/sets.
"""

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evm_chunk_mil_model import MLM8ViewMultiSlotMIL  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs  # noqa: E402
from train_chunk_mil import load_config as load_mil_config  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    raw = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    base_path = raw.get("base_config", "configs/retrieval_validation.yaml")
    base = yaml.safe_load(resolve(base_path).read_text(encoding="utf-8"))
    base.update({key: value for key, value in raw.items() if key != "base_config"})
    return base


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data(config):
    # Only validation features are needed: all checkpoint/influence artifacts
    # are train-originated, and loading the multi-GB train cache is unnecessary.
    payload = torch.load(resolve(config["feature_dir"]) / "valid.pt", map_location="cpu")
    features = payload["features"].float()
    mask = payload["chunk_mask"].bool()
    labels = payload["multi_labels"].float()
    valid = {"ids": [str(x) for x in payload["ids"]], "features": features, "mask": mask, "labels": labels}
    if valid["features"].ndim != 4 or valid["features"].shape[2:] != (8, int(config["feature_dim"])):
        raise ValueError(f"valid: unexpected feature shape {tuple(valid['features'].shape)}")
    if valid["mask"].shape != valid["features"].shape[:2] or valid["labels"].shape[1] != int(config["num_labels"]):
        raise ValueError("valid cache shape mismatch")
    if (~valid["mask"]).all(dim=1).any() or not torch.isfinite(valid["features"]).all():
        raise ValueError("valid cache has empty sample or NaN/Inf")
    expected = []
    with (resolve(config["data_dir"]) / "valid.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected.append(str(json.loads(line)["id"]))
    if expected != valid["ids"]:
        raise ValueError("valid feature cache IDs are not aligned with valid.jsonl")
    return valid


def build_model(config, device):
    model_config = load_mil_config(resolve(config["baseline_config"]), config.get("baseline_variant", "mlm8_slot3"))
    return MLM8ViewMultiSlotMIL(model_config).to(device)


def load_checkpoint(config, path, device):
    model = build_model(config, device)
    checkpoint = torch.load(resolve(path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError(f"checkpoint has no model state: {path}")
    model.load_state_dict(state, strict=True)
    return model.eval()


def infer(model, features, masks, device, batch_size, attention=False):
    logits, weights = [], []
    with torch.no_grad():
        for left in range(0, len(features), int(batch_size)):
            right = min(left + int(batch_size), len(features))
            output = model(features[left:right].to(device), masks[left:right].to(device=device, dtype=torch.bool), return_attention=attention)
            logits.append(output["recognition_logits"].float().cpu())
            if attention:
                weights.append(output["chunk_attention"].float().cpu())
    result = {"logits": torch.cat(logits)}
    if attention:
        result["attention"] = torch.cat(weights)
    return result


def delete_batch(data, row, chunks):
    chunks = [int(x) for x in chunks]
    masks = data["mask"][row:row + 1].expand(len(chunks), -1).clone()
    for item, chunk in enumerate(chunks):
        masks[item, chunk] = False
    if (~masks).all(dim=1).any():
        raise ValueError("cannot delete the only active chunk")
    features = data["features"][row:row + 1].expand(len(chunks), -1, -1, -1).contiguous()
    return features, masks


def compute_influence(config, data, model, device):
    batch_size = int(config["influence_batch_size"])
    baseline = infer(model, data["features"], data["mask"], device, batch_size, True)
    baseline_probs = torch.sigmoid(baseline["logits"])
    values = torch.full((len(data["ids"]), data["mask"].shape[1], int(config["num_labels"])), float("nan"))
    for row in range(len(data["ids"])):
        active = torch.where(data["mask"][row])[0].tolist()
        if len(active) <= 1:
            continue
        deleted = []
        for left in range(0, len(active), batch_size):
            current = active[left:left + batch_size]
            features, masks = delete_batch(data, row, current)
            deleted.append(torch.sigmoid(infer(model, features, masks, device, len(current))["logits"]))
        values[row, active] = baseline_probs[row].unsqueeze(0) - torch.cat(deleted)
        if (row + 1) % int(config.get("influence_progress_every", 100)) == 0:
            print(f"[influence] model_contracts={row + 1}/{len(data['ids'])}", flush=True)
    return {"baseline_probs": baseline_probs, "baseline_attention": baseline["attention"], "influence": values}


def jaccard(a, b):
    a, b = set(a), set(b)
    return float(len(a & b) / len(a | b)) if a or b else 1.0


def rank(values, active, k):
    candidates = [i for i in active if np.isfinite(float(values[i]))]
    return sorted(candidates, key=lambda i: float(values[i]), reverse=True)[:min(int(k), len(candidates))]


def regions(chunks, adjacency=1):
    ordered = sorted(set(int(x) for x in chunks))
    output = []
    for chunk in ordered:
        if not output or chunk - output[-1][-1] > int(adjacency):
            output.append([chunk])
        else:
            output[-1].append(chunk)
    return output


def interval_iou(left, right):
    left = (min(left), max(left))
    right = (min(right), max(right))
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = max(left[1], right[1]) - min(left[0], right[0]) + 1
    return float(intersection / union) if union else 0.0


def region_overlap(left, right):
    """Symmetric mean best interval IoU; unlike chunk Jaccard, near regions overlap."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    forward = np.mean([max(interval_iou(a, b) for b in right) for a in left])
    backward = np.mean([max(interval_iou(b, a) for a in left) for b in right])
    return float((forward + backward) / 2.0)


def coverage_iou(left, right):
    return jaccard([x for region in left for x in region], [x for region in right for x in region])


def region_distance(left, right):
    if not left or not right:
        return float("nan")
    return float(np.mean([min(abs((a[0] + a[-1]) / 2 - (b[0] + b[-1]) / 2) for b in right) for a in left]))


def stability(config, data, reports):
    rows = []
    ks = [int(x) for x in config["ranking_top_k_values"]]
    for left, right in itertools.combinations(range(len(reports)), 2):
        for label_id, label_name in enumerate(config["label_names"]):
            for k in ks:
                a, b = [], []
                for row in range(len(data["ids"])):
                    active = torch.where(data["mask"][row])[0].tolist()
                    ca = rank(reports[left]["influence"][row, :, label_id], active, k)
                    cb = rank(reports[right]["influence"][row, :, label_id], active, k)
                    if ca and cb:
                        a.append(regions(ca, config["region_adjacency"]))
                        b.append(regions(cb, config["region_adjacency"]))
                rows.append({"model_left": left, "model_right": right, "label": label_name, "top_k": k, "chunk_jaccard": float(np.mean([jaccard([x for z in aa for x in z], [x for z in bb for x in z]) for aa, bb in zip(a, b)])) if a else 0.0, "region_interval_iou": float(np.mean([region_overlap(aa, bb) for aa, bb in zip(a, b)])) if a else 0.0, "region_coverage_iou": float(np.mean([coverage_iou(aa, bb) for aa, bb in zip(a, b)])) if a else 0.0, "contracts": len(a)})
    return rows


def summarize_stability(rows):
    result = {}
    for k in sorted({int(r["top_k"]) for r in rows}):
        subset = [r for r in rows if int(r["top_k"]) == k]
        result[str(k)] = {name: float(np.mean([r[name] for r in subset])) for name in ("chunk_jaccard", "region_interval_iou", "region_coverage_iou")}
    return result


def region_stats(config, data, report):
    rows = []
    adjacency = int(config["region_adjacency"])
    for label_id, label_name in enumerate(config["label_names"]):
        positive = torch.where(data["labels"][:, label_id] > 0.5)[0].tolist()
        for k in config["ranking_top_k_values"]:
            stats = []
            for row in positive:
                active = torch.where(data["mask"][row])[0].tolist()
                selected = rank(report["influence"][row, :, label_id], active, int(k))
                if selected:
                    rr = regions(selected, adjacency)
                    stats.append((len(rr), np.mean([len(x) for x in rr]), len(selected) / max(1, len(active))))
            rows.append({"label": label_name, "top_k": int(k), "mean_region_count": float(np.mean([x[0] for x in stats])) if stats else 0.0, "mean_region_length": float(np.mean([x[1] for x in stats])) if stats else 0.0, "mean_evidence_density": float(np.mean([x[2] for x in stats])) if stats else 0.0, "positive_contracts": len(stats)})
    return rows


def select_budget_regions(values, active, region_count, budget, adjacency):
    """Select at most region_count disjoint regions using at most budget chunks."""
    seed_chunks = rank(values, active, max(int(budget), int(region_count), 10))
    candidates = regions(seed_chunks, adjacency)
    candidates.sort(key=lambda r: (sum(float(values[x]) for x in r), -r[0]), reverse=True)
    chosen, used = [], 0
    for candidate in candidates:
        if len(chosen) >= int(region_count) or used >= int(budget):
            break
        remaining = int(budget) - used
        if len(candidate) > remaining:
            if remaining <= 0:
                continue
            best = max((candidate[i:i + remaining] for i in range(len(candidate) - remaining + 1)), key=lambda r: sum(float(values[x]) for x in r))
            candidate = best
        if candidate and not any(set(candidate) & set(x) for x in chosen):
            chosen.append(candidate)
            used += len(candidate)
    if not chosen:
        fallback = rank(values, active, 1)
        return [[fallback[0]]] if fallback else []
    return chosen


def metric_from_logits(data, logits):
    metric = compute_multilabel_metrics_from_probs(data["labels"].numpy(), torch.sigmoid(logits).numpy(), 0.5)
    return {"macro_f1": float(metric["recognition_macro_f1"]), "micro_f1": float(metric["recognition_micro_f1"]), "per_label_f1": [float(x) for x in metric["per_label_f1"]]}


def isolation(config, data, report, model, device):
    results = []
    for method, ks in (("influence_chunks", config["ranking_top_k_values"]), ("influence_regions", config["region_count_values"])):
        for k in ks:
            joined = torch.zeros((len(data["ids"]), int(config["num_labels"])))
            densities, selected_counts, budget_utilization = [], [], []
            for label_id in range(int(config["num_labels"])):
                masks = torch.zeros_like(data["mask"])
                for row in range(len(data["ids"])):
                    active = torch.where(data["mask"][row])[0].tolist()
                    values = report["influence"][row, :, label_id]
                    if method == "influence_chunks":
                        selected = rank(values, active, int(k))
                    else:
                        selected = [x for r in select_budget_regions(values, active, int(k), int(k), int(config["region_adjacency"])) for x in r]
                    if not selected:
                        selected = active[:1]
                    masks[row, selected] = True
                    densities.append(len(selected) / max(1, len(active)))
                    selected_counts.append(len(selected))
                    budget_utilization.append(len(selected) / max(1, int(k)))
                output = infer(model, data["features"], masks, device, int(config["phase4_batch_size"]), False)
                joined[:, label_id] = output["logits"][:, label_id]
            metric = metric_from_logits(data, joined)
            metric.update({"method": method, "top_k": int(k), "chunk_budget": int(k), "mean_selected_chunks": float(np.mean(selected_counts)), "mean_evidence_density": float(np.mean(densities)), "mean_budget_utilization": float(np.mean(budget_utilization))})
            results.append(metric)
            print(f"[isolation] method={method} k={k} macro_f1={metric['macro_f1']:.6f}", flush=True)
    return results


def specificity(config, data, report):
    rows = []
    adjacency = int(config["region_adjacency"])
    pairs = config.get("label_pairs") or list(itertools.combinations(config["label_names"], 2))
    name_to_id = {name: i for i, name in enumerate(config["label_names"])}
    for left_name, right_name in pairs:
        left_id, right_id = name_to_id[left_name], name_to_id[right_name]
        both = torch.where((data["labels"][:, left_id] > 0.5) & (data["labels"][:, right_id] > 0.5))[0].tolist()
        for k in config["ranking_top_k_values"]:
            chunk_values, region_values, distances, overlaps = [], [], [], []
            for row in both:
                active = torch.where(data["mask"][row])[0].tolist()
                a = regions(rank(report["influence"][row, :, left_id], active, int(k)), adjacency)
                b = regions(rank(report["influence"][row, :, right_id], active, int(k)), adjacency)
                chunk_values.append(jaccard([x for z in a for x in z], [x for z in b for x in z]))
                region_values.append(region_overlap(a, b))
                distances.append(region_distance(a, b))
                overlaps.append(len(set(x for z in a for x in z) & set(x for z in b for x in z)) / max(1, min(sum(map(len, a)), sum(map(len, b)))))
            rows.append({"label_left": left_name, "label_right": right_name, "top_k": int(k), "chunk_jaccard": float(np.mean(chunk_values)) if chunk_values else 0.0, "region_interval_iou": float(np.mean(region_values)) if region_values else 0.0, "mean_region_distance": float(np.mean(distances)) if distances else 0.0, "overlap_ratio": float(np.mean(overlaps)) if overlaps else 0.0, "contracts_with_both_labels": len(both)})
    return rows


def load_or_compute_reports(config, data, device):
    source = resolve(config.get("source_result_dir", config["result_dir"]))
    reports, paths = [], []
    for index in range(3):
        cache_path = source / f"influence_model_{index}.pt"
        checkpoint_path = config["model_checkpoints"][index]
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            report = {"baseline_probs": payload["baseline_probs"].float(), "baseline_attention": payload["baseline_attention"].float(), "influence": payload["influence"].float()}
            if report["influence"].shape[:2] != data["mask"].shape:
                raise ValueError(f"influence cache shape mismatch: {cache_path}")
        else:
            model = load_checkpoint(config, checkpoint_path, device)
            report = compute_influence(config, data, model, device)
        reports.append(report)
        paths.append(str(checkpoint_path))
    return reports, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evidence_definition_final.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if bool(config.get("allow_test", False)) or str(config.get("allow_test", "0")) == "1":
        raise ValueError("final evidence definition diagnosis must keep test locked")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(config)
    root = resolve(config["result_dir"])
    root.mkdir(parents=True, exist_ok=True)
    reports, paths = load_or_compute_reports(config, data, device)
    model = load_checkpoint(config, paths[0], device)
    stability_rows = stability(config, data, reports)
    spatial_rows = region_stats(config, data, reports[0])
    specificity_rows = specificity(config, data, reports[0])
    isolation_rows = isolation(config, data, reports[0], model, device)
    write_csv(root / "chunk_region_stability.csv", stability_rows)
    write_csv(root / "region_spatial_distribution.csv", spatial_rows)
    write_csv(root / "label_specificity_chunk_region.csv", specificity_rows)
    write_json(root / "chunk_vs_region_isolation.json", {"label_names": config["label_names"], "results": isolation_rows, "test_checked": False, "diagnostic_upper_bound": True})
    summary = summarize_stability(stability_rows)
    influence_margin = {str(k): next((r["macro_f1"] for r in isolation_rows if r["method"] == "influence_regions" and r["top_k"] == k), 0.0) - next((r["macro_f1"] for r in isolation_rows if r["method"] == "influence_chunks" and r["top_k"] == k), 0.0) for k in config["region_count_values"]}
    region_stats_by_k = {str(k): [r for r in spatial_rows if int(r["top_k"]) == int(k)] for k in config["ranking_top_k_values"]}
    top5_summary = summary.get("5", {})
    main_region_gains = [influence_margin[str(k)] for k in (3, 5) if str(k) in influence_margin]
    mean_region_gain = float(np.mean(main_region_gains)) if main_region_gains else 0.0
    region_stability_pass = float(top5_summary.get("region_interval_iou", 0.0)) >= float(config["go_min_top5_region_iou"])
    region_gain_pass = mean_region_gain >= float(config["go_min_region_macro_margin"])
    decision = "GO" if region_stability_pass and region_gain_pass else "NO-GO"
    report = {
        "route": config["route_name"], "dataset": "DIVE Main6 random split", "seed": int(config["seed"]),
        "test_checked": False, "phase5_started": False, "diagnostic_upper_bound": True,
        "region_rule": {"adjacency": "chunk index difference <= 1", "expanded_interval_iou": "symmetric mean best interval IoU", "coverage_iou": "Jaccard of selected chunk coverage"},
        "m0_checkpoints": paths,
        "stability_summary": summary,
        "region_spatial_summary": region_stats_by_k,
        "label_specificity_summary": {"mean_top5_chunk_jaccard": float(np.mean([r["chunk_jaccard"] for r in specificity_rows if r["top_k"] == 5])) if specificity_rows else 0.0, "mean_top5_region_interval_iou": float(np.mean([r["region_interval_iou"] for r in specificity_rows if r["top_k"] == 5])) if specificity_rows else 0.0},
        "budget_matched_region_minus_chunk_macro_f1": influence_margin,
        "decision": decision,
        "decision_criteria": {"mean_top5_region_interval_iou": float(top5_summary.get("region_interval_iou", 0.0)), "min_top5_region_interval_iou": float(config["go_min_top5_region_iou"]), "mean_region_macro_gain_at_k3_k5": mean_region_gain, "min_region_macro_margin": float(config["go_min_region_macro_margin"]), "region_stability_pass": region_stability_pass, "region_gain_pass": region_gain_pass},
        "next_evidence_unit": "Multi-Chunk Evidence Set",
        "recommendation": "Proceed to a minimal weakly-supervised retriever only if GO. Regardless of the decision, influence remains a diagnostic sensitivity signal and not pseudo ground truth.",
        "artifacts": {"stability": "chunk_region_stability.csv", "spatial": "region_spatial_distribution.csv", "specificity": "label_specificity_chunk_region.csv", "isolation": "chunk_vs_region_isolation.json"},
        "warnings": ["Influence is M0 decision sensitivity, not vulnerability ground-truth evidence.", "Isolation uses validation labels to join label-specific masks and is an offline diagnostic upper bound, not deployable classification.", "No model was trained and no test data, labels, cache, or prediction was read or created."],
    }
    write_json(root / "evidence_definition_final_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
