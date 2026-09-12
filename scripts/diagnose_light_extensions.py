"""Validation-only mechanism diagnostics for B2 and its independent extensions."""

import argparse
import json
import math
import sys
from pathlib import Path
from functools import partial

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_extensions import LightExtensionNet  # noqa: E402
from light_label_data import LightLabelDataset, collate_light_label  # noqa: E402
from light_label_model import LabelGuidedOpcodeNet  # noqa: E402
from light_label_runtime import merge_runtime_config  # noqa: E402


LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
VARIANTS = [
    ("e0_b2", "configs/light_label/b2_label_attention.yaml", "b2_label_attention"),
    ("e1_vcfm", "configs/light_label/extensions/e1_vcfm.yaml", "e1_vcfm"),
    ("e2_vrop", "configs/light_label/extensions/e2_vrop.yaml", "e2_vrop"),
    ("e3_vasm", "configs/light_label/extensions/e3_vasm.yaml", "e3_vasm"),
    ("e4_pgvr", "configs/light_label/extensions/e4_pgvr.yaml", "e4_pgvr"),
    ("e5_tdvp", "configs/light_label/extensions/e5_tdvp.yaml", "e5_tdvp"),
]


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


def make_model(config, variant, tokenizer):
    args = (variant, len(tokenizer), tokenizer.pad_token_id, config["embedding_dim"],
            config["gru_hidden_size"], config["num_labels"], config["bidirectional"])
    if variant == "b2_label_attention":
        return LabelGuidedOpcodeNet(*args, local_radius=config.get("local_radius", 8),
                                    gru_layers=config.get("gru_layers", 1))
    return LightExtensionNet(
        *args,
        local_radius=config.get("local_radius", 8),
        gru_layers=config.get("gru_layers", 1),
        erase_mass=config.get("erase_mass", 0.30),
        propagation_k=config.get("propagation_k", 4),
        segment_kappa=config.get("segment_kappa", 1.0),
        segment_gap=config.get("segment_gap", 2),
        segment_min_len=config.get("segment_min_len", 2),
        segment_max=config.get("segment_max", 8),
        lambda_consistency=config.get("lambda_consistency", 0.01),
        confounder_clusters=config.get("confounder_clusters", 16),
    )


def mean_or_none(values):
    return float(sum(values) / len(values)) if values else None


def entropy(values):
    values = values.float()
    values = values / values.sum().clamp_min(1e-12)
    return float(-(values * values.clamp_min(1e-12).log()).sum())


def top_mass(values, k):
    k = min(int(k), len(values))
    return float(torch.topk(values.float(), k).values.sum()) if k else 0.0


def cosine(left, right):
    left = left.float(); right = right.float()
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))


def pairwise_offdiag(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.shape[0] < 2:
        return None
    norm = values / values.norm(dim=1, keepdim=True).clamp_min(1e-12)
    matrix = norm @ norm.T
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    return float(matrix[indices[0], indices[1]].mean())


def run_diagnostics(config, variant, run, device, batch_size):
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    data = LightLabelDataset(resolve(config["cache_dir"]) / f"valid_max{config['max_len']}.pt", runtime_max_len=config["max_len"])
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate_light_label, pad_id=tokenizer.pad_token_id))
    model = make_model(config, variant, tokenizer).to(device)
    checkpoint_path = resolve(config["result_dir"]) / run / ".." / ".." / ".." / "checkpoints"  # replaced below
    if variant == "b2_label_attention":
        checkpoint_path = resolve("checkpoints/light_label/b2_label_attention") / run / "best.pt"
    else:
        checkpoint_path = resolve(config["checkpoint_dir"]) / run / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    loaded = model.load_state_dict(payload["model_state_dict"], strict=False)
    if config.get("allow_test"):
        raise ValueError("test is locked")
    model.eval()
    amp = device.type == "cuda" and bool(config.get("amp", True))
    counters = {"samples": 0, "single_label": 0, "multi_label": 0}
    rows = {name: {"count": 0, "attention_entropy": [], "top5_mass": [], "top10_mass": [],
                   "representation_cosine": [], "attention_cosine": []} for name in LABELS}
    extension = {"e1_vcfm": {"top10_overlap": [], "representation_cosine_base_final": [], "beta": []},
                 "e2_vrop": {"anchor_coverage": [], "top10_overlap_base_refined": [], "gamma": []},
                 "e3_vasm": {"segment_count": [], "segment_length": [], "segment_coverage": [], "gamma": []},
                 "e4_pgvr": {"entropy_base": [], "entropy_fused": [], "top5_mass_base": [], "top5_mass_fused": [], "sigma": [], "eta": []},
                 "e5_tdvp": {"context_entropy": [], "representation_context_cosine": [], "dictionary_attention": []}}
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(inputs, batch["lengths"], mask)
            attention = output.get("base_attention", output["attention"]).float().cpu()
            representations = output["representations"].float().cpu()
            labels = batch["labels"].long()
            counters["samples"] += inputs.shape[0]
            counters["single_label"] += int((labels.sum(1) <= 1).sum())
            counters["multi_label"] += int((labels.sum(1) >= 2).sum())
            for row in range(inputs.shape[0]):
                valid = int(batch["lengths"][row])
                values = attention[row, :, :valid]
                for label_id, name in enumerate(LABELS):
                    rows[name]["count"] += 1
                    rows[name]["attention_entropy"].append(entropy(values[label_id]))
                    rows[name]["top5_mass"].append(top_mass(values[label_id], 5))
                    rows[name]["top10_mass"].append(top_mass(values[label_id], 10))
                rows_for_rep = representations[row]
                mean_rep = rows_for_rep.mean(dim=0)
                mean_attention = values.mean(dim=0)
                for label_id, name in enumerate(LABELS):
                    rows[name]["representation_cosine"].append(cosine(rows_for_rep[label_id], mean_rep))
                    rows[name]["attention_cosine"].append(cosine(values[label_id], mean_attention))
                if variant == "e1_vcfm":
                    second = output["complement_attention"][row].float().cpu()[:, :valid]
                    for label_id in range(len(LABELS)):
                        top_a = set(torch.argsort(values[label_id], descending=True)[:min(10, valid)].tolist())
                        top_b = set(torch.argsort(second[label_id], descending=True)[:min(10, valid)].tolist())
                        extension[variant]["top10_overlap"].append(len(top_a & top_b) / max(len(top_a | top_b), 1))
                        extension[variant]["representation_cosine_base_final"].append(cosine(output["base_representations"][row, label_id].float().cpu(), rows_for_rep[label_id]))
                    extension[variant]["beta"].append(float(output["complement_beta"].item()))
                elif variant == "e2_vrop":
                    refined = output["refined_attention"][row].float().cpu()[:, :valid]
                    anchors = output["representative_indices"][row].cpu()
                    for label_id in range(len(LABELS)):
                        top_a = set(torch.argsort(values[label_id], descending=True)[:min(10, valid)].tolist())
                        top_b = set(torch.argsort(refined[label_id], descending=True)[:min(10, valid)].tolist())
                        extension[variant]["top10_overlap_base_refined"].append(len(top_a & top_b) / max(len(top_a | top_b), 1))
                        extension[variant]["anchor_coverage"].append(len(set(int(x) for x in anchors[label_id] if int(x) < valid)) / max(valid, 1))
                    extension[variant]["gamma"].append(float(output["propagation_gamma"].item()))
                elif variant == "e3_vasm":
                    counts = output["segment_counts"][row]
                    lengths = output["segment_lengths"][row]
                    coverage = output["segment_coverage"][row]
                    extension[variant]["segment_count"].extend(counts)
                    extension[variant]["segment_length"].extend([item for row_lengths in lengths for item in row_lengths])
                    extension[variant]["segment_coverage"].extend(coverage)
                    extension[variant]["gamma"].append(float(output["segment_gamma"].item()))
                elif variant == "e4_pgvr":
                    fused = output["fused_attention"][row].float().cpu()[:, :valid]
                    for label_id in range(len(LABELS)):
                        extension[variant]["entropy_base"].append(entropy(values[label_id]))
                        extension[variant]["entropy_fused"].append(entropy(fused[label_id]))
                        extension[variant]["top5_mass_base"].append(top_mass(values[label_id], 5))
                        extension[variant]["top5_mass_fused"].append(top_mass(fused[label_id], 5))
                    extension[variant]["sigma"].extend(output["sigma"].float().cpu().tolist())
                    extension[variant]["eta"].extend(output["eta"].float().cpu().tolist())
                elif variant == "e5_tdvp":
                    context_attention = output["context_attention"][row].float().cpu()
                    context = output["context"][row].float().cpu()
                    for label_id in range(len(LABELS)):
                        extension[variant]["context_entropy"].append(entropy(context_attention[label_id]))
                        extension[variant]["representation_context_cosine"].append(cosine(rows_for_rep[label_id], context[label_id]))
                    extension[variant]["dictionary_attention"].extend(context_attention.reshape(-1).tolist())

    compact_rows = {}
    for name, values in rows.items():
        compact_rows[name] = {key: (value if key == "count" else mean_or_none(value)) for key, value in values.items()}
    compact_extension = {key: {name: mean_or_none(values) for name, values in item.items()} for key, item in extension.items() if key == variant}
    return {
        "variant": variant,
        "run": run,
        "checkpoint": str(checkpoint_path),
        "loaded_missing_keys": list(loaded.missing_keys),
        "loaded_unexpected_keys": list(loaded.unexpected_keys),
        "samples": counters,
        "per_label": compact_rows,
        "global_representation_cosine_mean": mean_or_none([value for item in rows.values() for value in item["representation_cosine"]]),
        "extension": compact_extension.get(variant, {}),
        "test_checked": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="full", choices=["smoke", "full"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--variants", nargs="*", default=[item[0] for item in VARIANTS])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports = []
    for short_name, config_path, variant in VARIANTS:
        if short_name not in args.variants:
            continue
        config = load_config(config_path)
        report = run_diagnostics(config, variant, args.run, device, args.batch_size)
        reports.append(report)
        output_dir = resolve(config["result_dir"]) / args.run
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"variant": variant, "samples": report["samples"]["samples"], "test_checked": False}, indent=2), flush=True)
    root = resolve("results/light_label/extensions")
    root.mkdir(parents=True, exist_ok=True)
    (root / f"diagnostics_{args.run}.json").write_text(json.dumps({"reports": reports, "dataset": "DIVE_main6_opcode_process01", "test_checked": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(reports), "device": str(device), "test_checked": False}, indent=2))


if __name__ == "__main__":
    main()
