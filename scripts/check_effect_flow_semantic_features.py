import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Check effect-flow semantic feature cache.")
    parser.add_argument("--semantic_dir", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--expected_max_chunks", type=int, required=True)
    parser.add_argument("--expected_efpp_dim", type=int, default=22)
    parser.add_argument("--expected_etp_dim", type=int, default=16)
    parser.add_argument("--expected_relation_dim", type=int, default=6)
    parser.add_argument("--expected_global_template_dim", type=int, default=None)
    parser.add_argument("--expected_num_labels", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def count_jsonl(path):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def check_split(split, args):
    semantic_path = resolve_path(args.semantic_dir) / f"{split}.pt"
    feature_path = resolve_path(args.feature_dir) / f"{split}.pt"
    if not semantic_path.exists():
        raise FileNotFoundError(f"missing semantic cache: {semantic_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"missing feature cache: {feature_path}")
    semantic = torch.load(semantic_path, map_location="cpu")
    feature = torch.load(feature_path, map_location="cpu")
    expected_file = "train_mlsmote.jsonl" if split == "train" else f"{split}.jsonl"
    expected_samples = count_jsonl(resolve_path(args.data_dir) / expected_file)

    efpp = semantic["efpp_probs"]
    etp = semantic["etp_distribution"]
    relation = semantic["relation_distribution"]
    evidence = semantic["vulnerability_evidence_probs"]
    template = semantic["template_match_scores"]
    pseudo_evidence = semantic["chunk_vulnerability_evidence"]
    pseudo_template = semantic["vulnerability_template_matches"]
    active_mask = semantic["active_vulnerability_label_mask"]
    chunk_mask = semantic["chunk_mask"].bool()
    ids_match = [str(v) for v in semantic["ids"]] == [str(v) for v in feature["ids"]]
    chunk_mask_matches = torch.equal(chunk_mask, feature["chunk_mask"].bool())
    labels_match = torch.equal(
        semantic["multi_labels"].float(),
        feature["multi_labels"].float(),
    )

    if efpp.shape != (expected_samples, args.expected_max_chunks, args.expected_efpp_dim):
        raise ValueError(f"{semantic_path} efpp shape mismatch: {tuple(efpp.shape)}")
    if etp.shape != (expected_samples, args.expected_max_chunks, args.expected_etp_dim):
        raise ValueError(f"{semantic_path} etp shape mismatch: {tuple(etp.shape)}")
    if relation.shape != (expected_samples, args.expected_max_chunks, args.expected_relation_dim):
        raise ValueError(f"{semantic_path} relation shape mismatch: {tuple(relation.shape)}")
    global_template_dim = (
        int(args.expected_global_template_dim)
        if args.expected_global_template_dim is not None
        else int(template.shape[-1])
    )
    for name, tensor in (
        ("vulnerability_evidence_probs", evidence),
        ("template_match_scores", template),
        ("chunk_vulnerability_evidence", pseudo_evidence),
        ("vulnerability_template_matches", pseudo_template),
    ):
        if tensor.shape != (expected_samples, args.expected_max_chunks, global_template_dim):
            raise ValueError(f"{semantic_path} {name} shape mismatch: {tuple(tensor.shape)}")
    if active_mask.shape != (expected_samples, global_template_dim):
        raise ValueError(f"{semantic_path} active_vulnerability_label_mask shape mismatch: {tuple(active_mask.shape)}")
    if semantic["multi_labels"].shape[1] != args.expected_num_labels:
        raise ValueError(f"{semantic_path} label width mismatch")
    if not ids_match:
        raise ValueError(f"{split}: ids do not match feature cache")
    if not chunk_mask_matches:
        raise ValueError(f"{split}: chunk_mask does not match feature cache")
    if not labels_match:
        raise ValueError(f"{split}: labels do not match feature cache")
    if torch.isnan(efpp.float()).any() or torch.isinf(efpp.float()).any():
        raise ValueError(f"{semantic_path} contains NaN/Inf efpp_probs")
    if torch.isnan(etp.float()).any() or torch.isinf(etp.float()).any():
        raise ValueError(f"{semantic_path} contains NaN/Inf etp_distribution")
    if torch.isnan(relation.float()).any() or torch.isinf(relation.float()).any():
        raise ValueError(f"{semantic_path} contains NaN/Inf relation_distribution")
    if torch.isnan(evidence.float()).any() or torch.isinf(evidence.float()).any():
        raise ValueError(f"{semantic_path} contains NaN/Inf vulnerability_evidence_probs")
    if torch.isnan(template.float()).any() or torch.isinf(template.float()).any():
        raise ValueError(f"{semantic_path} contains NaN/Inf template_match_scores")

    etp_sum = etp.float().sum(dim=-1)
    relation_sum = relation.float().sum(dim=-1)
    active = chunk_mask
    return {
        "split": split,
        "semantic_path": project_relative(semantic_path),
        "feature_path": project_relative(feature_path),
        "samples": int(efpp.shape[0]),
        "expected_samples": expected_samples,
        "efpp_shape": list(efpp.shape),
        "etp_distribution_shape": list(etp.shape),
        "relation_distribution_shape": list(relation.shape),
        "vulnerability_evidence_probs_shape": list(evidence.shape),
        "template_match_scores_shape": list(template.shape),
        "ids_match_feature_cache": ids_match,
        "chunk_mask_matches_feature_cache": chunk_mask_matches,
        "labels_match_feature_cache": labels_match,
        "efpp_nan_count": int(torch.isnan(efpp.float()).sum().item()),
        "efpp_inf_count": int(torch.isinf(efpp.float()).sum().item()),
        "etp_nan_count": int(torch.isnan(etp.float()).sum().item()),
        "etp_inf_count": int(torch.isinf(etp.float()).sum().item()),
        "relation_nan_count": int(torch.isnan(relation.float()).sum().item()),
        "relation_inf_count": int(torch.isinf(relation.float()).sum().item()),
        "vep_nan_count": int(torch.isnan(evidence.float()).sum().item()),
        "vep_inf_count": int(torch.isinf(evidence.float()).sum().item()),
        "vtm_nan_count": int(torch.isnan(template.float()).sum().item()),
        "vtm_inf_count": int(torch.isinf(template.float()).sum().item()),
        "mean_real_chunks": float(chunk_mask.sum(dim=1).float().mean().item()),
        "max_real_chunks": int(chunk_mask.sum(dim=1).max().item()),
        "mean_active_etp_distribution_sum": float(etp_sum[active].mean().item()),
        "mean_active_relation_distribution_sum": float(relation_sum[active].mean().item()),
        "mean_efpp_probability": float(efpp.float()[active].mean().item()),
        "mean_vulnerability_evidence_probability": float(evidence.float()[active].mean().item()),
        "mean_template_match_score": float(template.float()[active].mean().item()),
        "global_template_dim": global_template_dim,
    }


def main():
    args = parse_args()
    report = {
        "status": "ok",
        "semantic_dir": project_relative(resolve_path(args.semantic_dir)),
        "feature_dir": project_relative(resolve_path(args.feature_dir)),
        "expected_max_chunks": args.expected_max_chunks,
        "expected_efpp_dim": args.expected_efpp_dim,
        "expected_etp_dim": args.expected_etp_dim,
        "expected_relation_dim": args.expected_relation_dim,
        "expected_global_template_dim": args.expected_global_template_dim,
        "expected_num_labels": args.expected_num_labels,
        "splits": {split: check_split(split, args) for split in ["train", "valid", "test"]},
    }
    txt_path = resolve_path(args.output)
    json_path = txt_path.with_suffix(".json")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Effect-flow semantic feature cache check", ""]
    for key, value in report.items():
        if key == "splits":
            lines.append("splits:")
            for split, row in value.items():
                lines.append(f"- {split}:")
                for skey, svalue in row.items():
                    lines.append(f"  {skey}: {svalue}")
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {project_relative(txt_path)}")
    print(f"[OK] wrote {project_relative(json_path)}")


if __name__ == "__main__":
    main()
