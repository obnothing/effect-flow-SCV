import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from transformers import BertForMaskedLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evm_bert_classification_dataset import validate_evm_vocab  # noqa: E402
from effect_flow_ontology import (  # noqa: E402
    RELATION_TYPES,
    load_ontology,
    load_vulnerability_templates,
    pseudo_evidence_vector,
    template_score_vector,
)
from effect_flow_schema import (  # noqa: E402
    annotate_effect_relations,
    annotate_effect_types,
    annotate_efpp_patterns,
    build_token_units,
    effect_type_chunk_histogram,
)
from effect_flow_utils import GLOBAL_VULNERABILITY_LABELS  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract effect-flow semantic probabilities for chunk MIL."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
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


def load_yaml(path):
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_jsonl(path):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                yield idx, json.loads(line)


def encode_chunk(tokenizer, content_tokens, max_len):
    tokens = [tokenizer.cls_token] + content_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    padding = max_len - len(input_ids)
    if padding < 0:
        raise ValueError(f"chunk length {len(input_ids)} exceeds max_len={max_len}")
    if padding:
        input_ids.extend([tokenizer.pad_token_id] * padding)
        attention_mask.extend([0] * padding)
    content_mask = [0] + [1] * len(content_tokens) + [0]
    if padding:
        content_mask.extend([0] * padding)
    return input_ids, attention_mask, content_mask


def build_contract_chunks(tokenizer, opcode, config):
    max_len = int(config["max_len"])
    content_size = int(config.get("chunk_content_size", max_len - 2))
    stride = int(config["chunk_stride"])
    max_chunks = int(config["max_chunks_per_contract"])
    if content_size != max_len - 2:
        raise ValueError(
            f"chunk_content_size must be max_len - 2 ({max_len - 2}), got {content_size}"
        )
    tokens = tokenizer.tokenize(opcode, add_special_tokens=False)
    original_len = len(tokens)
    num_chunks_before = max(1, math.ceil(original_len / stride)) if original_len else 1
    chunks = []
    start = 0
    while start < original_len and len(chunks) < max_chunks:
        chunk_tokens = tokens[start : start + content_size]
        input_ids, attention_mask, content_mask = encode_chunk(tokenizer, chunk_tokens, max_len)
        chunks.append(
            {
                "chunk_tokens": chunk_tokens,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "content_mask": content_mask,
            }
        )
        start += stride
    if not chunks:
        input_ids, attention_mask, content_mask = encode_chunk(tokenizer, [], max_len)
        chunks.append(
            {
                "chunk_tokens": [],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "content_mask": content_mask,
            }
        )
    num_kept = len(chunks)
    covered_tokens = min(original_len, content_size + stride * max(0, num_kept - 1))
    coverage = 1.0 if original_len == 0 else covered_tokens / original_len
    return {
        "chunks": chunks,
        "original_token_length": original_len,
        "num_chunks_before_truncation": num_chunks_before,
        "num_chunks_kept": num_kept,
        "coverage_ratio": float(min(coverage, 1.0)),
        "truncated": num_chunks_before > max_chunks,
    }


class EffectFlowSemanticExtractor(nn.Module):
    def __init__(self, hf_model_path):
        super().__init__()
        path = resolve_path(hf_model_path)
        model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
        heads_path = path / "effect_flow_heads.pt"
        if not heads_path.exists():
            raise FileNotFoundError(f"effect_flow_heads.pt not found: {heads_path}")
        heads = torch.load(heads_path, map_location="cpu")
        hidden_size = int(model.config.hidden_size)
        self.bert = model.bert
        self.etp_head = nn.Linear(hidden_size, int(heads["num_effect_types"]))
        self.efpp_head = nn.Linear(hidden_size, int(heads["num_efpp_patterns"]))
        self.err_head = nn.Linear(hidden_size, int(heads["num_relation_types"]))
        self.vep_head = nn.Linear(hidden_size, int(heads["num_vulnerability_templates"]))
        self.vtm_head = nn.Linear(hidden_size, int(heads["num_vulnerability_templates"]))
        self.etp_head.load_state_dict(heads["etp_head_state_dict"])
        self.efpp_head.load_state_dict(heads["efpp_head_state_dict"])
        self.err_head.load_state_dict(heads["err_head_state_dict"])
        self.vep_head.load_state_dict(heads["vep_head_state_dict"])
        self.vtm_head.load_state_dict(heads["vtm_head_state_dict"])
        self.num_effect_types = int(heads["num_effect_types"])
        self.num_efpp_patterns = int(heads["num_efpp_patterns"])
        self.num_relation_types = int(heads["num_relation_types"])
        self.num_vulnerability_templates = int(heads["num_vulnerability_templates"])
        self.eval()

    def forward(self, input_ids, attention_mask, content_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        etp_probs = torch.softmax(self.etp_head(hidden), dim=-1)
        content_mask = content_mask.bool()
        weights = content_mask.unsqueeze(-1).to(etp_probs.dtype)
        etp_distribution = (etp_probs * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooling_mask = attention_mask.bool()
        pooling_weights = pooling_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * pooling_weights).sum(dim=1) / pooling_weights.sum(dim=1).clamp_min(1.0)
        efpp_probs = torch.sigmoid(self.efpp_head(pooled))
        relation_distribution = torch.softmax(self.err_head(pooled), dim=-1)
        vulnerability_evidence_probs = torch.sigmoid(self.vep_head(pooled))
        template_match_scores = torch.sigmoid(self.vtm_head(pooled))
        return (
            efpp_probs,
            etp_distribution,
            relation_distribution,
            vulnerability_evidence_probs,
            template_match_scores,
        )


def flush_batch(
    model,
    input_ids,
    masks,
    content_masks,
    targets,
    efpp_probs,
    etp_distribution,
    relation_distribution,
    vulnerability_evidence_probs,
    template_match_scores,
    device,
    use_fp16,
):
    if not input_ids:
        return
    ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention = torch.tensor(masks, dtype=torch.long, device=device)
    content = torch.tensor(content_masks, dtype=torch.long, device=device)
    with torch.no_grad():
        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                (
                    batch_efpp,
                    batch_etp,
                    batch_relation,
                    batch_vep,
                    batch_vtm,
                ) = model(ids, attention, content)
        else:
            (
                batch_efpp,
                batch_etp,
                batch_relation,
                batch_vep,
                batch_vtm,
            ) = model(ids, attention, content)
    batch_efpp = batch_efpp.detach().cpu().to(efpp_probs.dtype)
    batch_etp = batch_etp.detach().cpu().to(etp_distribution.dtype)
    batch_relation = batch_relation.detach().cpu().to(relation_distribution.dtype)
    batch_vep = batch_vep.detach().cpu().to(vulnerability_evidence_probs.dtype)
    batch_vtm = batch_vtm.detach().cpu().to(template_match_scores.dtype)
    for row_idx, chunk_idx, pooled_idx in targets:
        efpp_probs[row_idx, chunk_idx] = batch_efpp[pooled_idx]
        etp_distribution[row_idx, chunk_idx] = batch_etp[pooled_idx]
        relation_distribution[row_idx, chunk_idx] = batch_relation[pooled_idx]
        vulnerability_evidence_probs[row_idx, chunk_idx] = batch_vep[pooled_idx]
        template_match_scores[row_idx, chunk_idx] = batch_vtm[pooled_idx]
    input_ids.clear()
    masks.clear()
    content_masks.clear()
    targets.clear()


def annotate_chunk_targets(
    tokenizer,
    chunk_tokens,
    label_names,
    ontology,
    templates,
    contract_multi_labels,
):
    units = build_token_units(" ".join(chunk_tokens), tokenizer)
    effects = annotate_effect_types(units)
    pattern_labels, _ = annotate_efpp_patterns(units)
    relation_labels, _, _ = annotate_effect_relations(units)
    effect_histogram = effect_type_chunk_histogram(effects)
    template_scores = template_score_vector(
        label_names,
        templates,
        ontology,
        effect_histogram,
        pattern_labels,
        relation_labels,
    )
    chunk_evidence = pseudo_evidence_vector(
        template_scores,
        contract_multi_labels,
        label_names,
    )
    active_mask = [
        1 if label_name in label_names else 0
        for label_name in GLOBAL_VULNERABILITY_LABELS
    ]
    return {
        "relation_labels": relation_labels,
        "template_scores": template_scores,
        "chunk_evidence": chunk_evidence,
        "active_mask": active_mask,
    }


def load_reference(split, config):
    reference_dir = config.get("reference_feature_dir")
    if not reference_dir:
        return None
    path = resolve_path(reference_dir) / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"reference feature cache not found: {path}")
    return torch.load(path, map_location="cpu")


def validate_against_reference(split, payload, reference):
    if reference is None:
        return
    if [str(v) for v in payload["ids"]] != [str(v) for v in reference["ids"]]:
        raise ValueError(f"{split}: semantic ids do not match reference feature cache")
    if not torch.equal(payload["chunk_mask"], reference["chunk_mask"].bool()):
        raise ValueError(f"{split}: semantic chunk_mask does not match reference feature cache")
    if not torch.equal(payload["binary_labels"], reference["binary_labels"].float()):
        raise ValueError(f"{split}: semantic binary labels do not match reference feature cache")
    if not torch.equal(payload["multi_labels"], reference["multi_labels"].float()):
        raise ValueError(f"{split}: semantic multi-labels do not match reference feature cache")


def extract_split(split, input_path, output_path, tokenizer, model, config, device, overwrite):
    if output_path.exists() and not overwrite:
        print(f"[OK] {project_relative(output_path)} already exists, skipping")
        return torch.load(output_path, map_location="cpu").get("report", {})
    sample_count = count_jsonl(input_path)
    max_chunks = int(config["max_chunks_per_contract"])
    num_labels = int(config["num_labels"])
    dtype = torch.float16 if bool(config.get("fp16", True)) else torch.float32
    efpp_probs = torch.zeros(sample_count, max_chunks, model.num_efpp_patterns, dtype=dtype)
    etp_distribution = torch.zeros(sample_count, max_chunks, model.num_effect_types, dtype=dtype)
    relation_distribution = torch.zeros(sample_count, max_chunks, model.num_relation_types, dtype=dtype)
    vulnerability_evidence_probs = torch.zeros(sample_count, max_chunks, model.num_vulnerability_templates, dtype=dtype)
    template_match_scores = torch.zeros(sample_count, max_chunks, model.num_vulnerability_templates, dtype=dtype)
    chunk_mask = torch.zeros(sample_count, max_chunks, dtype=torch.bool)
    binary_labels = torch.zeros(sample_count, dtype=torch.float32)
    multi_labels = torch.zeros(sample_count, num_labels, dtype=torch.float32)
    chunk_vulnerability_evidence = torch.zeros(sample_count, max_chunks, model.num_vulnerability_templates, dtype=torch.float32)
    vulnerability_template_matches = torch.zeros(sample_count, max_chunks, model.num_vulnerability_templates, dtype=torch.float32)
    active_vulnerability_label_mask = torch.zeros(sample_count, model.num_vulnerability_templates, dtype=torch.float32)
    ids = []
    metadata = []
    input_ids = []
    masks = []
    content_masks = []
    targets = []
    total_chunks = 0
    truncated_count = 0
    coverage_values = []
    chunks_kept_values = []
    batch_size = int(config["batch_size"])
    use_fp16 = bool(config.get("fp16", True))
    ontology = load_ontology(resolve_path(config["ontology_path"]))
    templates = load_vulnerability_templates(
        resolve_path(config["template_path"]),
        ontology,
        expected_label_names=config["label_names"],
    )

    pbar = tqdm(total=sample_count, desc=f"effect_sem:{split}")
    for row_idx, (line_idx, item) in enumerate(iter_jsonl(input_path)):
        labels = item["multi_labels"]
        if len(labels) != num_labels:
            raise ValueError(f"{input_path}:{line_idx + 1} expected {num_labels} labels")
        contract_id = item.get("id") or item.get("address") or f"{split}_{line_idx}"
        chunk_info = build_contract_chunks(tokenizer, item.get("opcode", ""), config)
        ids.append(str(contract_id))
        binary_labels[row_idx] = float(item["binary_label"])
        multi_labels[row_idx] = torch.tensor(labels, dtype=torch.float32)
        contract_multi_labels = labels if split == "train" else [0] * len(labels)
        truncated_count += int(chunk_info["truncated"])
        coverage_values.append(chunk_info["coverage_ratio"])
        chunks_kept_values.append(chunk_info["num_chunks_kept"])
        metadata.append(
            {
                "id": str(contract_id),
                "source_split": split,
                "original_token_length": chunk_info["original_token_length"],
                "num_chunks_before_truncation": chunk_info["num_chunks_before_truncation"],
                "num_chunks_kept": chunk_info["num_chunks_kept"],
                "coverage_ratio": chunk_info["coverage_ratio"],
            }
        )
        for chunk_idx, chunk in enumerate(chunk_info["chunks"]):
            chunk_mask[row_idx, chunk_idx] = True
            input_ids.append(chunk["input_ids"])
            masks.append(chunk["attention_mask"])
            content_masks.append(chunk["content_mask"])
            targets.append((row_idx, chunk_idx, len(targets)))
            total_chunks += 1
            heuristic = annotate_chunk_targets(
                tokenizer,
                chunk["chunk_tokens"],
                config["label_names"],
                ontology,
                templates,
                contract_multi_labels,
            )
            chunk_vulnerability_evidence[row_idx, chunk_idx] = torch.tensor(
                heuristic["chunk_evidence"],
                dtype=torch.float32,
            )
            vulnerability_template_matches[row_idx, chunk_idx] = torch.tensor(
                heuristic["template_scores"],
                dtype=torch.float32,
            )
            active_vulnerability_label_mask[row_idx] = torch.tensor(
                heuristic["active_mask"],
                dtype=torch.float32,
            )
            if len(input_ids) >= batch_size:
                flush_batch(
                    model,
                    input_ids,
                    masks,
                    content_masks,
                    targets,
                    efpp_probs,
                    etp_distribution,
                    relation_distribution,
                    vulnerability_evidence_probs,
                    template_match_scores,
                    device,
                    use_fp16,
                )
        pbar.update(1)
    pbar.close()
    flush_batch(
        model,
        input_ids,
        masks,
        content_masks,
        targets,
        efpp_probs,
        etp_distribution,
        relation_distribution,
        vulnerability_evidence_probs,
        template_match_scores,
        device,
        use_fp16,
    )
    report = {
        "split": split,
        "samples": sample_count,
        "max_chunks_per_contract": max_chunks,
        "generated_chunks": int(total_chunks),
        "mean_chunks_per_contract": float(np.mean(chunks_kept_values)),
        "truncated_contract_count": int(truncated_count),
        "mean_coverage_ratio": float(np.mean(coverage_values)),
        "p50_coverage_ratio": float(np.percentile(coverage_values, 50)),
        "p90_coverage_ratio": float(np.percentile(coverage_values, 90)),
        "p95_coverage_ratio": float(np.percentile(coverage_values, 95)),
        "efpp_shape": list(efpp_probs.shape),
        "etp_distribution_shape": list(etp_distribution.shape),
        "relation_distribution_shape": list(relation_distribution.shape),
        "vulnerability_evidence_probs_shape": list(vulnerability_evidence_probs.shape),
        "template_match_scores_shape": list(template_match_scores.shape),
        "dtype": str(dtype),
        "efpp_nan_count": int(torch.isnan(efpp_probs.float()).sum().item()),
        "efpp_inf_count": int(torch.isinf(efpp_probs.float()).sum().item()),
        "etp_nan_count": int(torch.isnan(etp_distribution.float()).sum().item()),
        "etp_inf_count": int(torch.isinf(etp_distribution.float()).sum().item()),
        "relation_nan_count": int(torch.isnan(relation_distribution.float()).sum().item()),
        "relation_inf_count": int(torch.isinf(relation_distribution.float()).sum().item()),
        "vep_nan_count": int(torch.isnan(vulnerability_evidence_probs.float()).sum().item()),
        "vep_inf_count": int(torch.isinf(vulnerability_evidence_probs.float()).sum().item()),
        "vtm_nan_count": int(torch.isnan(template_match_scores.float()).sum().item()),
        "vtm_inf_count": int(torch.isinf(template_match_scores.float()).sum().item()),
        "hf_model_path": config["hf_model_path"],
        "num_effect_types": int(model.num_effect_types),
        "num_efpp_patterns": int(model.num_efpp_patterns),
        "num_relation_types": int(model.num_relation_types),
        "num_vulnerability_templates": int(model.num_vulnerability_templates),
        "reference_feature_dir": config.get("reference_feature_dir"),
        "ontology_path": config["ontology_path"],
        "template_path": config["template_path"],
    }
    payload = {
        "ids": ids,
        "efpp_probs": efpp_probs,
        "etp_distribution": etp_distribution,
        "relation_distribution": relation_distribution,
        "vulnerability_evidence_probs": vulnerability_evidence_probs,
        "template_match_scores": template_match_scores,
        "chunk_mask": chunk_mask,
        "binary_labels": binary_labels,
        "multi_labels": multi_labels,
        "chunk_vulnerability_evidence": chunk_vulnerability_evidence,
        "vulnerability_template_matches": vulnerability_template_matches,
        "active_vulnerability_label_mask": active_vulnerability_label_mask,
        "global_vulnerability_label_names": GLOBAL_VULNERABILITY_LABELS,
        "metadata": metadata,
        "report": report,
    }
    validate_against_reference(split, payload, load_reference(split, config))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    report["disk_size_bytes"] = output_path.stat().st_size
    payload["report"] = report
    torch.save(payload, output_path)
    print(f"[OK] wrote {project_relative(output_path)}")
    return report


def write_report(config, reports):
    report_dir = resolve_path(config.get("report_dir", "data/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_name": config["experiment_name"],
        "output_dir": config["output_dir"],
        "hf_model_path": config["hf_model_path"],
        "reference_feature_dir": config.get("reference_feature_dir"),
        "ontology_path": config["ontology_path"],
        "template_path": config["template_path"],
        "max_len": config["max_len"],
        "chunk_stride": config["chunk_stride"],
        "max_chunks_per_contract": config["max_chunks_per_contract"],
        "num_labels": config["num_labels"],
        "splits": reports,
    }
    lines = ["Effect-flow semantic cache extraction report", ""]
    for key, value in report.items():
        if key == "splits":
            lines.append("splits:")
            for split, split_report in value.items():
                lines.append(f"- {split}:")
                for skey, svalue in split_report.items():
                    lines.append(f"  {skey}: {svalue}")
        else:
            lines.append(f"{key}: {value}")
    for base in [f"{config['experiment_name']}_report", "extract_effect_flow_semantic_features_report"]:
        txt_path = report_dir / f"{base}.txt"
        json_path = report_dir / f"{base}.json"
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] wrote {project_relative(txt_path)}")
        print(f"[OK] wrote {project_relative(json_path)}")


def main():
    args = parse_args()
    config = load_yaml(args.config)
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve_path(config["vocab_path"]))
    validate_evm_vocab(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EffectFlowSemanticExtractor(config["hf_model_path"]).to(device)
    reports = {}
    for split in ["train", "valid", "test"]:
        reports[split] = extract_split(
            split,
            resolve_path(config[f"{split}_path"]),
            resolve_path(config["output_dir"]) / f"{split}.pt",
            tokenizer,
            model,
            config,
            device,
            args.overwrite,
        )
    write_report(config, reports)


if __name__ == "__main__":
    main()
