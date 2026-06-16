import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import BertForMaskedLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evm_bert_classification_dataset import validate_evm_vocab  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frozen pretrained EVM-BERT chunk features."
    )
    parser.add_argument(
        "--config",
        default="configs/extract_evm_bert_chunk_features.yaml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate feature files even if they already exist.",
    )
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
            if not line:
                continue
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
    return input_ids, attention_mask


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
        chunks.append(encode_chunk(tokenizer, chunk_tokens, max_len))
        start += stride
    if not chunks:
        chunks.append(encode_chunk(tokenizer, [], max_len))
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


def pool_encoder_output(outputs, attention_mask, pooling):
    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "masked_mean":
        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"Unsupported pooling: {pooling}")


def flush_chunk_batch(
    encoder,
    chunk_input_ids,
    chunk_attention_masks,
    chunk_targets,
    features,
    config,
    device,
):
    if not chunk_input_ids:
        return
    input_ids = torch.tensor(chunk_input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(chunk_attention_masks, dtype=torch.long, device=device)
    token_type_ids = torch.zeros_like(input_ids, device=device)
    use_fp16 = bool(config.get("fp16", False)) and device.type == "cuda"
    with torch.no_grad():
        if use_fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                pooled = pool_encoder_output(
                    outputs,
                    attention_mask,
                    config.get("pooling", "cls"),
                )
        else:
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pooled = pool_encoder_output(outputs, attention_mask, config.get("pooling", "cls"))
    pooled = pooled.detach().cpu().to(features.dtype)
    for row_idx, chunk_idx, pooled_idx in chunk_targets:
        features[row_idx, chunk_idx] = pooled[pooled_idx]
    chunk_input_ids.clear()
    chunk_attention_masks.clear()
    chunk_targets.clear()


def extract_split(split_name, input_path, output_path, tokenizer, encoder, config, device, overwrite):
    if output_path.exists() and not overwrite:
        print(f"[OK] {project_relative(output_path)} already exists, skipping")
        return torch.load(output_path, map_location="cpu").get("report", {})

    sample_count = count_jsonl(input_path)
    max_chunks = int(config["max_chunks_per_contract"])
    hidden_size = int(encoder.config.hidden_size)
    output_dtype = torch.float16 if bool(config.get("fp16", False)) else torch.float32
    features = torch.zeros(sample_count, max_chunks, hidden_size, dtype=output_dtype)
    chunk_mask = torch.zeros(sample_count, max_chunks, dtype=torch.bool)
    binary_labels = torch.zeros(sample_count, dtype=torch.float32)
    multi_labels = torch.zeros(sample_count, 10, dtype=torch.float32)
    ids = []
    metadata = []

    batch_size = int(config["batch_size"])
    chunk_input_ids = []
    chunk_attention_masks = []
    chunk_targets = []
    total_chunks = 0
    truncated_count = 0
    coverage_values = []
    chunks_kept_values = []

    pbar = tqdm(total=sample_count, desc=f"extract:{split_name}")
    for row_idx, (line_idx, item) in enumerate(iter_jsonl(input_path)):
        contract_id = item.get("id") or item.get("address") or f"{split_name}_{line_idx}"
        labels = item["multi_labels"]
        if len(labels) != 10:
            raise ValueError(f"{input_path}:{line_idx + 1} expected 10 labels")
        chunk_info = build_contract_chunks(tokenizer, item.get("opcode", ""), config)
        ids.append(str(contract_id))
        binary_labels[row_idx] = float(item["binary_label"])
        multi_labels[row_idx] = torch.tensor(labels, dtype=torch.float32)
        if chunk_info["truncated"]:
            truncated_count += 1
        coverage_values.append(chunk_info["coverage_ratio"])
        chunks_kept_values.append(chunk_info["num_chunks_kept"])
        metadata.append(
            {
                "id": str(contract_id),
                "source_split": split_name,
                "original_token_length": chunk_info["original_token_length"],
                "num_chunks_before_truncation": chunk_info["num_chunks_before_truncation"],
                "num_chunks_kept": chunk_info["num_chunks_kept"],
                "coverage_ratio": chunk_info["coverage_ratio"],
            }
        )
        for chunk_idx, (input_ids, mask) in enumerate(chunk_info["chunks"]):
            chunk_mask[row_idx, chunk_idx] = True
            chunk_input_ids.append(input_ids)
            chunk_attention_masks.append(mask)
            chunk_targets.append((row_idx, chunk_idx, len(chunk_targets)))
            total_chunks += 1
            if len(chunk_input_ids) >= batch_size:
                flush_chunk_batch(
                    encoder,
                    chunk_input_ids,
                    chunk_attention_masks,
                    chunk_targets,
                    features,
                    config,
                    device,
                )
        pbar.update(1)
    pbar.close()
    flush_chunk_batch(
        encoder,
        chunk_input_ids,
        chunk_attention_masks,
        chunk_targets,
        features,
        config,
        device,
    )

    report = {
        "split": split_name,
        "samples": sample_count,
        "max_chunks_per_contract": max_chunks,
        "generated_chunks": int(total_chunks),
        "mean_chunks_per_contract": float(np.mean(chunks_kept_values)),
        "truncated_contract_count": int(truncated_count),
        "mean_coverage_ratio": float(np.mean(coverage_values)),
        "p50_coverage_ratio": float(np.percentile(coverage_values, 50)),
        "p90_coverage_ratio": float(np.percentile(coverage_values, 90)),
        "p95_coverage_ratio": float(np.percentile(coverage_values, 95)),
        "feature_shape": list(features.shape),
        "dtype": str(features.dtype),
        "hf_model_path": config["hf_model_path"],
        "is_transductive_pretraining": bool(config.get("is_transductive_pretraining", True)),
    }
    payload = {
        "ids": ids,
        "features": features,
        "chunk_mask": chunk_mask,
        "binary_labels": binary_labels,
        "multi_labels": multi_labels,
        "metadata": metadata,
        "report": report,
    }
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
        "experiment_name": config.get("experiment_name"),
        "output_dir": config["output_dir"],
        "pooling": config.get("pooling", "cls"),
        "max_len": config["max_len"],
        "chunk_content_size": config["chunk_content_size"],
        "chunk_stride": config["chunk_stride"],
        "max_chunks_per_contract": config["max_chunks_per_contract"],
        "hf_model_path": config["hf_model_path"],
        "is_transductive_pretraining": bool(config.get("is_transductive_pretraining", True)),
        "splits": reports,
    }
    txt_path = report_dir / "extract_evm_bert_chunk_features_report.txt"
    json_path = report_dir / "extract_evm_bert_chunk_features_report.json"
    lines = ["EVM-BERT chunk feature extraction report", ""]
    for key, value in report.items():
        if key == "splits":
            lines.append("splits:")
            for split, split_report in value.items():
                lines.append(f"- {split}:")
                for skey, svalue in split_report.items():
                    lines.append(f"  {skey}: {svalue}")
        else:
            lines.append(f"{key}: {value}")
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
    mlm_model = BertForMaskedLM.from_pretrained(
        resolve_path(config["hf_model_path"]),
        local_files_only=True,
    )
    if mlm_model.config.vocab_size != len(tokenizer):
        raise ValueError(
            f"vocab mismatch: model={mlm_model.config.vocab_size}, tokenizer={len(tokenizer)}"
        )
    encoder = mlm_model.bert.to(device)
    encoder.eval()

    output_dir = resolve_path(config["output_dir"])
    split_paths = {
        "train": resolve_path(config["train_path"]),
        "valid": resolve_path(config["valid_path"]),
        "test": resolve_path(config["test_path"]),
    }
    reports = {}
    for split_name, path in split_paths.items():
        reports[split_name] = extract_split(
            split_name,
            path,
            output_dir / f"{split_name}.pt",
            tokenizer,
            encoder,
            config,
            device,
            args.overwrite,
        )
    write_report(config, reports)


if __name__ == "__main__":
    main()

