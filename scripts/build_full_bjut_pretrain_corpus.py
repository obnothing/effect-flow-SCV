import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TRANSDUCTIVE_WARNING = (
    "This pretraining corpus uses opcode inputs from train/valid/test without "
    "labels. It is a transductive self-supervised setting."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build full-corpus unlabeled BJUT SC01 MLM chunks."
    )
    parser.add_argument(
        "--config",
        default="configs/pretrain_evm_bert_base_full_bjut.yaml",
        help="Path to EVM-BERT pretraining YAML config.",
    )
    return parser.parse_args()


def resolve_project_path(path):
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


def load_config(path):
    with resolve_project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_tokenizer(tokenizer):
    missing = [token for token in SPECIAL_TOKENS if token not in tokenizer.vocab]
    if missing:
        raise ValueError(f"EVM vocab missing required special tokens: {missing}")


def iter_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            yield line_no, item


def hash_opcode(opcode):
    return hashlib.sha256(str(opcode).encode("utf-8")).hexdigest()


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (rank - low))


def covered_token_count(total_tokens, ranges):
    if total_tokens <= 0 or not ranges:
        return 0
    clipped = []
    for start, end in ranges:
        start = max(0, min(start, total_tokens))
        end = max(0, min(end, total_tokens))
        if end > start:
            clipped.append((start, end))
    if not clipped:
        return 0
    clipped.sort()
    merged = []
    cur_start, cur_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return sum(end - start for start, end in merged)


def make_chunk(tokenizer, tokens, token_ids, start, chunk_content_size, max_len):
    content_tokens = tokens[start : start + chunk_content_size]
    content_ids = token_ids[start : start + chunk_content_size]
    chunk_tokens = [tokenizer.cls_token] + content_tokens + [tokenizer.sep_token]
    chunk_ids = (
        [tokenizer.vocab[tokenizer.cls_token]]
        + content_ids
        + [tokenizer.vocab[tokenizer.sep_token]]
    )
    attention_mask = [1] * len(chunk_ids)
    padding = max_len - len(chunk_ids)
    if padding < 0:
        raise ValueError(
            f"Chunk length {len(chunk_ids)} exceeds max_len={max_len}; "
            "check chunk_content_size."
        )
    if padding:
        chunk_ids.extend([tokenizer.pad_token_id] * padding)
        attention_mask.extend([0] * padding)
    return chunk_tokens, chunk_ids, attention_mask


def build_corpus(config):
    vocab_path = resolve_project_path(config["vocab_path"])
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
    validate_tokenizer(tokenizer)

    max_len = int(config.get("max_len", 512))
    chunk_content_size = int(config.get("chunk_content_size", max_len - 2))
    chunk_stride = int(config.get("chunk_stride", chunk_content_size))
    max_chunks_per_contract = int(config.get("max_chunks_per_contract", 16))
    debug_num_contracts = config.get("debug_num_contracts")
    debug_num_contracts = (
        int(debug_num_contracts) if debug_num_contracts is not None else None
    )

    if chunk_content_size != max_len - 2:
        raise ValueError(
            f"chunk_content_size must equal max_len - 2. Got "
            f"{chunk_content_size} for max_len={max_len}."
        )
    if chunk_stride <= 0:
        raise ValueError("chunk_stride must be positive.")
    if max_chunks_per_contract <= 0:
        raise ValueError("max_chunks_per_contract must be positive.")

    output_path = resolve_project_path(config["pretrain_corpus_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_paths = config.get("data_paths") or {
        "train": "data/processed/BJUT_SC01/train.jsonl",
        "valid": "data/processed/BJUT_SC01/valid.jsonl",
        "test": "data/processed/BJUT_SC01/test.jsonl",
    }

    split_contract_counts = Counter()
    split_chunk_counts = Counter()
    opcode_hash_counts = Counter()
    covered_ratios = []
    chunks_per_contract = []
    token_counts = []
    empty_opcode_count = 0
    truncated_contract_count = 0
    original_contract_samples = 0
    generated_mlm_chunks = 0

    with output_path.open("w", encoding="utf-8") as out:
        for split, raw_path in data_paths.items():
            path = resolve_project_path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Missing {split} path: {path}")
            progress = tqdm(
                iter_jsonl(path),
                desc=f"build_pretrain:{split}",
                unit="contract",
            )
            for line_no, item in progress:
                if debug_num_contracts is not None and (
                    original_contract_samples >= debug_num_contracts
                ):
                    break

                # Labels are intentionally ignored in this self-supervised stage.
                opcode = item.get("opcode", "")
                contract_id = item.get("id") or f"{split}_{line_no}"
                opcode_hash = hash_opcode(opcode)
                opcode_hash_counts[opcode_hash] += 1
                original_contract_samples += 1
                split_contract_counts[split] += 1

                tokens = tokenizer.tokenize(opcode, add_special_tokens=False)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                total_tokens = len(token_ids)
                token_counts.append(total_tokens)
                if total_tokens == 0:
                    empty_opcode_count += 1
                    chunks_per_contract.append(0)
                    covered_ratios.append(0.0)
                    continue

                all_starts = list(range(0, total_tokens, chunk_stride))
                selected_starts = all_starts[:max_chunks_per_contract]
                if len(all_starts) > len(selected_starts):
                    truncated_contract_count += 1
                num_chunks = len(selected_starts)
                chunks_per_contract.append(num_chunks)

                ranges = [
                    (start, start + chunk_content_size) for start in selected_starts
                ]
                covered = covered_token_count(total_tokens, ranges)
                covered_ratio = covered / total_tokens if total_tokens else 0.0
                covered_ratios.append(covered_ratio)

                for chunk_id, start in enumerate(selected_starts):
                    chunk_tokens, chunk_ids, attention_mask = make_chunk(
                        tokenizer,
                        tokens,
                        token_ids,
                        start,
                        chunk_content_size,
                        max_len,
                    )
                    input_length = int(sum(attention_mask))
                    payload = {
                        "id": f"{contract_id}::chunk_{chunk_id}",
                        "contract_id": str(contract_id),
                        "chunk_id": chunk_id,
                        "num_chunks": num_chunks,
                        "source_split": split,
                        "opcode_hash": opcode_hash,
                        "token_ids": chunk_ids,
                        "attention_mask": attention_mask,
                        "input_length": input_length,
                        "original_token_count": total_tokens,
                        "covered_token_ratio": covered_ratio,
                    }
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    generated_mlm_chunks += 1
                    split_chunk_counts[split] += 1

    duplicate_opcode_hash_count = sum(
        count - 1 for count in opcode_hash_counts.values() if count > 1
    )
    report = {
        "experiment_name": config.get(
            "experiment_name", "pretrain_evm_bert_base_full_bjut"
        ),
        "pretrain_corpus_path": project_relative(output_path),
        "vocab_path": project_relative(vocab_path),
        "is_transductive_pretraining": True,
        "use_labels": False,
        "transductive_warning": TRANSDUCTIVE_WARNING,
        "max_len": max_len,
        "chunk_content_size": chunk_content_size,
        "chunk_stride": chunk_stride,
        "max_chunks_per_contract_limit": max_chunks_per_contract,
        "debug_num_contracts": debug_num_contracts,
        "train_samples": split_contract_counts.get("train", 0),
        "valid_samples": split_contract_counts.get("valid", 0),
        "test_samples": split_contract_counts.get("test", 0),
        "total_samples": original_contract_samples,
        "original_contract_samples": original_contract_samples,
        "generated_mlm_chunks": generated_mlm_chunks,
        "split_chunk_counts": dict(split_chunk_counts),
        "unique_opcode_hash_count": len(opcode_hash_counts),
        "duplicate_opcode_hash_count": duplicate_opcode_hash_count,
        "empty_opcode_count": empty_opcode_count,
        "average_chunks_per_contract": (
            generated_mlm_chunks / original_contract_samples
            if original_contract_samples
            else 0.0
        ),
        "max_chunks_per_contract": max(chunks_per_contract)
        if chunks_per_contract
        else 0,
        "truncated_contract_count": truncated_contract_count,
        "covered_token_ratio_mean": statistics.fmean(covered_ratios)
        if covered_ratios
        else 0.0,
        "covered_token_ratio_p50": percentile(covered_ratios, 0.50),
        "covered_token_ratio_p90": percentile(covered_ratios, 0.90),
        "original_token_count_mean": statistics.fmean(token_counts)
        if token_counts
        else 0.0,
        "original_token_count_p50": percentile(token_counts, 0.50),
        "original_token_count_p90": percentile(token_counts, 0.90),
    }
    return report


def write_report(report, config):
    report_dir = resolve_project_path(config.get("report_dir", "data/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    name = config.get("experiment_name", "pretrain_evm_bert_base_full_bjut")
    txt_path = report_dir / f"{name}_corpus_report.txt"
    json_path = report_dir / f"{name}_corpus_report.json"

    lines = [
        "Full BJUT SC01 unlabeled EVM-BERT pretraining corpus report",
        "",
        f"experiment_name: {report['experiment_name']}",
        f"pretrain_corpus_path: {report['pretrain_corpus_path']}",
        f"vocab_path: {report['vocab_path']}",
        f"is_transductive_pretraining: {report['is_transductive_pretraining']}",
        f"use_labels: {report['use_labels']}",
        f"transductive_warning: {report['transductive_warning']}",
        "",
        f"train samples: {report['train_samples']}",
        f"valid samples: {report['valid_samples']}",
        f"test samples: {report['test_samples']}",
        f"total samples: {report['total_samples']}",
        f"original_contract_samples: {report['original_contract_samples']}",
        f"generated_mlm_chunks: {report['generated_mlm_chunks']}",
        f"average_chunks_per_contract: {report['average_chunks_per_contract']:.6f}",
        f"max_chunks_per_contract: {report['max_chunks_per_contract']}",
        f"max_chunks_per_contract_limit: {report['max_chunks_per_contract_limit']}",
        f"truncated_contract_count: {report['truncated_contract_count']}",
        f"unique opcode hash count: {report['unique_opcode_hash_count']}",
        f"duplicate opcode hash count: {report['duplicate_opcode_hash_count']}",
        f"empty_opcode_count: {report['empty_opcode_count']}",
        f"covered_token_ratio_mean: {report['covered_token_ratio_mean']:.6f}",
        f"covered_token_ratio_p50: {report['covered_token_ratio_p50']:.6f}",
        f"covered_token_ratio_p90: {report['covered_token_ratio_p90']:.6f}",
        "",
        "Split chunk counts:",
    ]
    for split, count in report["split_chunk_counts"].items():
        lines.append(f"- {split}: {count}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {project_relative(txt_path)}")
    print(f"[OK] wrote {project_relative(json_path)}")

    # Compatibility report names requested in the stage description for the full run.
    if report["experiment_name"] == "pretrain_evm_bert_base_full_bjut":
        fixed_txt = report_dir / "pretrain_corpus_full_bjut_report.txt"
        fixed_json = report_dir / "pretrain_corpus_full_bjut_report.json"
        fixed_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fixed_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[OK] wrote {project_relative(fixed_txt)}")
        print(f"[OK] wrote {project_relative(fixed_json)}")


def main():
    args = parse_args()
    config = load_config(args.config)
    report = build_corpus(config)
    write_report(report, config)
    print(f"[OK] wrote {report['pretrain_corpus_path']}")
    print(f"[INFO] {TRANSDUCTIVE_WARNING}")


if __name__ == "__main__":
    main()
