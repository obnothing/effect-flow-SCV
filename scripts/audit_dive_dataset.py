import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from dive_common import (  # noqa: E402
    DIVE_LABEL_NAMES,
    PROJECT_ROOT,
    REPORT_DIR,
    chunk_coverage,
    default_label_path,
    default_opcode_path,
    iter_opcodes,
    load_labels,
    opcode_hash,
    resolve,
    summarize_values,
    write_json_txt,
)

try:
    from evm_tokenizer import EVMOpcodeTokenizer
except Exception:  # pragma: no cover
    EVMOpcodeTokenizer = None


def parse_args():
    parser = argparse.ArgumentParser(description="Audit DIVE opcode and label files.")
    parser.add_argument("--opcode_jsonl", default=None)
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--vocab_path", default="data/processed/BJUT_SC01/evm_vocab.json")
    return parser.parse_args()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def summarize_chunk_stats(lengths):
    configs = [
        {"max_len": 512, "chunk_content_size": 510, "stride": 510, "max_chunks": 16},
        {"max_len": 512, "chunk_content_size": 510, "stride": 510, "max_chunks": 32},
        {"max_len": 512, "chunk_content_size": 510, "stride": 256, "max_chunks": 16},
        {"max_len": 512, "chunk_content_size": 510, "stride": 256, "max_chunks": 32},
    ]
    reports = {}
    for cfg in configs:
        coverages = []
        kept = []
        needed = []
        truncated = 0
        for length in lengths:
            stat = chunk_coverage(length, cfg["chunk_content_size"], cfg["stride"], cfg["max_chunks"])
            coverages.append(int(round(stat["coverage_ratio"] * 1_000_000)))
            kept.append(stat["chunks_kept"])
            needed.append(stat["chunks_needed"])
            truncated += int(stat["truncated"])
        key = f"stride_{cfg['stride']}_max_chunks_{cfg['max_chunks']}"
        coverage_summary = summarize_values(coverages)
        reports[key] = {
            **cfg,
            "chunks_needed": summarize_values(needed),
            "chunks_kept": summarize_values(kept),
            "truncated_contract_count": truncated,
            "coverage_ratio_mean": coverage_summary["mean"] / 1_000_000,
            "coverage_ratio_median": coverage_summary["median"] / 1_000_000,
            "coverage_ratio_p90": coverage_summary["p90"] / 1_000_000,
            "coverage_ratio_p95": coverage_summary["p95"] / 1_000_000,
            "coverage_ratio_p99": coverage_summary["p99"] / 1_000_000,
        }
    return reports


def main():
    args = parse_args()
    opcode_path = resolve(args.opcode_jsonl) if args.opcode_jsonl else default_opcode_path()
    label_path = resolve(args.label_csv) if args.label_csv else default_label_path()
    labels = load_labels(label_path)
    tokenizer = None
    vocab_path = resolve(args.vocab_path)
    if EVMOpcodeTokenizer is not None and vocab_path.exists():
        tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)

    label_counts = [0] * len(DIVE_LABEL_NAMES)
    cooccurrence = [[0] * len(DIVE_LABEL_NAMES) for _ in DIVE_LABEL_NAMES]
    labels_with_opcode = set()
    opcodes_with_label = set()
    raw_lengths = []
    tokenized_lengths = []
    hash_counts = Counter()
    label_count_per_sample = []
    opcode_samples = 0

    for contract_id, opcode in iter_opcodes(opcode_path):
        opcode_samples += 1
        hash_counts[opcode_hash(opcode)] += 1
        raw_len = len(opcode.split())
        raw_lengths.append(raw_len)
        if tokenizer is not None:
            tokenized_lengths.append(len(tokenizer.tokenize(opcode, add_special_tokens=True)))
        if contract_id in labels:
            opcodes_with_label.add(contract_id)
            labels_with_opcode.add(contract_id)
            vector = labels[contract_id]
            label_count_per_sample.append(sum(vector))
            for idx, value in enumerate(vector):
                label_counts[idx] += value
            for i, left in enumerate(vector):
                if not left:
                    continue
                for j, right in enumerate(vector):
                    if right:
                        cooccurrence[i][j] += 1

    label_ids = set(labels)
    opcode_label_aligned = len(labels_with_opcode)
    duplicate_group_sizes = [count for count in hash_counts.values() if count > 1]
    report = {
        "status": "ok",
        "opcode_jsonl": opcode_path.relative_to(PROJECT_ROOT).as_posix(),
        "label_csv": label_path.relative_to(PROJECT_ROOT).as_posix(),
        "opcode_samples": opcode_samples,
        "label_samples": len(labels),
        "id_field_opcode": "contractID",
        "id_field_label": "contractID",
        "aligned_samples": opcode_label_aligned,
        "labels_missing_opcode": len(label_ids - labels_with_opcode),
        "opcodes_missing_label": opcode_samples - len(opcodes_with_label),
        "label_names": DIVE_LABEL_NAMES,
        "label_positive_counts": dict(zip(DIVE_LABEL_NAMES, label_counts)),
        "label_positive_ratios": {
            name: ratio(count, opcode_label_aligned)
            for name, count in zip(DIVE_LABEL_NAMES, label_counts)
        },
        "label_cooccurrence_matrix": cooccurrence,
        "average_labels_per_sample": ratio(sum(label_count_per_sample), len(label_count_per_sample)),
        "all_zero_label_samples": sum(1 for value in label_count_per_sample if value == 0),
        "opcode_hash_unique_count": len(hash_counts),
        "opcode_hash_duplicate_group_count": len(duplicate_group_sizes),
        "opcode_hash_duplicate_sample_count": sum(duplicate_group_sizes),
        "duplicate_group_size_distribution": summarize_values(duplicate_group_sizes),
        "opcode_token_length": summarize_values(raw_lengths),
        "evm_tokenized_length": summarize_values(tokenized_lengths),
        "chunk_statistics_raw_opcode": summarize_chunk_stats(raw_lengths),
        "chunk_statistics_evm_tokenized": summarize_chunk_stats(tokenized_lengths)
        if tokenized_lengths
        else None,
    }
    write_json_txt(
        report,
        REPORT_DIR / "dive_dataset_audit.json",
        REPORT_DIR / "dive_dataset_audit.txt",
        "DIVE dataset audit",
    )


if __name__ == "__main__":
    main()
