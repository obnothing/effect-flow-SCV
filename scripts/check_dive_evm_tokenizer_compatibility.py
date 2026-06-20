import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from dive_common import (  # noqa: E402
    PROJECT_ROOT,
    REPORT_DIR,
    chunk_coverage,
    resolve,
    summarize_values,
    write_json_txt,
)
from evm_tokenizer import (  # noqa: E402
    EVMOpcodeTokenizer,
    MNEMONIC_TOKENS,
    NORMALIZED_OPERAND_TOKENS,
    is_hex_literal,
)


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def parse_args():
    parser = argparse.ArgumentParser(description="Check BJUT EVM tokenizer compatibility on DIVE.")
    parser.add_argument("--vocab_path", default="data/processed/BJUT_SC01/evm_vocab.json")
    parser.add_argument("--data_dir", default="data/processed/DIVE")
    parser.add_argument("--max_unk_ratio", type=float, default=0.01)
    parser.add_argument("--max_chunks_per_contract", type=int, default=16)
    return parser.parse_args()


def iter_processed(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    args = parse_args()
    vocab_path = resolve(args.vocab_path)
    data_dir = resolve(args.data_dir)
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
    vocab = tokenizer.vocab

    special_present = {token: token in vocab for token in SPECIAL_TOKENS}
    mnemonic_oov = Counter()
    operand_oov = Counter()
    normalized_counts = Counter()
    token_lengths = []
    unk_count = 0
    total_tokens = 0
    raw_operand_count = 0
    raw_mnemonic_count = 0
    split_samples = {}
    chunk_rows = []

    for split in ["train", "valid", "test"]:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            split_samples[split] = 0
            continue
        count = 0
        for item in iter_processed(path):
            count += 1
            opcode = item["opcode"]
            raw_tokens = opcode.split()
            for raw in raw_tokens:
                if is_hex_literal(raw):
                    raw_operand_count += 1
                    normalized = tokenizer.tokenize(raw, add_special_tokens=False)[0]
                    normalized_counts[normalized] += 1
                    if normalized not in vocab:
                        operand_oov[normalized] += 1
                else:
                    raw_mnemonic_count += 1
                    upper = raw.upper()
                    if upper not in vocab and upper not in MNEMONIC_TOKENS:
                        mnemonic_oov[upper] += 1
            tokens = tokenizer.tokenize(opcode, add_special_tokens=True)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            token_lengths.append(len(ids))
            unk_count += sum(1 for value in ids if value == tokenizer.unk_token_id)
            total_tokens += len(ids)
            for stride in [510, 256]:
                row = chunk_coverage(
                    len(ids) - 2,
                    chunk_content_size=510,
                    stride=stride,
                    max_chunks=args.max_chunks_per_contract,
                )
                row["stride"] = stride
                chunk_rows.append(row)
        split_samples[split] = count

    unk_ratio = float(unk_count / total_tokens) if total_tokens else 0.0
    normalized_total = sum(normalized_counts.values())
    normalized_ratios = {
        token: float(normalized_counts[token] / normalized_total)
        if normalized_total
        else 0.0
        for token in NORMALIZED_OPERAND_TOKENS
    }
    warnings = []
    if unk_ratio > args.max_unk_ratio:
        warnings.append(f"UNK ratio is high: {unk_ratio:.6f}")
    if sum(mnemonic_oov.values()) > 0:
        warnings.append("DIVE contains opcode-like tokens outside the BJUT vocab.")
    can_reuse = all(special_present.values()) and unk_ratio <= args.max_unk_ratio
    chunk_statistics = {}
    for stride in [510, 256]:
        rows = [row for row in chunk_rows if row["stride"] == stride]
        chunk_statistics[str(stride)] = {
            "chunk_content_size": 510,
            "chunk_stride": stride,
            "max_chunks_per_contract": args.max_chunks_per_contract,
            "generated_chunks": int(sum(row["chunks_kept"] for row in rows)),
            "mean_chunks_per_contract": (
                float(sum(row["chunks_kept"] for row in rows) / len(rows))
                if rows
                else 0.0
            ),
            "truncated_contract_count": int(sum(row["truncated"] for row in rows)),
            "covered_token_ratio_mean": (
                float(sum(row["coverage_ratio"] for row in rows) / len(rows))
                if rows
                else 0.0
            ),
        }
    report = {
        "status": "ok" if can_reuse else "warning",
        "vocab_path": vocab_path.relative_to(PROJECT_ROOT).as_posix(),
        "data_dir": data_dir.relative_to(PROJECT_ROOT).as_posix(),
        "vocab_size": len(tokenizer),
        "special_tokens_present": special_present,
        "split_samples": split_samples,
        "dive_opcode_mnemonic_oov_count": int(sum(mnemonic_oov.values())),
        "dive_opcode_mnemonic_oov_top20": mnemonic_oov.most_common(20),
        "dive_operand_oov_count": int(sum(operand_oov.values())),
        "dive_operand_oov_top20": operand_oov.most_common(20),
        "raw_mnemonic_count": raw_mnemonic_count,
        "raw_operand_count": raw_operand_count,
        "normalized_operand_count": dict(normalized_counts),
        "normalized_operand_ratios": normalized_ratios,
        "unk_count": unk_count,
        "total_tokenized_tokens": total_tokens,
        "unk_ratio": unk_ratio,
        "max_allowed_unk_ratio": args.max_unk_ratio,
        "token_length_statistics": summarize_values(token_lengths),
        "chunk_statistics": chunk_statistics,
        "can_reuse_bjut_evm_bert": can_reuse,
        "warnings": warnings,
    }
    write_json_txt(
        report,
        REPORT_DIR / "dive_evm_tokenizer_compatibility.json",
        REPORT_DIR / "dive_evm_tokenizer_compatibility.txt",
        "DIVE EVM tokenizer compatibility report",
    )
    if not can_reuse:
        raise SystemExit(
            "DIVE tokenizer compatibility failed; continued MLM is blocked. "
            "Inspect data/reports/dive_evm_tokenizer_compatibility.txt."
        )


if __name__ == "__main__":
    main()
