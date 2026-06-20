import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


STRICT_NOTE = (
    "This corpus uses only DIVE train opcode inputs without labels. "
    "It is suitable for strict DIVE downstream evaluation."
)
TRANSDUCTIVE_NOTE = (
    "This corpus uses DIVE train/valid/test opcode inputs without labels. "
    "It is a transductive self-supervised setting and should not be reported "
    "as strict generalization."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build unlabeled DIVE continued-MLM corpus.")
    parser.add_argument("--mode", choices=["train_only", "full_corpus"], required=True)
    parser.add_argument("--data_dir", default="data/processed/DIVE")
    parser.add_argument("--vocab_path", default="data/processed/BJUT_SC01/evm_vocab.json")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=510)
    parser.add_argument("--max_chunks_per_contract", type=int, default=16)
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path):
    try:
        return Path(path).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (position - low))


def opcode_hash(opcode):
    normalized = " ".join(str(opcode).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def output_paths(mode):
    data_dir = PROJECT_ROOT / "data/processed/DIVE"
    report_dir = PROJECT_ROOT / "data/reports"
    if mode == "train_only":
        return (
            data_dir / "pretrain_corpus_dive_train_only.jsonl",
            report_dir / "dive_continued_mlm_corpus_train_only_report.txt",
            report_dir / "dive_continued_mlm_corpus_train_only_report.json",
        )
    return (
        data_dir / "pretrain_corpus_dive_full.jsonl",
        report_dir / "dive_continued_mlm_corpus_full_report.txt",
        report_dir / "dive_continued_mlm_corpus_full_report.json",
    )


def write_report(report, txt_path, json_path):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["DIVE continued MLM corpus report", ""]
    for key, value in report.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(value)}")
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {relative(txt_path)}")
    print(f"[OK] wrote {relative(json_path)}")


def main():
    args = parse_args()
    data_dir = resolve(args.data_dir)
    vocab_path = resolve(args.vocab_path)
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
    chunk_content_size = args.max_len - 2
    splits = ["train"] if args.mode == "train_only" else ["train", "valid", "test"]
    output_path, txt_path, json_path = output_paths(args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_samples = Counter()
    split_chunks = Counter()
    token_counts = []
    chunks_per_contract = []
    coverage_ratios = []
    hashes = Counter()
    truncated_contract_count = 0
    empty_opcode_count = 0

    with output_path.open("w", encoding="utf-8") as out:
        for split in splits:
            path = data_dir / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing DIVE split: {path}. Run Stage 12A first and stop."
                )
            for line_no, item in tqdm(iter_jsonl(path), desc=f"dive_mlm:{split}"):
                opcode = str(item.get("opcode", "")).strip()
                if not opcode:
                    empty_opcode_count += 1
                    continue
                contract_id = str(item.get("id") or f"{split}_{line_no}")
                digest = opcode_hash(opcode)
                payload = {
                    "id": contract_id,
                    "opcode": opcode,
                    "source_split": split,
                    "opcode_hash": digest,
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")

                token_count = len(tokenizer.tokenize(opcode, add_special_tokens=False))
                starts = list(range(0, token_count, args.chunk_stride)) or [0]
                selected = starts[: args.max_chunks_per_contract]
                covered = min(
                    token_count,
                    selected[-1] + chunk_content_size if token_count else 0,
                )
                split_samples[split] += 1
                split_chunks[split] += len(selected)
                token_counts.append(token_count)
                chunks_per_contract.append(len(selected))
                coverage_ratios.append(covered / token_count if token_count else 0.0)
                hashes[digest] += 1
                if len(starts) > len(selected):
                    truncated_contract_count += 1

    original_contract_samples = sum(split_samples.values())
    generated_mlm_chunks = sum(split_chunks.values())
    is_transductive = args.mode == "full_corpus"
    report = {
        "mode": args.mode,
        "corpus_path": relative(output_path),
        "vocab_path": relative(vocab_path),
        "use_labels": False,
        "is_dive_transductive": is_transductive,
        "setting_note": TRANSDUCTIVE_NOTE if is_transductive else STRICT_NOTE,
        "source_splits": splits,
        "split_contract_samples": dict(split_samples),
        "split_generated_mlm_chunks": dict(split_chunks),
        "original_contract_samples": original_contract_samples,
        "generated_mlm_chunks": generated_mlm_chunks,
        "average_chunks_per_contract": (
            generated_mlm_chunks / original_contract_samples
            if original_contract_samples
            else 0.0
        ),
        "max_chunks_per_contract": max(chunks_per_contract) if chunks_per_contract else 0,
        "max_chunks_per_contract_limit": args.max_chunks_per_contract,
        "truncated_contract_count": truncated_contract_count,
        "empty_opcode_count": empty_opcode_count,
        "unique_opcode_hash_count": len(hashes),
        "duplicate_opcode_sample_count": sum(v - 1 for v in hashes.values() if v > 1),
        "covered_token_ratio_mean": statistics.fmean(coverage_ratios) if coverage_ratios else 0.0,
        "covered_token_ratio_p50": percentile(coverage_ratios, 0.50),
        "covered_token_ratio_p90": percentile(coverage_ratios, 0.90),
        "token_count_mean": statistics.fmean(token_counts) if token_counts else 0.0,
        "token_count_p50": percentile(token_counts, 0.50),
        "token_count_p90": percentile(token_counts, 0.90),
        "max_len": args.max_len,
        "chunk_content_size": chunk_content_size,
        "chunk_stride": args.chunk_stride,
    }
    write_report(report, txt_path, json_path)
    print(f"[OK] wrote {relative(output_path)}")
    print(f"[INFO] {report['setting_note']}")


if __name__ == "__main__":
    main()
