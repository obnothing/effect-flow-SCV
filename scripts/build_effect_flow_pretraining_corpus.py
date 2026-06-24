import argparse
import json
import multiprocessing as mp
import statistics
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from effect_flow_schema import (  # noqa: E402
    EFFECT_TYPES,
    EFPP_PATTERNS,
    annotate_effect_types,
    annotate_efpp_patterns,
    build_token_units,
    effect_events,
)
from effect_flow_utils import (  # noqa: E402
    DATASET_SPECS,
    REPORT_DIR,
    iter_jsonl,
    opcode_hash,
    percentile,
    relative,
    resolve,
    strict_split_paths,
    write_json,
    write_text,
)
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


FORBIDDEN_OUTPUT_FIELDS = {"binary_label", "multi_labels", "labels", "vulnerability"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build label-free effect-flow chunks.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument(
        "--vocab_path", default="data/processed/BJUT_SC01/evm_vocab.json"
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=256)
    parser.add_argument(
        "--max_chunks_per_contract",
        type=int,
        default=None,
        help="Maximum chunks per contract. Defaults to 32 for BJUT and 64 for DIVE.",
    )
    parser.add_argument("--debug_num_contracts", type=int, default=None)
    parser.add_argument("--include_other_operands", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def covered_count(total, ranges):
    if not ranges:
        return 0
    ranges = sorted((max(0, a), min(total, b)) for a, b in ranges if b > a)
    merged = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def pad(values, length, pad_value):
    return values + [pad_value] * (length - len(values))


def make_payload(
    tokenizer,
    dataset,
    split,
    contract_id,
    opcode_digest,
    content_units,
    chunk_index,
    num_chunks,
    start,
    total_tokens,
    coverage_ratio,
    max_len,
    include_other_operands,
):
    effects = annotate_effect_types(
        content_units, include_other_operands=include_other_operands
    )
    pattern_labels, pattern_matches = annotate_efpp_patterns(content_units)
    content_tokens = [unit.token for unit in content_units]
    tokens = [tokenizer.cls_token] + content_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(tokens)
    zero_multihot = [0] * len(EFFECT_TYPES)
    effect_type_ids = [-100] + [effect["primary_id"] for effect in effects] + [-100]
    effect_type_multihot = (
        [list(zero_multihot)]
        + [effect["multihot"] for effect in effects]
        + [list(zero_multihot)]
    )
    etp_loss_mask = [0] + [effect["loss_mask"] for effect in effects] + [0]
    tokens = pad(tokens, max_len, tokenizer.pad_token)
    input_ids = pad(input_ids, max_len, tokenizer.pad_token_id)
    attention_mask = pad(attention_mask, max_len, 0)
    effect_type_ids = pad(effect_type_ids, max_len, -100)
    effect_type_multihot = pad(
        effect_type_multihot, max_len, list(zero_multihot)
    )
    etp_loss_mask = pad(etp_loss_mask, max_len, 0)
    payload = {
        "id": str(contract_id),
        "source_dataset": dataset,
        "source_split": split,
        "chunk_index": chunk_index,
        "num_chunks": num_chunks,
        "opcode_hash": opcode_digest,
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "effect_type_ids": effect_type_ids,
        "effect_type_multihot": effect_type_multihot,
        "etp_loss_mask": etp_loss_mask,
        "efpp_pattern_labels": pattern_labels,
        "efpp_matches": pattern_matches,
        "effect_events": effect_events(content_units, effects, offset=1),
        "coverage_start": start,
        "coverage_end": min(total_tokens, start + len(content_units)),
        "contract_token_count": total_tokens,
        "contract_coverage_ratio": coverage_ratio,
    }
    forbidden = FORBIDDEN_OUTPUT_FIELDS.intersection(payload)
    if forbidden:
        raise AssertionError(f"Vulnerability labels leaked into corpus: {forbidden}")
    return payload


_WORKER_TOKENIZER = None
_WORKER_CONFIG = None


def init_worker(vocab_path, config):
    global _WORKER_TOKENIZER, _WORKER_CONFIG
    _WORKER_TOKENIZER = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
    _WORKER_CONFIG = config


def process_contract(task):
    line_no, item, split = task
    config = _WORKER_CONFIG
    tokenizer = _WORKER_TOKENIZER
    raw_id = item.get("id")
    if raw_id is None:
        raw_id = item.get("address")
    contract_id = raw_id if raw_id is not None else f"{split}_{line_no}"
    opcode = item.get("opcode")
    if opcode is None:
        opcode = item.get("opcodes")
    if opcode is None:
        opcode = item.get("opcode_sequence")
    opcode = opcode or ""
    units = build_token_units(opcode, tokenizer)
    total_tokens = len(units)
    if not units:
        return {
            "lines": [],
            "chunks": 0,
            "empty": 1,
            "truncated": 0,
            "coverage_ratio": 0.0,
        }
    starts = list(range(0, total_tokens, config["chunk_stride"]))
    selected_starts = starts[: config["max_chunks_per_contract"]]
    ranges = [
        (start, min(total_tokens, start + config["content_size"]))
        for start in selected_starts
    ]
    coverage_ratio = covered_count(total_tokens, ranges) / total_tokens
    digest = opcode_hash(opcode)
    lines = []
    for chunk_index, start in enumerate(selected_starts):
        content_units = units[start : start + config["content_size"]]
        payload = make_payload(
            tokenizer,
            config["dataset"],
            split,
            contract_id,
            digest,
            content_units,
            chunk_index,
            len(selected_starts),
            start,
            total_tokens,
            coverage_ratio,
            config["max_len"],
            config["include_other_operands"],
        )
        lines.append(json.dumps(payload, ensure_ascii=False))
    return {
        "lines": lines,
        "chunks": len(lines),
        "empty": 0,
        "truncated": int(len(starts) > len(selected_starts)),
        "coverage_ratio": coverage_ratio,
    }


def build_split(args, tokenizer, split, input_path, output_path):
    max_len = args.max_len
    content_size = max_len - 2
    if content_size <= 0:
        raise ValueError("max_len must leave room for [CLS] and [SEP].")
    if args.chunk_stride <= 0 or args.max_chunks_per_contract <= 0:
        raise ValueError("chunk_stride and max_chunks_per_contract must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    contract_count = chunk_count = empty_count = truncated_count = 0
    chunks_per_contract = []
    coverage_ratios = []

    worker_config = {
        "dataset": args.dataset,
        "max_len": max_len,
        "content_size": content_size,
        "chunk_stride": args.chunk_stride,
        "max_chunks_per_contract": args.max_chunks_per_contract,
        "include_other_operands": args.include_other_operands,
    }

    def tasks():
        for index, (line_no, item) in enumerate(iter_jsonl(input_path)):
            if args.debug_num_contracts is not None and index >= args.debug_num_contracts:
                break
            yield line_no, item, split

    workers = max(1, min(4, int(args.num_workers)))
    init_args = (str(resolve(args.vocab_path)), worker_config)
    if workers == 1:
        init_worker(*init_args)
        result_iterator = map(process_contract, tasks())
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            processes=workers, initializer=init_worker, initargs=init_args
        )
        result_iterator = pool.imap(process_contract, tasks(), chunksize=1)

    try:
        with temp_path.open("w", encoding="utf-8") as output:
            for result in tqdm(
                result_iterator,
                desc=f"effect_flow:{args.dataset}:{split}",
                unit="contract",
            ):
                contract_count += 1
                empty_count += result["empty"]
                truncated_count += result["truncated"]
                chunk_count += result["chunks"]
                chunks_per_contract.append(result["chunks"])
                coverage_ratios.append(result["coverage_ratio"])
                for line in result["lines"]:
                    output.write(line + "\n")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    temp_path.replace(output_path)
    return {
        "input_path": relative(input_path),
        "output_path": relative(output_path),
        "contracts": contract_count,
        "chunks": chunk_count,
        "empty_contracts": empty_count,
        "truncated_contracts": truncated_count,
        "mean_chunks_per_contract": statistics.fmean(chunks_per_contract)
        if chunks_per_contract
        else 0.0,
        "max_chunks_per_contract": max(chunks_per_contract, default=0),
        "coverage_ratio_mean": statistics.fmean(coverage_ratios)
        if coverage_ratios
        else 0.0,
        "coverage_ratio_p50": percentile(coverage_ratios, 0.5),
        "coverage_ratio_p90": percentile(coverage_ratios, 0.9),
    }


def main():
    args = parse_args()
    if args.max_chunks_per_contract is None:
        args.max_chunks_per_contract = 64 if args.dataset == "DIVE" else 32
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(args.vocab_path))
    input_paths = strict_split_paths(args.dataset)
    missing = [relative(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing strict split paths: {missing}. Run find_effect_flow_inputs.py first."
        )
    output_root = resolve(args.output_dir or DATASET_SPECS[args.dataset]["output_dir"])
    split_reports = {}
    for split, input_path in input_paths.items():
        output_path = output_root / f"{split}_effect_flow_chunks.jsonl"
        if output_path.exists() and not args.force:
            raise FileExistsError(f"{output_path} exists; pass --force to rebuild.")
        split_reports[split] = build_split(
            args, tokenizer, split, input_path, output_path
        )
    report = {
        "dataset": args.dataset,
        "input_policy": "strict opcode-hash grouped splits only",
        "vulnerability_labels_written": False,
        "max_len": args.max_len,
        "chunk_content_size": args.max_len - 2,
        "chunk_stride": args.chunk_stride,
        "max_chunks_per_contract": args.max_chunks_per_contract,
        "num_workers": max(1, min(4, int(args.num_workers))),
        "effect_type_count": len(EFFECT_TYPES),
        "efpp_pattern_count": len(EFPP_PATTERNS),
        "splits": split_reports,
        "total_contracts": sum(row["contracts"] for row in split_reports.values()),
        "total_chunks": sum(row["chunks"] for row in split_reports.values()),
    }
    json_path = REPORT_DIR / f"effect_flow_corpus_build_{args.dataset}.json"
    txt_path = REPORT_DIR / f"effect_flow_corpus_build_{args.dataset}.txt"
    write_json(json_path, report)
    lines = [
        f"Effect-flow corpus build report: {args.dataset}",
        "",
        "vulnerability_labels_written: false",
        f"total_contracts: {report['total_contracts']}",
        f"total_chunks: {report['total_chunks']}",
    ]
    for split, row in split_reports.items():
        lines.extend(["", f"[{split}]"] + [f"{key}: {value}" for key, value in row.items()])
    write_text(txt_path, lines)
    print(f"[OK] wrote effect-flow corpus to {relative(output_root)}")
    print(f"[OK] wrote {relative(txt_path)}")


if __name__ == "__main__":
    main()
