import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_NAMES = [
    "approve_selector",
    "transfer_from_selector",
    "transfer_selector",
    "allowance_selector",
    "balance_of_selector",
    "swap_selector_family",
    "buy_sell_bid_order_text",
    "price_amount_balance_text",
    "calldata_to_state_write",
    "calldata_to_external_call",
    "external_call_then_state_write",
    "state_write_then_external_call",
    "deadline_or_time_guard_text",
    "slippage_or_min_amount_text",
    "commit_reveal_text",
]

SELECTORS = {
    "approve_selector": {"0x095ea7b3"},
    "transfer_from_selector": {"0x23b872dd"},
    "transfer_selector": {"0xa9059cbb"},
    "allowance_selector": {"0xdd62ed3e"},
    "balance_of_selector": {"0x70a08231"},
    "swap_selector_family": {
        "0x38ed1739",
        "0x8803dbee",
        "0x7ff36ab5",
        "0x4a25d94a",
        "0x18cbafe5",
        "0xfb3bdb41",
        "0x414bf389",
        "0xc04b8d59",
        "0xdb3e2198",
        "0xf28c0498",
        "0x3593564c",
    },
}

TEXT_KEYWORDS = {
    "buy_sell_bid_order_text": {
        "buy",
        "sell",
        "bid",
        "order",
        "auction",
        "swap",
        "trade",
    },
    "price_amount_balance_text": {
        "price",
        "amount",
        "balance",
        "reserve",
        "rate",
        "token",
    },
    "deadline_or_time_guard_text": {
        "deadline",
        "expire",
        "expiry",
        "timeout",
        "timestamp",
        "time",
    },
    "slippage_or_min_amount_text": {
        "slippage",
        "amountoutmin",
        "amountinmax",
        "minamount",
        "minimum",
        "min",
    },
    "commit_reveal_text": {
        "commit",
        "reveal",
        "commitment",
        "hashlock",
    },
}

CALL_OPS = {"CALL", "DELEGATECALL", "STATICCALL", "CALLCODE"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build DIVE Front Running special chunk feature cache."
    )
    parser.add_argument("--dataset", default="DIVE", choices=["DIVE"])
    parser.add_argument("--data_dir", default="data/processed/DIVE_random_split")
    parser.add_argument(
        "--output_dir",
        default="data/features/front_running_special/dive_random_stride256_max64",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=256)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--debug_num_contracts", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def iter_jsonl(path, debug_limit=None):
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if debug_limit is not None and index >= debug_limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def split_paths(data_dir):
    return {
        "train": data_dir / "train_mlsmote.jsonl",
        "valid": data_dir / "valid.jsonl",
        "test": data_dir / "test.jsonl",
    }


def normalize_selector(text):
    text = str(text).strip().lower()
    if not text.startswith("0x"):
        text = f"0x{text}"
    return text


def decode_hex_text(token):
    text = str(token).strip().lower()
    if not text.startswith("0x"):
        return ""
    hex_text = text[2:]
    if len(hex_text) < 4 or len(hex_text) % 2 != 0:
        return ""
    try:
        raw = bytes.fromhex(hex_text)
    except ValueError:
        return ""
    chars = []
    for value in raw:
        if 32 <= value <= 126:
            chars.append(chr(value))
        else:
            chars.append(" ")
    decoded = "".join(chars).strip().lower()
    return " ".join(decoded.split())


def positions(tokens, candidates):
    return [idx for idx, token in enumerate(tokens) if token in candidates]


def has_order(left, right, window):
    for left_idx in left:
        for right_idx in right:
            if 0 < right_idx - left_idx <= window:
                return True
    return False


def chunk_tokens(tokens, chunk_index, chunk_size, chunk_stride):
    start = chunk_index * chunk_stride
    end = min(len(tokens), start + chunk_size)
    if start >= len(tokens):
        return []
    return tokens[start:end]


def compute_chunk_features(tokens):
    values = {name: 0.0 for name in FEATURE_NAMES}
    upper_tokens = [token.upper() for token in tokens]
    lower_tokens = [token.lower() for token in tokens]

    for idx, token in enumerate(upper_tokens[:-1]):
        if token == "PUSH4":
            selector = normalize_selector(lower_tokens[idx + 1])
            for name, selector_set in SELECTORS.items():
                if selector in selector_set:
                    values[name] = 1.0

    decoded_text = " ".join(decode_hex_text(token) for token in lower_tokens)
    for name, keywords in TEXT_KEYWORDS.items():
        if any(keyword in decoded_text for keyword in keywords):
            values[name] = 1.0

    calldataload = positions(upper_tokens, {"CALLDATALOAD", "CALLDATACOPY"})
    state_write = positions(upper_tokens, {"SSTORE"})
    external_call = positions(upper_tokens, CALL_OPS)
    if has_order(calldataload, state_write, 160):
        values["calldata_to_state_write"] = 1.0
    if has_order(calldataload, external_call, 160):
        values["calldata_to_external_call"] = 1.0
    if has_order(external_call, state_write, 96):
        values["external_call_then_state_write"] = 1.0
    if has_order(state_write, external_call, 96):
        values["state_write_then_external_call"] = 1.0
    return [values[name] for name in FEATURE_NAMES]


def process_item(task):
    item, chunk_size, chunk_stride, max_chunks = task
    tokens = str(item.get("opcode", "")).split()
    chunk_count = min(
        max_chunks,
        max(1, (max(0, len(tokens) - 1) // chunk_stride) + 1) if tokens else 0,
    )
    features = torch.zeros((max_chunks, len(FEATURE_NAMES)), dtype=torch.float32)
    chunk_mask = torch.zeros(max_chunks, dtype=torch.bool)
    for chunk_index in range(chunk_count):
        current = chunk_tokens(tokens, chunk_index, chunk_size, chunk_stride)
        if not current:
            continue
        chunk_mask[chunk_index] = True
        features[chunk_index] = torch.tensor(
            compute_chunk_features(current),
            dtype=torch.float32,
        )
    return {
        "id": str(item.get("id")),
        "chunk_mask": chunk_mask,
        "front_special_features": features,
        "binary_label": int(item.get("binary_label", 0)),
        "multi_labels": item.get("multi_labels", []),
        "token_length": len(tokens),
        "num_chunks": int(chunk_mask.sum().item()),
    }


def build_split(split, input_path, output_path, args):
    if output_path.exists() and not args.force:
        print(f"[OK] {output_path.relative_to(PROJECT_ROOT)} already exists, skipping")
        return
    rows = list(iter_jsonl(input_path, args.debug_num_contracts))
    tasks = [
        (row, args.chunk_size, args.chunk_stride, args.max_chunks)
        for row in rows
    ]
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            processed = list(
                tqdm(
                    pool.imap(process_item, tasks, chunksize=128),
                    total=len(tasks),
                    desc=f"front-special:{split}",
                )
            )
    else:
        processed = [
            process_item(task)
            for task in tqdm(tasks, desc=f"front-special:{split}")
        ]

    ids = [row["id"] for row in processed]
    chunk_mask = torch.stack([row["chunk_mask"] for row in processed])
    features = torch.stack([row["front_special_features"] for row in processed])
    label_width = len(processed[0]["multi_labels"]) if processed else 0
    multi_labels = torch.tensor(
        [row["multi_labels"] for row in processed],
        dtype=torch.float32,
    ) if processed else torch.empty((0, label_width), dtype=torch.float32)
    binary_labels = torch.tensor(
        [row["binary_label"] for row in processed],
        dtype=torch.float32,
    ) if processed else torch.empty((0,), dtype=torch.float32)
    feature_counts = features.sum(dim=(0, 1)).tolist() if processed else []
    report = {
        "split": split,
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "sample_count": len(processed),
        "max_chunks": args.max_chunks,
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "feature_names": FEATURE_NAMES,
        "feature_positive_chunk_counts": {
            name: int(feature_counts[idx])
            for idx, name in enumerate(FEATURE_NAMES)
        },
        "mean_active_chunks": float(chunk_mask.float().sum(dim=1).mean().item())
        if processed
        else 0.0,
        "max_active_chunks": int(chunk_mask.sum(dim=1).max().item())
        if processed
        else 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": ids,
            "chunk_mask": chunk_mask,
            "front_special_features": features,
            "feature_names": FEATURE_NAMES,
            "binary_labels": binary_labels,
            "multi_labels": multi_labels,
            "report": report,
        },
        output_path,
    )
    print(f"[OK] wrote {output_path.relative_to(PROJECT_ROOT)}")


def main():
    args = parse_args()
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)
    for split, input_path in split_paths(data_dir).items():
        if not input_path.exists():
            raise FileNotFoundError(f"missing input split: {input_path}")
        build_split(split, input_path, output_dir / f"{split}.pt", args)


if __name__ == "__main__":
    main()
