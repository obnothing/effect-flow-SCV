import argparse
import json
import math
import multiprocessing as mp
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_NAMES = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation",
]
TARGET_LABEL_NAMES = ["Front Running", "Bad Randomness"]
TARGET_LABEL_IDS = [LABEL_NAMES.index(name) for name in TARGET_LABEL_NAMES]
FRONT_LOCAL_ID = TARGET_LABEL_NAMES.index("Front Running")
BAD_LOCAL_ID = TARGET_LABEL_NAMES.index("Bad Randomness")

TYPED_FEATURE_NAMES = [
    "front_selector_swap_or_order",
    "front_selector_bid_or_auction",
    "front_selector_erc20_trade",
    "front_price_reserve_balance_text",
    "front_calldata_to_sstore",
    "front_calldata_to_call",
    "front_calldata_to_log",
    "front_calldata_to_branch_then_sstore",
    "front_calldata_to_amount_arithmetic",
    "front_deadline_guard_present",
    "front_slippage_guard_present",
    "front_commit_reveal_proxy_present",
    "front_missing_deadline_guard",
    "front_missing_slippage_guard",
    "front_missing_commit_reveal_proxy",
    "front_combined_typed_score",
    "bad_env_source_present",
    "bad_env_to_sha3",
    "bad_env_to_mod",
    "bad_env_to_arithmetic",
    "bad_env_to_index_like_access",
    "bad_env_to_transfer_amount",
    "bad_env_to_sstore_result",
    "bad_env_to_branch_then_sstore",
    "bad_random_text",
    "bad_no_oracle_like_external_call",
    "bad_time_condition_only_counter",
    "bad_combined_typed_score",
]

FEATURE_INDEX = {name: idx for idx, name in enumerate(TYPED_FEATURE_NAMES)}

CALL_OPS = {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}
VALUE_CALL_OPS = {"CALL", "CALLCODE"}
ENV_OPS = {"TIMESTAMP", "NUMBER", "BLOCKHASH", "DIFFICULTY", "PREVRANDAO", "COINBASE"}
BRANCH_OPS = {"LT", "GT", "SLT", "SGT", "EQ", "ISZERO", "JUMPI"}
ARITH_OPS = {"ADD", "SUB", "MUL", "DIV", "SDIV", "MOD", "SMOD", "EXP", "AND", "OR", "XOR"}
HASH_OPS = {"SHA3", "KECCAK256"}
STATE_OPS = {"SLOAD", "SSTORE"}
LOG_OPS = {"LOG0", "LOG1", "LOG2", "LOG3", "LOG4"}
INDEX_LIKE_OPS = {"MLOAD", "SLOAD", "SHA3", "KECCAK256"}

ERC20_SELECTORS = {
    "0xa9059cbb",  # transfer(address,uint256)
    "0x23b872dd",  # transferFrom(address,address,uint256)
    "0x095ea7b3",  # approve(address,uint256)
    "0xdd62ed3e",  # allowance(address,address)
    "0x70a08231",  # balanceOf(address)
}

SWAP_ORDER_SELECTORS = {
    "0x38ed1739",  # swapExactTokensForTokens
    "0x8803dbee",  # swapTokensForExactTokens
    "0x7ff36ab5",  # swapExactETHForTokens
    "0x4a25d94a",  # swapTokensForExactETH
    "0x18cbafe5",  # swapExactTokensForETH
    "0xfb3bdb41",  # swapETHForExactTokens
    "0x414bf389",  # exactInputSingle
    "0xc04b8d59",  # exactInput
    "0xdb3e2198",  # exactOutputSingle
    "0xf28c0498",  # exactOutput
    "0x5c11d795",  # swapExactTokensForTokensSupportingFeeOnTransferTokens
    "0x791ac947",  # swapExactTokensForETHSupportingFeeOnTransferTokens
    "0xb6f9de95",  # swapExactETHForTokensSupportingFeeOnTransferTokens
    "0x3593564c",  # UniversalRouter execute(bytes,bytes[],uint256)
    "0x24856bc3",  # UniversalRouter execute(bytes,bytes[])
}

BID_AUCTION_SELECTORS = {
    "0x1998aeef",  # bid()
    "0x454a2ab3",  # bid(uint256)
    "0x2e1a7d4d",  # withdraw(uint256), common auction/market flow proxy
}

ORACLE_LIKE_SELECTORS = {
    "0xfeaf968c",  # latestRoundData()
    "0x50d25bcd",  # latestAnswer()
    "0x313ce567",  # decimals()
    "0x54fd4d50",  # version()
}

TEXT_KEYWORDS = {
    "front_price_state": {
        "price",
        "reserve",
        "amount",
        "amountout",
        "amountin",
        "balance",
        "allowance",
        "order",
        "swap",
        "trade",
        "bid",
        "auction",
    },
    "front_deadline": {"deadline", "expire", "expiry", "timestamp"},
    "front_slippage": {
        "slippage",
        "amountoutmin",
        "amountinmax",
        "minout",
        "maxin",
        "minimum",
        "minamount",
    },
    "front_commit_reveal": {"commit", "reveal", "sealed"},
    "bad_random": {"random", "rand", "seed", "lottery", "winner", "dice", "draw"},
    "oracle": {"oracle", "chainlink", "vrf", "latestrounddata", "randomness"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build typed Front/Bad mechanism evidence audit for DIVE."
    )
    parser.add_argument("--data_dir", default="data/processed/DIVE_random_split")
    parser.add_argument(
        "--feature_dir",
        default=None,
        help="Optional EVM-BERT feature cache for id/chunk alignment.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/features/typed_mechanism_evidence/dive_random_stride256_max64",
    )
    parser.add_argument(
        "--baseline_prediction_dir",
        default="results/dive_side_scale_search/train_dive_side_scale_150_ep50",
        help="Optional directory containing test_predictions_per_label.jsonl.",
    )
    parser.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=256)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--num_labels", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug_num_contracts", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def split_paths(data_dir):
    return {
        "train": data_dir / "train_mlsmote.jsonl",
        "valid": data_dir / "valid.jsonl",
        "test": data_dir / "test.jsonl",
    }


def iter_jsonl(path, limit=None):
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_selector(value):
    value = str(value).strip().lower()
    if not value.startswith("0x"):
        value = f"0x{value}"
    return value


def decode_hex_text(value):
    value = str(value).strip().lower()
    if not value.startswith("0x"):
        return ""
    hex_text = value[2:]
    if len(hex_text) < 4 or len(hex_text) % 2 != 0:
        return ""
    try:
        raw = bytes.fromhex(hex_text)
    except ValueError:
        return ""
    chars = [chr(byte) if 32 <= byte <= 126 else " " for byte in raw]
    return " ".join("".join(chars).lower().split())


def clip01(value, scale=1.0):
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(scale)))


def bool_score(value):
    return 1.0 if value else 0.0


def parse_instructions(opcode):
    tokens = str(opcode or "").split()
    instructions = []
    pc = 0
    token_index = 0
    while token_index < len(tokens):
        op = tokens[token_index].upper()
        start = token_index
        token_index += 1
        arg = None
        width = 1
        match = re.fullmatch(r"PUSH(\d+)", op)
        if op == "PUSH0":
            width = 1
        elif match:
            push_width = int(match.group(1))
            width = 1 + push_width
            if push_width > 0 and token_index < len(tokens):
                arg = tokens[token_index]
                token_index += 1
        instructions.append(
            {
                "pc": pc,
                "op": op,
                "arg": arg,
                "token_start": start,
                "token_end": token_index,
            }
        )
        pc += width
    return tokens, instructions


def chunks_for_token(token_index, chunk_size, chunk_stride, max_chunks):
    chunks = []
    for chunk_id in range(max_chunks):
        start = chunk_id * chunk_stride
        end = start + chunk_size
        if start <= token_index < end:
            chunks.append(chunk_id)
        if start > token_index:
            break
    return chunks


def chunks_for_pair(left, right, chunk_size, chunk_stride, max_chunks):
    chunks = set()
    for inst in (left, right):
        chunks.update(
            chunks_for_token(inst["token_start"], chunk_size, chunk_stride, max_chunks)
        )
    return chunks


def real_chunk_count(token_count, chunk_stride, max_chunks):
    if token_count <= 0:
        return 0
    return min(max_chunks, max(1, (max(0, token_count - 1) // chunk_stride) + 1))


def safe_mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def binary_mi(targets, hits):
    total = len(targets)
    if total == 0:
        return 0.0
    counts = defaultdict(int)
    y_counts = defaultdict(int)
    x_counts = defaultdict(int)
    for y, x in zip(targets, hits):
        y = int(bool(y))
        x = int(bool(x))
        counts[(y, x)] += 1
        y_counts[y] += 1
        x_counts[x] += 1
    mi = 0.0
    for (y, x), count in counts.items():
        pxy = count / total
        py = y_counts[y] / total
        px = x_counts[x] / total
        if pxy > 0 and py > 0 and px > 0:
            mi += pxy * math.log2(pxy / (py * px))
    return mi


class TypedMechanismAnalyzer:
    def __init__(self, opcode, chunk_size, chunk_stride, max_chunks):
        self.tokens, self.instructions = parse_instructions(opcode)
        self.ops = [inst["op"] for inst in self.instructions]
        self.counts = Counter(self.ops)
        self.chunk_size = int(chunk_size)
        self.chunk_stride = int(chunk_stride)
        self.max_chunks = int(max_chunks)
        self.selectors = self._extract_selectors()
        self.decoded_text = self._decode_text()
        self.chunk_evidence = torch.zeros(
            (self.max_chunks, len(TARGET_LABEL_NAMES), len(TYPED_FEATURE_NAMES)),
            dtype=torch.float32,
        )

    def _extract_selectors(self):
        selectors = set()
        for inst in self.instructions:
            if inst["op"] == "PUSH4" and inst["arg"]:
                selector = normalize_selector(inst["arg"])
                if re.fullmatch(r"0x[0-9a-f]{8}", selector):
                    selectors.add(selector)
        return selectors

    def _decode_text(self):
        decoded = [decode_hex_text(token) for token in self.tokens]
        return " ".join(text for text in decoded if text)

    def keyword_hit(self, key):
        return any(keyword in self.decoded_text for keyword in TEXT_KEYWORDS[key])

    def positions(self, ops):
        ops = set(ops)
        return [
            (idx, inst)
            for idx, inst in enumerate(self.instructions)
            if inst["op"] in ops
        ]

    def first_forward_pairs(self, source_ops, target_ops, window):
        targets = self.positions(target_ops)
        if not targets:
            return []
        pairs = []
        for source_idx, source in self.positions(source_ops):
            for target_idx, target in targets:
                delta = target_idx - source_idx
                if 0 < delta <= window:
                    pairs.append((source, target))
                    break
        return pairs

    def chain_exists(self, ops_a, ops_b, ops_c, window_ab, window_bc):
        pairs = []
        b_positions = self.positions(ops_b)
        c_positions = self.positions(ops_c)
        if not b_positions or not c_positions:
            return False, pairs
        for a_idx, a_inst in self.positions(ops_a):
            for b_idx, b_inst in b_positions:
                if not (0 < b_idx - a_idx <= window_ab):
                    continue
                for c_idx, c_inst in c_positions:
                    if 0 < c_idx - b_idx <= window_bc:
                        pairs.append((a_inst, c_inst))
                        return True, pairs
        return False, pairs

    def set_feature(self, label_local_id, feature_name, value, pairs=None, selectors=None):
        value = float(max(0.0, min(1.0, value)))
        feature_id = FEATURE_INDEX[feature_name]
        if pairs:
            for left, right in pairs:
                for chunk_id in chunks_for_pair(
                    left,
                    right,
                    self.chunk_size,
                    self.chunk_stride,
                    self.max_chunks,
                ):
                    self.chunk_evidence[chunk_id, label_local_id, feature_id] = max(
                        float(self.chunk_evidence[chunk_id, label_local_id, feature_id]),
                        value,
                    )
        if selectors:
            for inst in self.instructions:
                if inst["op"] == "PUSH4" and inst["arg"]:
                    selector = normalize_selector(inst["arg"])
                    if selector not in selectors:
                        continue
                    for chunk_id in chunks_for_token(
                        inst["token_start"],
                        self.chunk_size,
                        self.chunk_stride,
                        self.max_chunks,
                    ):
                        self.chunk_evidence[chunk_id, label_local_id, feature_id] = max(
                            float(self.chunk_evidence[chunk_id, label_local_id, feature_id]),
                            value,
                        )

    def analyze_front(self):
        features = {name: 0.0 for name in TYPED_FEATURE_NAMES}
        calldata_sources = {"CALLDATALOAD", "CALLDATACOPY", "CALLDATASIZE", "CALLVALUE"}
        calldata_to_sstore = self.first_forward_pairs(calldata_sources, {"SSTORE"}, 220)
        calldata_to_call = self.first_forward_pairs(calldata_sources, VALUE_CALL_OPS, 220)
        calldata_to_log = self.first_forward_pairs(calldata_sources, LOG_OPS, 220)
        calldata_to_arith = self.first_forward_pairs(calldata_sources, ARITH_OPS, 96)
        calldata_branch_then_sstore, branch_sstore_pairs = self.chain_exists(
            calldata_sources,
            BRANCH_OPS,
            {"SSTORE"},
            120,
            160,
        )
        timestamp_branch_pairs = self.first_forward_pairs({"TIMESTAMP"}, BRANCH_OPS, 96)

        features["front_selector_swap_or_order"] = bool_score(
            bool(self.selectors & SWAP_ORDER_SELECTORS)
        )
        features["front_selector_bid_or_auction"] = bool_score(
            bool(self.selectors & BID_AUCTION_SELECTORS)
            or any(word in self.decoded_text for word in ("bid", "auction"))
        )
        features["front_selector_erc20_trade"] = bool_score(bool(self.selectors & ERC20_SELECTORS))
        features["front_price_reserve_balance_text"] = bool_score(
            self.keyword_hit("front_price_state")
        )
        features["front_calldata_to_sstore"] = clip01(len(calldata_to_sstore), 3)
        features["front_calldata_to_call"] = clip01(len(calldata_to_call), 3)
        features["front_calldata_to_log"] = clip01(len(calldata_to_log), 2)
        features["front_calldata_to_branch_then_sstore"] = bool_score(
            calldata_branch_then_sstore
        )
        features["front_calldata_to_amount_arithmetic"] = clip01(len(calldata_to_arith), 4)
        features["front_deadline_guard_present"] = max(
            bool_score(self.keyword_hit("front_deadline")),
            clip01(len(timestamp_branch_pairs), 2),
        )
        features["front_slippage_guard_present"] = max(
            bool_score(self.keyword_hit("front_slippage")),
            0.5 * bool_score(bool(calldata_to_arith))
            * bool_score(bool(self.first_forward_pairs(calldata_sources, BRANCH_OPS, 96))),
        )
        features["front_commit_reveal_proxy_present"] = max(
            bool_score(self.keyword_hit("front_commit_reveal")),
            0.5
            * bool_score(bool(self.first_forward_pairs(calldata_sources, HASH_OPS, 120)))
            * bool_score(bool(self.first_forward_pairs(HASH_OPS, {"SSTORE"}, 160))),
        )

        input_path_score = max(
            features["front_calldata_to_sstore"],
            features["front_calldata_to_call"],
            features["front_calldata_to_log"],
            features["front_calldata_to_branch_then_sstore"],
        )
        selector_score = max(
            features["front_selector_swap_or_order"],
            features["front_selector_bid_or_auction"],
            0.5 * features["front_selector_erc20_trade"],
            features["front_price_reserve_balance_text"],
        )
        features["front_missing_deadline_guard"] = (
            input_path_score * (1.0 - features["front_deadline_guard_present"])
        )
        features["front_missing_slippage_guard"] = (
            max(input_path_score, selector_score)
            * (1.0 - features["front_slippage_guard_present"])
        )
        features["front_missing_commit_reveal_proxy"] = (
            max(input_path_score, selector_score)
            * (1.0 - features["front_commit_reveal_proxy_present"])
        )
        features["front_combined_typed_score"] = max(
            min(1.0, 0.55 * input_path_score + 0.45 * selector_score),
            min(
                1.0,
                0.45 * input_path_score
                + 0.25 * selector_score
                + 0.15 * features["front_missing_deadline_guard"]
                + 0.15 * features["front_missing_slippage_guard"],
            ),
        )

        self.set_feature(
            FRONT_LOCAL_ID,
            "front_selector_swap_or_order",
            features["front_selector_swap_or_order"],
            selectors=SWAP_ORDER_SELECTORS,
        )
        self.set_feature(
            FRONT_LOCAL_ID,
            "front_selector_bid_or_auction",
            features["front_selector_bid_or_auction"],
            selectors=BID_AUCTION_SELECTORS,
        )
        self.set_feature(
            FRONT_LOCAL_ID,
            "front_selector_erc20_trade",
            features["front_selector_erc20_trade"],
            selectors=ERC20_SELECTORS,
        )
        self.set_feature(
            FRONT_LOCAL_ID,
            "front_calldata_to_sstore",
            features["front_calldata_to_sstore"],
            pairs=calldata_to_sstore,
        )
        self.set_feature(
            FRONT_LOCAL_ID,
            "front_calldata_to_call",
            features["front_calldata_to_call"],
            pairs=calldata_to_call,
        )
        self.set_feature(
            FRONT_LOCAL_ID,
            "front_calldata_to_log",
            features["front_calldata_to_log"],
            pairs=calldata_to_log,
        )
        self.set_feature(
            FRONT_LOCAL_ID,
            "front_calldata_to_branch_then_sstore",
            features["front_calldata_to_branch_then_sstore"],
            pairs=branch_sstore_pairs,
        )
        return features

    def analyze_bad_randomness(self):
        features = {name: 0.0 for name in TYPED_FEATURE_NAMES}
        env_to_sha3 = self.first_forward_pairs(ENV_OPS, HASH_OPS, 128)
        env_to_mod = self.first_forward_pairs(ENV_OPS, {"MOD", "SMOD"}, 128)
        env_to_arith = self.first_forward_pairs(ENV_OPS, ARITH_OPS, 128)
        env_to_index = self.first_forward_pairs(ENV_OPS, INDEX_LIKE_OPS, 160)
        env_to_call = self.first_forward_pairs(ENV_OPS, VALUE_CALL_OPS, 220)
        env_to_sstore = self.first_forward_pairs(ENV_OPS, {"SSTORE"}, 220)
        env_branch_then_sstore, env_branch_sstore_pairs = self.chain_exists(
            ENV_OPS,
            BRANCH_OPS,
            {"SSTORE", "CALL"},
            120,
            180,
        )
        env_to_branch = self.first_forward_pairs(ENV_OPS, BRANCH_OPS, 128)
        oracle_like = bool(self.selectors & ORACLE_LIKE_SELECTORS) or self.keyword_hit("oracle")

        features["bad_env_source_present"] = bool_score(
            sum(self.counts[op] for op in ENV_OPS) > 0
        )
        features["bad_env_to_sha3"] = clip01(len(env_to_sha3), 2)
        features["bad_env_to_mod"] = clip01(len(env_to_mod), 2)
        features["bad_env_to_arithmetic"] = clip01(len(env_to_arith), 4)
        features["bad_env_to_index_like_access"] = clip01(len(env_to_index), 3)
        features["bad_env_to_transfer_amount"] = clip01(len(env_to_call), 2)
        features["bad_env_to_sstore_result"] = clip01(len(env_to_sstore), 2)
        features["bad_env_to_branch_then_sstore"] = bool_score(env_branch_then_sstore)
        features["bad_random_text"] = bool_score(self.keyword_hit("bad_random"))
        random_path = max(
            features["bad_env_to_sha3"],
            features["bad_env_to_mod"],
            features["bad_env_to_arithmetic"],
            features["bad_env_to_index_like_access"],
            features["bad_env_to_transfer_amount"],
            features["bad_env_to_sstore_result"],
        )
        features["bad_no_oracle_like_external_call"] = random_path * (1.0 - bool_score(oracle_like))
        features["bad_time_condition_only_counter"] = max(
            0.0,
            clip01(len(env_to_branch), 3)
            - max(features["bad_env_to_sha3"], features["bad_env_to_mod"]),
        )
        features["bad_combined_typed_score"] = min(
            1.0,
            0.35 * max(features["bad_env_to_sha3"], features["bad_env_to_mod"])
            + 0.25 * max(
                features["bad_env_to_index_like_access"],
                features["bad_env_to_transfer_amount"],
                features["bad_env_to_sstore_result"],
            )
            + 0.20 * features["bad_env_to_arithmetic"]
            + 0.20 * max(
                features["bad_random_text"],
                features["bad_no_oracle_like_external_call"],
            ),
        )

        self.set_feature(
            BAD_LOCAL_ID,
            "bad_env_to_sha3",
            features["bad_env_to_sha3"],
            pairs=env_to_sha3,
        )
        self.set_feature(
            BAD_LOCAL_ID,
            "bad_env_to_mod",
            features["bad_env_to_mod"],
            pairs=env_to_mod,
        )
        self.set_feature(
            BAD_LOCAL_ID,
            "bad_env_to_arithmetic",
            features["bad_env_to_arithmetic"],
            pairs=env_to_arith,
        )
        self.set_feature(
            BAD_LOCAL_ID,
            "bad_env_to_index_like_access",
            features["bad_env_to_index_like_access"],
            pairs=env_to_index,
        )
        self.set_feature(
            BAD_LOCAL_ID,
            "bad_env_to_transfer_amount",
            features["bad_env_to_transfer_amount"],
            pairs=env_to_call,
        )
        self.set_feature(
            BAD_LOCAL_ID,
            "bad_env_to_sstore_result",
            features["bad_env_to_sstore_result"],
            pairs=env_to_sstore,
        )
        self.set_feature(
            BAD_LOCAL_ID,
            "bad_env_to_branch_then_sstore",
            features["bad_env_to_branch_then_sstore"],
            pairs=env_branch_sstore_pairs,
        )
        return features

    def build(self):
        features = torch.zeros(
            (len(TARGET_LABEL_NAMES), len(TYPED_FEATURE_NAMES)),
            dtype=torch.float32,
        )
        front = self.analyze_front()
        bad = self.analyze_bad_randomness()
        for feature_name, value in front.items():
            features[FRONT_LOCAL_ID, FEATURE_INDEX[feature_name]] = float(value)
        for feature_name, value in bad.items():
            features[BAD_LOCAL_ID, FEATURE_INDEX[feature_name]] = float(value)
        return features, self.chunk_evidence


def process_item(task):
    item, args_dict = task
    analyzer = TypedMechanismAnalyzer(
        item.get("opcode", ""),
        args_dict["chunk_size"],
        args_dict["chunk_stride"],
        args_dict["max_chunks"],
    )
    contract_evidence, chunk_evidence = analyzer.build()
    token_count = len(analyzer.tokens)
    chunk_mask = torch.zeros(args_dict["max_chunks"], dtype=torch.bool)
    count = real_chunk_count(
        token_count,
        args_dict["chunk_stride"],
        args_dict["max_chunks"],
    )
    if count > 0:
        chunk_mask[:count] = True
    multi_labels = torch.tensor(
        item.get("multi_labels", [0] * args_dict["num_labels"]),
        dtype=torch.float32,
    )
    if multi_labels.numel() != args_dict["num_labels"]:
        raise ValueError(f"Invalid multi_labels width for id={item.get('id')}")
    return {
        "id": str(item.get("id")),
        "chunk_mask": chunk_mask,
        "binary_label": float(item.get("binary_label", 0.0)),
        "multi_labels": multi_labels,
        "typed_contract_evidence": contract_evidence,
        "typed_chunk_evidence": chunk_evidence,
        "token_count": token_count,
        "real_chunks": int(chunk_mask.sum().item()),
    }


def load_feature_payload(feature_dir, split):
    if feature_dir is None:
        return None
    path = feature_dir / f"{split}.pt"
    if not path.exists():
        print(f"[WARN] feature cache not found for alignment: {display_path(path)}")
        return None
    return torch.load(path, map_location="cpu")


def align_with_feature_cache(split, processed, feature_payload):
    if feature_payload is None:
        return processed
    feature_ids = [str(value) for value in feature_payload["ids"]]
    processed_ids = [str(row["id"]) for row in processed]
    if len(feature_ids) != len(processed_ids):
        raise ValueError(
            f"{split}: typed rows {len(processed_ids)} do not match feature rows {len(feature_ids)}"
        )
    mismatches = [
        idx for idx, (left, right) in enumerate(zip(processed_ids, feature_ids))
        if left != right
    ]
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"{split}: ids are not aligned at {first}: "
            f"{processed_ids[first]} != {feature_ids[first]}"
        )
    feature_mask = feature_payload["chunk_mask"].bool()
    feature_labels = feature_payload["multi_labels"].float()
    feature_binary = feature_payload["binary_labels"].float()
    for idx, row in enumerate(processed):
        row["id"] = feature_payload["ids"][idx]
        row["chunk_mask"] = feature_mask[idx]
        row["multi_labels"] = feature_labels[idx]
        row["binary_label"] = float(feature_binary[idx].item())
        row["typed_chunk_evidence"] = row["typed_chunk_evidence"].masked_fill(
            ~feature_mask[idx].view(-1, 1, 1),
            0.0,
        )
    return processed


def load_prediction_groups(path):
    if path is None or not path.exists():
        return {}
    groups = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            labels = [int(value) for value in row["multi_true"]]
            preds = [int(value) for value in row["multi_pred"]]
            item = {}
            for label_name, label_id in zip(TARGET_LABEL_NAMES, TARGET_LABEL_IDS):
                true = labels[label_id]
                pred = preds[label_id]
                if true and pred:
                    group = "tp"
                elif not true and pred:
                    group = "fp"
                elif true and not pred:
                    group = "fn"
                else:
                    group = "tn"
                item[label_name] = group
            groups[sample_id] = item
    return groups


def feature_rows_for_label(values, targets, label_name):
    target_bool = targets.bool()
    neg_bool = ~target_bool
    rows = []
    y = [int(value) for value in target_bool.tolist()]
    for feature_id, feature_name in enumerate(TYPED_FEATURE_NAMES):
        feature_values = values[:, feature_id]
        hit = feature_values > 0.5
        pos_mean = float(feature_values[target_bool].mean().item()) if target_bool.any() else 0.0
        neg_mean = float(feature_values[neg_bool].mean().item()) if neg_bool.any() else 0.0
        pos_hit = float(hit[target_bool].float().mean().item()) if target_bool.any() else 0.0
        neg_hit = float(hit[neg_bool].float().mean().item()) if neg_bool.any() else 0.0
        odds_ratio = ((pos_hit + 1e-3) / (1.0 - pos_hit + 1e-3)) / (
            (neg_hit + 1e-3) / (1.0 - neg_hit + 1e-3)
        )
        rows.append(
            {
                "label_name": label_name,
                "feature_name": feature_name,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "delta": pos_mean - neg_mean,
                "positive_hit_rate": pos_hit,
                "negative_hit_rate": neg_hit,
                "hit_rate_delta": pos_hit - neg_hit,
                "odds_ratio": odds_ratio,
                "mutual_information": binary_mi(y, [int(value) for value in hit.tolist()]),
            }
        )
    return rows


def confusion_audit(values, ids, prediction_groups, label_name):
    if not prediction_groups:
        return []
    label_groups = defaultdict(list)
    for idx, sample_id in enumerate(ids):
        group = prediction_groups.get(str(sample_id), {}).get(label_name)
        if group:
            label_groups[group].append(idx)
    rows = []
    for feature_id, feature_name in enumerate(TYPED_FEATURE_NAMES):
        group_stats = {}
        for group in ("tp", "fp", "fn", "tn"):
            indices = label_groups.get(group, [])
            if indices:
                feature_values = values[indices, feature_id]
                group_stats[group] = {
                    "count": len(indices),
                    "mean": float(feature_values.mean().item()),
                    "hit_rate": float((feature_values > 0.5).float().mean().item()),
                }
            else:
                group_stats[group] = {"count": 0, "mean": 0.0, "hit_rate": 0.0}
        rows.append(
            {
                "label_name": label_name,
                "feature_name": feature_name,
                "groups": group_stats,
                "fp_minus_tp_mean": group_stats["fp"]["mean"] - group_stats["tp"]["mean"],
                "fn_minus_tp_mean": group_stats["fn"]["mean"] - group_stats["tp"]["mean"],
                "fp_minus_tp_hit": group_stats["fp"]["hit_rate"]
                - group_stats["tp"]["hit_rate"],
                "fn_minus_tp_hit": group_stats["fn"]["hit_rate"]
                - group_stats["tp"]["hit_rate"],
            }
        )
    return rows


def feature_audit(split, ids, contract_evidence, multi_labels, prediction_groups):
    audit = {
        "split": split,
        "feature_names": TYPED_FEATURE_NAMES,
        "target_label_names": TARGET_LABEL_NAMES,
        "labels": [],
    }
    labels = multi_labels.float()
    for local_label_id, (label_name, global_label_id) in enumerate(
        zip(TARGET_LABEL_NAMES, TARGET_LABEL_IDS)
    ):
        targets = labels[:, global_label_id] > 0.5
        values = contract_evidence[:, local_label_id, :]
        rows = feature_rows_for_label(values, targets, label_name)
        audit["labels"].append(
            {
                "label_name": label_name,
                "support": int(targets.sum().item()),
                "negative_count": int((~targets).sum().item()),
                "features": rows,
                "confusion_features": confusion_audit(
                    values,
                    ids,
                    prediction_groups,
                    label_name,
                ),
            }
        )
    return audit


def write_audit_text(path, audit):
    lines = [f"DIVE typed mechanism evidence audit: {audit['split']}", ""]
    for label in audit["labels"]:
        lines.append(
            f"{label['label_name']} support={label['support']} "
            f"negatives={label['negative_count']}"
        )
        lines.append("Top positive-vs-negative typed features:")
        rows = sorted(
            label["features"],
            key=lambda row: (
                abs(float(row["hit_rate_delta"])),
                abs(float(row["delta"])),
                float(row["mutual_information"]),
            ),
            reverse=True,
        )
        for row in rows[:12]:
            lines.append(
                f"  {row['feature_name']}: pos_mean={row['positive_mean']:.4f} "
                f"neg_mean={row['negative_mean']:.4f} delta={row['delta']:.4f} "
                f"pos_hit={row['positive_hit_rate']:.4f} "
                f"neg_hit={row['negative_hit_rate']:.4f} "
                f"hit_delta={row['hit_rate_delta']:.4f} "
                f"odds={row['odds_ratio']:.2f} mi={row['mutual_information']:.5f}"
            )
        if label.get("confusion_features"):
            lines.append("Top baseline confusion differences:")
            confusion_rows = sorted(
                label["confusion_features"],
                key=lambda row: (
                    abs(float(row["fp_minus_tp_hit"])),
                    abs(float(row["fn_minus_tp_hit"])),
                    abs(float(row["fp_minus_tp_mean"])),
                ),
                reverse=True,
            )
            for row in confusion_rows[:12]:
                groups = row["groups"]
                lines.append(
                    f"  {row['feature_name']}: "
                    f"tp={groups['tp']['mean']:.4f}/{groups['tp']['hit_rate']:.4f}"
                    f"(n={groups['tp']['count']}) "
                    f"fp={groups['fp']['mean']:.4f}/{groups['fp']['hit_rate']:.4f}"
                    f"(n={groups['fp']['count']}) "
                    f"fn={groups['fn']['mean']:.4f}/{groups['fn']['hit_rate']:.4f}"
                    f"(n={groups['fn']['count']}) "
                    f"fp-tp-hit={row['fp_minus_tp_hit']:.4f} "
                    f"fn-tp-hit={row['fn_minus_tp_hit']:.4f}"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_split(split, input_path, output_path, feature_payload, prediction_groups, args):
    if output_path.exists() and not args.force:
        print(f"[OK] {display_path(output_path)} exists, skipping")
        return
    args_dict = {
        "chunk_size": int(args.chunk_size),
        "chunk_stride": int(args.chunk_stride),
        "max_chunks": int(args.max_chunks),
        "num_labels": int(args.num_labels),
    }
    tasks = ((row, args_dict) for row in iter_jsonl(input_path, args.debug_num_contracts))
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            processed = list(
                tqdm(
                    pool.imap(process_item, tasks, chunksize=16),
                    desc=f"typed {split}",
                )
            )
    else:
        processed = [
            process_item(task)
            for task in tqdm(tasks, desc=f"typed {split}")
        ]
    processed = align_with_feature_cache(split, processed, feature_payload)
    ids = [row["id"] for row in processed]
    chunk_mask = torch.stack([row["chunk_mask"].bool() for row in processed])
    multi_labels = torch.stack([row["multi_labels"].float() for row in processed])
    binary_labels = torch.tensor(
        [float(row["binary_label"]) for row in processed],
        dtype=torch.float32,
    )
    contract_evidence = torch.stack(
        [row["typed_contract_evidence"].float() for row in processed]
    )
    chunk_evidence = torch.stack(
        [row["typed_chunk_evidence"].float() for row in processed]
    )
    audit = feature_audit(split, ids, contract_evidence, multi_labels, prediction_groups)
    report = {
        "split": split,
        "samples": len(ids),
        "max_chunks": int(args.max_chunks),
        "chunk_size": int(args.chunk_size),
        "chunk_stride": int(args.chunk_stride),
        "target_label_names": TARGET_LABEL_NAMES,
        "target_label_ids": TARGET_LABEL_IDS,
        "typed_feature_dim": len(TYPED_FEATURE_NAMES),
        "typed_feature_names": TYPED_FEATURE_NAMES,
        "mean_real_chunks": float(chunk_mask.sum(dim=1).float().mean().item())
        if len(ids)
        else 0.0,
        "mean_contract_evidence": contract_evidence.mean(dim=(0, 1)).tolist()
        if len(ids)
        else [],
        "audit": audit,
    }
    payload = {
        "ids": ids,
        "chunk_mask": chunk_mask,
        "binary_labels": binary_labels,
        "multi_labels": multi_labels,
        "typed_contract_evidence": contract_evidence,
        "typed_chunk_evidence": chunk_evidence,
        "typed_feature_names": TYPED_FEATURE_NAMES,
        "target_label_names": TARGET_LABEL_NAMES,
        "target_label_ids": TARGET_LABEL_IDS,
        "metadata": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    report_json = output_path.with_suffix(".audit.json")
    report_txt = output_path.with_suffix(".audit.txt")
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_audit_text(report_txt, audit)
    print(f"[OK] wrote {display_path(output_path)}")
    print(f"[OK] wrote {display_path(report_txt)}")


def write_summary(output_dir, splits):
    lines = [
        "DIVE typed mechanism evidence cache",
        "",
        f"target_labels: {TARGET_LABEL_NAMES}",
        f"feature_dim: {len(TYPED_FEATURE_NAMES)}",
        "",
        "features:",
    ]
    lines.extend(f"- {name}" for name in TYPED_FEATURE_NAMES)
    lines.append("")
    lines.append("splits:")
    summary = {
        "target_label_names": TARGET_LABEL_NAMES,
        "target_label_ids": TARGET_LABEL_IDS,
        "typed_feature_names": TYPED_FEATURE_NAMES,
        "splits": {},
    }
    for split in splits:
        audit_path = output_dir / f"{split}.audit.json"
        if not audit_path.exists():
            continue
        audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
        summary["splits"][split] = {
            "samples": audit_report["samples"],
            "mean_real_chunks": audit_report["mean_real_chunks"],
        }
        lines.append(
            f"- {split}: samples={audit_report['samples']} "
            f"mean_real_chunks={audit_report['mean_real_chunks']:.2f}"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {display_path(output_dir / 'summary.txt')}")


def main():
    args = parse_args()
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)
    feature_dir = resolve(args.feature_dir)
    baseline_dir = resolve(args.baseline_prediction_dir)
    paths = split_paths(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_groups = {}
    if baseline_dir is not None:
        prediction_groups = load_prediction_groups(
            baseline_dir / "test_predictions_per_label.jsonl"
        )
        if prediction_groups:
            print(f"[OK] loaded baseline test prediction groups: {len(prediction_groups)}")
        else:
            print("[WARN] no baseline test predictions found for confusion audit")

    for split in args.splits:
        if split not in paths:
            raise ValueError(f"Unknown split {split}; expected one of {sorted(paths)}")
        split_predictions = prediction_groups if split == "test" else {}
        build_split(
            split,
            paths[split],
            output_dir / f"{split}.pt",
            load_feature_payload(feature_dir, split),
            split_predictions,
            args,
        )
    write_summary(output_dir, args.splits)


if __name__ == "__main__":
    main()
