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

GRAPH_FEATURE_NAMES = [
    "source_to_sink_path",
    "guard_condition_path",
    "call_state_order_path",
    "environment_flow_path",
    "selector_text_signal",
    "loop_sensitive_path",
    "state_rw_signal",
    "external_call_signal",
    "protective_signal",
    "hash_arithmetic_signal",
    "revert_failure_signal",
    "combined_path_score",
]

CALL_OPS = {"CALL", "DELEGATECALL", "STATICCALL", "CALLCODE"}
ENV_OPS = {"TIMESTAMP", "NUMBER", "DIFFICULTY", "PREVRANDAO", "BLOCKHASH", "COINBASE"}
BRANCH_OPS = {"JUMPI", "LT", "GT", "SLT", "SGT", "EQ", "ISZERO"}
HASH_ARITH_OPS = {
    "SHA3",
    "KECCAK256",
    "MOD",
    "SMOD",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "SDIV",
    "AND",
    "OR",
    "XOR",
}
STATE_OPS = {"SLOAD", "SSTORE"}
SENSITIVE_SINK_OPS = {"SSTORE", "CALL", "DELEGATECALL", "CALLCODE", "SELFDESTRUCT"}
LOG_OPS = {"LOG0", "LOG1", "LOG2", "LOG3", "LOG4"}

ERC20_SELECTORS = {
    "0xa9059cbb",
    "0x23b872dd",
    "0x095ea7b3",
    "0xdd62ed3e",
    "0x70a08231",
    "0x18160ddd",
}
OWNER_SELECTORS = {
    "0x8da5cb5b",
    "0xf2fde38b",
    "0x715018a6",
}
SWAP_ORDER_SELECTORS = {
    "0x38ed1739",
    "0x8803dbee",
    "0x7ff36ab5",
    "0x4a25d94a",
    "0x18cbafe5",
    "0xfb3bdb41",
    "0x414bf389",
    "0x5c11d795",
    "0x3593564c",
    "0xd0e30db0",
    "0x2e1a7d4d",
}

TEXT_KEYWORDS = {
    "front": {
        "swap",
        "trade",
        "buy",
        "sell",
        "bid",
        "order",
        "auction",
        "price",
        "reserve",
        "amount",
        "balance",
    },
    "front_protective": {
        "deadline",
        "slippage",
        "amountoutmin",
        "amountinmax",
        "minamount",
        "commit",
        "reveal",
    },
    "access": {"owner", "admin", "role", "onlyowner", "authority", "permission"},
    "reentrancy": {"withdraw", "claim", "deposit", "send", "transfer"},
    "reentrancy_protective": {"reentrant", "nonreentrant", "locked", "mutex"},
    "bad_randomness": {"random", "rand", "seed", "lottery", "winner", "dice"},
    "time": {"time", "timestamp", "deadline", "expire", "expiry", "lock"},
    "dos": {"batch", "airdrop", "distribute", "iterate", "loop", "all"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build contract-level graph-derived DIVE evidence cache."
    )
    parser.add_argument("--data_dir", default="data/processed/DIVE_random_split")
    parser.add_argument(
        "--feature_dir",
        default=None,
        help="Optional EVM-BERT feature cache dir for strict id/chunk alignment.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/features/graph_evidence/dive_random_stride256_max64",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=256)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--num_labels", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug_num_contracts", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(path):
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
        raw_op = tokens[token_index]
        op = raw_op.upper()
        start = token_index
        token_index += 1
        arg = None
        width = 1
        match = re.fullmatch(r"PUSH(\d+)", op)
        if op == "PUSH0" and token_index < len(tokens) and tokens[token_index].lower() == "0x":
            arg = tokens[token_index]
            token_index += 1
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
            chunks_for_token(
                inst["token_start"],
                chunk_size,
                chunk_stride,
                max_chunks,
            )
        )
    return chunks


def real_chunk_count(token_count, chunk_stride, max_chunks):
    if token_count <= 0:
        return 0
    return min(max_chunks, max(1, (max(0, token_count - 1) // chunk_stride) + 1))


class ContractGraphEvidence:
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
            (self.max_chunks, len(LABEL_NAMES)),
            dtype=torch.float32,
        )
        self.signals = self._base_signals()

    def _extract_selectors(self):
        selectors = []
        for inst in self.instructions:
            if inst["op"] == "PUSH4" and inst["arg"]:
                selector = normalize_selector(inst["arg"])
                if re.fullmatch(r"0x[0-9a-f]{8}", selector):
                    selectors.append(selector)
        return set(selectors)

    def _decode_text(self):
        decoded = [decode_hex_text(token) for token in self.tokens]
        return " ".join(text for text in decoded if text)

    def keyword_score(self, key):
        keywords = TEXT_KEYWORDS[key]
        return bool_score(any(keyword in self.decoded_text for keyword in keywords))

    def positions(self, ops):
        ops = set(ops)
        return [
            (idx, inst)
            for idx, inst in enumerate(self.instructions)
            if inst["op"] in ops
        ]

    def pairs_forward(self, source_ops, target_ops, window):
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

    def pairs_near(self, source_ops, target_ops, window):
        targets = self.positions(target_ops)
        if not targets:
            return []
        pairs = []
        for source_idx, source in self.positions(source_ops):
            for target_idx, target in targets:
                if 0 < abs(target_idx - source_idx) <= window:
                    pairs.append((source, target))
                    break
        return pairs

    def mark_pairs(self, label_name, pairs, value):
        label_id = LABEL_NAMES.index(label_name)
        for left, right in pairs:
            for chunk_id in chunks_for_pair(
                left,
                right,
                self.chunk_size,
                self.chunk_stride,
                self.max_chunks,
            ):
                self.chunk_evidence[chunk_id, label_id] = max(
                    float(self.chunk_evidence[chunk_id, label_id]),
                    float(value),
                )

    def mark_selector_chunks(self, label_name, selector_set, value):
        label_id = LABEL_NAMES.index(label_name)
        for idx, inst in enumerate(self.instructions):
            if inst["op"] == "PUSH4" and inst["arg"]:
                selector = normalize_selector(inst["arg"])
                if selector not in selector_set:
                    continue
                for chunk_id in chunks_for_token(
                    inst["token_start"],
                    self.chunk_size,
                    self.chunk_stride,
                    self.max_chunks,
                ):
                    self.chunk_evidence[chunk_id, label_id] = max(
                        float(self.chunk_evidence[chunk_id, label_id]),
                        float(value),
                    )

    def jump_target(self, instruction_index):
        if instruction_index <= 0:
            return None
        prev = self.instructions[instruction_index - 1]
        if not prev["op"].startswith("PUSH") or prev["arg"] is None:
            return None
        try:
            return int(str(prev["arg"]), 16)
        except ValueError:
            return None

    def loop_ranges(self):
        pc_to_index = {inst["pc"]: idx for idx, inst in enumerate(self.instructions)}
        ranges = []
        for idx, inst in enumerate(self.instructions):
            if inst["op"] not in {"JUMP", "JUMPI"}:
                continue
            target_pc = self.jump_target(idx)
            if target_pc is None or target_pc > inst["pc"]:
                continue
            start_idx = pc_to_index.get(target_pc)
            if start_idx is None:
                candidates = [
                    pos for pc, pos in pc_to_index.items() if target_pc <= pc <= inst["pc"]
                ]
                start_idx = min(candidates) if candidates else None
            if start_idx is not None and start_idx < idx:
                ranges.append((start_idx, idx))
        return ranges

    def loop_sensitive_signal(self):
        ranges = self.loop_ranges()
        if not ranges:
            return 0.0, 0.0, []
        sensitive_ranges = []
        call_or_revert_ranges = []
        for start, end in ranges:
            ops = set(self.ops[start : end + 1])
            if ops & (CALL_OPS | {"SSTORE", "SLOAD"}):
                sensitive_ranges.append((start, end))
            if ops & (CALL_OPS | {"REVERT", "INVALID"}):
                call_or_revert_ranges.append((start, end))
        return (
            clip01(len(sensitive_ranges), 2),
            clip01(len(call_or_revert_ranges), 2),
            sensitive_ranges,
        )

    def mark_loop_chunks(self, label_name, ranges, value):
        label_id = LABEL_NAMES.index(label_name)
        for start, end in ranges:
            for inst in (self.instructions[start], self.instructions[end]):
                for chunk_id in chunks_for_token(
                    inst["token_start"],
                    self.chunk_size,
                    self.chunk_stride,
                    self.max_chunks,
                ):
                    self.chunk_evidence[chunk_id, label_id] = max(
                        float(self.chunk_evidence[chunk_id, label_id]),
                        float(value),
                    )

    def _base_signals(self):
        calldata_state_call_pairs = self.pairs_forward(
            {"CALLDATALOAD", "CALLDATACOPY"},
            STATE_OPS | CALL_OPS | LOG_OPS,
            160,
        )
        calldata_branch_pairs = self.pairs_forward(
            {"CALLDATALOAD", "CALLDATACOPY"},
            BRANCH_OPS,
            96,
        )
        guard_pairs = self.pairs_forward(
            {"CALLER", "ORIGIN"},
            BRANCH_OPS,
            96,
        )
        guard_sink_pairs = self.pairs_forward(
            {"CALLER", "ORIGIN"},
            SENSITIVE_SINK_OPS | BRANCH_OPS,
            160,
        )
        call_before_state_pairs = self.pairs_forward(CALL_OPS, {"SSTORE"}, 200)
        state_before_call_pairs = self.pairs_forward({"SSTORE"}, CALL_OPS, 200)
        env_branch_pairs = self.pairs_forward(ENV_OPS, BRANCH_OPS, 96)
        env_hash_pairs = self.pairs_forward(ENV_OPS, HASH_ARITH_OPS, 96)
        env_state_call_pairs = self.pairs_forward(ENV_OPS, STATE_OPS | CALL_OPS | LOG_OPS, 160)
        call_revert_pairs = self.pairs_forward(CALL_OPS, {"REVERT", "INVALID"}, 96)
        call_check_pairs = self.pairs_forward(CALL_OPS, {"ISZERO", "JUMPI"}, 64)
        loop_sensitive, loop_call_revert, loop_ranges = self.loop_sensitive_signal()

        front_selector = bool_score(
            bool(self.selectors & (ERC20_SELECTORS | SWAP_ORDER_SELECTORS))
        )
        access_selector = bool_score(bool(self.selectors & OWNER_SELECTORS))
        random_source = bool_score(
            self.counts["BLOCKHASH"] > 0
            or self.counts["NUMBER"] > 0
            or self.counts["DIFFICULTY"] > 0
            or self.counts["PREVRANDAO"] > 0
        )
        state_rw = clip01(self.counts["SLOAD"] + self.counts["SSTORE"], 80)
        external_call = clip01(sum(self.counts[op] for op in CALL_OPS), 8)
        hash_arith = clip01(sum(self.counts[op] for op in HASH_ARITH_OPS), 80)
        revert_failure = clip01(self.counts["REVERT"] + self.counts["INVALID"], 20)

        return {
            "calldata_state_call": clip01(len(calldata_state_call_pairs), 4),
            "calldata_branch": clip01(len(calldata_branch_pairs), 4),
            "guard": clip01(len(guard_pairs), 3),
            "guard_sensitive": clip01(len(guard_sink_pairs), 3),
            "call_before_state": clip01(len(call_before_state_pairs), 3),
            "state_before_call": clip01(len(state_before_call_pairs), 3),
            "env_branch": clip01(len(env_branch_pairs), 3),
            "env_hash": clip01(len(env_hash_pairs), 3),
            "env_state_call": clip01(len(env_state_call_pairs), 3),
            "call_revert": clip01(len(call_revert_pairs), 2),
            "call_checked": clip01(len(call_check_pairs), 3),
            "loop_sensitive": loop_sensitive,
            "loop_call_revert": loop_call_revert,
            "front_selector_text": max(front_selector, self.keyword_score("front")),
            "front_protective": self.keyword_score("front_protective"),
            "access_selector_text": max(access_selector, self.keyword_score("access")),
            "reentrancy_text": self.keyword_score("reentrancy"),
            "reentrancy_protective": self.keyword_score("reentrancy_protective"),
            "bad_randomness_text": self.keyword_score("bad_randomness"),
            "time_text": self.keyword_score("time"),
            "dos_text": self.keyword_score("dos"),
            "random_source": random_source,
            "timestamp_source": bool_score(self.counts["TIMESTAMP"] > 0),
            "state_rw": state_rw,
            "external_call": external_call,
            "hash_arith": hash_arith,
            "revert_failure": revert_failure,
            "erc20_selector": bool_score(bool(self.selectors & ERC20_SELECTORS)),
            "unchecked_call": max(0.0, external_call - clip01(len(call_check_pairs), 3)),
            "_pairs": {
                "calldata_state_call": calldata_state_call_pairs,
                "guard_sensitive": guard_sink_pairs,
                "call_before_state": call_before_state_pairs,
                "state_before_call": state_before_call_pairs,
                "env_branch": env_branch_pairs,
                "env_hash": env_hash_pairs,
                "env_state_call": env_state_call_pairs,
                "call_revert": call_revert_pairs,
                "loop_ranges": loop_ranges,
            },
        }

    def evidence_rows(self):
        s = self.signals
        rows = {}
        rows["Reentrancy"] = [
            s["external_call"],
            s["guard"],
            s["call_before_state"],
            0.0,
            s["reentrancy_text"],
            s["loop_call_revert"],
            s["state_rw"],
            s["external_call"],
            s["reentrancy_protective"],
            0.0,
            s["call_revert"],
            max(s["call_before_state"], 0.5 * s["call_revert"]),
        ]
        rows["Access Control"] = [
            s["guard_sensitive"],
            s["guard"],
            0.0,
            0.0,
            s["access_selector_text"],
            0.0,
            s["state_rw"],
            s["external_call"],
            0.0,
            0.0,
            s["revert_failure"],
            max(s["guard_sensitive"], s["access_selector_text"]),
        ]
        rows["Arithmetic"] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            s["state_rw"],
            0.0,
            0.0,
            s["hash_arith"],
            s["revert_failure"],
            max(s["hash_arith"], s["revert_failure"]),
        ]
        rows["Unchecked Return Values"] = [
            s["external_call"],
            0.0,
            s["unchecked_call"],
            0.0,
            0.0,
            s["loop_call_revert"],
            0.0,
            s["external_call"],
            s["call_checked"],
            0.0,
            s["call_revert"],
            s["unchecked_call"],
        ]
        rows["DoS"] = [
            s["calldata_state_call"],
            s["calldata_branch"],
            max(s["call_before_state"], s["state_before_call"]),
            0.0,
            s["dos_text"],
            s["loop_sensitive"],
            s["state_rw"],
            s["external_call"],
            0.0,
            0.0,
            max(s["revert_failure"], s["loop_call_revert"]),
            max(s["loop_sensitive"], s["call_revert"], 0.5 * s["external_call"]),
        ]
        rows["Bad Randomness"] = [
            max(s["env_hash"], s["env_state_call"]),
            s["env_branch"],
            0.0,
            max(s["env_hash"], s["random_source"]),
            s["bad_randomness_text"],
            0.0,
            s["state_rw"],
            s["external_call"],
            s["front_protective"],
            s["hash_arith"],
            0.0,
            max(s["env_hash"], s["bad_randomness_text"], s["random_source"]),
        ]
        rows["Front Running"] = [
            s["calldata_state_call"],
            s["calldata_branch"],
            max(s["call_before_state"], s["state_before_call"]),
            0.0,
            s["front_selector_text"],
            0.0,
            s["state_rw"],
            s["external_call"],
            s["front_protective"],
            s["hash_arith"] * s["calldata_state_call"],
            0.0,
            max(s["calldata_state_call"], s["front_selector_text"]),
        ]
        rows["Time manipulation"] = [
            max(s["env_branch"], s["env_state_call"]),
            s["env_branch"],
            0.0,
            max(s["env_branch"], s["timestamp_source"]),
            s["time_text"],
            0.0,
            s["state_rw"],
            s["external_call"],
            0.0,
            s["env_hash"],
            0.0,
            max(s["env_branch"], s["env_state_call"], s["time_text"]),
        ]
        return torch.tensor(
            [rows[name] for name in LABEL_NAMES],
            dtype=torch.float32,
        )

    def build(self):
        s = self.signals
        self.mark_pairs("Front Running", s["_pairs"]["calldata_state_call"], s["calldata_state_call"])
        self.mark_selector_chunks(
            "Front Running",
            ERC20_SELECTORS | SWAP_ORDER_SELECTORS,
            max(0.25, s["front_selector_text"]),
        )
        self.mark_pairs("Access Control", s["_pairs"]["guard_sensitive"], s["guard_sensitive"])
        self.mark_selector_chunks("Access Control", OWNER_SELECTORS, max(0.25, s["access_selector_text"]))
        self.mark_pairs("Reentrancy", s["_pairs"]["call_before_state"], s["call_before_state"])
        self.mark_pairs("Unchecked Return Values", s["_pairs"]["call_revert"], s["unchecked_call"])
        self.mark_pairs("Bad Randomness", s["_pairs"]["env_hash"], max(s["env_hash"], s["random_source"]))
        self.mark_pairs("Bad Randomness", s["_pairs"]["env_state_call"], s["env_state_call"])
        self.mark_pairs("Time manipulation", s["_pairs"]["env_branch"], s["env_branch"])
        self.mark_pairs("Time manipulation", s["_pairs"]["env_state_call"], s["env_state_call"])
        self.mark_pairs("DoS", s["_pairs"]["call_revert"], s["call_revert"])
        self.mark_loop_chunks("DoS", s["_pairs"]["loop_ranges"], s["loop_sensitive"])
        self.mark_pairs("Arithmetic", s["_pairs"]["env_hash"], 0.25 * s["hash_arith"])
        return self.evidence_rows(), self.chunk_evidence


def process_item(task):
    item, chunk_size, chunk_stride, max_chunks, num_labels = task
    analyzer = ContractGraphEvidence(
        item.get("opcode", ""),
        chunk_size,
        chunk_stride,
        max_chunks,
    )
    contract_evidence, chunk_evidence = analyzer.build()
    token_count = len(analyzer.tokens)
    chunk_mask = torch.zeros(max_chunks, dtype=torch.bool)
    count = real_chunk_count(token_count, chunk_stride, max_chunks)
    if count > 0:
        chunk_mask[:count] = True
    multi_labels = torch.tensor(item.get("multi_labels", [0] * num_labels), dtype=torch.float32)
    if multi_labels.numel() != num_labels:
        raise ValueError(f"Invalid multi_labels width for id={item.get('id')}")
    return {
        "id": str(item.get("id")),
        "chunk_mask": chunk_mask,
        "binary_label": float(item.get("binary_label", 0.0)),
        "multi_labels": multi_labels,
        "graph_contract_evidence": contract_evidence,
        "graph_chunk_evidence": chunk_evidence,
        "token_count": token_count,
        "real_chunks": int(chunk_mask.sum().item()),
    }


def load_feature_payload(feature_dir, split):
    if feature_dir is None:
        return None
    path = feature_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature cache for graph alignment: {path}")
    return torch.load(path, map_location="cpu")


def align_with_feature_cache(split, processed, feature_payload):
    if feature_payload is None:
        return processed
    feature_ids = [str(value) for value in feature_payload["ids"]]
    processed_ids = [str(row["id"]) for row in processed]
    if len(feature_ids) != len(processed_ids):
        raise ValueError(
            f"{split}: graph rows {len(processed_ids)} do not match feature rows {len(feature_ids)}"
        )
    mismatches = [
        idx for idx, (left, right) in enumerate(zip(processed_ids, feature_ids))
        if left != right
    ]
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"{split}: graph ids are not aligned at {first}: "
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
        row["graph_chunk_evidence"] = row["graph_chunk_evidence"].masked_fill(
            ~feature_mask[idx].unsqueeze(-1),
            0.0,
        )
    return processed


def feature_audit(split, contract_evidence, multi_labels):
    audit = {
        "split": split,
        "feature_names": GRAPH_FEATURE_NAMES,
        "labels": [],
    }
    labels = multi_labels.float()
    for label_id, label_name in enumerate(LABEL_NAMES):
        targets = labels[:, label_id] > 0.5
        support = int(targets.sum().item())
        negatives = int((~targets).sum().item())
        rows = []
        for feature_id, feature_name in enumerate(GRAPH_FEATURE_NAMES):
            values = contract_evidence[:, label_id, feature_id]
            positive_mean = float(values[targets].mean().item()) if support else 0.0
            negative_mean = float(values[~targets].mean().item()) if negatives else 0.0
            rows.append(
                {
                    "feature_name": feature_name,
                    "positive_mean": positive_mean,
                    "negative_mean": negative_mean,
                    "delta": positive_mean - negative_mean,
                    "ratio": (positive_mean + 1e-6) / (negative_mean + 1e-6),
                }
            )
        audit["labels"].append(
            {
                "label_name": label_name,
                "support": support,
                "negative_count": negatives,
                "features": rows,
            }
        )
    return audit


def write_audit_text(path, audit):
    lines = [f"Graph evidence audit: {audit['split']}", ""]
    for label in audit["labels"]:
        lines.append(
            f"{label['label_name']} support={label['support']} "
            f"negatives={label['negative_count']}"
        )
        rows = sorted(
            label["features"],
            key=lambda row: abs(float(row["delta"])),
            reverse=True,
        )
        for row in rows[:8]:
            lines.append(
                f"  {row['feature_name']}: pos={row['positive_mean']:.4f} "
                f"neg={row['negative_mean']:.4f} delta={row['delta']:.4f} "
                f"ratio={row['ratio']:.2f}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_split(split, input_path, output_path, feature_payload, args):
    if output_path.exists() and not args.force:
        print(f"[OK] {display_path(output_path)} exists, skipping")
        return
    tasks = (
        (row, args.chunk_size, args.chunk_stride, args.max_chunks, args.num_labels)
        for row in iter_jsonl(input_path, args.debug_num_contracts)
    )
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            processed = list(
                tqdm(
                    pool.imap(process_item, tasks, chunksize=16),
                    desc=f"graph {split}",
                )
            )
    else:
        processed = [
            process_item(task)
            for task in tqdm(tasks, desc=f"graph {split}")
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
        [row["graph_contract_evidence"].float() for row in processed]
    )
    chunk_evidence = torch.stack(
        [row["graph_chunk_evidence"].float() for row in processed]
    )
    audit = feature_audit(split, contract_evidence, multi_labels)
    report = {
        "split": split,
        "samples": len(ids),
        "max_chunks": int(args.max_chunks),
        "chunk_size": int(args.chunk_size),
        "chunk_stride": int(args.chunk_stride),
        "num_labels": int(args.num_labels),
        "graph_evidence_dim": len(GRAPH_FEATURE_NAMES),
        "feature_names": GRAPH_FEATURE_NAMES,
        "label_names": LABEL_NAMES,
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
        "graph_contract_evidence": contract_evidence,
        "graph_chunk_evidence": chunk_evidence,
        "graph_feature_names": GRAPH_FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "report": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    audit_json = output_path.with_suffix(".audit.json")
    audit_txt = output_path.with_suffix(".audit.txt")
    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_audit_text(audit_txt, audit)
    print(f"[OK] wrote {display_path(output_path)}")
    print(f"[OK] wrote {display_path(audit_txt)}")


def main():
    args = parse_args()
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)
    feature_dir = resolve(args.feature_dir) if args.feature_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output_dir": display_path(output_dir),
        "data_dir": display_path(data_dir),
        "feature_dir": display_path(feature_dir) if feature_dir else None,
        "chunk_size": int(args.chunk_size),
        "chunk_stride": int(args.chunk_stride),
        "max_chunks": int(args.max_chunks),
        "num_labels": int(args.num_labels),
        "graph_evidence_dim": len(GRAPH_FEATURE_NAMES),
        "feature_names": GRAPH_FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "splits": {},
    }
    for split, input_path in split_paths(data_dir).items():
        feature_payload = load_feature_payload(feature_dir, split)
        output_path = output_dir / f"{split}.pt"
        build_split(split, input_path, output_path, feature_payload, args)
        manifest["splits"][split] = {
            "input": display_path(input_path),
            "output": display_path(output_path),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {display_path(manifest_path)}")


if __name__ == "__main__":
    main()
