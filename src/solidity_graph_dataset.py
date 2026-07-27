"""GraphCodeBERT-ready Solidity source-unit cache and dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from solidity_graph_utils import extract_source_units, select_units, source_file_text


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _find_subsequence(values, pattern):
    if not pattern:
        return []
    for start in range(0, len(values) - len(pattern) + 1):
        if values[start:start + len(pattern)] == pattern:
            return list(range(start, start + len(pattern)))
    return []


def encode_unit(unit, tokenizer, max_code_tokens: int, max_dfg_nodes: int, max_length: int):
    code_tokens = tokenizer.tokenize(unit.source)[:max_code_tokens]
    node_tokens = []
    node_code_positions = []
    source_node_ids = []
    for node_id, (name, _) in enumerate(unit.dfg_nodes):
        pieces = tokenizer.tokenize(name)
        if not pieces or len(node_tokens) >= max_dfg_nodes:
            continue
        positions = _find_subsequence(code_tokens, pieces)
        if not positions:
            continue
        source_node_ids.append(node_id)
        node_code_positions.append(positions[0])
        node_tokens.append(pieces[0])
    special = 3
    available_nodes = max(0, max_length - len(code_tokens) - special)
    node_tokens = node_tokens[:available_nodes]
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token] + node_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    real = len(input_ids)
    code_end = min(1 + len(code_tokens) + 1, real)
    node_to_code = torch.full((max_dfg_nodes,), -1, dtype=torch.int16)
    node_to_code[:len(node_code_positions)] = torch.tensor(node_code_positions, dtype=torch.int16)
    remap = {old: new for new, old in enumerate(source_node_ids[:len(node_code_positions)])}
    edge_src = torch.full((max_dfg_nodes,), -1, dtype=torch.int16)
    edge_dst = torch.full((max_dfg_nodes,), -1, dtype=torch.int16)
    edge_index = 0
    for left, right in unit.dfg_edges:
        if left not in remap or right not in remap:
            continue
        if edge_index >= max_dfg_nodes:
            break
        edge_src[edge_index], edge_dst[edge_index] = remap[left], remap[right]
        edge_index += 1
    input_ids += [tokenizer.pad_token_id] * (max_length - real)
    token_mask = [1] * real + [0] * (max_length - real)
    return (torch.tensor(input_ids), torch.tensor(token_mask, dtype=torch.bool),
            torch.tensor(code_end, dtype=torch.int16), node_to_code, edge_src, edge_dst)


def encode_record(record, tokenizer, config, root: Path):
    source_path = root / record["source_path"]
    units = select_units(extract_source_units(source_file_text(source_path)), int(config["max_units"]))
    max_units, max_length = int(config["max_units"]), int(config["max_length"])
    input_ids = torch.full((max_units, max_length), tokenizer.pad_token_id, dtype=torch.long)
    token_mask = torch.zeros((max_units, max_length), dtype=torch.bool)
    code_ends = torch.zeros(max_units, dtype=torch.int16)
    node_to_code = torch.full((max_units, int(config["max_dfg_nodes"])), -1, dtype=torch.int16)
    edge_src = torch.full_like(node_to_code, -1)
    edge_dst = torch.full_like(node_to_code, -1)
    unit_mask = torch.zeros(max_units, dtype=torch.bool)
    for index, unit in enumerate(units):
        ids, mask, code_end, unit_nodes, unit_src, unit_dst = encode_unit(
            unit, tokenizer, int(config["max_code_tokens"]), int(config["max_dfg_nodes"]), max_length
        )
        input_ids[index], token_mask[index], unit_mask[index] = ids, mask, True
        code_ends[index], node_to_code[index], edge_src[index], edge_dst[index] = code_end, unit_nodes, unit_src, unit_dst
    return {
        "id": record["original_id"], "input_ids": input_ids, "token_mask": token_mask,
        "code_ends": code_ends, "node_to_code": node_to_code, "edge_src": edge_src,
        "edge_dst": edge_dst, "unit_mask": unit_mask,
        "multi_labels": torch.tensor(record["multi_labels"], dtype=torch.float32),
        "binary_label": torch.tensor(record["binary_label"], dtype=torch.float32),
    }


def build_cache(records_path: Path, output_path: Path, tokenizer, config, root: Path):
    records = load_records(records_path)
    rows = [encode_record(record, tokenizer, config, root) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": rows, "ids": [row["id"] for row in rows], "schema": "solidity_graph_v1"}, output_path)
    return len(rows)


class SolidityGraphDataset(Dataset):
    def __init__(self, cache_path: str | Path):
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("schema") != "solidity_graph_v1":
            raise ValueError(f"Unsupported source graph cache schema: {payload.get('schema')}")
        self.rows = payload["records"]
        if not self.rows:
            raise ValueError("Source graph cache is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_solidity_graph(batch):
    input_ids = torch.stack([row["input_ids"] for row in batch])
    token_mask = torch.stack([row["token_mask"] for row in batch])
    code_ends = torch.stack([row["code_ends"] for row in batch])
    node_to_code = torch.stack([row["node_to_code"] for row in batch])
    edge_src = torch.stack([row["edge_src"] for row in batch])
    edge_dst = torch.stack([row["edge_dst"] for row in batch])
    return {
        "ids": [row["id"] for row in batch],
        "input_ids": input_ids, "token_mask": token_mask,
        "graph_mask": build_graph_attention(token_mask, code_ends, node_to_code, edge_src, edge_dst),
        "unit_mask": torch.stack([row["unit_mask"] for row in batch]),
        "multi_labels": torch.stack([row["multi_labels"] for row in batch]),
        "binary_label": torch.stack([row["binary_label"] for row in batch]),
    }


def build_graph_attention(token_mask, code_ends, node_to_code, edge_src, edge_dst):
    """Restore sparse GraphCodeBERT attention immediately before model execution."""
    batch, units, length = token_mask.shape
    output = torch.zeros((batch, units, length, length), dtype=torch.bool)
    for batch_id in range(batch):
        for unit_id in range(units):
            code_end = int(code_ends[batch_id, unit_id])
            if code_end <= 0:
                continue
            output[batch_id, unit_id, :code_end, :code_end] = True
            nodes = node_to_code[batch_id, unit_id]
            for node_id, code_index in enumerate(nodes.tolist()):
                node_position = code_end + node_id
                if code_index < 0 or node_position >= length or not token_mask[batch_id, unit_id, node_position]:
                    continue
                code_position = 1 + code_index
                output[batch_id, unit_id, node_position, node_position] = True
                output[batch_id, unit_id, node_position, code_position] = True
                output[batch_id, unit_id, code_position, node_position] = True
            for src, dst in zip(edge_src[batch_id, unit_id].tolist(), edge_dst[batch_id, unit_id].tolist()):
                if src < 0 or dst < 0:
                    continue
                src_position, dst_position = code_end + src, code_end + dst
                if src_position < length and dst_position < length:
                    output[batch_id, unit_id, src_position, dst_position] = True
                    output[batch_id, unit_id, dst_position, src_position] = True
    return output
