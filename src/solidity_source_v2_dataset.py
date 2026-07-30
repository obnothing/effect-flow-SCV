"""Ragged, coverage-preserving Solidity source-window cache for Source-Main6 v2.

This is deliberately a sequence baseline.  Contract relation edges are not
silently approximated here; they will be introduced by the later graph route.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from solidity_contract_graph_v2 import extract_audit_units, normalize_source_ast, source_file_text


SCHEMA = "solidity_source_windows_v2"


def _windows(token_ids: list[int], width: int, stride: int):
    if not token_ids:
        return [[]]
    if len(token_ids) <= width:
        return [token_ids]
    starts = list(range(0, len(token_ids) - width + 1, stride))
    final = len(token_ids) - width
    if starts[-1] != final:
        starts.append(final)
    return [token_ids[start:start + width] for start in starts]


def _select_windows(windows, mandatory_count: int, budget: int):
    """Keep every mandatory window; uniformly cover ordinary windows in budget."""
    mandatory, ordinary = windows[:mandatory_count], windows[mandatory_count:]
    remaining = max(0, budget - len(mandatory))
    if len(ordinary) <= remaining:
        return mandatory + ordinary
    if remaining == 0:
        return mandatory
    indices = sorted({round(index * (len(ordinary) - 1) / max(1, remaining - 1)) for index in range(remaining)})
    return mandatory + [ordinary[index] for index in indices]


def encode_record(record, tokenizer, config, root: Path):
    source = normalize_source_ast(source_file_text(root / record["source_path"]))
    units = extract_audit_units(source)
    width, stride = int(config["max_code_tokens"]), int(config["window_stride"])
    mandatory, ordinary = [], []
    for unit in units:
        # The complete unit may exceed the model limit; it is deliberately
        # windowed below, so suppress the tokenizer's pre-window warning.
        ids = tokenizer(unit.source, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        unit_windows = _windows(ids, width, stride)
        if unit.mandatory:
            mandatory.extend(unit_windows)
        else:
            ordinary.extend(unit_windows)
    encoded = mandatory + ordinary
    mandatory_count = len(mandatory)
    selected = _select_windows(encoded, mandatory_count, int(config["unit_budget"]))
    max_length = int(config["max_length"])
    padded = torch.full((len(selected), max_length), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(selected), max_length), dtype=torch.bool)
    for index, window in enumerate(selected):
        sequence = tokenizer.build_inputs_with_special_tokens(window)
        if len(sequence) > max_length:
            raise ValueError("Source window exceeds configured maximum sequence length")
        padded[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[index, :len(sequence)] = True
    return {
        "id": record["original_id"],
        "input_ids": padded,
        "token_mask": mask,
        "multi_labels": torch.tensor(record["multi_labels"], dtype=torch.float32),
        "binary_label": torch.tensor(record["binary_label"], dtype=torch.float32),
        "window_count_before_budget": len(encoded),
        "mandatory_window_count": mandatory_count,
    }


def build_cache(records_path: Path, output_path: Path, tokenizer, config, root: Path):
    rows = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(encode_record(json.loads(line), tokenizer, config, root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "records": rows, "ids": [row["id"] for row in rows]}, output_path)
    return len(rows)


class SoliditySourceV2Dataset(torch.utils.data.Dataset):
    def __init__(self, cache_path: str | Path):
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported Source-Main6 v2 cache: {payload.get('schema')}")
        self.rows = payload["records"]
        if not self.rows:
            raise ValueError("Source-Main6 v2 cache is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_source_v2(batch):
    units = max(row["input_ids"].shape[0] for row in batch)
    length = batch[0]["input_ids"].shape[1]
    input_ids = torch.zeros((len(batch), units, length), dtype=torch.long)
    token_mask = torch.zeros((len(batch), units, length), dtype=torch.bool)
    unit_mask = torch.zeros((len(batch), units), dtype=torch.bool)
    for index, row in enumerate(batch):
        count = row["input_ids"].shape[0]
        input_ids[index, :count] = row["input_ids"]
        token_mask[index, :count] = row["token_mask"]
        unit_mask[index, :count] = True
    return {
        "ids": [row["id"] for row in batch], "input_ids": input_ids,
        "token_mask": token_mask, "graph_mask": token_mask,
        "unit_mask": unit_mask,
        "multi_labels": torch.stack([row["multi_labels"] for row in batch]),
        "binary_label": torch.stack([row["binary_label"] for row in batch]),
    }


def coverage_report(cache_path: Path):
    dataset = SoliditySourceV2Dataset(cache_path)
    rows = dataset.rows
    counts = [row["window_count_before_budget"] for row in rows]
    kept = [row["input_ids"].shape[0] for row in rows]
    return {
        "schema": SCHEMA, "samples": len(rows), "max_windows_before_budget": max(counts),
        "max_windows_kept": max(kept), "mandatory_windows_preserved": all(
            row["input_ids"].shape[0] >= row["mandatory_window_count"] for row in rows
        ),
        "within_frozen_budget": sum(count <= 168 for count in counts) / len(counts),
    }
