"""Audit opcode-native CSDG coverage without reading test data."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evm_control_stack_graph import EDGE_TYPES, build_evm_graph  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def iter_jsonl(path):
    with resolve(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                yield line_number, json.loads(line)


def percentile(values, value):
    return float(np.percentile(values, value)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--output", default="data/reports/main6_opcode_csdg/graph_audit.json")
    args = parser.parse_args()
    payload = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    config = {**payload.get("common", {}), **payload.get("graph_cache", {})}
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    report = {
        "route": config["route_name"],
        "seed": 42,
        "splits_read": ["train", "valid"],
        "test_read": False,
        "edge_types": EDGE_TYPES,
        "splits": {},
    }
    for split in ("train", "valid"):
        block_counts = []
        edge_counts = []
        instruction_counts = []
        direct = 0
        unresolved = 0
        stack_edges = 0
        storage_edges = 0
        token_coverage = []
        edge_counter = Counter()
        samples = 0
        for _, item in tqdm(iter_jsonl(config[f"{split}_path"]), desc=f"audit:{split}"):
            graph = build_evm_graph(
                item.get("opcode", ""),
                tokenizer=tokenizer,
                include_storage_edges=bool(config.get("include_storage_edges", False)),
                max_producers=int(config.get("max_stack_producers", 4)),
            )
            samples += 1
            stats = graph["report"]
            block_counts.append(stats["basic_block_count"])
            edge_counts.append(stats["edge_count"])
            instruction_counts.append(stats["instruction_count"])
            direct += stats["direct_jump_count"]
            unresolved += stats["unresolved_jump_count"]
            stack_edges += stats["stack_edge_count"]
            storage_edges += stats["storage_edge_count"]
            token_coverage.append(stats["token_coverage"])
            for edge in graph["edges"]:
                edge_counter[str(edge["type"])] += 1
        report["splits"][split] = {
            "samples": samples,
            "basic_blocks_mean": float(np.mean(block_counts)) if block_counts else 0.0,
            "basic_blocks_p50": percentile(block_counts, 50),
            "basic_blocks_p90": percentile(block_counts, 90),
            "basic_blocks_p95": percentile(block_counts, 95),
            "basic_blocks_max": int(max(block_counts)) if block_counts else 0,
            "edges_mean": float(np.mean(edge_counts)) if edge_counts else 0.0,
            "instructions_mean": float(np.mean(instruction_counts)) if instruction_counts else 0.0,
            "direct_jump_resolution_rate": direct / max(1, direct + unresolved),
            "direct_jump_count": direct,
            "unresolved_jump_count": unresolved,
            "stack_edge_count": stack_edges,
            "storage_edge_count": storage_edges,
            "mean_token_coverage": float(np.mean(token_coverage)) if token_coverage else 0.0,
            "edge_type_counts": dict(edge_counter),
            "batch_budget_over_256": int(sum(value > 256 for value in block_counts)),
        }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()

