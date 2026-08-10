"""Audit opcode-native CSDG coverage without reading test data."""

import argparse
import json
import multiprocessing as mp
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


def audit_opcode(task):
    """Build one label-agnostic graph and return only audit statistics."""
    opcode, max_producers, max_worklist_steps, max_instruction_visits = task
    graph = build_evm_graph(
        opcode,
        tokenizer=None,
        include_storage_edges=False,
        max_producers=max_producers,
        max_worklist_steps=max_worklist_steps,
        max_instruction_visits=max_instruction_visits,
    )
    return graph["report"], [edge["type"] for edge in graph["edges"]]


def split_tasks(path, max_producers, max_worklist_steps, max_instruction_visits):
    for _, item in iter_jsonl(path):
        yield item.get("opcode", ""), max_producers, max_worklist_steps, max_instruction_visits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--output", default="data/reports/main6_opcode_csdg/graph_audit.json")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU worker processes; each process audits independent opcode samples.",
    )
    args = parser.parse_args()
    payload = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    config = {**payload.get("common", {}), **payload.get("graph_cache", {})}
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
        worklist_steps = []
        instruction_visits = []
        capped_samples = 0
        edge_counter = Counter()
        samples = 0
        max_producers = int(config.get("max_stack_producers", 4))
        max_worklist_steps = int(config.get("max_stack_worklist_steps", 20000))
        max_instruction_visits = int(config.get("max_stack_instruction_visits", 250000))
        tasks = split_tasks(
            config[f"{split}_path"], max_producers, max_worklist_steps, max_instruction_visits
        )
        worker_count = max(1, args.workers)
        with mp.Pool(processes=worker_count) as pool:
            results = pool.imap_unordered(audit_opcode, tasks, chunksize=1)
            for stats, edge_types in tqdm(results, desc=f"audit:{split}"):
                samples += 1
                block_counts.append(stats["basic_block_count"])
                edge_counts.append(stats["edge_count"])
                instruction_counts.append(stats["instruction_count"])
                direct += stats["direct_jump_count"]
                unresolved += stats["unresolved_jump_count"]
                stack_edges += stats["stack_edge_count"]
                storage_edges += stats["storage_edge_count"]
                token_coverage.append(stats["token_coverage"])
                worklist_steps.append(stats["stack_worklist_steps"])
                instruction_visits.append(stats["stack_instruction_visits"])
                capped_samples += int(stats["stack_analysis_capped"])
                edge_counter.update(str(edge_type) for edge_type in edge_types)
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
            "stack_analysis_capped_samples": capped_samples,
            "stack_analysis_capped_rate": capped_samples / max(1, samples),
            "stack_worklist_steps_max": int(max(worklist_steps)) if worklist_steps else 0,
            "stack_instruction_visits_max": int(max(instruction_visits)) if instruction_visits else 0,
            "mean_token_coverage": float(np.mean(token_coverage)) if token_coverage else 0.0,
            "edge_type_counts": dict(edge_counter),
            "batch_budget_over_256": int(sum(value > 256 for value in block_counts)),
        }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] audit workers: {max(1, args.workers)}")
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
