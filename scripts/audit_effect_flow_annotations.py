import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from effect_flow_schema import (  # noqa: E402
    EFFECT_TYPES,
    EFPP_PATTERNS,
    NOISY_PATTERNS,
    pattern_metadata,
)
from effect_flow_utils import (  # noqa: E402
    REPORT_DIR,
    corpus_split_paths,
    iter_jsonl,
    percentile,
    relative,
    resolve,
    write_json,
    write_text,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Stage 16A annotations.")
    parser.add_argument("--dataset", choices=("BJUT", "DIVE"), required=True)
    parser.add_argument("--corpus_dir", default=None)
    parser.add_argument("--report_suffix", default="")
    return parser.parse_args()


def flush_contract(pattern_or, split, contract_pattern_counts, split_contract_pattern_counts):
    if pattern_or is None:
        return
    for index, value in enumerate(pattern_or):
        if value:
            contract_pattern_counts[index] += 1
            split_contract_pattern_counts[split][index] += 1


def audit(dataset, corpus_dir=None):
    paths = corpus_split_paths(dataset)
    if corpus_dir:
        root = resolve(corpus_dir)
        paths = {
            split: root / f"{split}_effect_flow_chunks.jsonl"
            for split in ("train", "valid", "test")
        }
    missing = [relative(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing effect-flow corpus files: {missing}")

    primary_counts = Counter()
    primary_chunk_counts = Counter()
    multihot_counts = Counter()
    multihot_chunk_counts = Counter()
    pattern_chunk_counts = [0] * len(EFPP_PATTERNS)
    pattern_contract_counts = [0] * len(EFPP_PATTERNS)
    split_chunk_counts = Counter()
    split_contract_counts = Counter()
    split_pattern_chunks = {
        split: [0] * len(EFPP_PATTERNS) for split in paths
    }
    split_pattern_contracts = {
        split: [0] * len(EFPP_PATTERNS) for split in paths
    }
    cooccurrence = [
        [0] * len(EFPP_PATTERNS) for _ in range(len(EFPP_PATTERNS))
    ]
    chain3 = Counter()
    chain4 = Counter()
    examples = {name: [] for name in EFPP_PATTERNS}
    chunks_per_contract = []
    coverage_ratios = []
    active_token_total = 0
    total_chunks = 0
    total_contracts = 0

    for split, path in paths.items():
        current_id = None
        current_pattern_or = None
        seen_contract_chunks = 0
        current_coverage = 0.0
        for _, item in tqdm(
            iter_jsonl(path), desc=f"audit:{dataset}:{split}", unit="chunk"
        ):
            contract_id = str(item["id"])
            if current_id is not None and contract_id != current_id:
                flush_contract(
                    current_pattern_or,
                    split,
                    pattern_contract_counts,
                    split_pattern_contracts,
                )
                total_contracts += 1
                split_contract_counts[split] += 1
                chunks_per_contract.append(seen_contract_chunks)
                coverage_ratios.append(current_coverage)
                current_pattern_or = None
                seen_contract_chunks = 0
            if current_pattern_or is None:
                current_pattern_or = [0] * len(EFPP_PATTERNS)
                current_id = contract_id
                current_coverage = float(item.get("contract_coverage_ratio", 0.0))
            seen_contract_chunks += 1
            total_chunks += 1
            split_chunk_counts[split] += 1

            ids = item["effect_type_ids"]
            multihot = item["effect_type_multihot"]
            mask = item["etp_loss_mask"]
            chunk_primary = set()
            chunk_multi = set()
            for effect_id, vector, active in zip(ids, multihot, mask):
                if not active:
                    continue
                active_token_total += 1
                primary_counts[effect_id] += 1
                chunk_primary.add(effect_id)
                for effect_index, value in enumerate(vector):
                    if value:
                        multihot_counts[effect_index] += 1
                        chunk_multi.add(effect_index)
            for effect_id in chunk_primary:
                primary_chunk_counts[effect_id] += 1
            for effect_id in chunk_multi:
                multihot_chunk_counts[effect_id] += 1

            patterns = item["efpp_pattern_labels"]
            positive = [index for index, value in enumerate(patterns) if value]
            for index in positive:
                pattern_chunk_counts[index] += 1
                split_pattern_chunks[split][index] += 1
                current_pattern_or[index] = 1
                name = EFPP_PATTERNS[index]
                if len(examples[name]) < 3:
                    match = item.get("efpp_matches", {}).get(name, {})
                    examples[name].append(
                        {
                            "contract_id": contract_id,
                            "split": split,
                            "chunk_index": item["chunk_index"],
                            "matched_rule": match.get("matched_rule", "rule match"),
                            "local_opcode_snippet": match.get("local_opcode_snippet", ""),
                        }
                    )
            for left in positive:
                for right in positive:
                    cooccurrence[left][right] += 1

            event_names = [event["effect_type"] for event in item["effect_events"]]
            chain3.update(tuple(event_names[i : i + 3]) for i in range(len(event_names) - 2))
            chain4.update(tuple(event_names[i : i + 4]) for i in range(len(event_names) - 3))

        if current_id is not None:
            flush_contract(
                current_pattern_or,
                split,
                pattern_contract_counts,
                split_pattern_contracts,
            )
            total_contracts += 1
            split_contract_counts[split] += 1
            chunks_per_contract.append(seen_contract_chunks)
            coverage_ratios.append(current_coverage)

    contracts_with_chunks = total_contracts
    split_contracts_with_chunks = dict(split_contract_counts)
    build_report_path = REPORT_DIR / f"effect_flow_corpus_build_{dataset}.json"
    if build_report_path.exists():
        build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
        total_contracts = int(build_report.get("total_contracts", total_contracts))
        for split in paths:
            split_contract_counts[split] = int(
                build_report.get("splits", {}).get(split, {}).get(
                    "contracts", split_contract_counts[split]
                )
            )

    primary_distribution = []
    multihot_distribution = []
    for index, name in enumerate(EFFECT_TYPES):
        primary_distribution.append(
            {
                "effect_type_id": index,
                "effect_type": name,
                "token_count": primary_counts[index],
                "token_ratio": primary_counts[index] / max(1, active_token_total),
                "chunk_count_containing_effect": primary_chunk_counts[index],
                "chunk_ratio": primary_chunk_counts[index] / max(1, total_chunks),
            }
        )
        multihot_distribution.append(
            {
                "effect_type_id": index,
                "effect_type": name,
                "token_count": multihot_counts[index],
                "token_ratio_per_active_token": multihot_counts[index]
                / max(1, active_token_total),
                "chunk_count_containing_effect": multihot_chunk_counts[index],
                "chunk_ratio": multihot_chunk_counts[index] / max(1, total_chunks),
            }
        )

    pattern_distribution = []
    usable_count = 0
    for index, name in enumerate(EFPP_PATTERNS):
        ratio = pattern_chunk_counts[index] / max(1, total_chunks)
        warnings = []
        if ratio < 0.001:
            warnings.append("too_sparse")
        if ratio > 0.80:
            warnings.append("too_frequent_low_discrimination")
        if name in NOISY_PATTERNS:
            warnings.append("noisy_but_candidate")
        usable = 0.001 <= ratio <= 0.80
        usable_count += int(usable)
        pattern_distribution.append(
            {
                "pattern_id": index,
                "pattern_name": name,
                "positive_chunk_count": pattern_chunk_counts[index],
                "positive_chunk_ratio": ratio,
                "positive_contract_count": pattern_contract_counts[index],
                "positive_contract_ratio": pattern_contract_counts[index]
                / max(1, total_contracts),
                "per_split": {
                    split: {
                        "positive_chunk_count": split_pattern_chunks[split][index],
                        "positive_chunk_ratio": split_pattern_chunks[split][index]
                        / max(1, split_chunk_counts[split]),
                        "positive_contract_count": split_pattern_contracts[split][index],
                        "positive_contract_ratio": split_pattern_contracts[split][index]
                        / max(1, split_contract_counts[split]),
                    }
                    for split in paths
                },
                "warnings": warnings,
                "usable_for_pretraining": usable,
                "examples": examples[name],
            }
        )

    report = {
        "dataset": dataset,
        "annotation_semantics": "weak rule-based signals, not vulnerability labels",
        "total_contracts": total_contracts,
        "contracts_with_chunks": contracts_with_chunks,
        "total_chunks": total_chunks,
        "mean_chunks_per_contract": total_chunks / max(1, total_contracts),
        "max_chunks_per_contract": max(chunks_per_contract, default=0),
        "token_coverage_ratio_mean": statistics.fmean(coverage_ratios)
        if coverage_ratios
        else 0.0,
        "token_coverage_ratio_p50": percentile(coverage_ratios, 0.5),
        "token_coverage_ratio_p90": percentile(coverage_ratios, 0.9),
        "split_contract_counts": dict(split_contract_counts),
        "split_contracts_with_chunks": split_contracts_with_chunks,
        "split_chunk_counts": dict(split_chunk_counts),
        "active_etp_tokens": active_token_total,
        "etp_primary_distribution": primary_distribution,
        "etp_multihot_distribution": multihot_distribution,
        "efpp_pattern_distribution": pattern_distribution,
        "efpp_cooccurrence_matrix": {
            "pattern_names": EFPP_PATTERNS,
            "counts": cooccurrence,
        },
        "top_effect_event_chains_length_3": [
            {"chain": list(chain), "count": count} for chain, count in chain3.most_common(20)
        ],
        "top_effect_event_chains_length_4": [
            {"chain": list(chain), "count": count} for chain, count in chain4.most_common(20)
        ],
        "pattern_schema": pattern_metadata(),
        "usable_for_pretraining": usable_count > 0,
        "usable_pattern_count": usable_count,
        "usable_reason": f"{usable_count}/{len(EFPP_PATTERNS)} patterns fall in the 0.1%-80% chunk prevalence range",
    }
    return report


def render(report):
    lines = [
        f"Effect-flow annotation audit: {report['dataset']}",
        "",
        "annotation_semantics: weak rule-based signals, not vulnerability labels",
        f"total_contracts: {report['total_contracts']}",
        f"total_chunks: {report['total_chunks']}",
        f"mean_chunks_per_contract: {report['mean_chunks_per_contract']:.6f}",
        f"max_chunks_per_contract: {report['max_chunks_per_contract']}",
        f"token_coverage_ratio_mean: {report['token_coverage_ratio_mean']:.6f}",
        f"token_coverage_ratio_p50: {report['token_coverage_ratio_p50']:.6f}",
        f"token_coverage_ratio_p90: {report['token_coverage_ratio_p90']:.6f}",
        f"usable_for_pretraining: {report['usable_for_pretraining']}",
        f"usable_reason: {report['usable_reason']}",
        "",
        "ETP primary effect type distribution:",
        "effect_type | token_count | token_ratio | chunks | chunk_ratio",
    ]
    for row in report["etp_primary_distribution"]:
        lines.append(
            f"{row['effect_type']} | {row['token_count']} | {row['token_ratio']:.6f} | "
            f"{row['chunk_count_containing_effect']} | {row['chunk_ratio']:.6f}"
        )
    lines.extend(
        [
            "",
            "ETP multi-hot effect type distribution:",
            "effect_type | token_count | token_ratio_per_active_token | chunks | chunk_ratio",
        ]
    )
    for row in report["etp_multihot_distribution"]:
        lines.append(
            f"{row['effect_type']} | {row['token_count']} | "
            f"{row['token_ratio_per_active_token']:.6f} | "
            f"{row['chunk_count_containing_effect']} | {row['chunk_ratio']:.6f}"
        )
    lines.extend(
        [
            "",
            "EFPP pattern distribution:",
            "pattern | chunk_count | chunk_ratio | contract_count | contract_ratio | usable | warnings",
        ]
    )
    for row in report["efpp_pattern_distribution"]:
        lines.append(
            f"{row['pattern_name']} | {row['positive_chunk_count']} | "
            f"{row['positive_chunk_ratio']:.6f} | {row['positive_contract_count']} | "
            f"{row['positive_contract_ratio']:.6f} | {row['usable_for_pretraining']} | {row['warnings']}"
        )
    lines.extend(["", "Top effect event chains length 3:"])
    for row in report["top_effect_event_chains_length_3"]:
        lines.append(f"{' -> '.join(row['chain'])}: {row['count']}")
    lines.extend(["", "Top effect event chains length 4:"])
    for row in report["top_effect_event_chains_length_4"]:
        lines.append(f"{' -> '.join(row['chain'])}: {row['count']}")
    lines.extend(["", "EFPP co-occurrence matrix (chunk counts):"])
    lines.append("pattern | " + " | ".join(report["efpp_cooccurrence_matrix"]["pattern_names"]))
    for name, counts in zip(
        report["efpp_cooccurrence_matrix"]["pattern_names"],
        report["efpp_cooccurrence_matrix"]["counts"],
    ):
        lines.append(name + " | " + " | ".join(str(value) for value in counts))
    lines.extend(["", "Examples by EFPP pattern:"])
    for row in report["efpp_pattern_distribution"]:
        lines.append(f"[{row['pattern_name']}]")
        for example in row["examples"]:
            lines.append(
                f"- {example['contract_id']} | {example['split']} | chunk={example['chunk_index']} | "
                f"{example['matched_rule']} | {example['local_opcode_snippet']}"
            )
    return lines


def main():
    args = parse_args()
    report = audit(args.dataset, args.corpus_dir)
    suffix = f"_{args.report_suffix}" if args.report_suffix else ""
    json_path = REPORT_DIR / f"effect_flow_annotation_audit_{args.dataset}{suffix}.json"
    txt_path = REPORT_DIR / f"effect_flow_annotation_audit_{args.dataset}{suffix}.txt"
    write_json(json_path, report)
    write_text(txt_path, render(report))
    print(f"[OK] wrote {relative(txt_path)}")
    print(f"[OK] wrote {relative(json_path)}")


if __name__ == "__main__":
    main()
