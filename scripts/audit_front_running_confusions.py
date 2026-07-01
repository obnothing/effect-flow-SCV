import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation",
]
CONFUSION_LABELS = ["Access Control", "Reentrancy", "Time manipulation"]
FRONT_LABEL = "Front Running"

EXTERNAL_CALL_OPS = {"CALL", "DELEGATECALL", "STATICCALL", "CALLCODE"}
ENV_OPS = {
    "TIMESTAMP",
    "NUMBER",
    "BLOCKHASH",
    "COINBASE",
    "DIFFICULTY",
    "PREVRANDAO",
    "GASLIMIT",
    "ORIGIN",
    "GASPRICE",
}
COND_OPS = {"JUMPI", "LT", "GT", "SLT", "SGT", "EQ", "ISZERO"}
ARITH_OR_HASH_OPS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "SDIV",
    "MOD",
    "SMOD",
    "ADDMOD",
    "MULMOD",
    "EXP",
    "SHA3",
    "KECCAK256",
}
AUTH_HINT_OPS = {"CALLER", "ORIGIN"}
GUARD_OPS = {"EQ", "JUMPI", "REVERT", "INVALID", "ISZERO"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit Front Running FP/FN top chunks and confusion with Access Control, "
            "Reentrancy, and Time manipulation."
        )
    )
    parser.add_argument(
        "--data",
        default="data/processed/DIVE_random_split/test.jsonl",
        help="Processed split jsonl containing id/opcode/multi_labels.",
    )
    parser.add_argument(
        "--predictions",
        default=(
            "results/dive_side_scale_search/train_dive_side_scale_150_ep50/"
            "test_predictions_per_label.jsonl"
        ),
        help="Per-label calibrated prediction jsonl.",
    )
    parser.add_argument(
        "--top_chunks",
        default=(
            "results/dive_side_scale_search/train_dive_side_scale_150_ep50/"
            "top_chunks_test_calibrated.jsonl"
        ),
        help="Top chunk jsonl from evaluate_chunk_mil.py.",
    )
    parser.add_argument(
        "--pattern_config",
        default="configs/effect_flow_efpp_conservative_22.json",
    )
    parser.add_argument(
        "--semantic_cache",
        default="data/features/effect_flow_semantics/dive_random_stride256_max64/test.pt",
        help="Optional semantic cache. If absent, the audit falls back to opcode rules.",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=256)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--examples_per_group", type=int, default=20)
    parser.add_argument(
        "--output_prefix",
        default="data/reports/front_running_confusion_audit",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_pattern_names(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["included_pattern_names"])


def op_positions(tokens, ops):
    return [idx for idx, token in enumerate(tokens) if token in ops]


def has_near(indices_a, indices_b, radius):
    if not indices_a or not indices_b:
        return False
    for a in indices_a:
        for b in indices_b:
            if abs(a - b) <= radius:
                return True
    return False


def has_after(indices_a, indices_b, radius):
    if not indices_a or not indices_b:
        return False
    for a in indices_a:
        for b in indices_b:
            if 0 < b - a <= radius:
                return True
    return False


def compute_opcode_features(tokens):
    external_calls = op_positions(tokens, EXTERNAL_CALL_OPS)
    delegatecalls = op_positions(tokens, {"DELEGATECALL"})
    state_reads = op_positions(tokens, {"SLOAD"})
    state_writes = op_positions(tokens, {"SSTORE"})
    env_ops = op_positions(tokens, ENV_OPS)
    cond_ops = op_positions(tokens, COND_OPS)
    arith_hash_ops = op_positions(tokens, ARITH_OR_HASH_OPS)
    div_ops = op_positions(tokens, {"DIV", "SDIV"})
    mul_ops = op_positions(tokens, {"MUL", "MULMOD"})
    auth_ops = op_positions(tokens, AUTH_HINT_OPS)
    guard_ops = op_positions(tokens, GUARD_OPS)
    gas_or_value = op_positions(tokens, {"CALLVALUE", "GAS", "GASPRICE"})
    hardcoded = [
        idx
        for idx, token in enumerate(tokens)
        if token.startswith("PUSH20") or token.startswith("PUSH32")
    ]
    return_check = has_after(external_calls, cond_ops + op_positions(tokens, {"REVERT"}), 10)
    auth_guard = bool(auth_ops and guard_ops and has_near(auth_ops, guard_ops, 32))
    dense_ops = len(external_calls) + len(state_reads) + len(state_writes)
    lower_joined = " ".join(tokens).lower()
    return {
        "has_external_call": bool(external_calls),
        "has_delegatecall": bool(delegatecalls),
        "has_state_read": bool(state_reads),
        "has_state_write": bool(state_writes),
        "call_before_state_write": bool(
            external_calls and state_writes and min(external_calls) < max(state_writes)
        ),
        "state_write_before_call": bool(
            external_calls and state_writes and min(state_writes) < max(external_calls)
        ),
        "call_without_return_check": bool(external_calls and not return_check),
        "call_with_return_check": bool(external_calls and return_check),
        "state_write_with_auth_guard": bool(state_writes and auth_guard),
        "state_write_without_auth_guard": bool(state_writes and not auth_guard),
        "sensitive_call_with_auth_guard": bool(external_calls and auth_guard),
        "sensitive_call_without_auth_guard": bool(external_calls and not auth_guard),
        "env_used_in_condition": bool(env_ops and cond_ops),
        "env_used_in_arithmetic_or_hash": bool(env_ops and arith_hash_ops),
        "div_before_mul": bool(div_ops and mul_ops and min(div_ops) < max(mul_ops)),
        "has_hardcoded_address": bool(hardcoded),
        "hardcoded_address_near_call": bool(hardcoded and has_near(hardcoded, external_calls, 32)),
        "value_or_gas_sensitive_call": bool(
            external_calls and (gas_or_value or "CALLVALUE" in tokens)
        ),
        "dense_storage_or_call_ops": dense_ops >= 6,
        "panic_selector_present": "0x4e487b71" in lower_joined,
        "selfdestruct_present": "SELFDESTRUCT" in tokens,
        "create_contract_present": "CREATE" in tokens or "CREATE2" in tokens,
        "_counts": {
            "external_call_count": len(external_calls),
            "state_read_count": len(state_reads),
            "state_write_count": len(state_writes),
            "env_opcode_count": len(env_ops),
            "auth_hint_count": len(auth_ops),
            "guard_opcode_count": len(guard_ops),
            "gas_or_value_count": len(gas_or_value),
            "dense_storage_or_call_count": dense_ops,
        },
    }


def chunk_tokens(opcode, chunk_idx, chunk_size, chunk_stride, max_chunks):
    tokens = opcode.split()
    if chunk_idx >= max_chunks:
        return [], (0, 0), len(tokens)
    start = chunk_idx * chunk_stride
    end = min(start + chunk_size, len(tokens))
    if start >= len(tokens):
        return [], (start, end), len(tokens)
    return tokens[start:end], (start, end), len(tokens)


def mean(values):
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def precision_recall_f1(tp, fp, fn):
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def load_semantic_cache(path, ids, pattern_names):
    if not path.exists():
        return None, f"semantic cache not found: {project_relative(path)}"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - only used when cache exists.
        return None, f"semantic cache exists but torch import failed: {exc}"
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover
        return None, f"semantic cache load failed: {exc}"
    if "ids" not in payload or "efpp_probs" not in payload:
        return None, "semantic cache missing ids or efpp_probs"
    cache_ids = [str(v) for v in payload["ids"]]
    index = {sample_id: idx for idx, sample_id in enumerate(cache_ids)}
    missing = [sample_id for sample_id in ids if sample_id not in index]
    if missing:
        return None, f"semantic cache id mismatch; missing {len(missing)} prediction ids"
    semantic = {
        "index": index,
        "efpp_probs": payload.get("efpp_probs"),
        "etp_distribution": payload.get("etp_distribution"),
        "relation_distribution": payload.get("relation_distribution"),
        "pattern_names": pattern_names,
    }
    return semantic, f"semantic cache loaded: {project_relative(path)}"


def semantic_pattern_values(semantic, sample_id, chunk_idx):
    if semantic is None:
        return {}
    row = semantic["index"].get(str(sample_id))
    if row is None:
        return {}
    efpp = semantic.get("efpp_probs")
    if efpp is None:
        return {}
    if chunk_idx >= int(efpp.shape[1]):
        return {}
    values = efpp[row, chunk_idx].detach().cpu().float().tolist()
    return {
        name: float(values[idx])
        for idx, name in enumerate(semantic["pattern_names"])
        if idx < len(values)
    }


def build_maps(data_path, prediction_path, top_chunk_path):
    data_records = {str(obj["id"]): obj for obj in read_jsonl(data_path)}
    predictions = {str(obj["id"]): obj for obj in read_jsonl(prediction_path)}
    top_chunks = {}
    for obj in read_jsonl(top_chunk_path):
        top_chunks[(str(obj["id"]), obj["label_name"])] = obj
    return data_records, predictions, top_chunks


def group_name(front_true, front_pred):
    if front_true == 1 and front_pred == 1:
        return "tp"
    if front_true == 0 and front_pred == 1:
        return "fp"
    if front_true == 1 and front_pred == 0:
        return "fn"
    return "tn"


def summarize_confusion(predictions, label_names, front_id, confusion_ids):
    groups = {key: [] for key in ("tp", "fp", "fn", "tn")}
    for sample_id, pred in predictions.items():
        true = int(pred["multi_true"][front_id])
        out = int(pred["multi_pred"][front_id])
        groups[group_name(true, out)].append((sample_id, pred))

    summary = {}
    for group, rows in groups.items():
        row_count = len(rows)
        entry = {"count": row_count, "labels": {}}
        for label_id in confusion_ids:
            label = label_names[label_id]
            true_count = sum(int(pred["multi_true"][label_id]) for _, pred in rows)
            pred_count = sum(int(pred["multi_pred"][label_id]) for _, pred in rows)
            true_and_pred_count = sum(
                int(pred["multi_true"][label_id] and pred["multi_pred"][label_id])
                for _, pred in rows
            )
            probs = [float(pred["multi_prob"][label_id]) for _, pred in rows]
            entry["labels"][label] = {
                "true_positive_label_count": true_count,
                "predicted_positive_label_count": pred_count,
                "true_and_predicted_positive_label_count": true_and_pred_count,
                "true_rate_in_group": safe_ratio(true_count, row_count),
                "predicted_rate_in_group": safe_ratio(pred_count, row_count),
                "mean_probability": mean(probs),
            }
        summary[group] = entry
    return summary, groups


def collect_chunk_audit(
    groups,
    label_names,
    target_labels,
    data_records,
    predictions,
    top_chunks,
    pattern_names,
    semantic,
    args,
):
    rows = []
    group_label_pattern_hits = defaultdict(Counter)
    group_label_semantic_sums = defaultdict(lambda: defaultdict(float))
    group_label_counts = Counter()
    group_label_opcode_counts = defaultdict(Counter)
    group_label_chunk_overlap = Counter()
    group_label_contract_count = Counter()

    for group, group_rows in groups.items():
        for sample_id, pred in group_rows:
            data = data_records.get(sample_id)
            if not data:
                continue
            front_top = top_chunks.get((sample_id, FRONT_LABEL), {})
            front_indices = set(front_top.get("top_chunk_indices") or [])
            for label in target_labels:
                label_id = label_names.index(label)
                top = top_chunks.get((sample_id, label))
                if top is None:
                    continue
                indices = list(top.get("top_chunk_indices") or [])[:2]
                scores = list(top.get("top_chunk_scores") or [])[: len(indices)]
                if not indices:
                    continue
                key = (group, label)
                group_label_contract_count[key] += 1
                if label != FRONT_LABEL and front_indices.intersection(indices):
                    group_label_chunk_overlap[key] += 1
                for rank, chunk_idx in enumerate(indices):
                    tokens, span, total_tokens = chunk_tokens(
                        data.get("opcode", ""),
                        int(chunk_idx),
                        args.chunk_size,
                        args.chunk_stride,
                        args.max_chunks,
                    )
                    features = compute_opcode_features(tokens)
                    pattern_hits = {
                        name: bool(features.get(name, False)) for name in pattern_names
                    }
                    for name, hit in pattern_hits.items():
                        if hit:
                            group_label_pattern_hits[key][name] += 1
                    for count_key, value in features["_counts"].items():
                        group_label_opcode_counts[key][count_key] += int(value)
                    semantic_values = semantic_pattern_values(semantic, sample_id, int(chunk_idx))
                    for name, value in semantic_values.items():
                        group_label_semantic_sums[key][name] += float(value)
                    group_label_counts[key] += 1
                    if label == FRONT_LABEL and rank < args.examples_per_group:
                        pass
                    rows.append(
                        {
                            "id": sample_id,
                            "front_group": group,
                            "audited_label": label,
                            "audited_label_id": label_id,
                            "true_front_running": int(pred["multi_true"][label_names.index(FRONT_LABEL)]),
                            "pred_front_running": int(pred["multi_pred"][label_names.index(FRONT_LABEL)]),
                            "prob_front_running": float(pred["multi_prob"][label_names.index(FRONT_LABEL)]),
                            "true_confusion_labels": {
                                name: int(pred["multi_true"][label_names.index(name)])
                                for name in CONFUSION_LABELS
                            },
                            "pred_confusion_labels": {
                                name: int(pred["multi_pred"][label_names.index(name)])
                                for name in CONFUSION_LABELS
                            },
                            "prob_confusion_labels": {
                                name: float(pred["multi_prob"][label_names.index(name)])
                                for name in CONFUSION_LABELS
                            },
                            "chunk_index": int(chunk_idx),
                            "chunk_rank": rank,
                            "chunk_score": float(scores[rank]) if rank < len(scores) else None,
                            "chunk_token_start": int(span[0]),
                            "chunk_token_end": int(span[1]),
                            "contract_token_length": int(total_tokens),
                            "pattern_hits": [
                                name for name in pattern_names if pattern_hits.get(name)
                            ],
                            "opcode_counts": features["_counts"],
                            "semantic_pattern_probs": semantic_values,
                            "opcode_snippet": " ".join(tokens[:120]),
                        }
                    )

    pattern_summary = {}
    for key, total_chunks in group_label_counts.items():
        group, label = key
        contract_count = group_label_contract_count[key]
        item = {
            "front_group": group,
            "audited_label": label,
            "contracts_with_top_chunks": int(contract_count),
            "top_chunk_count": int(total_chunks),
            "front_chunk_overlap_contracts": int(group_label_chunk_overlap[key]),
            "front_chunk_overlap_rate": safe_ratio(group_label_chunk_overlap[key], contract_count),
            "pattern_hit_rate": {
                name: group_label_pattern_hits[key][name] / total_chunks
                for name in pattern_names
            },
            "mean_opcode_counts": {
                name: value / total_chunks
                for name, value in group_label_opcode_counts[key].items()
            },
        }
        if group_label_semantic_sums[key]:
            item["semantic_pattern_mean"] = {
                name: value / total_chunks
                for name, value in group_label_semantic_sums[key].items()
            }
        pattern_summary[f"{group}::{label}"] = item
    return rows, pattern_summary


def top_deltas(pattern_summary, group_a, group_b, label, pattern_names, limit=12):
    key_a = f"{group_a}::{label}"
    key_b = f"{group_b}::{label}"
    if key_a not in pattern_summary or key_b not in pattern_summary:
        return []
    rates_a = pattern_summary[key_a]["pattern_hit_rate"]
    rates_b = pattern_summary[key_b]["pattern_hit_rate"]
    deltas = [
        {
            "pattern": name,
            f"{group_a}_rate": rates_a.get(name, 0.0),
            f"{group_b}_rate": rates_b.get(name, 0.0),
            "delta": rates_a.get(name, 0.0) - rates_b.get(name, 0.0),
        }
        for name in pattern_names
    ]
    deltas.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return deltas[:limit]


def select_examples(rows, examples_per_group):
    selected = []
    counts = Counter()
    priority = {"fp": 0, "fn": 1, "tp": 2, "tn": 3}
    front_rows = [row for row in rows if row["audited_label"] == FRONT_LABEL]
    front_rows.sort(
        key=lambda row: (
            priority.get(row["front_group"], 99),
            row["chunk_rank"],
            -abs(row["prob_front_running"] - 0.5),
        )
    )
    for row in front_rows:
        group = row["front_group"]
        if group not in {"fp", "fn", "tp"}:
            continue
        if counts[group] >= examples_per_group:
            continue
        selected.append(row)
        counts[group] += 1
    return selected


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt_float(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def write_text_report(path, report):
    lines = []
    lines.append("Front Running FP/FN top-chunk confusion audit")
    lines.append("")
    lines.append("Sources")
    for key, value in report["sources"].items():
        lines.append(f"- {key}: {value}")
    lines.append(f"- semantic_cache_status: {report['semantic_cache_status']}")
    lines.append("")
    metrics = report["front_running_metrics"]
    lines.append("Front Running contract-level metrics")
    lines.append(
        "TP={tp} FP={fp} FN={fn} TN={tn} precision={precision} recall={recall} f1={f1}".format(
            tp=metrics["tp"],
            fp=metrics["fp"],
            fn=metrics["fn"],
            tn=metrics["tn"],
            precision=fmt_float(metrics["precision"]),
            recall=fmt_float(metrics["recall"]),
            f1=fmt_float(metrics["f1"]),
        )
    )
    lines.append("")
    lines.append("Confusion-label co-occurrence inside Front Running groups")
    lines.append(
        "front_group | count | label | true_count | true_rate | pred_count | pred_rate | mean_prob"
    )
    for group, item in report["confusion_summary"].items():
        for label, values in item["labels"].items():
            lines.append(
                " | ".join(
                    [
                        group,
                        str(item["count"]),
                        label,
                        str(values["true_positive_label_count"]),
                        fmt_float(values["true_rate_in_group"]),
                        str(values["predicted_positive_label_count"]),
                        fmt_float(values["predicted_rate_in_group"]),
                        fmt_float(values["mean_probability"]),
                    ]
                )
            )
    lines.append("")
    lines.append("Front Running top-chunk opcode/effect-flow pattern hit rates")
    lines.append("group | chunks | contracts | top patterns")
    for group in ("tp", "fp", "fn", "tn"):
        key = f"{group}::{FRONT_LABEL}"
        item = report["pattern_summary"].get(key)
        if not item:
            continue
        ranked = sorted(
            item["pattern_hit_rate"].items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:10]
        pattern_text = ", ".join(f"{name}={value:.3f}" for name, value in ranked)
        lines.append(
            f"{group} | {item['top_chunk_count']} | {item['contracts_with_top_chunks']} | {pattern_text}"
        )
    lines.append("")
    lines.append("Front Running FP vs TP largest pattern deltas")
    for item in report["front_fp_minus_tp_pattern_deltas"]:
        lines.append(
            f"- {item['pattern']}: FP={item['fp_rate']:.3f}, TP={item['tp_rate']:.3f}, delta={item['delta']:.3f}"
        )
    lines.append("")
    lines.append("Front Running FN vs TP missing-signal deltas")
    for item in report["front_fn_minus_tp_pattern_deltas"]:
        lines.append(
            f"- {item['pattern']}: FN={item['fn_rate']:.3f}, TP={item['tp_rate']:.3f}, delta={item['delta']:.3f}"
        )
    lines.append("")
    lines.append("Confusion label top-chunk overlap on Front Running FP/FN contracts")
    lines.append("group | label | contracts | overlap_with_front_top_chunks | overlap_rate | top patterns")
    for group in ("fp", "fn"):
        for label in CONFUSION_LABELS:
            key = f"{group}::{label}"
            item = report["pattern_summary"].get(key)
            if not item:
                continue
            ranked = sorted(
                item["pattern_hit_rate"].items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:8]
            pattern_text = ", ".join(f"{name}={value:.3f}" for name, value in ranked)
            lines.append(
                " | ".join(
                    [
                        group,
                        label,
                        str(item["contracts_with_top_chunks"]),
                        str(item["front_chunk_overlap_contracts"]),
                        fmt_float(item["front_chunk_overlap_rate"]),
                        pattern_text,
                    ]
                )
            )
    lines.append("")
    lines.append("Interpretation")
    for text in report["interpretation"]:
        lines.append(f"- {text}")
    lines.append("")
    lines.append("Artifacts")
    lines.append(f"- json: {report['artifacts']['json']}")
    lines.append(f"- top_chunks_jsonl: {report['artifacts']['top_chunks_jsonl']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_interpretation(confusion_summary, pattern_summary, fp_deltas, fn_deltas):
    texts = []
    fp_count = confusion_summary["fp"]["count"]
    fn_count = confusion_summary["fn"]["count"]
    if fp_count:
        fp_access = confusion_summary["fp"]["labels"]["Access Control"]
        fp_reent = confusion_summary["fp"]["labels"]["Reentrancy"]
        fp_time = confusion_summary["fp"]["labels"]["Time manipulation"]
        texts.append(
            "Front Running FP contracts frequently carry other positive labels: "
            f"Access Control true_rate={fmt_float(fp_access['true_rate_in_group'])}, "
            f"Reentrancy true_rate={fmt_float(fp_reent['true_rate_in_group'])}, "
            f"Time manipulation true_rate={fmt_float(fp_time['true_rate_in_group'])}."
        )
    if fn_count:
        fn_access = confusion_summary["fn"]["labels"]["Access Control"]
        fn_reent = confusion_summary["fn"]["labels"]["Reentrancy"]
        fn_time = confusion_summary["fn"]["labels"]["Time manipulation"]
        texts.append(
            "Front Running FN contracts also overlap with high-prevalence labels: "
            f"Access Control true_rate={fmt_float(fn_access['true_rate_in_group'])}, "
            f"Reentrancy true_rate={fmt_float(fn_reent['true_rate_in_group'])}, "
            f"Time manipulation true_rate={fmt_float(fn_time['true_rate_in_group'])}."
        )
    high_fp = [item for item in fp_deltas if item["delta"] > 0.15]
    if high_fp:
        names = ", ".join(item["pattern"] for item in high_fp[:5])
        texts.append(
            "Patterns higher in Front Running FP than TP suggest false-positive drivers: "
            f"{names}."
        )
    missing = [item for item in fn_deltas if item["delta"] < -0.15]
    if missing:
        names = ", ".join(item["pattern"] for item in missing[:5])
        texts.append(
            "Patterns lower in Front Running FN than TP suggest missing true-positive signals: "
            f"{names}."
        )
    front_fp = pattern_summary.get("fp::Front Running", {})
    rates = front_fp.get("pattern_hit_rate", {})
    generic = [
        name
        for name in (
            "has_external_call",
            "value_or_gas_sensitive_call",
            "sensitive_call_without_auth_guard",
            "state_write_without_auth_guard",
            "env_used_in_condition",
        )
        if rates.get(name, 0.0) >= 0.5
    ]
    if generic:
        texts.append(
            "Front Running FP top chunks contain generic risk evidence "
            f"({', '.join(generic)}), which is not specific enough to distinguish "
            "transaction-order dependence from Access/Reentrancy/Time behavior."
        )
    texts.append(
        "Recommended next step: keep Front Running risk evidence conservative and add "
        "more specific order/value/state-update proxies before increasing its side-evidence weight."
    )
    return texts


def main():
    args = parse_args()
    data_path = resolve_path(args.data)
    prediction_path = resolve_path(args.predictions)
    top_chunk_path = resolve_path(args.top_chunks)
    pattern_config_path = resolve_path(args.pattern_config)
    semantic_path = resolve_path(args.semantic_cache)
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pattern_names = load_pattern_names(pattern_config_path)
    data_records, predictions, top_chunks = build_maps(
        data_path, prediction_path, top_chunk_path
    )
    label_names = DEFAULT_LABELS
    front_id = label_names.index(FRONT_LABEL)
    confusion_ids = [label_names.index(name) for name in CONFUSION_LABELS]

    semantic, semantic_status = load_semantic_cache(
        semantic_path, predictions.keys(), pattern_names
    )
    confusion_summary, groups = summarize_confusion(
        predictions, label_names, front_id, confusion_ids
    )

    rows, pattern_summary = collect_chunk_audit(
        groups=groups,
        label_names=label_names,
        target_labels=[FRONT_LABEL] + CONFUSION_LABELS,
        data_records=data_records,
        predictions=predictions,
        top_chunks=top_chunks,
        pattern_names=pattern_names,
        semantic=semantic,
        args=args,
    )
    examples = select_examples(rows, args.examples_per_group)

    tp = confusion_summary["tp"]["count"]
    fp = confusion_summary["fp"]["count"]
    fn = confusion_summary["fn"]["count"]
    tn = confusion_summary["tn"]["count"]
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    fp_deltas = top_deltas(pattern_summary, "fp", "tp", FRONT_LABEL, pattern_names)
    fn_deltas = top_deltas(pattern_summary, "fn", "tp", FRONT_LABEL, pattern_names)

    json_path = output_prefix.with_suffix(".json")
    txt_path = output_prefix.with_suffix(".txt")
    examples_path = output_prefix.parent / f"{output_prefix.name}_top_chunks.jsonl"
    report = {
        "sources": {
            "data": project_relative(data_path),
            "predictions": project_relative(prediction_path),
            "top_chunks": project_relative(top_chunk_path),
            "pattern_config": project_relative(pattern_config_path),
            "semantic_cache": project_relative(semantic_path),
        },
        "semantic_cache_status": semantic_status,
        "label_names": label_names,
        "front_label_id": front_id,
        "confusion_labels": CONFUSION_LABELS,
        "chunking": {
            "chunk_size": args.chunk_size,
            "chunk_stride": args.chunk_stride,
            "max_chunks": args.max_chunks,
        },
        "front_running_metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "confusion_summary": confusion_summary,
        "pattern_summary": pattern_summary,
        "front_fp_minus_tp_pattern_deltas": fp_deltas,
        "front_fn_minus_tp_pattern_deltas": fn_deltas,
        "example_count": len(examples),
        "interpretation": [],
        "artifacts": {
            "json": project_relative(json_path),
            "txt": project_relative(txt_path),
            "top_chunks_jsonl": project_relative(examples_path),
        },
    }
    report["interpretation"] = build_interpretation(
        confusion_summary, pattern_summary, fp_deltas, fn_deltas
    )

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(examples_path, examples)
    write_text_report(txt_path, report)
    print(f"[OK] wrote {project_relative(txt_path)}")
    print(f"[OK] wrote {project_relative(json_path)}")
    print(f"[OK] wrote {project_relative(examples_path)}")


if __name__ == "__main__":
    main()
