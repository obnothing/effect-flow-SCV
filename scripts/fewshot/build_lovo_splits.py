"""Build strict train-only leave-one-vulnerability-out pilot splits."""

import json
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/fewshot_query_pilot/pilot0.yaml"
ID_RE = re.compile(r'"id"\s*:\s*"([^"]+)"')


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_ids_only(path):
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                match = ID_RE.search(line)
                if not match:
                    raise ValueError(f"ID field not found in {path}")
                ids.append(match.group(1))
    return ids


def make_groups(rows, novel_index, quantile_edges):
    old_indices = [i for i in range(8) if i != novel_index]
    groups = {}
    for row in rows:
        length = len(row["opcode"].split())
        length_bin = min(9, int(np.searchsorted(quantile_edges[1:-1], length, side="right")))
        old_label_count = int(sum(row["multi_labels"][i] for i in old_indices))
        groups[row["id"]] = (length_bin, old_label_count)
    return groups


def choose_matched_negatives(pos_rows, negative_rows, groups, seed):
    rng = np.random.default_rng(seed)
    available = {row["id"]: row for row in negative_rows}
    selected = []
    for positive in pos_rows:
        target_group = groups[positive["id"]]
        candidates = list(available.values())
        if not candidates:
            raise ValueError("Insufficient negative contracts for matching")
        exact = [row for row in candidates if groups[row["id"]] == target_group]
        if exact:
            pool = exact
        else:
            distance = lambda row: (abs(groups[row["id"]][0]-target_group[0]), abs(groups[row["id"]][1]-target_group[1]))
            best = min(distance(row) for row in candidates)
            pool = [row for row in candidates if distance(row) == best]
        choice = pool[int(rng.integers(len(pool)))]
        selected.append(choice)
        del available[choice["id"]]
    return selected


def build_one(rows, novel_label, config, valid_ids, test_ids):
    names = config["label_names"]
    novel_index = names.index(novel_label)
    train_ids = [row["id"] for row in rows]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Duplicate IDs in original train")
    if set(train_ids) & set(valid_ids):
        raise ValueError("Original train intersects official valid IDs")
    if set(train_ids) & set(test_ids):
        raise ValueError("Original train intersects official test IDs")
    positives = [row for row in rows if row["multi_labels"][novel_index] == 1]
    negatives = [row for row in rows if row["multi_labels"][novel_index] == 0]
    if len(positives) < 50:
        return {"novel_label": novel_label, "positive_count": len(positives), "status": "blocked_positive_count_below_50"}
    lengths = np.asarray([len(row["opcode"].split()) for row in rows])
    edges = np.quantile(lengths, np.linspace(0, 1, 11)).tolist()
    groups = make_groups(rows, novel_index, edges)
    rng = np.random.default_rng(config["base_seed"] + novel_index)
    pool_count = min(len(positives), max(config["support_pool_minimum"], 5 * config["support_k"]))
    pos_order = rng.permutation(len(positives))[:pool_count]
    support_pos = [positives[int(i)] for i in pos_order]
    support_neg = choose_matched_negatives(support_pos, negatives, groups, config["base_seed"] + 100 + novel_index)
    support_pos_ids = {row["id"] for row in support_pos}
    support_neg_ids = {row["id"] for row in support_neg}
    support_ids = support_pos_ids | support_neg_ids
    eligible_base = [row for row in negatives if row["id"] not in support_neg_ids]
    dev_rng = np.random.default_rng(config["base_seed"] + 200 + novel_index)
    dev_count = max(1, int(round(len(eligible_base) * config["base_dev_fraction"])))
    dev_idx = set(int(i) for i in dev_rng.choice(len(eligible_base), size=dev_count, replace=False))
    base_dev = [row for i, row in enumerate(eligible_base) if i in dev_idx]
    base_train = [row for i, row in enumerate(eligible_base) if i not in dev_idx]
    base_train_ids = {row["id"] for row in base_train}; base_dev_ids = {row["id"] for row in base_dev}
    sets = {"base_train": base_train_ids, "base_dev": base_dev_ids, "support_pool": support_ids,
            "official_valid": set(valid_ids), "official_test_ids_only": set(test_ids)}
    intersection_audit = {f"{a}_intersects_{b}": len(sets[a] & sets[b]) for a, b in (
        ("base_train", "base_dev"), ("base_train", "support_pool"), ("base_dev", "support_pool"),
        ("base_train", "official_valid"), ("base_dev", "official_valid"), ("support_pool", "official_valid"),
        ("base_train", "official_test_ids_only"), ("base_dev", "official_test_ids_only"),
        ("support_pool", "official_test_ids_only"))}
    if any(intersection_audit.values()):
        raise ValueError(f"LOVO leakage detected: {intersection_audit}")
    if any(row["multi_labels"][novel_index] for row in base_train + base_dev):
        raise ValueError("Strict novel mode permits no positive held-out label in base training or dev")
    base_labels = [name for i, name in enumerate(names) if i != novel_index]
    base_indices = [i for i in range(8) if i != novel_index]
    episodes = []
    for support_seed in config["support_seeds"]:
        ep_rng = np.random.default_rng(int(support_seed) + 3000 + novel_index)
        chosen_pos = [support_pos[int(i)] for i in ep_rng.choice(len(support_pos), config["support_k"], replace=False)]
        chosen_neg = choose_matched_negatives(chosen_pos, support_neg, groups, int(support_seed) + 5000 + novel_index)
        episodes.append({"support_seed": int(support_seed), "positive_ids": [r["id"] for r in chosen_pos],
                         "negative_ids": [r["id"] for r in chosen_neg],
                         "length_bin_match_rate": float(np.mean([groups[p["id"]][0] == groups[n["id"]][0] for p,n in zip(chosen_pos,chosen_neg)])),
                         "old_label_count_match_rate": float(np.mean([groups[p["id"]][1] == groups[n["id"]][1] for p,n in zip(chosen_pos,chosen_neg)]))})
    return {"novel_label": novel_label, "novel_label_index": novel_index, "novel_positive_count_original_train": len(positives),
        "original_train_count": len(rows), "base_train_count": len(base_train), "base_dev_count": len(base_dev),
        "positive_support_pool_count": len(support_pos), "negative_support_pool_count": len(support_neg),
        "base_train_ids": [r["id"] for r in base_train],
        "base_dev_ids": [r["id"] for r in base_dev], "support_positive_pool_ids": [r["id"] for r in support_pos],
        "support_negative_pool_ids": [r["id"] for r in support_neg], "base_labels": base_labels,
        "base_label_indices": base_indices, "episodes": episodes, "intersection_audit": intersection_audit,
        "test_checked": False, "test_ids_read_for_intersection_only": len(test_ids),
        "length_quantile_edges": edges, "status": "ready"}


def main():
    config = json.loads(json.dumps(__import__("yaml").safe_load(CONFIG_PATH.read_text(encoding="utf-8"))))
    data_dir = ROOT / config["dataset"]
    train_rows = read_rows(data_dir / "train.jsonl")
    # Only ID fields are extracted from official valid/test for leakage assertions.
    valid_ids = read_ids_only(data_dir / "valid.jsonl")
    test_ids = read_ids_only(data_dir / "test.jsonl")
    out = ROOT / config["result_root"] / "splits"; out.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": config["dataset"], "strict_novel": config["strict_novel"], "seed": config["base_seed"],
        "official_test_content_read": False, "official_test_ids_read_for_intersection_only": True, "labels": {}}
    for label in config["heldout_labels"]:
        split = build_one(train_rows, label, config, valid_ids, test_ids)
        summary["labels"][label] = {key: value for key, value in split.items() if key not in {"base_train_ids", "base_dev_ids", "support_positive_pool_ids", "support_negative_pool_ids"}}
        if split["status"] == "ready":
            (out / f"{label.replace(' ', '_')}.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (out / "split_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Few-shot Pilot Split Audit", "", f"Dataset: `{config['dataset']}`; strict_novel={config['strict_novel']}; seed={config['base_seed']}.",
        "Only official valid/test IDs were extracted for set-intersection checks; no valid/test labels or opcode content were read during split construction.", "",
        "| Novel label | Original train positive | Base train | Base dev | Positive support pool | Negative support pool | status |", "|---|---:|---:|---:|---:|---:|---|"]
    for label, result in summary["labels"].items():
        lines.append(f"| {label} | {result['novel_positive_count_original_train']} | {result.get('base_train_count', '-')} | {result.get('base_dev_count', '-')} | {result.get('positive_support_pool_count', '-')} | {result.get('negative_support_pool_count', '-')} | {result['status']} |")
    lines += ["", "Every LOVO split JSON contains the exact ID sets, zero pairwise intersections, 7-label mapping, and five support episodes.", ""]
    (ROOT / config["report_root"] / "split_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / config["report_root"] / "split_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
