import copy
import json
import random
from pathlib import Path

import numpy as np


def load_jsonl(path):
    samples = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "multi_labels" not in item:
                raise ValueError(f"{path}:{line_no} missing multi_labels")
            samples.append(item)
    return samples


def save_jsonl(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def get_label_matrix(samples):
    if not samples:
        return np.zeros((0, 0), dtype=np.int64)
    return np.asarray([sample["multi_labels"] for sample in samples], dtype=np.int64)


def compute_label_counts(Y):
    if Y.size == 0:
        return np.asarray([], dtype=np.int64)
    return Y.sum(axis=0).astype(np.int64)


def compute_irlbl(label_counts):
    label_counts = np.asarray(label_counts, dtype=np.float64)
    if label_counts.size == 0:
        return np.asarray([], dtype=np.float64)
    max_count = float(label_counts.max())
    irlbl = np.zeros_like(label_counts, dtype=np.float64)
    nonzero = label_counts > 0
    irlbl[nonzero] = max_count / label_counts[nonzero]
    irlbl[~nonzero] = float("inf")
    return irlbl


def compute_mean_ir(irlbl):
    finite_values = np.asarray(irlbl, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0
    return float(finite_values.mean())


def get_minority_labels(irlbl, mean_ir):
    return [int(idx) for idx, value in enumerate(irlbl) if value > mean_ir]


def compute_multilabel_stats(Y, label_names):
    counts = compute_label_counts(Y)
    irlbl = compute_irlbl(counts)
    mean_ir = compute_mean_ir(irlbl)
    minority_indices = get_minority_labels(irlbl, mean_ir)
    return {
        "label_counts": {
            label: int(counts[idx]) for idx, label in enumerate(label_names)
        },
        "irlbl": {
            label: float(irlbl[idx]) if np.isfinite(irlbl[idx]) else "inf"
            for idx, label in enumerate(label_names)
        },
        "mean_ir": mean_ir,
        "minority_labels": [label_names[idx] for idx in minority_indices],
    }


def _binary_label_from_multi_labels(multi_labels):
    return int(any(int(value) == 1 for value in multi_labels))


def _target_counts_from_strategy(samples, label_names, target_strategy):
    Y = get_label_matrix(samples)
    before_counts = compute_label_counts(Y)
    strategy_name = target_strategy.get("name", target_strategy.get("type"))
    if strategy_name != "paper_after_proportional":
        raise ValueError(f"Unsupported target_strategy: {strategy_name}")

    paper_before = target_strategy["paper_before_counts"]
    paper_after = target_strategy["paper_after_counts"]
    targets = before_counts.copy()
    for idx, label in enumerate(label_names):
        expected_before = int(paper_before[label])
        expected_after = int(paper_after[label])
        if expected_before <= 0:
            continue
        target = round(int(before_counts[idx]) * expected_after / expected_before)
        targets[idx] = max(int(before_counts[idx]), int(target))
    return targets.astype(np.int64)


def oversample_text_multilabel(samples, label_names, target_strategy, seed):
    rng = random.Random(seed)
    max_duplicate_per_sample = int(target_strategy.get("max_duplicate_per_sample", 5))
    target_counts = _target_counts_from_strategy(samples, label_names, target_strategy)

    original_samples = list(samples)
    augmented_samples = list(samples)
    Y = get_label_matrix(original_samples)
    current_counts = compute_label_counts(Y)
    duplicate_counts = [0 for _ in original_samples]
    candidate_indices_by_label = {
        label_idx: [
            sample_idx
            for sample_idx, row in enumerate(Y)
            if int(row[label_idx]) == 1
        ]
        for label_idx in range(len(label_names))
    }
    for indices in candidate_indices_by_label.values():
        rng.shuffle(indices)

    aug_index = 1
    max_iterations = max(1, len(original_samples) * max_duplicate_per_sample)
    iterations = 0

    while np.any(current_counts < target_counts) and iterations < max_iterations:
        deficits = target_counts - current_counts
        deficit_labels = [idx for idx, value in enumerate(deficits) if value > 0]
        if not deficit_labels:
            break

        target_label = max(
            deficit_labels,
            key=lambda idx: deficits[idx] / max(int(target_counts[idx]), 1),
        )
        candidates = [
            idx
            for idx in candidate_indices_by_label[target_label]
            if duplicate_counts[idx] < max_duplicate_per_sample
        ]
        if not candidates:
            deficits[target_label] = 0
            target_counts[target_label] = current_counts[target_label]
            if not np.any(current_counts < target_counts):
                break
            continue

        sample_idx = max(
            candidates,
            key=lambda idx: int(np.minimum(Y[idx], np.maximum(deficits, 0)).sum()),
        )
        new_sample = copy.deepcopy(original_samples[sample_idx])
        base_id = str(new_sample.get("id", sample_idx))
        new_sample["id"] = f"{base_id}__aug_{aug_index:06d}"
        new_sample["binary_label"] = _binary_label_from_multi_labels(
            new_sample["multi_labels"]
        )
        augmented_samples.append(new_sample)
        duplicate_counts[sample_idx] += 1
        current_counts += Y[sample_idx]
        aug_index += 1
        iterations += 1

    return augmented_samples, {
        "target_train_after_counts": {
            label: int(target_counts[idx]) for idx, label in enumerate(label_names)
        },
        "achieved_after_counts": {
            label: int(current_counts[idx]) for idx, label in enumerate(label_names)
        },
        "added_samples": len(augmented_samples) - len(original_samples),
        "max_duplicate_per_sample": max_duplicate_per_sample,
    }
