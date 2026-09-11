"""Train-only context dictionary construction and reproducibility metadata."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from light_label_data import collate_light_label


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256("\n".join(str(value) for value in ids).encode("utf-8")).hexdigest()


def sha256_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def normalized_mean(value):
    return F.normalize(value.float().mean(dim=0), dim=0)


@torch.no_grad()
def collect_train_contexts(model, train_data, config, tokenizer, device, amp):
    loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_light_label(items, pad_id=tokenizer.pad_token_id),
    )
    contexts = []
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            hidden = model.encode(input_ids, batch["lengths"])
        mask = batch["mask"].to(device, non_blocking=True)
        values = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / batch["lengths"].to(device).float().clamp_min(1).unsqueeze(-1)
        contexts.append(values.cpu())
    return torch.cat(contexts, dim=0)


def build_dictionary(contexts, mode, dictionary_size, seed):
    normalized = F.normalize(contexts.float(), dim=-1)
    count = len(normalized)
    if count < dictionary_size:
        raise ValueError(f"need at least {dictionary_size} train contracts, got {count}")
    assignments = np.zeros(count, dtype=np.int64)
    if mode == "d1_param_control":
        centroids = torch.zeros(dictionary_size, normalized.shape[1])
        global_context = torch.zeros(normalized.shape[1])
        method = "zero_context"
    elif mode == "d2_global_mean":
        global_context = normalized_mean(normalized)
        centroids = global_context.unsqueeze(0).repeat(dictionary_size, 1)
        method = "global_mean_repeated_dictionary"
    elif mode == "d3_shuffled_dictionary":
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(count)
        groups = np.array_split(order, dictionary_size)
        centroids = []
        for cluster_id, group in enumerate(groups):
            assignments[group] = cluster_id
            centroids.append(normalized[group].mean(dim=0))
        centroids = F.normalize(torch.stack(centroids), dim=-1)
        global_context = normalized_mean(normalized)
        method = "random_partition_train_means"
    elif mode == "d4_tdvp":
        from sklearn.cluster import KMeans
        clustering = KMeans(n_clusters=dictionary_size, random_state=int(seed), n_init=10).fit(normalized.numpy())
        assignments = clustering.labels_.astype(np.int64)
        centroids = F.normalize(torch.tensor(clustering.cluster_centers_, dtype=torch.float32), dim=-1)
        global_context = normalized_mean(normalized)
        method = "kmeans_train_contexts"
    else:
        raise ValueError(f"unknown dictionary mode: {mode}")
    counts = np.bincount(assignments, minlength=dictionary_size).tolist()
    metadata = {
        "mode": mode,
        "method": method,
        "seed": int(seed),
        "clusters": int(dictionary_size),
        "samples": int(count),
        "cluster_sizes": [int(value) for value in counts],
        "centroid_sha256": sha256_tensor(centroids),
    }
    return centroids, global_context, torch.tensor(assignments, dtype=torch.long), metadata


def save_dictionary(path, centroids, global_context, assignments, metadata, train_data, train_file):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "centroids": centroids.float(),
        "global_context": global_context.float(),
        "assignments": assignments.long(),
        "ids": [train_data.ids[index] for index in train_data.indices],
        "labels": train_data.labels[train_data.indices].float(),
        "original_lengths": train_data.original_lengths[train_data.indices].long(),
        "metadata": {
            **metadata,
            "train_file_sha256": sha256_file(train_file),
            "train_ids_sha256": sha256_ids([train_data.ids[index] for index in train_data.indices]),
        },
        "test_checked": False,
    }
    torch.save(payload, path)
    return payload["metadata"]

