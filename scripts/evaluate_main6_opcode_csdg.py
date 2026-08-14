"""Run the one explicitly unlocked Opcode-CSDG test evaluation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evm_opcode_graph_dataset import OpcodeGraphSequenceDataset, collate_opcode_graph, move_graph_batch  # noqa: E402
from evm_opcode_graph_residual_mil import OpcodeGraphResidualMIL, load_opcode_graph_state  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path):
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_metrics(logits, labels, thresholds):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    thresholds = np.asarray(thresholds, dtype=float)
    predictions = probabilities >= thresholds[None, :]
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "per_label_f1": f1_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_label_precision": precision_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_label_recall": recall_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_label_average_precision": [float(average_precision_score(labels[:, i], probabilities[:, i])) for i in range(labels.shape[1])],
        "tp": (predictions & labels.astype(bool)).sum(axis=0).astype(int).tolist(),
        "fp": (predictions & ~labels.astype(bool)).sum(axis=0).astype(int).tolist(),
        "fn": ((~predictions) & labels.astype(bool)).sum(axis=0).astype(int).tolist(),
        "thresholds": thresholds.tolist(),
    }, probabilities, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--split", choices=["test"], default="test")
    args = parser.parse_args()
    selection_path = resolve("results/main6_opcode_csdg/selection.json")
    artifact_path = resolve("results/main6_opcode_csdg/final_test_artifact.json")
    if artifact_path.exists():
        raise RuntimeError("Final test artifact already exists; refusing a second test")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("approved_for_single_test") or not selection.get("selected"):
        raise RuntimeError("Validation selection did not approve a test")
    variant = selection["selected"]["variant"]
    payload = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    config = dict(payload["common"])
    config.update(payload["variants"][variant])
    checkpoint_path = resolve(config["checkpoint_dir"]) / "best_macro_f1.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = OpcodeGraphResidualMIL(config)
    load_opcode_graph_state(model, checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = OpcodeGraphSequenceDataset(resolve(config["graph_cache_dir"]) / "test.pt", resolve(config["sequence_feature_dir"]) / "test.pt", config["label_names"])
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)), collate_fn=collate_opcode_graph)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_graph_batch(batch, device)
            output = model(batch["sequence_features"], batch["sequence_mask"], batch["node_features"], batch["node_mask"], batch["edge_index"], batch["edge_type"], node_local_features=batch["node_local_features"], node_local_offsets=batch["node_local_offsets"], node_type=batch["node_type"])
            all_logits.append(output["recognition_logits"].cpu())
            all_labels.append(batch["multi_labels"].cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics, probabilities, predictions = fixed_metrics(logits, labels, selection["thresholds"])
    result = {
        "route": "DIVE Main6 Opcode-CSDG",
        "split": "test",
        "test_labels_read": True,
        "single_test_unlock": "ALLOW_TEST=1",
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "test_cache_sha256": {
            "graph": file_sha256(resolve(config["graph_cache_dir"]) / "test.pt"),
            "sequence": file_sha256(resolve(config["sequence_feature_dir"]) / "test.pt"),
        },
        "selection": selection,
        "metrics": metrics,
        "ids": dataset.ids,
        "probabilities": probabilities.tolist(),
        "predictions": predictions.astype(int).tolist(),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"variant": variant, "macro_f1": metrics["macro_f1"], "micro_f1": metrics["micro_f1"], "artifact": str(artifact_path)}, indent=2))


if __name__ == "__main__":
    main()
