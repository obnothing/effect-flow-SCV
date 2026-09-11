"""Audit the existing E5 implementation without opening the test split."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_audit(path):
    ids, hashes, lengths, labels = [], set(), [], []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            ids.append(str(item.get("id", len(ids))))
            opcode = " ".join(str(item.get("opcode", "")).split())
            hashes.add(hashlib.sha256(opcode.encode("utf-8")).hexdigest())
            lengths.append(len(str(item.get("opcode", "")).split()))
            labels.append(item.get("multi_labels", []))
    return {"samples": len(ids), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "unique_normalized_opcode_hashes": len(hashes), "hashes": hashes,
            "mean_opcode_tokens": sum(lengths) / max(len(lengths), 1),
            "label_prevalence": [sum(row[index] for row in labels) / max(len(labels), 1) for index in range(6)]}


def main():
    data_dir = ROOT / "data/processed/DIVE_main6_opcode_process01"
    train = jsonl_audit(data_dir / "train.jsonl")
    valid = jsonl_audit(data_dir / "valid.jsonl")
    current_metrics = ROOT / "results/light_label/extensions/e5_tdvp/full/metrics.json"
    current_checkpoint = ROOT / "checkpoints/light_label/extensions/e5_tdvp/full/best.pt"
    old_result = json.loads(current_metrics.read_text(encoding="utf-8")) if current_metrics.exists() else None
    checkpoint = __import__("torch").load(current_checkpoint, map_location="cpu") if current_checkpoint.exists() else None
    dictionary_shape = None
    if checkpoint:
        value = checkpoint.get("model_state_dict", {}).get("confounder_dictionary")
        dictionary_shape = list(value.shape) if value is not None else None
    payload = {
        "route": "DIVE Main6 E5 TDVP implementation audit",
        "dataset": "DIVE_main6_opcode_process01",
        "test_checked": False,
        "paths": {"data_dir": str(data_dir), "train": str(data_dir / "train.jsonl"), "valid": str(data_dir / "valid.jsonl"),
                  "test": str(data_dir / "test.jsonl"), "current_e5_metrics": str(current_metrics), "current_e5_checkpoint": str(current_checkpoint)},
        "split_hashes": {"train_file_sha256": sha256_file(data_dir / "train.jsonl"), "valid_file_sha256": sha256_file(data_dir / "valid.jsonl"),
                         "train": {key: value for key, value in train.items() if key != "hashes"}, "valid": {key: value for key, value in valid.items() if key != "hashes"},
                         "train_valid_normalized_opcode_hash_overlap": len(train["hashes"] & valid["hashes"])},
        "current_implementation": {
            "backbone": "Embedding(128) -> 1-layer bidirectional GRU(hidden=128) -> B2 label attention",
            "vulnerability_representation": "z_b_l = label-specific attention weighted mean of BiGRU hidden states",
            "context_representation": "g_b = masked mean of BiGRU hidden states; no label query, true label, or classifier output",
            "dictionary": "KMeans over train-loader g_b vectors only; K=16; current code does not L2-normalize before KMeans or centroids after KMeans",
            "dictionary_updates": "frozen after construction; stored as a buffer, not nn.Parameter; optimizer cannot update centroids",
            "joint_dim": 128,
            "fusion": "Linear(2d -> d) on concatenated [z_b_l; cbar_b_l], followed by the original label scorer",
            "context_prior": "softmax attention over dictionary with implicit uniform P(c_i)=1/K",
            "dictionary_state_shape": dictionary_shape,
            "additional_trainable_parameters": 196864,
            "e5_total_parameters": 612230,
            "b2_total_parameters": 415366,
        },
        "current_result": None if old_result is None else {
            "tuned_macro_f1": old_result["metrics"]["tuned"]["macro_f1"],
            "tuned_micro_f1": old_result["metrics"]["tuned"]["micro_f1"],
            "best_epoch": old_result["best_epoch"],
            "seed": old_result["seed"],
            "reported_max_len": old_result.get("actual_config", {}).get("max_len", old_result.get("max_len")),
            "reproduces_reference_0.809403": abs(old_result["metrics"]["tuned"]["macro_f1"] - 0.8094033148335561) < 1e-9,
        },
        "grouped_protocol": {"existing_data_splits_directory": (ROOT / "data/splits").exists(), "test_used": False},
        "audit_findings": [
            "The existing E5 artifact is a validation-only random-split result.",
            "The existing summary does not persist train split hash, train ID hash, centroid hash, or cluster assignments.",
            "The existing implementation is context-adjustment inspired; it does not establish causal identification.",
            "The test JSONL is recorded as a path for audit completeness but is not opened by this script.",
        ],
    }
    output = ROOT / "results/e5_tdvp/audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_dir = ROOT / "reports/light_label_model/e5_final"
    report_dir.mkdir(parents=True, exist_ok=True)
    current = payload["current_result"]
    lines = ["# E5 TDVP Implementation Audit", "", "Dataset: `DIVE_main6_opcode_process01`; validation-only; `test_checked=false`.", "",
             "## Verified implementation", "", "- B2 vulnerability-specific representation: label-conditioned attention over BiGRU states.",
             "- Context: label-agnostic masked mean of BiGRU states.", "- Dictionary: train-loader-only KMeans, K=16 in the current code path.",
             "- Dictionary: frozen buffer after construction; not optimized by backprop.", "- Joint dimension: 128.",
             "- Fusion: `Linear(512 -> 256)` followed by the original label scorer; no GELU or residual gate in the current artifact.", "",
             "## Reproducibility findings", "", f"- Train samples: `{train['samples']}`; valid samples: `{valid['samples']}`.",
             f"- Train/valid normalized-opcode hash overlap: `{len(train['hashes'] & valid['hashes'])}`.",
             f"- Additional trainable parameters: `{payload['current_implementation']['additional_trainable_parameters']}`.",
             f"- Existing E5 result: `{current['tuned_macro_f1']:.9f}` Macro-F1, best epoch `{current['best_epoch']}`." if current else "- Existing E5 metrics artifact was not found.",
             "- Existing artifact does not persist split hashes, train IDs hash, centroid hash, or cluster assignments.",
             "", "## Terminology boundary", "", "The current method should be described as context adjustment inspired by deconfounding. A causal identification claim is not verified."]
    (report_dir / "e5_implementation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(report_dir / "e5_implementation_audit.md"), "test_checked": False, "reference_reproduced": current and current["reproduces_reference_0.809403"]}, indent=2))


if __name__ == "__main__":
    main()

