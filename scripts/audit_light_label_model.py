"""Local environment, dataset, and sequence-length audit for LabelGuidedOpcodeNet."""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def count_lines(path):
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def scan_split(path, tokenizer, num_labels, include_lengths):
    count = 0
    ids = set()
    positives = np.zeros(num_labels, dtype=np.int64)
    lengths = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            labels = item.get("multi_labels", [])
            if len(labels) != num_labels:
                raise ValueError(f"{path}:{line_no} expected {num_labels} labels")
            count += 1
            ids.add(str(item.get("id", line_no)))
            positives += np.asarray(labels, dtype=np.int64)
            if include_lengths:
                lengths.append(len(tokenizer.tokenize(item.get("opcode", ""), add_special_tokens=False)))
    result = {"records": count, "unique_ids": len(ids), "duplicate_ids": count - len(ids), "positive_counts": positives.tolist()}
    if include_lengths:
        values = np.asarray(lengths, dtype=np.int64)
        result["length"] = {
            "mean": float(values.mean()),
            "p50": int(np.percentile(values, 50)), "p75": int(np.percentile(values, 75)),
            "p90": int(np.percentile(values, 90)), "p95": int(np.percentile(values, 95)),
            "p99": int(np.percentile(values, 99)), "max": int(values.max()),
            "coverage": {str(limit): float((values <= limit).mean()) for limit in (2048, 4096, 8192, 16384)},
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/light_label/b0_mean.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if config.get("allow_test"):
        raise ValueError("local label experiment keeps test locked")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    data_dir = resolve(config["data_dir"])
    train = scan_split(data_dir / "train.jsonl", tokenizer, config["num_labels"], True)
    valid = scan_split(data_dir / "valid.jsonl", tokenizer, config["num_labels"], True)
    test_count = count_lines(data_dir / "test.jsonl")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    report = {
        "route": config["route_name"], "dataset": "DIVE_main6_opcode_process01", "seed": config["seed"],
        "environment": {"os": platform.platform(), "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "cuda_available": torch.cuda.is_available(), "gpu": gpu_name, "vram_gib": vram},
        "paths": {"data": str(data_dir), "vocab": str(resolve(config["vocab_path"]))},
        "labels": config["label_names"], "splits": {"train": train, "valid": valid, "test_records": test_count},
        "max_len_selection": {"selected": 8192, "selection_source": "train length distribution only",
                              "train_coverage": train["length"]["coverage"]["8192"],
                              "train_truncation_ratio": 1.0 - train["length"]["coverage"]["8192"]},
        "estimated_memory": "Embedding and one-layer BiGRU use packed sequences; batch 4 is tested first on the 8 GiB GPU.",
        "test_policy": "Only the number of test records was counted. Test opcode and labels were not used for max_len selection, training, or threshold tuning.",
        "test_checked": False,
    }
    output = resolve(config["report_dir"]) / "local_model_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = ["# Local Label Model Audit", "", f"- Dataset: `{data_dir}`", f"- Train/valid/test: {train['records']} / {valid['records']} / {test_count}",
          f"- Labels: {', '.join(config['label_names'])}", f"- Tokenizer: `{resolve(config['vocab_path'])}`", f"- GPU: {gpu_name}", f"- VRAM: {vram:.2f} GiB",
          f"- PyTorch/CUDA: {torch.__version__} / {torch.version.cuda}", f"- Train length p50/p75/p90/p95/p99/max: {train['length']['p50']} / {train['length']['p75']} / {train['length']['p90']} / {train['length']['p95']} / {train['length']['p99']} / {train['length']['max']}",
          f"- Coverage 4096/8192/16384: {train['length']['coverage']['4096']:.4f} / {train['length']['coverage']['8192']:.4f} / {train['length']['coverage']['16384']:.4f}",
          f"- Selected max_len: 8192; train truncation ratio: {report['max_len_selection']['train_truncation_ratio']:.4f}", "", report["test_policy"]]
    (output.parent / "local_model_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gpu": gpu_name, "train": train["records"], "valid": valid["records"], "test_checked": False}, indent=2))


if __name__ == "__main__":
    main()
