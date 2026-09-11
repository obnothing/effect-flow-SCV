"""Audit the current Main-6 light baseline before any AutoTune trial."""

import importlib.util
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config(path):
    value = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if value.get("base_config"):
        base = yaml.safe_load(resolve(value["base_config"]).read_text(encoding="utf-8"))
        base.update(value)
        value = base
    return value


def split_stats(cache_path, max_len):
    payload = torch.load(cache_path, map_location="cpu")
    lengths = payload["original_lengths"].numpy()
    labels = payload["labels"].numpy()
    return {"samples": int(len(lengths)), "stored_tokens": int(payload["token_ids"].numel()),
            "coverage_at_configured_max_len": float((lengths <= max_len).mean()),
            "truncated_at_configured_max_len": int((lengths > max_len).sum()),
            "p95_original_length": float(np.percentile(lengths, 95)),
            "p99_original_length": float(np.percentile(lengths, 99)),
            "label_prevalence": {name: float(labels[:, index].mean()) for index, name in enumerate(LABELS)}}


def main():
    baseline = config("configs/light_label/b2_label_attention.yaml")
    train_cache = resolve(baseline["cache_dir"]) / f"train_max{baseline['max_len']}.pt"
    valid_cache = resolve(baseline["cache_dir"]) / f"valid_max{baseline['max_len']}.pt"
    result_path = resolve("results/light_label/b2_label_attention/full/metrics.json")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    report = {
        "route": "SCVD AutoTune pre-search audit",
        "dataset": "DIVE_main6_opcode_process01",
        "protocol": "validation_only",
        "test_checked": False,
        "baseline_config": baseline,
        "train": split_stats(train_cache, int(baseline["max_len"])),
        "valid": split_stats(valid_cache, int(baseline["max_len"])),
        "baseline_result": None if result is None else {"tuned_macro_f1": result["metrics"]["tuned"]["macro_f1"], "tuned_micro_f1": result["metrics"]["tuned"]["micro_f1"], "best_epoch": result["best_epoch"], "params": result["total_params"]},
        "source_hashes": {"train_cache": sha256_file(train_cache), "valid_cache": sha256_file(valid_cache)},
        "dependencies": {"optuna_available": importlib.util.find_spec("optuna") is not None, "yaml_available": importlib.util.find_spec("yaml") is not None},
        "implementation_findings": [
            "The current light baseline uses a 1-layer bidirectional GRU in src/light_label_model.py.",
            "The existing gru_layers config field is not wired into LabelGuidedOpcodeNet and must not be tuned until fixed.",
            "The search route must keep DIVE Main-6 opcode-only scope and test locked.",
        ],
    }
    output = ROOT / "autotune/audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_dir = ROOT / "reports/autotune"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# SCVD AutoTune Pre-search Audit", "", "Dataset: `DIVE_main6_opcode_process01`; validation-only; `test_checked=false`.", "",
             "## Baseline", "", f"- Configured max length: `{baseline['max_len']}`.", f"- Train coverage: `{report['train']['coverage_at_configured_max_len']:.4f}`.", f"- Valid coverage: `{report['valid']['coverage_at_configured_max_len']:.4f}`.",
             f"- Existing tuned Macro-F1: `{report['baseline_result']['tuned_macro_f1']:.6f}`." if result else "- Existing baseline metrics were not found.",
             "", "## Blocking findings", "", "- `gru_layers` is currently a dormant configuration field; multi-layer trials are prohibited until the implementation is wired and smoke-tested.",
             f"- Optuna available in the current Python environment: `{report['dependencies']['optuna_available']}`.", "- No test file is opened by this audit."]
    (report_dir / "pre_search_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(output), "optuna_available": report["dependencies"]["optuna_available"], "test_checked": False}, indent=2))


if __name__ == "__main__":
    main()

