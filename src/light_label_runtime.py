"""Safe runtime configuration merging for light-label experiments."""

import json


RUNTIME_OVERRIDE_KEYS = {"batch_size", "gradient_accumulation_steps"}
RUNTIME_CHECKED_CONFIG_KEYS = {
    "embedding_dim", "gru_hidden_size", "gru_layers", "max_len", "bidirectional",
    "num_labels", "variant", "learning_rate", "weight_decay", "weighted_bce",
    "pos_weight_mode", "max_pos_weight", "amp", "epochs", "early_stopping_patience", "seed",
}


def merge_runtime_config(config, runtime_path):
    """Merge only hardware overrides and reject stale experiment settings."""
    merged = dict(config)
    runtime_path = runtime_path if hasattr(runtime_path, "exists") else None
    if runtime_path is None or not runtime_path.exists():
        return merged
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    for key in RUNTIME_CHECKED_CONFIG_KEYS:
        if key in runtime and key in merged and runtime[key] != merged[key]:
            raise ValueError(
                f"runtime/config mismatch for {key}: YAML={merged[key]!r}, "
                f"runtime={runtime[key]!r}; rerun the runtime resolver"
            )
    for key in RUNTIME_OVERRIDE_KEYS:
        if key in runtime:
            merged[key] = runtime[key]
    return merged
