#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-all}"
CONFIG="configs/train_main6_stack_adapter.yaml"

case "$MODE" in
  pretrain)
    python scripts/pretrain_main6_stack_adapter_mlm.py --config "$CONFIG"
    ;;
  cache)
    python scripts/extract_main6_stack_adapter_features.py --config "$CONFIG" --splits train valid
    python scripts/check_main6_stack_relational_feature_cache.py --config "$CONFIG"
    ;;
  train)
    python scripts/train_main6_stack_relational.py --config "$CONFIG"
    ;;
  select)
    python scripts/select_main6_stack_relational_valid.py --config "$CONFIG"
    ;;
  all)
    bash scripts/run_main6_stack_adapter.sh pretrain
    bash scripts/run_main6_stack_adapter.sh cache
    bash scripts/run_main6_stack_adapter.sh train
    bash scripts/run_main6_stack_adapter.sh select
    ;;
  *)
    echo "usage: bash scripts/run_main6_stack_adapter.sh {pretrain|cache|train|select|all}" >&2
    exit 2
    ;;
esac
