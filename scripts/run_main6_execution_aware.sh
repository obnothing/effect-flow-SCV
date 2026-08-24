#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
CONFIG="configs/train_main6_execution_aware_mil.yaml"
mkdir -p logs

case "$MODE" in
  extract)
    python scripts/extract_main6_execution_features.py --config "$CONFIG" --splits train valid
    python scripts/check_main6_execution_cache.py
    ;;
  train)
    python scripts/train_main6_execution_aware_mil.py --config "$CONFIG"
    ;;
  select)
    python scripts/select_main6_execution_aware_valid.py --config "$CONFIG"
    ;;
  all)
    bash scripts/run_main6_execution_aware.sh extract
    bash scripts/run_main6_execution_aware.sh train
    bash scripts/run_main6_execution_aware.sh select
    ;;
  *)
    echo "usage: bash scripts/run_main6_execution_aware.sh {extract|train|select|all}" >&2
    exit 2
    ;;
esac
