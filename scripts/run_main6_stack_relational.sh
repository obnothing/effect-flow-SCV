#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
CONFIG="configs/train_main6_stack_relational.yaml"

case "$MODE" in
  extract)
    python scripts/extract_main6_stack_relation_cache.py --config "$CONFIG" --splits train valid
    python scripts/check_main6_stack_relation_cache.py --config "$CONFIG"
    ;;
  train)
    python scripts/train_main6_stack_relational.py --config "$CONFIG"
    ;;
  pretrain)
    python scripts/pretrain_main6_stack_aware_mlm.py --config "$CONFIG"
    ;;
  select)
    python scripts/select_main6_stack_relational_valid.py --config "$CONFIG"
    ;;
  all)
    bash scripts/run_main6_stack_relational.sh extract
    bash scripts/run_main6_stack_relational.sh pretrain
    bash scripts/run_main6_stack_relational.sh train
    bash scripts/run_main6_stack_relational.sh select
    ;;
  *)
    echo "usage: bash scripts/run_main6_stack_relational.sh {extract|train|select|all}" >&2
    exit 2
    ;;
esac
