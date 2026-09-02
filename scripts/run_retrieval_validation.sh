#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-audit}"
CONFIG="configs/retrieval_validation.yaml"

case "$MODE" in
  audit|prepare|train)
    python scripts/run_retrieval_validation.py "$MODE" --config "$CONFIG"
    ;;
  all)
    python scripts/run_retrieval_validation.py audit --config "$CONFIG"
    python scripts/run_retrieval_validation.py prepare --config "$CONFIG"
    python scripts/run_retrieval_validation.py train --config "$CONFIG"
    ;;
  *)
    echo "usage: $0 {audit|prepare|train|all}" >&2
    exit 2
    ;;
esac
