#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-audit}"
CONFIG="configs/retrieval_validation.yaml"

case "$MODE" in
  audit|prepare|train)
    python scripts/run_retrieval_validation.py "$MODE" --config "$CONFIG"
    ;;
  learned)
    python scripts/train_learned_evidence_retrieval.py \
      --config configs/retrieval_learned_evidence.yaml
    ;;
  all)
    python scripts/run_retrieval_validation.py audit --config "$CONFIG"
    python scripts/run_retrieval_validation.py prepare --config "$CONFIG"
    python scripts/run_retrieval_validation.py train --config "$CONFIG"
    ;;
  *)
    echo "usage: $0 {audit|prepare|train|learned|all}" >&2
    exit 2
    ;;
esac
