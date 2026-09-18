#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source "${HOME}/.bashrc"
conda activate pytorch-2.1.1
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p logs

mode="${1:-all}"
case "${mode}" in
  audit)
    python -u scripts/run_relevance_polarity.py audit
    ;;
  smoke)
    python -u scripts/test_relevance_polarity.py
    python -u scripts/run_relevance_polarity.py smoke
    ;;
  train)
    exec 9>logs/relevance_polarity_train.lock
    flock -n 9 || { echo 'Relevance-polarity training already running'; exit 1; }
    python -u scripts/run_relevance_polarity.py train
    ;;
  diagnose)
    python -u scripts/diagnose_relevance_polarity.py
    ;;
  all)
    python -u scripts/run_relevance_polarity.py audit
    python -u scripts/test_relevance_polarity.py
    python -u scripts/run_relevance_polarity.py smoke
    exec 9>logs/relevance_polarity_train.lock
    flock -n 9 || { echo 'Relevance-polarity training already running'; exit 1; }
    python -u scripts/run_relevance_polarity.py train
    python -u scripts/diagnose_relevance_polarity.py
    ;;
  *)
    echo "usage: $0 {audit|smoke|train|diagnose|all}" >&2
    exit 2
    ;;
esac
