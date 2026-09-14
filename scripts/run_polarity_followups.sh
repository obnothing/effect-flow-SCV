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
exec 9>logs/polarity_followups.lock
flock -n 9 || { echo 'Polarity follow-up queue already running'; exit 1; }
echo "$$" > logs/polarity_followups.pid
trap 'rm -f logs/polarity_followups.pid' EXIT
python -u scripts/test_polarity_queries.py
python -u scripts/run_polarity_followups.py
