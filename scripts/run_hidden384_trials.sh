#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source "${HOME}/.bashrc"
conda activate pytorch-2.1.1
set -u
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs
exec 9>logs/hidden384_trials.lock
flock -n 9 || { echo 'Another trial queue is running'; exit 1; }
python -u scripts/test_hidden384_trials.py
python -u scripts/run_hidden384_trials.py --smoke
python -u scripts/run_hidden384_trials.py --start-index "${START_INDEX:-0}"
