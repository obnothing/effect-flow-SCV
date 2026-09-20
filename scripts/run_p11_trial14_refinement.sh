#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source "${HOME}/.bashrc"
conda activate pytorch-2.1.1
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p logs
exec 9>logs/p11_trial14_refinement.lock
flock -n 9 || { echo 'Trial 14 refinement queue is already running'; exit 1; }
python -u scripts/test_p11_hparam_stage1.py
python -u scripts/test_polarity_queries.py
python -u scripts/run_p11_trial14_refinement.py
