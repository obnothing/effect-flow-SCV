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
exec 9>logs/p11_width_study.lock
flock -n 9 || { echo 'P11 width study is already running'; exit 1; }
python -u scripts/test_p11_width_study.py
python -u scripts/test_polarity_queries.py
python -u scripts/run_p11_width_study.py
