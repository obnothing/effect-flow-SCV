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
exec 9>logs/encoder_study.lock
flock -n 9 || { echo 'Encoder study already running'; exit 1; }
python -u scripts/test_encoder_study.py
python -u scripts/test_polarity_queries.py
python -u scripts/smoke_encoder_study.py
python -u scripts/run_encoder_study.py
