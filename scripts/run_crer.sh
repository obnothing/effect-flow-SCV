#!/usr/bin/env bash
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
mkdir -p logs
test "${ALLOW_TEST:-0}" != "1"
python scripts/train_crer.py --config "${CONFIG:-configs/train_crer.yaml}" "${1:-all}"
