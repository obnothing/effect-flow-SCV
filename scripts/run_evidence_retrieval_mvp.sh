#!/usr/bin/env bash
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
test "${ALLOW_TEST:-0}" != "1"
python scripts/train_evidence_retrieval_mvp.py "${1:-all}" --config "${CONFIG:-configs/evidence_retrieval_mvp.yaml}"
