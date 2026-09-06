#!/usr/bin/env bash
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
test "${ALLOW_TEST:-0}" != "1"
python scripts/audit_main6_m0_errors.py \
  --config "${CONFIG:-configs/evidence_retrieval_mvp.yaml}" \
  --output "${OUTPUT:-results/main6_random_090_mlm8/error_audit/m0_error_audit.json}"
