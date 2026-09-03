#!/usr/bin/env bash
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
test "${ALLOW_TEST:-0}" != "1"
python scripts/diagnose_evidence_definition_final.py --config "${CONFIG:-configs/evidence_definition_final.yaml}"
