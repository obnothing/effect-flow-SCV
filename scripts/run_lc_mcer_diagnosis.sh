#!/usr/bin/env bash
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
test "${ALLOW_TEST:-0}" != "1"
python scripts/diagnose_lc_mcer.py --config "${CONFIG:-configs/lc_mcer_diagnosis.yaml}"
