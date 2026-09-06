#!/usr/bin/env bash
#SBATCH --job-name=main6-m0-audit
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/main6_m0_error_audit_%j.out
#SBATCH --error=logs/main6_m0_error_audit_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
mkdir -p logs
test "${ALLOW_TEST:-0}" != "1"
python scripts/audit_main6_m0_errors.py \
  --config "${CONFIG:-configs/evidence_retrieval_mvp.yaml}" \
  --output "${OUTPUT:-results/main6_random_090_mlm8/error_audit/m0_error_audit.json}"
