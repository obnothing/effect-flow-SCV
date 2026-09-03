#!/usr/bin/env bash
#SBATCH --job-name=main6-evidence-mvp
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/main6_evidence_mvp_%j.out
#SBATCH --error=logs/main6_evidence_mvp_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
mkdir -p logs
test "${ALLOW_TEST:-0}" != "1"
python scripts/train_evidence_retrieval_mvp.py all --config "${CONFIG:-configs/evidence_retrieval_mvp.yaml}"
