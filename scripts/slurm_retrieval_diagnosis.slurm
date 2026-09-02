#!/usr/bin/env bash
#SBATCH --job-name=main6-retrieval-diagnosis
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main6_retrieval_diagnosis_%j.out
#SBATCH --error=logs/main6_retrieval_diagnosis_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
mkdir -p logs
test "${ALLOW_TEST:-0}" != "1"
python scripts/diagnose_retrieval.py \
  --config "${CONFIG:-configs/retrieval_diagnosis.yaml}"
