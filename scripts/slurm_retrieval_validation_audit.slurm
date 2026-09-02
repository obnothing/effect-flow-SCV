#!/usr/bin/env bash
#SBATCH --job-name=main6-retrieval-audit
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_retrieval_audit_%j.out
#SBATCH --error=logs/main6_retrieval_audit_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
mkdir -p logs
python scripts/run_retrieval_validation.py audit --config configs/retrieval_validation.yaml
