#!/bin/bash
#SBATCH --job-name=main6-exec-train
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/main6_execution_aware_train_%j.out
#SBATCH --error=logs/main6_execution_aware_train_%j.err
set -euo pipefail
source "${HOME}/.bashrc"
conda activate correlascan_a40
cd "${SLURM_SUBMIT_DIR}"
python scripts/train_main6_execution_aware_mil.py --config configs/train_main6_execution_aware_mil.yaml
