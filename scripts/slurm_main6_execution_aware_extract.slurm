#!/bin/bash
#SBATCH --job-name=main6-exec-extract
#SBATCH --partition=gpu-l20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_execution_aware_extract_%j.out
#SBATCH --error=logs/main6_execution_aware_extract_%j.err
set -eo pipefail
source "${HOME}/.bashrc"
set -u
conda activate correlascan_a40
cd "${SLURM_SUBMIT_DIR}"
python scripts/extract_main6_execution_features.py --config configs/train_main6_execution_aware_mil.yaml --splits train valid
python scripts/check_main6_execution_cache.py --cache-dir data/features/main6_opcode_execution_aware/execution
