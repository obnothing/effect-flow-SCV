#!/bin/bash
#SBATCH --job-name=main6-exec-select
#SBATCH --partition=gpu-a30
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/main6_execution_aware_select_%j.out
#SBATCH --error=logs/main6_execution_aware_select_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
python scripts/select_main6_execution_aware_valid.py --config configs/train_main6_execution_aware_mil.yaml
