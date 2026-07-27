#!/usr/bin/env bash
#SBATCH --job-name=source_main6_final
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/source_main6_final_%j.out
#SBATCH --error=logs/source_main6_final_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs
ALLOW_TEST=1 bash scripts/run_dive_source_main6_graphcodebert.sh final
