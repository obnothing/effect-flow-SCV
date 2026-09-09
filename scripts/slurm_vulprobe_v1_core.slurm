#!/usr/bin/env bash
#SBATCH --job-name=vulprobe-v1-core
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vulprobe_v1_core_%j.out
#SBATCH --error=logs/vulprobe_v1_core_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
mkdir -p logs
bash scripts/run_vulprobe_v1.sh core

