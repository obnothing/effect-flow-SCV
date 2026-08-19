#!/usr/bin/env bash
#SBATCH --job-name=main6_lse_mil
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main6_lse_mil_%j.out
#SBATCH --error=logs/main6_lse_mil_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
set -u
mkdir -p logs
bash scripts/run_main6_opcode_lse_mil.sh train
