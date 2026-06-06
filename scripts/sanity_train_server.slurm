#!/usr/bin/env bash
#SBATCH --job-name=correlascan_sanity
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/slurm_sanity_%j.out
#SBATCH --error=logs/slurm_sanity_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"


source ~/.bashrc
conda activate correlascan_a40

bash scripts/sanity_train_server.sh
